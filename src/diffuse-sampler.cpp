#include "diffuse-sampler.h"
#include "diffuse-graph.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <chrono>

// ═══════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════

// Token transfer schedule: how many tokens to commit per step
// Matches dFactory's _get_num_transfer_tokens(block_length, steps)
static std::vector<int> get_num_transfer_tokens(int block_length, int steps) {
    if (steps == 0) return {};
    std::vector<int> schedule(steps);
    int base = block_length / steps;
    int remainder = block_length % steps;
    for (int i = 0; i < steps; i++) {
        schedule[i] = base + (i < remainder ? 1 : 0);
    }
    return schedule;
}

// Sample a token and its probability from logits
// temperature=0 → argmax (deterministic)
static void sample_token(const float * logits, int n_vocab, float temperature,
                         uint32_t seed, std::mt19937 & rng,
                         int & token_id, float & token_prob) {
    if (temperature <= 0.0f) {
        // Argmax
        int best = 0;
        float best_val = logits[0];
        for (int v = 1; v < n_vocab; v++) {
            if (logits[v] > best_val) {
                best_val = logits[v];
                best = v;
            }
        }
        token_id = best;
        // Compute probability via softmax
        float max_val = best_val;
        float sum = 0.0f;
        for (int v = 0; v < n_vocab; v++) {
            sum += expf(logits[v] - max_val);
        }
        token_prob = 1.0f / sum;  // prob of argmax = exp(0) / sum = 1/sum
    } else {
        // Temperature sampling
        float max_val = logits[0];
        for (int v = 1; v < n_vocab; v++) {
            if (logits[v] > max_val) max_val = logits[v];
        }
        std::vector<float> probs(n_vocab);
        float sum = 0.0f;
        for (int v = 0; v < n_vocab; v++) {
            probs[v] = expf((logits[v] - max_val) / temperature);
            sum += probs[v];
        }
        for (int v = 0; v < n_vocab; v++) probs[v] /= sum;

        std::discrete_distribution<int> dist(probs.begin(), probs.end());
        token_id = dist(rng);
        token_prob = probs[token_id];
    }
}

// Compute softmax probability of the argmax token
static float softmax_max_prob(const float * logits, int n_vocab) {
    float max_val = logits[0];
    for (int v = 1; v < n_vocab; v++) {
        if (logits[v] > max_val) max_val = logits[v];
    }
    float sum = 0.0f;
    for (int v = 0; v < n_vocab; v++) {
        sum += expf(logits[v] - max_val);
    }
    return 1.0f / sum;
}

// ═══════════════════════════════════════════════════════════════
// Block diffusion generation (LLaDA2.X)
// ═══════════════════════════════════════════════════════════════
//
// Generation flow (matching dFactory generate()):
// 1. Pad prompt + gen_length to be divisible by block_length
// 2. Fill generation positions with mask_id
// 3. For each block (left to right):
//    a. For each denoising step:
//       - Forward pass on tokens[:current_window_end]
//       - Get logits for the current block
//       - Sample x0 and p(x0) for each masked position
//       - Transfer tokens: commit high-confidence first (p > threshold)
//         or top-k by confidence
//       - Apply Levenshtein editing if enabled
//       - Early stop if no masks remain
//    b. Check for EOS

static std::vector<int32_t> generate_block_diffusion(
        diffuse_context * ctx,
        const std::vector<int32_t> & prompt_tokens,
        int n_generate,
        const diffuse_sampler_params & params,
        diffuse_step_callback callback) {

    const auto & model = *ctx->model;
    const auto & hp    = model.hparams;
    const int mask_id  = hp.mask_token_id;
    const int eos_id   = hp.eos_token_id;
    const int n_vocab  = hp.n_vocab;
    const int block_len = (int)hp.block_length;
    const int del_id    = (int)model.delete_token_id;
    const int ins_id    = (int)model.split_token_id;
    const bool editing  = params.enable_editing && del_id > 0 && ins_id > 0;

    int prompt_len = (int)prompt_tokens.size();

    // Blocks tile from absolute position 0 (matches the reference generate(),
    // and keeps every fed window a multiple of block_len, which the MoE block
    // router requires).
    int num_blocks   = (prompt_len + n_generate + block_len - 1) / block_len;
    int total_length = num_blocks * block_len;

    // Build token sequence: prompt + mask tokens for generation
    std::vector<int32_t> seq(total_length, mask_id);
    for (int i = 0; i < prompt_len && i < total_length; i++) {
        seq[i] = prompt_tokens[i];
    }

    // Floor, not ceil: the block that straddles the end of the prompt still has
    // mask positions to denoise. Rounding up skips it and leaves them masked
    // forever, which poisons the context for every later block.
    int prefill_blocks = prompt_len / block_len;

    // Denoising schedule
    int steps = std::min(params.n_steps, block_len);
    auto transfer_schedule = get_num_transfer_tokens(block_len, steps);

    std::mt19937 rng(params.seed);

    DIFFUSE_LOG("block diffusion: %d blocks (prefill=%d), prompt=%d, block_len=%d, steps=%d, threshold=%.2f, editing=%s",
                num_blocks, prefill_blocks, prompt_len, block_len, steps, params.threshold,
                editing ? "yes" : "no");

    using clk = std::chrono::steady_clock;
    auto gen_start = clk::now();
    double total_forward_ms = 0.0;

    // Process blocks left to right
    for (int blk = prefill_blocks; blk < num_blocks; blk++) {
        int block_start = blk * block_len;
        int cur_window_end = std::min(block_start + block_len, total_length);

        // Token buffer for this window (grows as we process more blocks)
        // We only feed tokens up to the current block's end
        std::vector<float> logits(cur_window_end * n_vocab);

        for (int step = 0; step < steps; step++) {
            // Check if there are still masked tokens in the current block
            int block_offset = block_start;
            int block_n = std::min(block_len, total_length - block_start);
            bool any_masked = false;
            for (int i = 0; i < block_n; i++) {
                if (seq[block_offset + i] == mask_id) { any_masked = true; break; }
            }
            if (!any_masked) break;

            // Forward pass
            auto t0 = clk::now();
            if (!diffuse_forward_moe(ctx, seq.data(), cur_window_end, logits.data())) {
                DIFFUSE_DIE("forward pass failed at block %d, step %d", blk, step);
            }
            auto t1 = clk::now();
            total_forward_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

            // Sample tokens for masked positions in the current block
            struct Candidate {
                int pos;       // absolute position in seq
                int token;     // sampled token
                float prob;    // probability of sampled token
            };
            std::vector<Candidate> candidates;

            for (int i = 0; i < block_n; i++) {
                int abs_pos = block_offset + i;
                if (seq[abs_pos] != mask_id) continue;

                const float * logit_row = logits.data() + (size_t)abs_pos * n_vocab;
                int sampled_token;
                float sampled_prob;
                sample_token(logit_row, n_vocab, params.temperature, params.seed, rng,
                             sampled_token, sampled_prob);

                // For low_confidence: use the sampled probability as confidence
                // For random: assign random confidence
                float confidence;
                if (params.remasking == diffuse_remasking::RANDOM) {
                    confidence = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
                } else {
                    confidence = sampled_prob;
                }

                candidates.push_back({abs_pos, sampled_token, confidence});
            }

            if (candidates.empty()) break;

            // Determine how many tokens to transfer this step
            int num_to_transfer = transfer_schedule[step];

            // Sort candidates by confidence (highest first)
            std::sort(candidates.begin(), candidates.end(),
                      [](const Candidate & a, const Candidate & b) {
                          return a.prob > b.prob;
                      });

            // Transfer: commit tokens above threshold, or top-k
            int transferred = 0;
            for (const auto & c : candidates) {
                if (transferred >= num_to_transfer && c.prob < params.threshold) {
                    break;
                }
                if (transferred >= num_to_transfer) {
                    // Already transferred enough, but check if more exceed threshold
                    if (c.prob < params.threshold) break;
                }
                seq[c.pos] = c.token;
                transferred++;
            }

            // Ensure at least 1 token transferred if there are masked positions
            if (transferred == 0 && !candidates.empty()) {
                seq[candidates[0].pos] = candidates[0].token;
                transferred = 1;
            }

            // Levenshtein editing: apply DELETE and INSERT
            if (editing) {
                for (int i = 0; i < block_n; i++) {
                    int abs_pos = block_offset + i;
                    if (seq[abs_pos] == del_id) {
                        // DELETE: mark for deletion (shift left later)
                        // For simplicity, replace with mask and handle in post-processing
                        seq[abs_pos] = mask_id;
                    } else if (seq[abs_pos] == ins_id) {
                        // INSERT: create a new mask position
                        // Replace INSERT with mask (creates an editable slot)
                        seq[abs_pos] = mask_id;
                    }
                }
            }

            if (callback) {
                callback(blk, num_blocks, step + 1, steps, seq);
            }
        }

        // Check for EOS in completed block. LLaDA2 ends an assistant turn with
        // <|role_end|>, which is a separate id from <|endoftext|>, so both stop.
        if (params.eos_early_stop && (eos_id > 0 || params.stop_token_2 >= 0)) {
            for (int i = block_start; i < cur_window_end; i++) {
                if (seq[i] == eos_id || (params.stop_token_2 >= 0 && seq[i] == params.stop_token_2)) {
                    // Check all positions before EOS are unmasked
                    bool all_clear = true;
                    for (int j = prompt_len; j < i; j++) {
                        if (seq[j] == mask_id) { all_clear = false; break; }
                    }
                    if (all_clear) {
                        // Return up to and including EOS
                        auto gen_end = clk::now();
                        double total_ms = std::chrono::duration<double, std::milli>(gen_end - gen_start).count();
                        DIFFUSE_LOG("generation complete (EOS): %d tokens in %.0fms (%.1f tok/s, fwd=%.0fms)",
                                    i - prompt_len + 1, total_ms,
                                    1000.0 * (i - prompt_len + 1) / total_ms, total_forward_ms);
                        return std::vector<int32_t>(seq.begin() + prompt_len, seq.begin() + i + 1);
                    }
                }
            }
        }
    }

    // ── Finalize ───────────────────────────────────────────────
    auto gen_end = clk::now();
    double total_ms = std::chrono::duration<double, std::milli>(gen_end - gen_start).count();
    int actual_gen = std::min(n_generate, total_length - prompt_len);

    DIFFUSE_LOG("generation complete: %d tokens in %.0fms (%.1f tok/s, fwd=%.0fms)",
                actual_gen, total_ms, 1000.0 * actual_gen / total_ms, total_forward_ms);

    // Extract generated tokens, filtering out mask tokens
    std::vector<int32_t> result;
    result.reserve(n_generate);
    for (int i = prompt_len; i < prompt_len + n_generate && i < total_length; i++) {
        if (seq[i] != mask_id) {
            result.push_back(seq[i]);
        }
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════
// Flat diffusion generation (LLaDA 1.0 / Dream)
// ═══════════════════════════════════════════════════════════════
//
// Generates all tokens simultaneously with full bidirectional attention.
// Iterative unmasking across the full sequence.

static std::vector<int32_t> generate_flat_diffusion(
        diffuse_context * ctx,
        const std::vector<int32_t> & prompt_tokens,
        int n_generate,
        const diffuse_sampler_params & params,
        diffuse_step_callback callback) {

    const auto & hp = ctx->model->hparams;
    const int mask_id  = hp.mask_token_id;
    const int n_vocab  = hp.n_vocab;

    int prompt_len = (int)prompt_tokens.size();
    int total_len = prompt_len + n_generate;

    // Build sequence: prompt + mask tokens
    std::vector<int32_t> seq = prompt_tokens;
    seq.resize(total_len, mask_id);

    std::vector<bool> is_masked(total_len, false);
    for (int i = prompt_len; i < total_len; i++) is_masked[i] = true;
    int n_masked = n_generate;

    std::vector<float> logits(total_len * n_vocab);
    std::mt19937 rng(params.seed);

    using clk = std::chrono::steady_clock;
    auto gen_start = clk::now();

    for (int step = 0; step < params.n_steps && n_masked > 0; step++) {
        // Forward pass
        auto t0 = clk::now();
        if (!diffuse_forward_dense(ctx, seq.data(), total_len, logits.data())) {
            DIFFUSE_DIE("forward pass failed at step %d", step);
        }
        auto t1 = clk::now();
        double fwd_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        // Sample candidates for masked positions
        struct Candidate { int pos; int token; float confidence; };
        std::vector<Candidate> candidates;

        for (int i = 0; i < total_len; i++) {
            if (!is_masked[i]) continue;

            const float * logit_row = logits.data() + (size_t)i * n_vocab;
            int sampled_token;
            float sampled_prob;
            sample_token(logit_row, n_vocab, params.temperature, params.seed, rng,
                         sampled_token, sampled_prob);

            float confidence = (params.remasking == diffuse_remasking::RANDOM)
                ? std::uniform_real_distribution<float>(0.0f, 1.0f)(rng)
                : sampled_prob;

            candidates.push_back({i, sampled_token, confidence});
        }

        // Determine how many to unmask this step
        float t0f = (float)step / params.n_steps;
        float t1f = (float)(step + 1) / params.n_steps;
        float frac;
        if (params.schedule == diffuse_schedule::COSINE) {
            float cos0 = cosf(t0f * M_PI * 0.5f);
            float cos1 = cosf(t1f * M_PI * 0.5f);
            frac = (cos0 - cos1) / cos0;
        } else {
            frac = 1.0f / (params.n_steps - step);
        }
        int n_unmask = std::max(1, (int)roundf(frac * n_masked));
        n_unmask = std::min(n_unmask, (int)candidates.size());

        // Sort by confidence
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate & a, const Candidate & b) {
                      return a.confidence > b.confidence;
                  });

        // Unmask top candidates
        for (int j = 0; j < n_unmask; j++) {
            seq[candidates[j].pos] = candidates[j].token;
            is_masked[candidates[j].pos] = false;
            n_masked--;
        }

        DIFFUSE_LOG("  step %d/%d: unmasked %d, %d remaining (fwd=%.1fms)",
                    step + 1, params.n_steps, n_unmask, n_masked, fwd_ms);

        if (callback) {
            callback(0, 1, step + 1, params.n_steps, seq);
        }
    }

    auto gen_end = clk::now();
    double total_ms = std::chrono::duration<double, std::milli>(gen_end - gen_start).count();
    DIFFUSE_LOG("generation complete: %d tokens in %.0fms (%.1f tok/s)",
                n_generate, total_ms, 1000.0 * n_generate / total_ms);

    return std::vector<int32_t>(seq.begin() + prompt_len, seq.end());
}

// ═══════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════

std::vector<int32_t> diffuse_sample(
        diffuse_context * ctx,
        const std::vector<int32_t> & prompt_tokens,
        int n_generate,
        const diffuse_sampler_params & params,
        diffuse_step_callback callback) {

    const auto & model = *ctx->model;

    if (model.arch == diffuse_arch::LLaDA2_MoE && model.hparams.block_length > 0) {
        return generate_block_diffusion(ctx, prompt_tokens, n_generate, params, callback);
    } else {
        return generate_flat_diffusion(ctx, prompt_tokens, n_generate, params, callback);
    }
}

std::vector<int32_t> diffuse_generate(
        diffuse_context * ctx,
        const std::vector<int32_t> & prompt_tokens,
        int n_generate,
        const diffuse_sampler_params & params,
        diffuse_step_callback callback) {
    return diffuse_sample(ctx, prompt_tokens, n_generate, params, callback);
}
