#include "diffuse-tokenizer.h"

#include <algorithm>
#include <set>
#include <climits>
#include <cstring>
#include <sstream>

// ═══════════════════════════════════════════════════════════════
// Byte ↔ Unicode mapping (GPT-2 style)
// ═══════════════════════════════════════════════════════════════

void diffuse_tokenizer::init_byte_tables() {
    // Build the GPT-2 byte_to_unicode mapping:
    // Printable ranges (33-126, 161-172, 174-255) map to themselves.
    // All other bytes map to 256+n in sequence.
    std::set<int> printable;
    for (int b = 33; b <= 126; b++) printable.insert(b);
    for (int b = 161; b <= 172; b++) printable.insert(b);
    for (int b = 174; b <= 255; b++) printable.insert(b);

    // Initialize reverse map
    for (int i = 0; i < 512; i++) unicode_to_byte_map[i] = -1;

    int n = 0;
    for (int b = 0; b < 256; b++) {
        int cp;
        if (printable.count(b)) {
            cp = b;
        } else {
            cp = 256 + n;
            n++;
        }
        byte_to_unicode_str[b] = codepoint_to_utf8(cp);
        if (cp < 512) unicode_to_byte_map[cp] = b;
    }
}

std::string diffuse_tokenizer::codepoint_to_utf8(int cp) {
    std::string result;
    if (cp <= 0x7F) {
        result += (char)(unsigned char)cp;
    } else if (cp <= 0x7FF) {
        result += (char)(0xC0 | (cp >> 6));
        result += (char)(0x80 | (cp & 0x3F));
    } else if (cp <= 0xFFFF) {
        result += (char)(0xE0 | (cp >> 12));
        result += (char)(0x80 | ((cp >> 6) & 0x3F));
        result += (char)(0x80 | (cp & 0x3F));
    } else {
        result += (char)(0xF0 | (cp >> 18));
        result += (char)(0x80 | ((cp >> 12) & 0x3F));
        result += (char)(0x80 | ((cp >> 6) & 0x3F));
        result += (char)(0x80 | (cp & 0x3F));
    }
    return result;
}

// Extract first UTF-8 code point from a string
static int utf8_to_codepoint(const std::string & s, int & consumed) {
    if (s.empty()) { consumed = 0; return 0; }
    unsigned char c = (unsigned char)s[0];
    if (c < 0x80) {
        consumed = 1;
        return c;
    } else if ((c & 0xE0) == 0xC0 && s.size() >= 2) {
        consumed = 2;
        return ((c & 0x1F) << 6) | ((unsigned char)s[1] & 0x3F);
    } else if ((c & 0xF0) == 0xE0 && s.size() >= 3) {
        consumed = 3;
        return ((c & 0x0F) << 12) | (((unsigned char)s[1] & 0x3F) << 6) | ((unsigned char)s[2] & 0x3F);
    } else if (s.size() >= 4) {
        consumed = 4;
        return ((c & 0x07) << 18) | (((unsigned char)s[1] & 0x3F) << 12) |
               (((unsigned char)s[2] & 0x3F) << 6) | ((unsigned char)s[3] & 0x3F);
    }
    consumed = 1;
    return c;
}

std::vector<std::string> diffuse_tokenizer::utf8_split(const std::string & s) {
    std::vector<std::string> result;
    size_t i = 0;
    while (i < s.size()) {
        int consumed = 1;
        unsigned char c = (unsigned char)s[i];
        if (c >= 0xF0) consumed = 4;
        else if (c >= 0xE0) consumed = 3;
        else if (c >= 0xC0) consumed = 2;
        size_t len = std::min((size_t)consumed, s.size() - i);
        result.push_back(s.substr(i, len));
        i += len;
    }
    return result;
}

// ═══════════════════════════════════════════════════════════════
// Load vocabulary from GGUF
// ═══════════════════════════════════════════════════════════════

bool diffuse_tokenizer::load_from_gguf(struct gguf_context * gctx) {
    init_byte_tables();

    // Tokenizer model name
    {
        int64_t kid = gguf_find_key(gctx, "tokenizer.ggml.model");
        if (kid >= 0) {
            model_name = gguf_get_val_str(gctx, kid);
        }
    }

    // Tokens array (required)
    int64_t tokens_kid = gguf_find_key(gctx, "tokenizer.ggml.tokens");
    if (tokens_kid < 0) {
        DIFFUSE_LOG("tokenizer: no tokenizer.ggml.tokens found, tokenizer disabled");
        return false;
    }

    size_t n_tokens = gguf_get_arr_n(gctx, tokens_kid);
    id_to_token.resize(n_tokens);
    id_to_type.resize(n_tokens, DIFFUSE_TOKEN_TYPE_NORMAL);
    id_to_score.resize(n_tokens, 0.0f);
    token_to_id.reserve(n_tokens * 2);

    for (size_t i = 0; i < n_tokens; i++) {
        const char * tok_str = gguf_get_arr_str(gctx, tokens_kid, i);
        id_to_token[i] = tok_str ? tok_str : "";
        token_to_id[id_to_token[i]] = (int)i;
    }

    // Token types (optional)
    int64_t types_kid = gguf_find_key(gctx, "tokenizer.ggml.token_type");
    if (types_kid >= 0) {
        const int32_t * types = (const int32_t *)gguf_get_arr_data(gctx, types_kid);
        size_t n_types = gguf_get_arr_n(gctx, types_kid);
        for (size_t i = 0; i < n_types && i < n_tokens; i++) {
            id_to_type[i] = (diffuse_token_type)types[i];
        }
    }

    // Scores (optional)
    int64_t scores_kid = gguf_find_key(gctx, "tokenizer.ggml.scores");
    if (scores_kid >= 0) {
        const float * scores = (const float *)gguf_get_arr_data(gctx, scores_kid);
        size_t n_scores = gguf_get_arr_n(gctx, scores_kid);
        for (size_t i = 0; i < n_scores && i < n_tokens; i++) {
            id_to_score[i] = scores[i];
        }
    }

    // BPE merges (for GPT-2 style tokenizers)
    int64_t merges_kid = gguf_find_key(gctx, "tokenizer.ggml.merges");
    if (merges_kid >= 0) {
        size_t n_merges = gguf_get_arr_n(gctx, merges_kid);
        for (size_t i = 0; i < n_merges; i++) {
            const char * merge_str = gguf_get_arr_str(gctx, merges_kid, i);
            if (!merge_str) continue;
            // Parse "A B" format
            std::string s(merge_str);
            auto space_pos = s.find(' ');
            if (space_pos != std::string::npos) {
                std::string a = s.substr(0, space_pos);
                std::string b = s.substr(space_pos + 1);
                bpe_ranks[{a, b}] = (int)i;
            }
        }
        DIFFUSE_LOG("tokenizer: loaded %zu merges", n_merges);
    }

    // Special token IDs
    auto get_u32_key = [&](const char * key, int32_t def) -> int32_t {
        int64_t kid = gguf_find_key(gctx, key);
        if (kid >= 0) return (int32_t)gguf_get_val_u32(gctx, kid);
        return def;
    };
    bos_id  = get_u32_key("tokenizer.ggml.bos_token_id", -1);
    eos_id  = get_u32_key("tokenizer.ggml.eos_token_id", -1);
    unk_id  = get_u32_key("tokenizer.ggml.unknown_token_id", -1);
    pad_id  = get_u32_key("tokenizer.ggml.padding_token_id", -1);
    mask_id = get_u32_key("tokenizer.ggml.mask_token_id", -1);

    // Flags
    int64_t add_bos_kid = gguf_find_key(gctx, "tokenizer.ggml.add_bos_token");
    if (add_bos_kid >= 0) add_bos = gguf_get_val_bool(gctx, add_bos_kid);
    int64_t add_eos_kid = gguf_find_key(gctx, "tokenizer.ggml.add_eos_token");
    if (add_eos_kid >= 0) add_eos = gguf_get_val_bool(gctx, add_eos_kid);

    // Chat template (stored as a string in GGUF)
    int64_t ct_kid = gguf_find_key(gctx, "tokenizer.chat_template");
    if (ct_kid >= 0) {
        chat_template = gguf_get_val_str(gctx, ct_kid);
    }

    DIFFUSE_LOG("tokenizer: %zu tokens, model=%s, bos=%d, eos=%d, unk=%d, mask=%d, add_bos=%d, add_eos=%d",
                n_tokens, model_name.c_str(),
                bos_id, eos_id, unk_id, mask_id, add_bos, add_eos);

    initialized = true;
    return true;
}

// ═══════════════════════════════════════════════════════════════
// Pre-tokenization (GPT-2 regex approximation)
// ═══════════════════════════════════════════════════════════════

static inline bool is_ascii_alpha(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}
static inline bool is_ascii_digit(unsigned char c) {
    return c >= '0' && c <= '9';
}
static inline bool is_ascii_space(unsigned char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f';
}

std::vector<std::string> diffuse_tokenizer::pre_tokenize(const std::string & text) const {
    // Approximate GPT-2 regex:
    //   '(s|t|re|ve|m|ll|d)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    //
    // For C++ without Unicode regex, we classify each UTF-8 code point:
    //   - ASCII letters → LETTER
    //   - ASCII digits → DIGIT
    //   - Whitespace → SPACE
    //   - Everything else (including multi-byte UTF-8) → OTHER

    enum CharClass { LETTER, DIGIT, SPACE, OTHER };

    std::vector<std::string> result;
    size_t i = 0;
    size_t len = text.size();

    while (i < len) {
        unsigned char c = (unsigned char)text[i];

        // Determine code point length
        int cp_len = 1;
        if (c >= 0xF0) cp_len = 4;
        else if (c >= 0xE0) cp_len = 3;
        else if (c >= 0xC0) cp_len = 2;
        if (i + cp_len > len) cp_len = len - i;

        // Classify
        CharClass cls;
        if (cp_len > 1) {
            // Non-ASCII → OTHER (CJK, emoji, etc.)
            cls = OTHER;
        } else if (is_ascii_alpha(c)) {
            cls = LETTER;
        } else if (is_ascii_digit(c)) {
            cls = DIGIT;
        } else if (is_ascii_space(c)) {
            cls = SPACE;
        } else if (c == '\'') {
            // Handle contractions: 's 't 're 've 'm 'll 'd
            // Check if preceded by a letter (common case)
            cls = OTHER; // treat apostrophe as OTHER initially
        } else {
            cls = OTHER;
        }

        if (cls == SPACE) {
            // Collect whitespace run
            size_t start = i;
            while (i < len && cp_len == 1 && is_ascii_space((unsigned char)text[i])) {
                i++;
            }
            result.push_back(text.substr(start, i - start));
        } else if (cls == LETTER) {
            // " ?\p{L}+": optional leading space + letters
            size_t start = i;
            // Check for leading space already consumed; this handles inline
            while (i < len) {
                unsigned char ch = (unsigned char)text[i];
                int cl = 1;
                if (ch >= 0xC0) { cl = (ch >= 0xF0) ? 4 : (ch >= 0xE0) ? 3 : 2; }
                if (cl == 1 && is_ascii_alpha(ch)) {
                    i++;
                } else {
                    break;
                }
            }
            result.push_back(text.substr(start, i - start));
        } else if (cls == DIGIT) {
            // " ?\p{N}+": optional leading space + digits (max 3 for Qwen2)
            size_t start = i;
            int digit_count = 0;
            while (i < len && is_ascii_digit((unsigned char)text[i]) && digit_count < 3) {
                i++;
                digit_count++;
            }
            result.push_back(text.substr(start, i - start));
            // Continue with remaining digits if more than 3
        } else {
            // OTHER: " ?[^\s\p{L}\p{N}]+": optional leading space + other chars
            size_t start = i;
            while (i < len) {
                unsigned char ch = (unsigned char)text[i];
                int cl = 1;
                if (ch >= 0xC0) { cl = (ch >= 0xF0) ? 4 : (ch >= 0xE0) ? 3 : 2; }

                if (cl > 1) {
                    // Non-ASCII byte → part of OTHER group
                    i += cl;
                } else if (!is_ascii_alpha(ch) && !is_ascii_digit(ch) && !is_ascii_space(ch)) {
                    i++;
                } else {
                    break;
                }
            }
            result.push_back(text.substr(start, i - start));
        }
    }

    // Post-process: handle leading spaces by attaching them to the next token
    // (GPT-2 style: " word" becomes one pre-token)
    std::vector<std::string> final_result;
    for (size_t k = 0; k < result.size(); k++) {
        if (result[k].size() == 1 && result[k][0] == ' ') {
            // Standalone space: try to attach to next token
            if (k + 1 < result.size() && result[k+1][0] != ' ' &&
                result[k+1][0] != '\t' && result[k+1][0] != '\n') {
                final_result.push_back(" " + result[k+1]);
                k++; // skip next
            } else {
                final_result.push_back(result[k]);
            }
        } else if (result[k].size() > 1 && result[k][0] == ' ' &&
                   result[k].find_first_not_of(" \t\n\r", 1) != std::string::npos) {
            // Token starting with space followed by content - keep as is
            // But GPT-2 groups at most one leading space with content
            final_result.push_back(result[k]);
        } else {
            final_result.push_back(result[k]);
        }
    }

    return final_result;
}

// ═══════════════════════════════════════════════════════════════
// BPE merge algorithm
// ═══════════════════════════════════════════════════════════════

std::vector<std::string> diffuse_tokenizer::bpe(const std::string & token) const {
    // token is already byte-encoded (each byte → unicode char via byte_to_unicode)
    auto word = utf8_split(token);
    if (word.size() < 2 || bpe_ranks.empty()) return word;

    while (true) {
        int best_rank = INT_MAX;
        size_t best_idx = SIZE_MAX;

        for (size_t i = 0; i + 1 < word.size(); i++) {
            auto it = bpe_ranks.find({word[i], word[i+1]});
            if (it != bpe_ranks.end() && it->second < best_rank) {
                best_rank = it->second;
                best_idx = i;
            }
        }

        if (best_idx == SIZE_MAX) break;

        // Merge word[best_idx] + word[best_idx+1]
        std::vector<std::string> new_word;
        new_word.reserve(word.size() - 1);
        for (size_t i = 0; i < word.size(); i++) {
            if (i == best_idx) {
                new_word.push_back(word[i] + word[i+1]);
                i++;
            } else {
                new_word.push_back(word[i]);
            }
        }
        word = std::move(new_word);
    }

    return word;
}

// ═══════════════════════════════════════════════════════════════
// Encode
// ═══════════════════════════════════════════════════════════════

std::vector<int32_t> diffuse_tokenizer::encode(const std::string & text, bool add_special) const {
    std::vector<int32_t> result;

    if (!initialized) return result;

    // Add BOS if configured
    if (add_special && add_bos && bos_id >= 0) {
        result.push_back(bos_id);
    }

    // Pre-tokenize
    auto pre_tokens = pre_tokenize(text);

    for (const auto & pre_tok : pre_tokens) {
        // Byte-encode: convert each byte to its unicode character
        std::string byte_encoded;
        for (unsigned char c : pre_tok) {
            byte_encoded += byte_to_unicode_str[c];
        }

        // Apply BPE merges
        auto bpe_tokens = bpe(byte_encoded);

        // Look up each BPE token in vocab
        for (const auto & bt : bpe_tokens) {
            auto it = token_to_id.find(bt);
            if (it != token_to_id.end()) {
                result.push_back(it->second);
            } else if (unk_id >= 0) {
                result.push_back(unk_id);
            }
            // If no UNK token and not in vocab, skip (shouldn't happen with byte fallback)
        }
    }

    // Add EOS if configured
    if (add_special && add_eos && eos_id >= 0) {
        result.push_back(eos_id);
    }

    return result;
}

// ═══════════════════════════════════════════════════════════════
// Decode
// ═══════════════════════════════════════════════════════════════

std::string diffuse_tokenizer::decode(const std::vector<int32_t> & ids, bool skip_special) const {
    if (!initialized) return "";

    // First, concatenate all token strings
    std::string byte_encoded;
    for (int32_t id : ids) {
        if (id < 0 || (size_t)id >= id_to_token.size()) continue;

        if (skip_special) {
            auto type = id_to_type[id];
            if (type == DIFFUSE_TOKEN_TYPE_CONTROL || type == DIFFUSE_TOKEN_TYPE_UNKNOWN ||
                type == DIFFUSE_TOKEN_TYPE_UNUSED) {
                continue;
            }
            // Also skip known special token IDs
            if (id == bos_id || id == eos_id || id == pad_id) continue;
        }

        byte_encoded += id_to_token[id];
    }

    // Reverse the byte encoding: decode each code point back to a byte
    std::string result;
    result.reserve(byte_encoded.size());
    size_t i = 0;
    while (i < byte_encoded.size()) {
        // Parse one UTF-8 code point
        int consumed = 0;
        int cp = utf8_to_codepoint(byte_encoded.substr(i), consumed);
        if (consumed == 0) break;

        // Reverse lookup: code point → byte
        if (cp >= 0 && cp < 512 && unicode_to_byte_map[cp] >= 0) {
            result += (char)(unsigned char)unicode_to_byte_map[cp];
        } else {
            // Not in byte_to_unicode mapping: output the raw UTF-8
            result += byte_encoded.substr(i, consumed);
        }

        i += consumed;
    }

    return result;
}

// ═══════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════

bool diffuse_tokenizer::is_special(int32_t id) const {
    if (id < 0 || (size_t)id >= id_to_type.size()) return false;
    auto t = id_to_type[id];
    return t == DIFFUSE_TOKEN_TYPE_CONTROL || t == DIFFUSE_TOKEN_TYPE_UNKNOWN ||
           t == DIFFUSE_TOKEN_TYPE_UNUSED ||
           id == bos_id || id == eos_id || id == pad_id;
}

// ═══════════════════════════════════════════════════════════════
// Chat template (Qwen2 / ChatML style)
// ═══════════════════════════════════════════════════════════════

std::string chat_template_str;  // declared in header as chat_template

std::vector<int32_t> diffuse_apply_chat_template(
        const diffuse_tokenizer * tok,
        const std::vector<diffuse_chat_message> & messages,
        bool add_generation_prompt) {

    // Build the prompt text using ChatML format:
    //   <|im_start|>system\n{content}<|im_end|>
    //   <|im_start|>user\n{content}<|im_end|>
    //   <|im_start|>assistant\n

    std::string prompt;

    for (const auto & msg : messages) {
        prompt += "<|im_start|>" + msg.role + "\n" + msg.content + "<|im_end|>\n";
    }

    if (add_generation_prompt) {
        prompt += "<|im_start|>assistant\n";
    }

    return tok->encode(prompt, false);
}

std::string diffuse_extract_assistant_text(
        const diffuse_tokenizer * tok,
        const std::vector<int32_t> & tokens) {

    std::string text = tok->decode(tokens, true);

    // Find the assistant marker and extract content
    std::string marker = "assistant\n";
    auto pos = text.rfind(marker);
    if (pos != std::string::npos) {
        text = text.substr(pos + marker.size());
    }

    // Trim trailing <|im_end|> or similar
    auto end_pos = text.find("<|im_end|>");
    if (end_pos != std::string::npos) {
        text = text.substr(0, end_pos);
    }

    return text;
}


// ═══════════════════════════════════════════════════════════════
// Public C++ API wrappers
// ═══════════════════════════════════════════════════════════════

diffuse_tokenizer * diffuse_tokenizer_load(const std::string & gguf_path) {
    auto * tok = new diffuse_tokenizer();

    struct gguf_init_params gparams = { false, nullptr };
    struct gguf_context * gctx = gguf_init_from_file(gguf_path.c_str(), gparams);
    if (!gctx) {
        DIFFUSE_LOG("tokenizer: failed to open %s", gguf_path.c_str());
        delete tok;
        return nullptr;
    }

    bool ok = tok->load_from_gguf(gctx);
    gguf_free(gctx);

    if (!ok) {
        delete tok;
        return nullptr;
    }
    return tok;
}

void diffuse_tokenizer_free(diffuse_tokenizer * tok) {
    delete tok;
}

std::vector<int32_t> diffuse_tokenize(
        diffuse_tokenizer * tok,
        const std::string & text,
        bool add_special) {
    return tok->encode(text, add_special);
}

std::string diffuse_detokenize(
        diffuse_tokenizer * tok,
        const std::vector<int32_t> & ids,
        bool skip_special) {
    return tok->decode(ids, skip_special);
}

bool diffuse_tokenizer_ready(const diffuse_tokenizer * tok) {
    return tok && tok->initialized;
}

size_t diffuse_tokenizer_size(const diffuse_tokenizer * tok) {
    return tok ? tok->size() : 0;
}
