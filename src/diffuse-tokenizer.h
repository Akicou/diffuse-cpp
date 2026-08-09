#pragma once

// ── BPE Tokenizer for diffuse-cpp ──────────────────────────────
//
// Implements GPT-2 / Qwen2 style byte-level BPE tokenization
// that reads vocabulary and merge rules from GGUF metadata.
//
// Supported GGUF keys:
//   tokenizer.ggml.model       (string)  e.g. "gpt2"
//   tokenizer.ggml.tokens      (array of strings)
//   tokenizer.ggml.token_type  (array of int32)
//   tokenizer.ggml.scores      (array of float32)
//   tokenizer.ggml.merges      (array of strings)  "A B"
//   tokenizer.ggml.bos_token_id, eos_token_id, unknown_token_id
//   tokenizer.ggml.padding_token_id
//   tokenizer.ggml.add_bos_token, add_eos_token

#include "diffuse.h"
#include "diffuse-common.h"
#include "gguf.h"
#include "ggml.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <cstdint>

// ── Token types (matching llama.cpp / GGUF convention) ─────────
enum diffuse_token_type {
    DIFFUSE_TOKEN_TYPE_UNDEFINED   = 0,
    DIFFUSE_TOKEN_TYPE_NORMAL      = 1,
    DIFFUSE_TOKEN_TYPE_UNKNOWN     = 2,
    DIFFUSE_TOKEN_TYPE_CONTROL     = 3,
    DIFFUSE_TOKEN_TYPE_USER_DEFINED = 4,
    DIFFUSE_TOKEN_TYPE_UNUSED      = 5,
    DIFFUSE_TOKEN_TYPE_BYTE        = 6,
};

// ── Tokenizer / Vocabulary ─────────────────────────────────────
struct diffuse_tokenizer {

    // Vocabulary
    std::vector<std::string>          id_to_token;   // id → token text
    std::vector<diffuse_token_type>   id_to_type;
    std::vector<float>                id_to_score;
    std::unordered_map<std::string, int> token_to_id; // token text → id

    // BPE merge ranks: pair of token strings → rank (lower = higher priority)
    std::map<std::pair<std::string, std::string>, int> bpe_ranks;

    // Literal text of every control/user-defined token, longest first.
    // encode() matches these before BPE so markers like <|role_end|> map to
    // their own id instead of being shredded into byte-level pieces.
    std::vector<std::string> special_tokens;

    // Model name ("gpt2", "llama", etc.)
    std::string model_name = "gpt2";

    // Chat template (from GGUF if available)
    std::string chat_template;

    // Special token IDs (-1 = not defined)
    int32_t bos_id  = -1;
    int32_t eos_id  = -1;
    int32_t unk_id  = -1;
    int32_t pad_id  = -1;
    int32_t mask_id = -1;

    // Flags
    bool add_bos = false;
    bool add_eos = false;

    // ── Pre-computed byte ↔ unicode mapping (GPT-2 style) ────────
    std::string byte_to_unicode_str[256];      // byte → UTF-8 string
    int         unicode_to_byte_map[512];      // code point → byte (reverse lookup)

    bool initialized = false;

    // ── API ─────────────────────────────────────────────────────

    // Initialize the byte↔unicode lookup tables
    void init_byte_tables();

    // Load vocabulary from a GGUF context
    // Returns true on success
    bool load_from_gguf(struct gguf_context * gctx);

    // Encode text → token IDs
    std::vector<int32_t> encode(const std::string & text, bool add_special = true) const;

    // Decode token IDs → text
    std::string decode(const std::vector<int32_t> & ids, bool skip_special = true) const;

    // Check if a token is a special token (control, unknown, etc.)
    bool is_special(int32_t id) const;

    // Number of tokens in vocabulary
    size_t size() const { return id_to_token.size(); }

    // Pre-tokenize: split text into sub-words for BPE (public for testing)
    std::vector<std::string> pre_tokenize(const std::string & text) const;

    // Split UTF-8 string into individual code points (public for testing)
    static std::vector<std::string> utf8_split(const std::string & s);

    // Encode a single UTF-8 code point to its byte representation
    static std::string codepoint_to_utf8(int cp);

private:
    // GPT-2 BPE algorithm on a byte-encoded word
    std::vector<std::string> bpe(const std::string & token) const;
};

// ── Chat template support ──────────────────────────────────────
struct diffuse_chat_message {
    std::string role;    // "system", "user", "assistant"
    std::string content;
};

// Apply a chat template to messages → token IDs
// Uses Qwen2/Llama3 style template by default
std::vector<int32_t> diffuse_apply_chat_template(
    const diffuse_tokenizer * tok,
    const std::vector<diffuse_chat_message> & messages,
    bool add_generation_prompt = true);

// Extract assistant text from generated tokens
std::string diffuse_extract_assistant_text(
    const diffuse_tokenizer * tok,
    const std::vector<int32_t> & tokens);
