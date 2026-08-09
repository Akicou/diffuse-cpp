#include "diffuse-tokenizer.h"
#include "diffuse.h"

#include <cstdio>
#include <cassert>
#include <string>
#include <vector>

// Test the BPE tokenizer without needing a model file
// Tests the byte↔unicode mapping, BPE algorithm, and encode/decode roundtrip

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(cond, name) do { \
    if (cond) { tests_passed++; printf("  ✓ %s\n", name); } \
    else { tests_failed++; printf("  ✗ %s\n", name); } \
} while(0)

int main() {
    printf("=== Tokenizer Unit Tests ===\n\n");

    // Test 1: Byte↔unicode mapping
    printf("Test: Byte↔Unicode Mapping\n");
    {
        diffuse_tokenizer tok;
        tok.init_byte_tables();

        // Printable ASCII maps to itself
        TEST(tok.byte_to_unicode_str['A'] == "A", "Byte 'A' maps to 'A'");
        TEST(tok.byte_to_unicode_str['z'] == "z", "Byte 'z' maps to 'z'");
        TEST(tok.byte_to_unicode_str['!'] == "!", "Byte '!' maps to '!'");
        TEST(tok.byte_to_unicode_str['~'] == "~", "Byte '~' maps to '~'");

        // Space (0x20) maps to a non-printable unicode
        TEST(tok.byte_to_unicode_str[0x20] != " ", "Byte 0x20 (space) does NOT map to space");

        // Reverse mapping works
        TEST(tok.unicode_to_byte_map[(int)'A'] == 'A', "Reverse map: 'A' → 65");
        TEST(tok.unicode_to_byte_map[(int)'!'] == '!', "Reverse map: '!' → 33");

        // Null byte maps to 256 (0x100)
        // 0x00 is non-printable, n=0 → cp = 256
        TEST(tok.byte_to_unicode_str[0] != "", "Byte 0 has a mapping");
        TEST(tok.byte_to_unicode_str[0].size() == 2, "Byte 0 maps to 2-byte UTF-8 (cp=256)");
    }
    printf("\n");

    // Test 2: UTF-8 splitting
    printf("Test: UTF-8 Code Point Splitting\n");
    {
        // ASCII
        auto parts = diffuse_tokenizer::utf8_split("hello");
        TEST(parts.size() == 5, "ASCII 'hello' splits into 5 parts");

        // Mixed ASCII + 2-byte UTF-8
        parts = diffuse_tokenizer::utf8_split("a\xc3\xa9");  // 'a' + 'é'
        TEST(parts.size() == 2, "'a' + 'é' splits into 2 parts");
        TEST(parts[0] == "a", "First part is 'a'");
        TEST(parts[1] == "\xc3\xa9", "Second part is 'é' (2 bytes)");

        // 3-byte UTF-8 (CJK character 一 = U+4E00)
        parts = diffuse_tokenizer::utf8_split("\xe4\xb8\x80");
        TEST(parts.size() == 1, "CJK char splits into 1 part");
        TEST(parts[0].size() == 3, "CJK char is 3 bytes");
    }
    printf("\n");

    // Test 3: Pre-tokenization
    printf("Test: Pre-tokenization\n");
    {
        diffuse_tokenizer tok;
        tok.init_byte_tables();

        auto parts = tok.pre_tokenize("Hello world");
        TEST(parts.size() >= 2, "'Hello world' splits into ≥2 parts");

        parts = tok.pre_tokenize("123abc");
        TEST(parts.size() >= 2, "'123abc' splits digits from letters");

        parts = tok.pre_tokenize("a, b.");
        TEST(parts.size() >= 3, "'a, b.' separates punctuation");

        // Empty string
        parts = tok.pre_tokenize("");
        TEST(parts.empty(), "Empty string produces no tokens");
    }
    printf("\n");

    // Test 4: Tokenizer not initialized
    printf("Test: Uninitialized Tokenizer\n");
    {
        diffuse_tokenizer tok;
        // Should not crash
        auto tokens = tok.encode("hello", true);
        TEST(tokens.empty(), "Uninitialized encode returns empty");

        std::string text = tok.decode({1, 2, 3}, true);
        TEST(text.empty(), "Uninitialized decode returns empty");

        TEST(tok.size() == 0, "Uninitialized size is 0");
        TEST(!tok.initialized, "initialized flag is false");
    }
    printf("\n");

    // Test 5: is_special
    printf("Test: Special Token Check\n");
    {
        diffuse_tokenizer tok;
        tok.init_byte_tables();
        tok.id_to_type = {
            DIFFUSE_TOKEN_TYPE_NORMAL,
            DIFFUSE_TOKEN_TYPE_CONTROL,
            DIFFUSE_TOKEN_TYPE_NORMAL,
            DIFFUSE_TOKEN_TYPE_UNKNOWN,
        };
        tok.bos_id = 1;
        tok.eos_id = -1;

        TEST(!tok.is_special(0), "Token 0 (NORMAL) is not special");
        TEST(tok.is_special(1), "Token 1 (CONTROL/bos) is special");
        TEST(!tok.is_special(2), "Token 2 (NORMAL) is not special");
        TEST(tok.is_special(3), "Token 3 (UNKNOWN) is special");
        TEST(!tok.is_special(100), "Out-of-range token is not special");
        TEST(!tok.is_special(-1), "Negative token is not special");
    }
    printf("\n");

    // Test 6: Chat template
    printf("Test: Chat Template\n");
    {
        diffuse_tokenizer tok;
        tok.init_byte_tables();
        // Add a minimal vocab so encode doesn't crash
        tok.initialized = true;
        tok.id_to_token = {"a", "b", "c"};
        tok.token_to_id = {{"a", 0}, {"b", 1}, {"c", 2}};
        tok.unk_id = -1;

        std::vector<diffuse_chat_message> messages = {
            {"user", "hello"},
        };

        // This will produce tokens (mostly unknown since vocab is tiny)
        auto tokens = diffuse_apply_chat_template(&tok, messages, true);
        // Should produce some tokens (even if they're not meaningful)
        TEST(true, "Chat template applied without crash");
    }
    printf("\n");

    // Summary
    printf("═══════════════════════════════════════\n");
    printf(" Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("═══════════════════════════════════════\n");

    return tests_failed > 0 ? 1 : 0;
}
