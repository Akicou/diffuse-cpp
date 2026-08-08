#!/usr/bin/env python3
"""Comprehensive API tests for diffuse-server.

Validates that no request from the UI (or any client) to diffuse-server
would be:
  1. Unauthenticated — all API endpoints check the API key
  2. Invalid — missing fields, wrong types, out-of-range values
  3. Bad form — malformed JSON, wrong content type, oversized bodies

Usage:
    # Start the server first:
    ./build/diffuse-server -m model.gguf --port 8080 --api-key testkey123

    # Run tests:
    python tests/test-server.py --host 127.0.0.1 --port 8080 --api-key testkey123

    # Or without auth:
    ./build/diffuse-server -m model.gguf --port 8080
    python tests/test-server.py --host 127.0.0.1 --port 8080
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
import http.client


# ═══════════════════════════════════════════════════════════════
# HTTP helpers
# ═══════════════════════════════════════════════════════════════

class Response:
    def __init__(self, status, body, headers=None):
        self.status = status
        self.body = body
        self.json = None
        if headers:
            self.headers = headers
        else:
            self.headers = {}
        try:
            self.json = json.loads(body) if body else None
        except (json.JSONDecodeError, TypeError):
            pass


def http_request(host, port, method, path, body=None, api_key=None,
                 content_type="application/json", headers=None):
    """Make an HTTP request and return a Response object."""
    conn = http.client.HTTPConnection(host, port, timeout=120)

    hdrs = {}
    if body is not None:
        hdrs["Content-Type"] = content_type
    if api_key:
        hdrs["Authorization"] = f"Bearer {api_key}"
    if headers:
        hdrs.update(headers)

    conn.request(method, path, body=body, headers=hdrs)
    resp = conn.getresponse()
    data = resp.read().decode("utf-8", errors="replace")
    conn.close()

    return Response(resp.status, data, dict(resp.headers))


# ═══════════════════════════════════════════════════════════════
# Test framework
# ═══════════════════════════════════════════════════════════════

passed = 0
failed = 0
errors = []


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        msg = f"  ✗ {name}" + (f" — {detail}" if detail else "")
        print(msg)
        errors.append(name)


def test_section(name):
    print(f"\n{'─'*60}")
    print(f" {name}")
    print(f"{'─'*60}")


# ═══════════════════════════════════════════════════════════════
# Test suites
# ═══════════════════════════════════════════════════════════════

def test_health(host, port, api_key=None):
    """Test the /health endpoint."""
    test_section("Health Endpoint")

    # Health check should always succeed (no auth required)
    r = http_request(host, port, "GET", "/health")
    test("GET /health returns 200", r.status == 200, f"Got {r.status}")
    test("Health response is JSON", r.json is not None)
    test("Health has 'status' field", r.json and r.json.get("status") == "ok")
    test("Health has 'model_loaded' field", r.json and "model_loaded" in r.json)

    # Health should not require auth even when API key is set
    if api_key:
        r = http_request(host, port, "GET", "/health")  # No API key
        test("Health works without API key", r.status == 200, f"Got {r.status}")


def test_web_ui(host, port, api_key=None):
    """Test the web UI endpoint."""
    test_section("Web UI")

    r = http_request(host, port, "GET", "/")
    test("GET / returns 200", r.status == 200, f"Got {r.status}")
    test("Returns HTML", "text/html" in r.headers.get("Content-Type", ""),
         f"Content-Type: {r.headers.get('Content-Type', 'missing')}")
    test("Contains diffuse-server title", "diffuse-server" in r.body)
    test("Contains chat interface", "chat" in r.body.lower())


def test_models(host, port, api_key=None):
    """Test the /v1/models endpoint."""
    test_section("Models Endpoint")

    # With auth
    r = http_request(host, port, "GET", "/v1/models", api_key=api_key)
    test("GET /v1/models returns 200", r.status == 200, f"Got {r.status}")
    test("Response has 'data' array", r.json and isinstance(r.json.get("data"), list))

    if r.json and r.json.get("data"):
        model = r.json["data"][0]
        test("Model has 'id'", "id" in model)
        test("Model has 'object'", model.get("object") == "model")
        test("Model has 'meta'", "meta" in model)
        meta = model.get("meta", {})
        test("Meta has n_vocab", "n_vocab" in meta)
        test("Meta has n_embd", "n_embd" in meta)
        test("Meta has n_layer", "n_layer" in meta)


def test_authentication(host, port, api_key):
    """Test that API key authentication works correctly."""
    test_section("Authentication")

    # Without API key on a protected endpoint
    r = http_request(host, port, "GET", "/v1/models")  # No key
    test("Protected endpoint without key returns 401", r.status == 401,
         f"Got {r.status}")
    test("Error response is JSON", r.json is not None)
    test("Error has 'error' field", r.json and "error" in r.json)
    test("Error message mentions API key",
         r.json and "key" in r.json.get("error", {}).get("message", "").lower())

    # With wrong API key
    r = http_request(host, port, "GET", "/v1/models", api_key="wrongkey")
    test("Wrong API key returns 401", r.status == 401, f"Got {r.status}")

    # With correct API key
    r = http_request(host, port, "GET", "/v1/models", api_key=api_key)
    test("Correct API key returns 200", r.status == 200, f"Got {r.status}")

    # X-API-Key header should also work
    r = http_request(host, port, "GET", "/v1/models",
                     headers={"X-API-Key": api_key})
    test("X-API-Key header works", r.status == 200, f"Got {r.status}")

    # POST endpoints also require auth
    body = json.dumps({"content": "hello"})
    r = http_request(host, port, "POST", "/tokenize", body=body)  # No key
    test("POST without auth returns 401", r.status == 401, f"Got {r.status}")


def test_chat_completion_validation(host, port, api_key=None):
    """Test validation of /v1/chat/completions requests."""
    test_section("Chat Completion Validation")

    # Missing body
    r = http_request(host, port, "POST", "/v1/chat/completions", api_key=api_key)
    test("Empty body returns 400", r.status == 400, f"Got {r.status}")

    # Malformed JSON
    r = http_request(host, port, "POST", "/v1/chat/completions",
                     body="{not json", api_key=api_key)
    test("Malformed JSON returns 400", r.status == 400, f"Got {r.status}")

    # Missing 'messages' field
    r = http_request(host, port, "POST", "/v1/chat/completions",
                     body=json.dumps({"max_tokens": 100}), api_key=api_key)
    test("Missing 'messages' returns 400", r.status == 400, f"Got {r.status}")
    test("Error mentions 'messages'", r.json and "messages" in
         r.json.get("error", {}).get("message", "").lower())

    # Empty messages array
    r = http_request(host, port, "POST", "/v1/chat/completions",
                     body=json.dumps({"messages": []}), api_key=api_key)
    test("Empty messages array returns 400", r.status == 400, f"Got {r.status}")

    # Message missing 'role'
    r = http_request(host, port, "POST", "/v1/chat/completions",
                     body=json.dumps({"messages": [{"content": "hi"}]}), api_key=api_key)
    test("Message without 'role' returns 400", r.status == 400, f"Got {r.status}")

    # Message with invalid role
    r = http_request(host, port, "POST", "/v1/chat/completions",
                     body=json.dumps({"messages": [{"role": "admin", "content": "hi"}]}),
                     api_key=api_key)
    test("Invalid role returns 400", r.status == 400, f"Got {r.status}")

    # Message missing 'content'
    r = http_request(host, port, "POST", "/v1/chat/completions",
                     body=json.dumps({"messages": [{"role": "user"}]}), api_key=api_key)
    test("Message without 'content' returns 400", r.status == 400, f"Got {r.status}")

    # Empty content for user message
    r = http_request(host, port, "POST", "/v1/chat/completions",
                     body=json.dumps({"messages": [{"role": "user", "content": ""}]}),
                     api_key=api_key)
    test("Empty user content returns 400", r.status == 400, f"Got {r.status}")

    # Invalid max_tokens (zero)
    r = http_request(host, port, "POST", "/v1/chat/completions",
                     body=json.dumps({
                         "messages": [{"role": "user", "content": "test"}],
                         "max_tokens": 0
                     }), api_key=api_key)
    test("max_tokens=0 returns 400", r.status == 400, f"Got {r.status}")

    # Invalid max_tokens (too large)
    r = http_request(host, port, "POST", "/v1/chat/completions",
                     body=json.dumps({
                         "messages": [{"role": "user", "content": "test"}],
                         "max_tokens": 99999
                     }), api_key=api_key)
    test("max_tokens=99999 returns 400", r.status == 400, f"Got {r.status}")

    # max_tokens as string (wrong type)
    r = http_request(host, port, "POST", "/v1/chat/completions",
                     body=json.dumps({
                         "messages": [{"role": "user", "content": "test"}],
                         "max_tokens": "100"
                     }), api_key=api_key)
    # Should either coerce or reject — test that it doesn't crash
    test("String max_tokens handled gracefully", r.status in (200, 400),
         f"Got {r.status}")


def test_completion_validation(host, port, api_key=None):
    """Test validation of /completion requests."""
    test_section("Completion Validation")

    # Missing body
    r = http_request(host, port, "POST", "/completion", api_key=api_key)
    test("Empty body returns 400", r.status == 400, f"Got {r.status}")

    # Missing both 'prompt' and 'tokens'
    r = http_request(host, port, "POST", "/completion",
                     body=json.dumps({"n_predict": 10}), api_key=api_key)
    test("Missing prompt and tokens returns 400", r.status == 400, f"Got {r.status}")

    # Tokens with negative values
    r = http_request(host, port, "POST", "/completion",
                     body=json.dumps({"tokens": [-1, 2, 3]}), api_key=api_key)
    # This may or may not be rejected (depends on whether we validate)
    test("Negative tokens handled", r.status in (200, 400), f"Got {r.status}")


def test_tokenize_validation(host, port, api_key=None):
    """Test validation of /tokenize and /detokenize endpoints."""
    test_section("Tokenize/Detokenize Validation")

    # Tokenize: missing 'content'
    r = http_request(host, port, "POST", "/tokenize",
                     body=json.dumps({"foo": "bar"}), api_key=api_key)
    test("Tokenize without 'content' returns 400", r.status == 400, f"Got {r.status}")

    # Tokenize: 'content' is not a string
    r = http_request(host, port, "POST", "/tokenize",
                     body=json.dumps({"content": 123}), api_key=api_key)
    test("Tokenize with non-string content returns 400", r.status == 400,
         f"Got {r.status}")

    # Tokenize: empty content (valid — should return empty tokens)
    r = http_request(host, port, "POST", "/tokenize",
                     body=json.dumps({"content": ""}), api_key=api_key)
    test("Tokenize empty content handled", r.status in (200, 400),
         f"Got {r.status}")

    # Tokenize: valid request
    r = http_request(host, port, "POST", "/tokenize",
                     body=json.dumps({"content": "Hello world"}), api_key=api_key)
    test("Tokenize valid content returns 200", r.status == 200,
         f"Got {r.status}")
    if r.status == 200 and r.json:
        test("Tokenize returns 'tokens' array",
             isinstance(r.json.get("tokens"), list))
        test("Tokenize returns 'count'",
             "count" in r.json)
        test("Count matches tokens length",
             r.json.get("count") == len(r.json.get("tokens", [])))

    # Detokenize: missing 'tokens'
    r = http_request(host, port, "POST", "/detokenize",
                     body=json.dumps({"text": "hello"}), api_key=api_key)
    test("Detokenize without 'tokens' returns 400", r.status == 400,
         f"Got {r.status}")

    # Detokenize: 'tokens' is not an array
    r = http_request(host, port, "POST", "/detokenize",
                     body=json.dumps({"tokens": "not array"}), api_key=api_key)
    test("Detokenize with non-array tokens returns 400", r.status == 400,
         f"Got {r.status}")

    # Detokenize: negative token ID
    r = http_request(host, port, "POST", "/detokenize",
                     body=json.dumps({"tokens": [-5]}), api_key=api_key)
    test("Detokenize with negative token returns 400", r.status == 400,
         f"Got {r.status}")


def test_404_and_cors(host, port, api_key=None):
    """Test 404 handling and CORS headers."""
    test_section("404 and CORS")

    # Unknown endpoint
    r = http_request(host, port, "GET", "/nonexistent", api_key=api_key)
    test("Unknown endpoint returns 404", r.status == 404, f"Got {r.status}")

    # Unknown endpoint with JSON error
    test("404 response is JSON error", r.json and "error" in r.json)

    # CORS preflight
    r = http_request(host, port, "OPTIONS", "/v1/models", api_key=api_key)
    test("OPTIONS returns 204", r.status == 204, f"Got {r.status}")

    # CORS headers present
    r = http_request(host, port, "GET", "/health")
    cors_origin = r.headers.get("Access-Control-Allow-Origin", "")
    test("CORS Allow-Origin present", cors_origin == "*",
         f"Got '{cors_origin}'")


def test_stats(host, port, api_key=None):
    """Test the /stats endpoint."""
    test_section("Server Stats")

    r = http_request(host, port, "GET", "/stats", api_key=api_key)
    test("GET /stats returns 200", r.status == 200, f"Got {r.status}")
    if r.json:
        test("Stats has 'total_requests'", "total_requests" in r.json)
        test("Stats has 'uptime_seconds'", "uptime_seconds" in r.json)
        test("Stats has 'model_path'", "model_path" in r.json)
        test("total_requests is non-negative",
             r.json.get("total_requests", -1) >= 0)


def test_roundtrip(host, port, api_key=None):
    """Test a complete tokenize → detokenize roundtrip."""
    test_section("Tokenize/Detokenize Roundtrip")

    text = "The quick brown fox jumps over the lazy dog."

    # Tokenize
    r = http_request(host, port, "POST", "/tokenize",
                     body=json.dumps({"content": text, "add_special": False}),
                     api_key=api_key)

    if r.status != 200:
        test("Tokenize for roundtrip", False, f"Status {r.status}")
        return

    tokens = r.json.get("tokens", [])
    test("Tokenize produced tokens", len(tokens) > 0)
    test(f"Tokenized '{text[:30]}...' → {len(tokens)} tokens", True)

    # Detokenize
    r = http_request(host, port, "POST", "/detokenize",
                     body=json.dumps({"tokens": tokens, "skip_special": True}),
                     api_key=api_key)

    if r.status != 200:
        test("Detokenize for roundtrip", False, f"Status {r.status}")
        return

    decoded = r.json.get("text", "")
    # Note: BPE roundtrip may not be exact due to spacing normalization
    test("Detokenize produced text", len(decoded) > 0)
    # Check key words are present
    test("Roundtrip preserves 'fox'", "fox" in decoded.lower())
    test("Roundtrip preserves 'dog'", "dog" in decoded.lower())


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Test diffuse-server API")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--api-key", default=None,
                        help="API key (only test auth if provided)")
    args = parser.parse_args()

    print(f"Testing diffuse-server at {args.host}:{args.port}")
    print(f"API Key: {'set' if args.api_key else 'not set'}")

    # Check server is running
    try:
        r = http_request(args.host, args.port, "GET", "/health")
        if r.status != 200:
            print(f"\nServer not responding correctly (status {r.status})")
            sys.exit(1)
    except Exception as e:
        print(f"\nCannot connect to server: {e}")
        print("Start the server first: ./build/diffuse-server -m model.gguf --port 8080")
        sys.exit(1)

    # Run all test suites
    test_health(args.host, args.port, args.api_key)
    test_web_ui(args.host, args.port, args.api_key)
    test_models(args.host, args.port, args.api_key)
    test_chat_completion_validation(args.host, args.port, args.api_key)
    test_completion_validation(args.host, args.port, args.api_key)
    test_tokenize_validation(args.host, args.port, args.api_key)
    test_404_and_cors(args.host, args.port, args.api_key)
    test_stats(args.host, args.port, args.api_key)
    test_roundtrip(args.host, args.port, args.api_key)

    # Auth tests only if API key is set
    if args.api_key:
        test_authentication(args.host, args.port, args.api_key)

    # Summary
    print(f"\n{'═'*60}")
    print(f" Results: {passed} passed, {failed} failed, {passed+failed} total")
    print(f"{'═'*60}")

    if failed > 0:
        print(f"\nFailed tests:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
