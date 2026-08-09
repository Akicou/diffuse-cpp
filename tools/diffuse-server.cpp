// ═══════════════════════════════════════════════════════════════
// diffuse-server: HTTP inference server for diffusion LLMs
// ═══════════════════════════════════════════════════════════════
//
// Provides a llama-server compatible HTTP API and a web UI for
// interactive diffusion text generation.
//
// Endpoints:
//   GET  /                        Web UI (chat interface)
//   GET  /health                  Health check
//   GET  /v1/models               List models (OpenAI compatible)
//   POST /v1/chat/completions     Chat completion (OpenAI compatible)
//   POST /completion              Raw text completion
//   POST /tokenize                Tokenize text
//   POST /detokenize              Detokenize token IDs
//
// Usage:
//   diffuse-server -m model.gguf --host 0.0.0.0 --port 8080
//   diffuse-server -m model.gguf --api-key secret123

#include "diffuse.h"
#include "diffuse-tokenizer.h"
#include "diffuse-json.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <ctime>
#include <algorithm>
#include <sstream>
#include <set>

// ── Cross-platform socket headers ──────────────────────────────
#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "ws2_32.lib")
    typedef int socklen_t;
    #define DIFFUSE_CLOSE_SOCKET closesocket
    #define DIFFUSE_SOCKET_ERR  SOCKET_ERROR
    using socket_t = SOCKET;
    using recv_ret_t = int;
#else
    #include <sys/socket.h>
    #include <netinet/in.h>
    #include <arpa/inet.h>
    #include <unistd.h>
    #include <fcntl.h>
    #include <signal.h>
    typedef int socket_t;
    using recv_ret_t = ssize_t;
    #define DIFFUSE_CLOSE_SOCKET close
    #define DIFFUSE_SOCKET_ERR  (-1)
    #define INVALID_SOCKET (-1)
#endif

// ═══════════════════════════════════════════════════════════════
// HTTP types
// ═══════════════════════════════════════════════════════════════

struct HttpRequest {
    std::string method;
    std::string path;
    std::string version;
    std::map<std::string, std::string> headers;
    std::string body;

    std::string header(const std::string & key) const {
        auto it = headers.find(key);
        if (it != headers.end()) return it->second;
        // Try case-insensitive
        std::string lower_key = key;
        std::transform(lower_key.begin(), lower_key.end(), lower_key.begin(), ::tolower);
        for (const auto & [k, v] : headers) {
            std::string lk = k;
            std::transform(lk.begin(), lk.end(), lk.begin(), ::tolower);
            if (lk == lower_key) return v;
        }
        return "";
    }

    std::string query(const std::string & key) const {
        auto qpos = path.find('?');
        if (qpos == std::string::npos) return "";
        std::string query_str = path.substr(qpos + 1);
        std::string prefix = key + "=";
        auto kpos = query_str.find(prefix);
        if (kpos == std::string::npos) return "";
        size_t start = kpos + prefix.size();
        size_t end = query_str.find('&', start);
        if (end == std::string::npos) end = query_str.size();
        return query_str.substr(start, end - start);
    }
};

struct HttpResponse {
    int status = 200;
    std::string status_text = "OK";
    std::map<std::string, std::string> headers;
    std::string body;

    void set_json(const std::string & json_str) {
        body = json_str;
        headers["Content-Type"] = "application/json";
    }

    void set_html(const std::string & html_str) {
        body = html_str;
        headers["Content-Type"] = "text/html; charset=utf-8";
    }

    void set_error(int code, const std::string & msg) {
        status = code;
        status_text = (code == 400) ? "Bad Request" :
                      (code == 401) ? "Unauthorized" :
                      (code == 403) ? "Forbidden" :
                      (code == 404) ? "Not Found" :
                      (code == 500) ? "Internal Server Error" : "Error";
        json::Value err = json::Value::object();
        err("error") = json::Value::object();
        err("error")("message") = msg;
        err("error")("type") = (code == 401) ? "authentication_error" :
                               (code == 400) ? "invalid_request_error" :
                               "server_error";
        err("error")("code") = code;
        set_json(json::serialize(err));
    }

    std::string serialize() const {
        std::string resp;
        resp += "HTTP/1.1 " + std::to_string(status) + " " + status_text + "\r\n";
        for (const auto & [k, v] : headers) {
            resp += k + ": " + v + "\r\n";
        }
        resp += "Content-Length: " + std::to_string(body.size()) + "\r\n";
        resp += "Connection: close\r\n";
        resp += "Access-Control-Allow-Origin: *\r\n";
        resp += "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n";
        resp += "Access-Control-Allow-Headers: Content-Type, Authorization, X-API-Key\r\n";
        resp += "\r\n";
        resp += body;
        return resp;
    }
};

// ═══════════════════════════════════════════════════════════════
// HTTP request parsing
// ═══════════════════════════════════════════════════════════════

static bool parse_http_request(const std::string & raw, HttpRequest & req) {
    // Find header/body boundary
    auto hdr_end = raw.find("\r\n\r\n");
    if (hdr_end == std::string::npos) return false;

    std::string header_section = raw.substr(0, hdr_end);

    // Parse request line
    auto first_eol = header_section.find("\r\n");
    if (first_eol == std::string::npos) return false;
    std::string request_line = header_section.substr(0, first_eol);

    // Split request line
    std::istringstream rl(request_line);
    rl >> req.method >> req.path >> req.version;

    if (req.method.empty() || req.path.empty()) return false;

    // Parse headers
    std::string hdrs = header_section.substr(first_eol + 2);
    size_t pos = 0;
    while (pos < hdrs.size()) {
        auto eol = hdrs.find("\r\n", pos);
        if (eol == std::string::npos) eol = hdrs.size();
        std::string line = hdrs.substr(pos, eol - pos);
        auto colon = line.find(':');
        if (colon != std::string::npos) {
            std::string key = line.substr(0, colon);
            std::string val = line.substr(colon + 1);
            // Trim whitespace
            while (!val.empty() && val[0] == ' ') val = val.substr(1);
            req.headers[key] = val;
        }
        pos = eol + 2;
    }

    // Body is after the header boundary
    req.body = raw.substr(hdr_end + 4);

    return true;
}

// ═══════════════════════════════════════════════════════════════
// Socket initialization helpers
// ═══════════════════════════════════════════════════════════════

static bool init_sockets() {
#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) return false;
#endif
    return true;
}

static void cleanup_sockets() {
#ifdef _WIN32
    WSACleanup();
#endif
}

static std::string recv_all(socket_t sock, size_t max_bytes = 50 * 1024 * 1024) {
    std::string result;
    char buf[8192];
    bool got_headers = false;
    size_t content_length = 0;

    while (result.size() < max_bytes) {
        recv_ret_t n = recv(sock, buf, sizeof(buf), 0);
        if (n <= 0) break;
        result.append(buf, n);

        // Check for headers complete
        if (!got_headers) {
            auto hdr_end = result.find("\r\n\r\n");
            if (hdr_end != std::string::npos) {
                got_headers = true;
                // Parse Content-Length
                std::string hdrs = result.substr(0, hdr_end);
                auto cl_pos = hdrs.find("Content-Length:");
                if (cl_pos == std::string::npos) cl_pos = hdrs.find("content-length:");
                if (cl_pos != std::string::npos) {
                    size_t val_start = hdrs.find(':', cl_pos) + 1;
                    content_length = std::stoul(hdrs.substr(val_start));
                } else {
                    // No Content-Length: headers are all we need (GET requests)
                    break;
                }
            }
        }

        // Check if we have the full body
        if (got_headers && content_length > 0) {
            auto hdr_end = result.find("\r\n\r\n");
            if (hdr_end != std::string::npos) {
                size_t body_len = result.size() - hdr_end - 4;
                if (body_len >= content_length) break;
            }
        }
    }
    return result;
}

static void send_response(socket_t sock, const HttpResponse & resp) {
    std::string data = resp.serialize();
    send(sock, data.c_str(), (int)data.size(), 0);
}

// ═══════════════════════════════════════════════════════════════
// Server state
// ═══════════════════════════════════════════════════════════════

struct ServerState {
    diffuse_model * model = nullptr;
    diffuse_tokenizer tokenizer;
    std::mutex model_mutex;
    std::atomic<bool> running{true};
    std::atomic<int> active_requests{0};

    // Config
    std::string model_path;
    std::string host = "127.0.0.1";
    int port = 8080;
    int n_threads = 4;
    int n_gpu_layers = 0;
    int n_steps = 16;
    int n_generate_default = 256;
    float temperature = 0.0f;
    float threshold = 0.95f;
    std::string remasking = "low_confidence";
    std::string api_key;
    bool verbose = false;

    // Stats
    std::atomic<int64_t> total_requests{0};
    std::atomic<int64_t> total_tokens_generated{0};
    std::chrono::steady_clock::time_point start_time;
};

static ServerState g_state;

// ── Get the web UI HTML ────────────────────────────────────────
// (defined in diffuse-server-ui.h, included at bottom)
std::string get_web_ui_html();

// ── Authentication check ───────────────────────────────────────
static bool check_auth(const HttpRequest & req) {
    if (g_state.api_key.empty()) return true; // No auth required

    // Check Bearer token
    std::string auth = req.header("Authorization");
    if (!auth.empty()) {
        if (auth.substr(0, 7) == "Bearer ") {
            if (auth.substr(7) == g_state.api_key) return true;
        }
    }

    // Check X-API-Key header
    std::string api_key_hdr = req.header("X-API-Key");
    if (!api_key_hdr.empty() && api_key_hdr == g_state.api_key) return true;

    return false;
}

// ── Get current ISO timestamp ──────────────────────────────────
static std::string iso_timestamp() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&t));
    return buf;
}

// ═══════════════════════════════════════════════════════════════
// API endpoint handlers
// ═══════════════════════════════════════════════════════════════

static HttpResponse handle_health(const HttpRequest & req) {
    HttpResponse resp;
    json::Value body = json::Value::object();
    body("status") = "ok";
    body("model_loaded") = (g_state.model != nullptr);
    body("active_requests") = (int64_t)g_state.active_requests;
    resp.set_json(json::serialize(body));
    return resp;
}

static HttpResponse handle_models(const HttpRequest & req) {
    HttpResponse resp;
    json::Value body = json::Value::object();
    body("object") = "list";

    json::Value models = json::Value::array();
    json::Value model_obj = json::Value::object();
    model_obj("id") = g_state.model_path;
    model_obj("object") = "model";

    json::Value meta = json::Value::object();
    const auto & hp = diffuse_model_hparams(g_state.model);
    meta("n_vocab") = (int64_t)hp.n_vocab;
    meta("n_embd") = (int64_t)hp.n_embd;
    meta("n_layer") = (int64_t)hp.n_layer;
    meta("n_head") = (int64_t)hp.n_head;
    meta("context_length") = (int64_t)hp.n_ctx_max;
    model_obj("meta") = meta;

    model_obj("owned_by") = "diffuse-cpp";
    models.push_back(model_obj);
    body("data") = models;

    resp.set_json(json::serialize(body));
    return resp;
}

static diffuse_sampler_params parse_sampler_params(const json::Value & req_json) {
    diffuse_sampler_params sp;
    sp.n_steps = req_json["n_steps"].as_int(g_state.n_steps);
    sp.temperature = req_json["temperature"].as_number(g_state.temperature);
    sp.threshold = req_json["threshold"].as_number(g_state.threshold);
    sp.seed = req_json["seed"].as_int(42);
    sp.eos_early_stop = req_json["eos_early_stop"].as_bool(true);
    sp.enable_editing = req_json["enable_editing"].as_bool(true);

    std::string remasking = req_json["remasking"].as_string(g_state.remasking);
    if (remasking == "random") sp.remasking = diffuse_remasking::RANDOM;
    else sp.remasking = diffuse_remasking::LOW_CONFIDENCE;

    return sp;
}

static HttpResponse handle_chat_completion(const HttpRequest & req) {
    HttpResponse resp;

    // Parse JSON body
    json::Value body = json::parse(req.body);
    if (body.is_null() || !body.is_object()) {
        resp.set_error(400, "Invalid JSON body");
        return resp;
    }

    // Validate messages
    if (!body.has("messages") || !body["messages"].is_array()) {
        resp.set_error(400, "'messages' field is required and must be an array");
        return resp;
    }

    const auto & messages = body["messages"].arr_val;
    if (messages.empty()) {
        resp.set_error(400, "'messages' array must not be empty");
        return resp;
    }

    // Validate each message
    std::vector<diffuse_chat_message> chat_msgs;
    for (size_t i = 0; i < messages.size(); i++) {
        const auto & msg = messages[i];
        if (!msg.is_object() || !msg.has("role") || !msg.has("content")) {
            resp.set_error(400, "Each message must have 'role' and 'content' fields");
            return resp;
        }
        std::string role = msg["role"].as_string();
        if (role != "system" && role != "user" && role != "assistant") {
            resp.set_error(400, "Message role must be 'system', 'user', or 'assistant'");
            return resp;
        }
        if (msg["content"].as_string().empty() && role != "system") {
            resp.set_error(400, "Message content must not be empty");
            return resp;
        }
        chat_msgs.push_back({role, msg["content"].as_string()});
    }

    // Parameters
    int n_generate = body["max_tokens"].as_int(g_state.n_generate_default);
    if (n_generate <= 0 || n_generate > 4096) {
        resp.set_error(400, "'max_tokens' must be between 1 and 4096");
        return resp;
    }
    if (!g_state.tokenizer.initialized) {
        resp.set_error(400, "Model does not have a tokenizer. Re-convert with --embed-tokenizer.");
        return resp;
    }

    diffuse_sampler_params sp = parse_sampler_params(body);

    // Tokenize with chat template
    auto prompt_tokens = diffuse_apply_chat_template(&g_state.tokenizer, chat_msgs, true);

    if (prompt_tokens.empty()) {
        resp.set_error(400, "Tokenization produced no tokens");
        return resp;
    }

    // Check context window
    const auto & hp = diffuse_model_hparams(g_state.model);
    if ((int)prompt_tokens.size() + n_generate > (int)hp.n_ctx_max) {
        resp.set_error(400, "Prompt + max_tokens exceeds context window (" +
                        std::to_string(hp.n_ctx_max) + ")");
        return resp;
    }

    // Generate
    g_state.active_requests++;
    g_state.total_requests++;
    auto t0 = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lock(g_state.model_mutex);
    int n_ctx = (int)prompt_tokens.size() + n_generate;
    diffuse_context * ctx = diffuse_context_new_gpu(g_state.model, n_ctx, g_state.n_threads, g_state.n_gpu_layers);

    auto result = diffuse_generate(ctx, prompt_tokens, n_generate, sp,
        [](int blk, int total_blk, int step, int total_step, const std::vector<int32_t> &) {
        });

    auto t1 = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    diffuse_context_free(ctx);
    g_state.active_requests--;
    g_state.total_tokens_generated += n_generate;

    // Decode result
    std::string assistant_text = diffuse_extract_assistant_text(&g_state.tokenizer, result);

    // Build OpenAI-compatible response
    bool stream = body["stream"].as_bool(false);

    json::Value resp_json = json::Value::object();
    resp_json("id") = "chatcmpl-" + std::to_string(g_state.total_requests.load());
    resp_json("object") = "chat.completion";
    resp_json("created") = (int64_t)std::time(nullptr);
    resp_json("model") = g_state.model_path;

    json::Value choices = json::Value::array();
    json::Value choice = json::Value::object();
    choice("index") = 0;
    json::Value message = json::Value::object();
    message("role") = "assistant";
    message("content") = assistant_text;
    choice("message") = message;
    choice("finish_reason") = "stop";
    choices.push_back(choice);
    resp_json("choices") = choices;

    json::Value usage = json::Value::object();
    usage("prompt_tokens") = (int64_t)prompt_tokens.size();
    usage("completion_tokens") = (int64_t)result.size();
    usage("total_tokens") = (int64_t)(prompt_tokens.size() + result.size());
    resp_json("usage") = usage;

    // Extra diffusion-specific stats
    json::Value diffuse_info = json::Value::object();
    diffuse_info("elapsed_ms") = elapsed_ms;
    diffuse_info("tokens_per_second") = elapsed_ms > 0 ? 1000.0 * n_generate / elapsed_ms : 0.0;
    diffuse_info("diffusion_steps") = sp.n_steps;
    resp_json("diffuse_info") = diffuse_info;

    resp.set_json(json::serialize(resp_json));
    return resp;
}

static HttpResponse handle_completion(const HttpRequest & req) {
    HttpResponse resp;

    json::Value body = json::parse(req.body);
    if (!body.is_object()) {
        resp.set_error(400, "Invalid JSON body");
        return resp;
    }

    // Support both "prompt" and "tokens"
    std::vector<int32_t> prompt_tokens;

    if (body.has("tokens") && body["tokens"].is_array()) {
        for (const auto & t : body["tokens"].arr_val) {
            prompt_tokens.push_back(t.as_int());
        }
    } else if (body.has("prompt") && body["prompt"].is_string()) {
        if (!g_state.tokenizer.initialized) {
            resp.set_error(400, "Model has no tokenizer. Use 'tokens' field with pre-tokenized IDs.");
            return resp;
        }
        prompt_tokens = g_state.tokenizer.encode(body["prompt"].as_string(), true);
    } else {
        resp.set_error(400, "Either 'prompt' (text) or 'tokens' (array of ints) is required");
        return resp;
    }

    if (prompt_tokens.empty()) {
        resp.set_error(400, "Prompt is empty after tokenization");
        return resp;
    }

    int n_generate = body["n_predict"].as_int(g_state.n_generate_default);
    if (n_generate <= 0 || n_generate > 4096) {
        resp.set_error(400, "'n_predict' must be between 1 and 4096");
        return resp;
    }

    diffuse_sampler_params sp = parse_sampler_params(body);

    g_state.active_requests++;
    g_state.total_requests++;
    auto t0 = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lock(g_state.model_mutex);
    int n_ctx = (int)prompt_tokens.size() + n_generate;
    diffuse_context * ctx = diffuse_context_new_gpu(g_state.model, n_ctx, g_state.n_threads, g_state.n_gpu_layers);

    auto result = diffuse_generate(ctx, prompt_tokens, n_generate, sp, nullptr);

    auto t1 = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    diffuse_context_free(ctx);
    g_state.active_requests--;
    g_state.total_tokens_generated += n_generate;

    json::Value resp_json = json::Value::object();
    resp_json("model") = g_state.model_path;
    resp_json("created") = (int64_t)std::time(nullptr);

    json::Value choices = json::Value::array();
    json::Value choice = json::Value::object();
    choice("text") = g_state.tokenizer.initialized
        ? g_state.tokenizer.decode(result, true)
        : "";
    choice("tokens") = json::Value::array();
    for (int32_t t : result) {
        json::Value arr = json::Value::array();
        choice("tokens").push_back(t);
    }
    choices.push_back(choice);
    resp_json("choices") = choices;

    json::Value usage = json::Value::object();
    usage("prompt_tokens") = (int64_t)prompt_tokens.size();
    usage("completion_tokens") = (int64_t)result.size();
    resp_json("usage") = usage;
    resp_json("elapsed_ms") = elapsed_ms;
    resp_json("tokens_per_second") = elapsed_ms > 0 ? 1000.0 * n_generate / elapsed_ms : 0.0;

    resp.set_json(json::serialize(resp_json));
    return resp;
}

static HttpResponse handle_tokenize(const HttpRequest & req) {
    HttpResponse resp;

    json::Value body = json::parse(req.body);
    if (!body.is_object()) {
        resp.set_error(400, "Invalid JSON body");
        return resp;
    }

    if (!body.has("content") || !body["content"].is_string()) {
        resp.set_error(400, "'content' field (string) is required");
        return resp;
    }

    if (!g_state.tokenizer.initialized) {
        resp.set_error(400, "Model does not have a tokenizer");
        return resp;
    }

    std::string text = body["content"].as_string();
    bool add_special = body["add_special"].as_bool(true);
    auto tokens = g_state.tokenizer.encode(text, add_special);

    json::Value resp_json = json::Value::object();
    resp_json("tokens") = json::Value::array();
    for (int32_t t : tokens) {
        resp_json("tokens").push_back(t);
    }
    resp_json("count") = (int64_t)tokens.size();

    resp.set_json(json::serialize(resp_json));
    return resp;
}

static HttpResponse handle_detokenize(const HttpRequest & req) {
    HttpResponse resp;

    json::Value body = json::parse(req.body);
    if (!body.is_object()) {
        resp.set_error(400, "Invalid JSON body");
        return resp;
    }

    if (!body.has("tokens") || !body["tokens"].is_array()) {
        resp.set_error(400, "'tokens' field (array of integers) is required");
        return resp;
    }

    std::vector<int32_t> tokens;
    for (const auto & t : body["tokens"].arr_val) {
        int32_t id = t.as_int();
        if (id < 0) {
            resp.set_error(400, "Token IDs must be non-negative integers");
            return resp;
        }
        tokens.push_back(id);
    }

    if (!g_state.tokenizer.initialized) {
        resp.set_error(400, "Model does not have a tokenizer");
        return resp;
    }

    bool skip_special = body["skip_special"].as_bool(true);
    std::string text = g_state.tokenizer.decode(tokens, skip_special);

    json::Value resp_json = json::Value::object();
    resp_json("text") = text;

    resp.set_json(json::serialize(resp_json));
    return resp;
}

// ═══════════════════════════════════════════════════════════════
// Request router
// ═══════════════════════════════════════════════════════════════

static HttpResponse route_request(const HttpRequest & req) {
    // CORS preflight
    if (req.method == "OPTIONS") {
        HttpResponse resp;
        resp.status = 204;
        resp.status_text = "No Content";
        return resp;
    }

    // Strip query string for routing
    std::string path = req.path;
    auto qpos = path.find('?');
    if (qpos != std::string::npos) path = path.substr(0, qpos);

    // Web UI
    if (path == "/" && req.method == "GET") {
        HttpResponse resp;
        resp.set_html(get_web_ui_html());
        return resp;
    }

    // Health (no auth required)
    if (path == "/health" && req.method == "GET") {
        return handle_health(req);
    }

    // All remaining endpoints require authentication
    if (!check_auth(req)) {
        HttpResponse resp;
        resp.set_error(401, "Invalid or missing API key. Set the Authorization: Bearer <key> header.");
        return resp;
    }

    // API endpoints
    if (path == "/v1/models" && req.method == "GET") {
        return handle_models(req);
    }

    if (path == "/v1/chat/completions" && req.method == "POST") {
        if (req.body.empty()) {
            HttpResponse resp;
            resp.set_error(400, "Request body is empty");
            return resp;
        }
        return handle_chat_completion(req);
    }

    if (path == "/completion" && req.method == "POST") {
        if (req.body.empty()) {
            HttpResponse resp;
            resp.set_error(400, "Request body is empty");
            return resp;
        }
        return handle_completion(req);
    }

    if (path == "/tokenize" && req.method == "POST") {
        if (req.body.empty()) {
            HttpResponse resp;
            resp.set_error(400, "Request body is empty");
            return resp;
        }
        return handle_tokenize(req);
    }

    if (path == "/detokenize" && req.method == "POST") {
        if (req.body.empty()) {
            HttpResponse resp;
            resp.set_error(400, "Request body is empty");
            return resp;
        }
        return handle_detokenize(req);
    }

    // Server stats
    if (path == "/stats" && req.method == "GET") {
        HttpResponse resp;
        json::Value body = json::Value::object();
        body("total_requests") = (int64_t)g_state.total_requests.load();
        body("total_tokens_generated") = (int64_t)g_state.total_tokens_generated.load();
        body("active_requests") = (int64_t)g_state.active_requests.load();

        auto now = std::chrono::steady_clock::now();
        double uptime_s = std::chrono::duration<double>(now - g_state.start_time).count();
        body("uptime_seconds") = uptime_s;
        body("model_path") = g_state.model_path;
        body("n_threads") = g_state.n_threads;
        body("n_gpu_layers") = g_state.n_gpu_layers;

        resp.set_json(json::serialize(body));
        return resp;
    }

    // 404
    HttpResponse resp;
    resp.set_error(404, "Endpoint not found: " + req.method + " " + path);
    return resp;
}

// ═══════════════════════════════════════════════════════════════
// Connection handler
// ═══════════════════════════════════════════════════════════════

static void handle_connection(socket_t client_sock) {
    std::string raw = recv_all(client_sock);

    if (raw.empty()) {
        DIFFUSE_CLOSE_SOCKET(client_sock);
        return;
    }

    HttpRequest req;
    if (!parse_http_request(raw, req)) {
        HttpResponse resp;
        resp.set_error(400, "Malformed HTTP request");
        send_response(client_sock, resp);
        DIFFUSE_CLOSE_SOCKET(client_sock);
        return;
    }

    if (g_state.verbose) {
        fprintf(stderr, "[server] %s %s\n", req.method.c_str(), req.path.c_str());
    }

    HttpResponse resp = route_request(req);
    send_response(client_sock, resp);

    if (g_state.verbose && resp.status >= 400) {
        fprintf(stderr, "[server] %d %s %s\n", resp.status, req.method.c_str(), req.path.c_str());
    }

    DIFFUSE_CLOSE_SOCKET(client_sock);
}

// ═══════════════════════════════════════════════════════════════
// Usage
// ═══════════════════════════════════════════════════════════════

static void print_usage(const char * prog) {
    fprintf(stderr,
        "diffuse-server: HTTP inference server for diffusion LLMs\n\n"
        "Usage: %s [options]\n\n"
        "Options:\n"
        "  -m PATH          Model file (GGUF) [required]\n"
        "  --host HOST      Listen address (default: 127.0.0.1)\n"
        "  --port PORT      Listen port (default: 8080)\n"
        "  -t INT           CPU threads (default: 4)\n"
        "  -ngl INT         GPU layers to offload (default: 0)\n"
        "  -s INT           Default diffusion steps (default: 16)\n"
        "  -n INT           Default tokens to generate (default: 256)\n"
        "  --temp FLOAT     Default temperature (default: 0.0 = argmax)\n"
        "  --remasking STR  Default remasking: low_confidence|random (default: low_confidence)\n"
        "  --threshold F     Confidence threshold (default: 0.95)\n"
        "  --api-key KEY    Require API key for API endpoints (default: none)\n"
        "  --verbose        Log every request\n"
        "  -h, --help       Show this help\n\n"
        "Endpoints:\n"
        "  GET  /                  Web UI\n"
        "  GET  /health            Health check\n"
        "  GET  /v1/models         List models\n"
        "  POST /v1/chat/completions  Chat completion (OpenAI compatible)\n"
        "  POST /completion        Text completion\n"
        "  POST /tokenize          Tokenize text\n"
        "  POST /detokenize        Detokenize token IDs\n"
        "  GET  /stats             Server statistics\n"
    );
}

// ═══════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════

int main(int argc, char ** argv) {
    g_state.start_time = std::chrono::steady_clock::now();

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) g_state.model_path = argv[++i];
        else if (arg == "--host" && i + 1 < argc) g_state.host = argv[++i];
        else if (arg == "--port" && i + 1 < argc) g_state.port = atoi(argv[++i]);
        else if (arg == "-t" && i + 1 < argc) g_state.n_threads = atoi(argv[++i]);
        else if (arg == "-ngl" && i + 1 < argc) g_state.n_gpu_layers = atoi(argv[++i]);
        else if (arg == "-s" && i + 1 < argc) g_state.n_steps = atoi(argv[++i]);
        else if (arg == "-n" && i + 1 < argc) g_state.n_generate_default = atoi(argv[++i]);
        else if (arg == "--temp" && i + 1 < argc) g_state.temperature = atof(argv[++i]);
        else if (arg == "--remasking" && i + 1 < argc) g_state.remasking = argv[++i];
        else if (arg == "--threshold" && i + 1 < argc) g_state.threshold = atof(argv[++i]);
        else if (arg == "--api-key" && i + 1 < argc) g_state.api_key = argv[++i];
        else if (arg == "--verbose") g_state.verbose = true;
        else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (g_state.model_path.empty()) {
        fprintf(stderr, "Error: model path required (-m)\n\n");
        print_usage(argv[0]);
        return 1;
    }

    // Load model
    fprintf(stderr, "[server] Loading model: %s\n", g_state.model_path.c_str());
    g_state.model = diffuse_model_load(g_state.model_path, g_state.n_threads);
    if (!g_state.model) {
        fprintf(stderr, "[server] Failed to load model\n");
        return 1;
    }

    // Load tokenizer from the same GGUF file
    {
        struct gguf_init_params gparams = { false, nullptr };
        struct gguf_context * gctx = gguf_init_from_file(g_state.model_path.c_str(), gparams);
        if (gctx) {
            bool ok = g_state.tokenizer.load_from_gguf(gctx);
            if (!ok) {
                fprintf(stderr, "[server] WARNING: No tokenizer found in GGUF.\n");
                fprintf(stderr, "[server]          Text input will not work. Re-convert with:\n");
                fprintf(stderr, "[server]          python convert-llada2.py --input ... --output ... --embed-tokenizer\n");
            }
            gguf_free(gctx);
        }
    }

    const auto & hp = diffuse_model_hparams(g_state.model);
    fprintf(stderr, "[server] Model: vocab=%u embd=%u layers=%u heads=%u/%u\n",
            hp.n_vocab, hp.n_embd, hp.n_layer, hp.n_head, hp.n_head_kv);
    fprintf(stderr, "[server] Tokenizer: %s (%zu tokens)\n",
            g_state.tokenizer.initialized ? "loaded" : "not available",
            g_state.tokenizer.size());

    // Initialize sockets
    if (!init_sockets()) {
        fprintf(stderr, "[server] Failed to initialize sockets\n");
        return 1;
    }

#ifndef _WIN32
    signal(SIGPIPE, SIG_IGN);
#endif

    // Create listen socket
    socket_t server_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (server_sock == INVALID_SOCKET) {
        fprintf(stderr, "[server] Failed to create socket\n");
        cleanup_sockets();
        return 1;
    }

    // Allow address reuse
    int opt = 1;
    setsockopt(server_sock, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt, sizeof(opt));

    // Bind
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(g_state.port);
    inet_pton(AF_INET, g_state.host.c_str(), &addr.sin_addr);

    if (bind(server_sock, (struct sockaddr *)&addr, sizeof(addr)) == DIFFUSE_SOCKET_ERR) {
        fprintf(stderr, "[server] Failed to bind to %s:%d\n", g_state.host.c_str(), g_state.port);
        DIFFUSE_CLOSE_SOCKET(server_sock);
        cleanup_sockets();
        return 1;
    }

    if (listen(server_sock, 16) == DIFFUSE_SOCKET_ERR) {
        fprintf(stderr, "[server] Failed to listen\n");
        DIFFUSE_CLOSE_SOCKET(server_sock);
        cleanup_sockets();
        return 1;
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "[server] ════════════════════════════════════════════════\n");
    fprintf(stderr, "[server]  diffuse-server ready!\n");
    fprintf(stderr, "[server]  Web UI:  http://%s:%d\n", g_state.host.c_str(), g_state.port);
    fprintf(stderr, "[server]  API:     http://%s:%d/v1/chat/completions\n", g_state.host.c_str(), g_state.port);
    if (!g_state.api_key.empty()) {
        fprintf(stderr, "[server]  API Key: %s\n", g_state.api_key.c_str());
    }
    fprintf(stderr, "[server]  Threads: %d, GPU layers: %d\n", g_state.n_threads, g_state.n_gpu_layers);
    fprintf(stderr, "[server] ════════════════════════════════════════════════\n");
    fprintf(stderr, "\n");

    // Accept loop
    while (g_state.running) {
        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        socket_t client_sock = accept(server_sock, (struct sockaddr *)&client_addr, &client_len);

        if (client_sock == INVALID_SOCKET) {
            continue;
        }

        // Handle each connection in a separate thread
        std::thread(handle_connection, client_sock).detach();
    }

    DIFFUSE_CLOSE_SOCKET(server_sock);
    cleanup_sockets();
    diffuse_model_free(g_state.model);

    return 0;
}

// Include the web UI (defined as a function returning HTML)
#include "diffuse-server-ui.h"
