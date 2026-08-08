#pragma once

// ── Minimal JSON parser and serializer for diffuse-server ──────
//
// Supports: objects, arrays, strings, numbers, bools, null.
// Not a full RFC 8259 implementation — just enough for API use.

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <variant>
#include <cstdint>
#include <stdexcept>

namespace json {

class Value;
using Object = std::map<std::string, Value>;
using Array  = std::vector<Value>;

enum class Type { Null, Bool, Number, String, Array, Object };

class Value {
public:
    Type type = Type::Null;
    bool bool_val = false;
    double num_val = 0.0;
    std::string str_val;
    Array arr_val;
    Object obj_val;

    Value() {}
    Value(bool b) : type(Type::Bool), bool_val(b) {}
    Value(int v) : type(Type::Number), num_val((double)v) {}
    Value(int64_t v) : type(Type::Number), num_val((double)v) {}
    Value(double v) : type(Type::Number), num_val(v) {}
    Value(const char * s) : type(Type::String), str_val(s) {}
    Value(const std::string & s) : type(Type::String), str_val(s) {}
    Value(std::string && s) : type(Type::String), str_val(std::move(s)) {}
    Value(Array && a) : type(Type::Array), arr_val(std::move(a)) {}
    Value(Object && o) : type(Type::Object), obj_val(std::move(o)) {}

    static Value null() { return Value(); }
    static Value object() { Value v; v.type = Type::Object; return v; }
    static Value array() { Value v; v.type = Type::Array; return v; }

    bool is_null() const { return type == Type::Null; }
    bool is_bool() const { return type == Type::Bool; }
    bool is_number() const { return type == Type::Number; }
    bool is_string() const { return type == Type::String; }
    bool is_array() const { return type == Type::Array; }
    bool is_object() const { return type == Type::Object; }

    bool as_bool(bool def = false) const { return type == Type::Bool ? bool_val : def; }
    double as_number(double def = 0) const { return type == Type::Number ? num_val : def; }
    int as_int(int def = 0) const { return type == Type::Number ? (int)num_val : def; }
    const std::string & as_string(const std::string & def = "") const {
        return type == Type::String ? str_val : def;
    }

    const Value & operator[](const std::string & key) const {
        static Value null_val;
        auto it = obj_val.find(key);
        return it != obj_val.end() ? it->second : null_val;
    }
    const Value & operator[](size_t i) const {
        static Value null_val;
        return i < arr_val.size() ? arr_val[i] : null_val;
    }

    bool has(const std::string & key) const {
        return type == Type::Object && obj_val.count(key) > 0;
    }

    Value & operator()(const std::string & key) {
        type = Type::Object;
        return obj_val[key];
    }
    void push_back(Value v) {
        type = Type::Array;
        arr_val.push_back(std::move(v));
    }
};

// ── Serializer ─────────────────────────────────────────────────

inline std::string escape_string(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char)c;
                }
        }
    }
    return out;
}

inline std::string serialize(const Value & v, bool pretty = false, int indent = 0) {
    std::string pad = pretty ? std::string(indent * 2, ' ') : "";
    std::string nl  = pretty ? "\n" : "";
    std::string sep = pretty ? ",\n" : ",";
    std::string colon = pretty ? ": " : ":";

    switch (v.type) {
        case Type::Null:   return "null";
        case Type::Bool:   return v.bool_val ? "true" : "false";
        case Type::Number: {
            // Print integers without decimal
            double intpart;
            if (std::modf(v.num_val, &intpart) == 0.0 &&
                v.num_val >= -9e15 && v.num_val <= 9e15) {
                return std::to_string((int64_t)v.num_val);
            }
            char buf[64];
            snprintf(buf, sizeof(buf), "%.17g", v.num_val);
            return buf;
        }
        case Type::String:
            return "\"" + escape_string(v.str_val) + "\"";
        case Type::Array: {
            if (v.arr_val.empty()) return "[]";
            std::string s = "[" + nl;
            for (size_t i = 0; i < v.arr_val.size(); i++) {
                if (i > 0) s += sep;
                s += (pretty ? std::string((indent + 1) * 2, ' ') : "") +
                     serialize(v.arr_val[i], pretty, indent + 1);
            }
            s += nl + pad + "]";
            return s;
        }
        case Type::Object: {
            if (v.obj_val.empty()) return "{}";
            std::string s = "{" + nl;
            bool first = true;
            for (const auto & [k, val] : v.obj_val) {
                if (!first) s += sep;
                first = false;
                s += (pretty ? std::string((indent + 1) * 2, ' ') : "") +
                     "\"" + escape_string(k) + "\"" + colon +
                     serialize(val, pretty, indent + 1);
            }
            s += nl + pad + "}";
            return s;
        }
    }
    return "null";
}

// ── Parser ─────────────────────────────────────────────────────

class Parser {
public:
    Parser(const std::string & src) : src_(src), pos_(0) {}

    Value parse() {
        skip_ws();
        Value v = parse_value();
        skip_ws();
        return v;
    }

private:
    const std::string & src_;
    size_t pos_;

    void skip_ws() {
        while (pos_ < src_.size() &&
               (src_[pos_] == ' ' || src_[pos_] == '\t' ||
                src_[pos_] == '\n' || src_[pos_] == '\r')) {
            pos_++;
        }
    }

    Value parse_value() {
        skip_ws();
        if (pos_ >= src_.size()) return Value::null();

        char c = src_[pos_];
        if (c == '{') return parse_object();
        if (c == '[') return parse_array();
        if (c == '"') return Value(parse_string());
        if (c == 't' || c == 'f') return parse_bool();
        if (c == 'n') return parse_null();
        return parse_number();
    }

    Value parse_object() {
        Value v = Value::object();
        pos_++; // skip {
        skip_ws();
        if (pos_ < src_.size() && src_[pos_] == '}') { pos_++; return v; }

        while (pos_ < src_.size()) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            if (pos_ < src_.size() && src_[pos_] == ':') pos_++;
            skip_ws();
            v.obj_val[key] = parse_value();
            skip_ws();
            if (pos_ < src_.size() && src_[pos_] == ',') { pos_++; continue; }
            if (pos_ < src_.size() && src_[pos_] == '}') { pos_++; break; }
            break;
        }
        return v;
    }

    Value parse_array() {
        Value v = Value::array();
        pos_++; // skip [
        skip_ws();
        if (pos_ < src_.size() && src_[pos_] == ']') { pos_++; return v; }

        while (pos_ < src_.size()) {
            v.arr_val.push_back(parse_value());
            skip_ws();
            if (pos_ < src_.size() && src_[pos_] == ',') { pos_++; skip_ws(); continue; }
            if (pos_ < src_.size() && src_[pos_] == ']') { pos_++; break; }
            break;
        }
        return v;
    }

    std::string parse_string() {
        std::string result;
        if (pos_ >= src_.size() || src_[pos_] != '"') return result;
        pos_++; // skip opening quote

        while (pos_ < src_.size() && src_[pos_] != '"') {
            char c = src_[pos_++];
            if (c == '\\' && pos_ < src_.size()) {
                char esc = src_[pos_++];
                switch (esc) {
                    case '"': result += '"'; break;
                    case '\\': result += '\\'; break;
                    case '/': result += '/'; break;
                    case 'n': result += '\n'; break;
                    case 't': result += '\t'; break;
                    case 'r': result += '\r'; break;
                    case 'b': result += '\b'; break;
                    case 'f': result += '\f'; break;
                    case 'u': {
                        if (pos_ + 4 <= src_.size()) {
                            unsigned int cp = 0;
                            for (int i = 0; i < 4; i++) {
                                cp <<= 4;
                                char h = src_[pos_++];
                                if (h >= '0' && h <= '9') cp |= h - '0';
                                else if (h >= 'a' && h <= 'f') cp |= h - 'a' + 10;
                                else if (h >= 'A' && h <= 'F') cp |= h - 'A' + 10;
                            }
                            // UTF-8 encode
                            if (cp < 0x80) result += (char)cp;
                            else if (cp < 0x800) {
                                result += (char)(0xC0 | (cp >> 6));
                                result += (char)(0x80 | (cp & 0x3F));
                            } else {
                                result += (char)(0xE0 | (cp >> 12));
                                result += (char)(0x80 | ((cp >> 6) & 0x3F));
                                result += (char)(0x80 | (cp & 0x3F));
                            }
                        }
                        break;
                    }
                    default: result += esc;
                }
            } else {
                result += c;
            }
        }
        if (pos_ < src_.size()) pos_++; // skip closing quote
        return result;
    }

    Value parse_number() {
        size_t start = pos_;
        if (pos_ < src_.size() && (src_[pos_] == '-' || src_[pos_] == '+')) pos_++;
        while (pos_ < src_.size() &&
               (std::isdigit((unsigned char)src_[pos_]) ||
                src_[pos_] == '.' || src_[pos_] == 'e' || src_[pos_] == 'E' ||
                src_[pos_] == '+' || src_[pos_] == '-')) {
            pos_++;
        }
        try {
            return Value(std::stod(src_.substr(start, pos_ - start)));
        } catch (...) {
            return Value(0.0);
        }
    }

    Value parse_bool() {
        if (pos_ + 4 <= src_.size() && src_.substr(pos_, 4) == "true") {
            pos_ += 4;
            return Value(true);
        }
        if (pos_ + 5 <= src_.size() && src_.substr(pos_, 5) == "false") {
            pos_ += 5;
            return Value(false);
        }
        return Value(false);
    }

    Value parse_null() {
        if (pos_ + 4 <= src_.size() && src_.substr(pos_, 4) == "null") {
            pos_ += 4;
        }
        return Value::null();
    }
};

inline Value parse(const std::string & src) {
    Parser p(src);
    return p.parse();
}

} // namespace json
