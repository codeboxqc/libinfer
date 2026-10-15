/* ========================================================================== */
/* infer_core.cpp (v2.2.0 - Production Implementation)                       */
/* ========================================================================== */
#include "infer.h"

#include <llama.h>
#include <ggml-backend.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <io.h>
#else
#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#endif

namespace {

constexpr uint32_t STATE_MAGIC          = 0x494E4655; // "INFU"
constexpr uint32_t STATE_VERSION        = 3;          // Version 3 header with CRC & history
constexpr uint32_t DEFAULT_BATCH_CAP    = 512;
constexpr uint32_t DEFAULT_FALLBACK_CTX = 4096;
constexpr size_t   MAX_STATE_FILE_SIZE  = 64ULL * 1024ULL * 1024ULL * 1024ULL; // 64GB bound

std::mutex g_backend_mutex;
int32_t    g_backend_ref_count = 0;

infer_log_cb g_user_log_cb   = nullptr;
void*        g_user_log_data = nullptr;
std::mutex   g_log_mutex;


#if defined(_WIN32)
static HANDLE g_instance_mutex = NULL;
#else
static int g_instance_fd = -1;
#endif

static bool get_module_dir_utf8(std::string& out) {
#if defined(_WIN32)
    HMODULE hm = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                            GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&get_module_dir_utf8), &hm)) {
        return false;
    }
    std::vector<wchar_t> buf(32768);
    DWORD n = GetModuleFileNameW(hm, buf.data(), static_cast<DWORD>(buf.size()));
    if (n == 0 || n >= buf.size()) return false;
    std::wstring w(buf.data(), n);
    size_t slash = w.find_last_of(L"\\/");
    if (slash == std::wstring::npos) return false;
    w.resize(slash);
    int len = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), nullptr, 0, nullptr, nullptr);
    if (len <= 0) return false;
    out.assign(static_cast<size_t>(len), '\0');
    return WideCharToMultiByte(CP_UTF8, 0, w.c_str(), static_cast<int>(w.size()), &out[0], len, nullptr, nullptr) > 0;
#else
    Dl_info info{};
    if (!dladdr(reinterpret_cast<void*>(&get_module_dir_utf8), &info) || !info.dli_fname) return false;
    std::string p = info.dli_fname;
    size_t slash = p.find_last_of('/');
    if (slash == std::string::npos) return false;
    out = p.substr(0, slash);
    return true;
#endif
}

bool ref_llama_backend() {
    std::lock_guard<std::mutex> lock(g_backend_mutex);
    if (g_backend_ref_count == 0) {
#if defined(_WIN32)
        g_instance_mutex = CreateMutexA(NULL, FALSE, "Global\\InferCoreSingleInstance");
        if (GetLastError() == ERROR_ALREADY_EXISTS) {
            if (g_instance_mutex) { CloseHandle(g_instance_mutex); g_instance_mutex = NULL; }
            return false;
        }
#else
        g_instance_fd = ::open("/tmp/infer_core_single_instance.lock", O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
        if (g_instance_fd < 0) return false;
        struct flock fl{};
        fl.l_type = F_WRLCK;
        fl.l_whence = SEEK_SET;
        if (::fcntl(g_instance_fd, F_SETLK, &fl) < 0) {
            ::close(g_instance_fd);
            g_instance_fd = -1;
            return false;
        }
#endif
        std::string dir;
        if (!get_module_dir_utf8(dir)) {
#if defined(_WIN32)
            if (g_instance_mutex) { CloseHandle(g_instance_mutex); g_instance_mutex = NULL; }
#else
            if (g_instance_fd >= 0) { ::close(g_instance_fd); g_instance_fd = -1; }
#endif
            return false;
        }
        ggml_backend_load_all_from_path(dir.c_str());
        llama_backend_init();
    }
    g_backend_ref_count++;
    return true;
}

void unref_llama_backend() {
    std::lock_guard<std::mutex> lock(g_backend_mutex);
    g_backend_ref_count--;
    if (g_backend_ref_count <= 0) {
        llama_backend_free();
#if defined(_WIN32)
        if (g_instance_mutex) { CloseHandle(g_instance_mutex); g_instance_mutex = NULL; }
#else
        if (g_instance_fd >= 0) { ::close(g_instance_fd); g_instance_fd = -1; }
#endif
        g_backend_ref_count = 0;
    }
}



infer_log_level map_log_level(enum ggml_log_level level) {
    switch (level) {
        case GGML_LOG_LEVEL_NONE:  return INFER_LOG_NONE;
        case GGML_LOG_LEVEL_DEBUG: return INFER_LOG_DEBUG;
        case GGML_LOG_LEVEL_INFO:  return INFER_LOG_INFO;
        case GGML_LOG_LEVEL_WARN:  return INFER_LOG_WARN;
        case GGML_LOG_LEVEL_ERROR: return INFER_LOG_ERROR;
        case GGML_LOG_LEVEL_CONT:  return INFER_LOG_CONT;
        default:                   return INFER_LOG_INFO;
    }
}

void noop_ggml_log(enum ggml_log_level, const char*, void*) {}

void internal_ggml_log(enum ggml_log_level level, const char* text, void* user_data) {
    (void)user_data;
    infer_log_cb cb   = nullptr;
    void*        data = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        cb   = g_user_log_cb;
        data = g_user_log_data;
    }
    if (cb) {
        cb(map_log_level(level), text, data);
    }
}

static std::atomic<uint32_t> g_tmp_counter{0};

static std::string make_tmp_path(const char* dst) {
#if defined(_WIN32)
    unsigned long pid = GetCurrentProcessId();
#else
    unsigned long pid = static_cast<unsigned long>(getpid());
#endif
    return std::string(dst) + ".tmp." + std::to_string(pid) + "." +
           std::to_string(g_tmp_counter.fetch_add(1, std::memory_order_relaxed));
}

FILE* open_file_utf8(const char* path, const char* mode) {
#if defined(_WIN32)
    int wpath_len = MultiByteToWideChar(CP_UTF8, 0, path, -1, nullptr, 0);
    int wmode_len = MultiByteToWideChar(CP_UTF8, 0, mode, -1, nullptr, 0);
    if (wpath_len > 0 && wmode_len > 0) {
        std::vector<wchar_t> wpath(wpath_len);
        std::vector<wchar_t> wmode(wmode_len);
        MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath.data(), wpath_len);
        MultiByteToWideChar(CP_UTF8, 0, mode, -1, wmode.data(), wmode_len);
        return _wfopen(wpath.data(), wmode.data());
    }
    return nullptr;
#else
    return std::fopen(path, mode);
#endif
}

bool remove_file_utf8(const char* path) {
#if defined(_WIN32)
    int wpath_len = MultiByteToWideChar(CP_UTF8, 0, path, -1, nullptr, 0);
    if (wpath_len > 0) {
        std::vector<wchar_t> wpath(wpath_len);
        MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath.data(), wpath_len);
        return _wremove(wpath.data()) == 0;
    }
    return false;
#else
    return std::remove(path) == 0;
#endif
}

struct TmpGuard {
    std::string path;
    bool armed = false;
    ~TmpGuard() {
        if (armed) {
            remove_file_utf8(path.c_str());
        }
    }
};

static FILE* open_tmp_exclusive(const std::string& path) {
#if defined(_WIN32)
    int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    if (wlen <= 0) return nullptr;
    std::vector<wchar_t> w(static_cast<size_t>(wlen));
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), wlen);
    return _wfopen(w.data(), L"wbx");
#else
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0644);
    if (fd < 0) return nullptr;
    FILE* f = ::fdopen(fd, "wb");
    if (!f) { ::close(fd); ::unlink(path.c_str()); }
    return f;
#endif
}

static bool sync_file(FILE* f) {
    if (std::fflush(f) != 0) return false;
#if defined(_WIN32)
    return _commit(_fileno(f)) == 0;
#else
    int fd = ::fileno(f);
#if defined(__APPLE__) && defined(F_FULLFSYNC)
    if (::fcntl(fd, F_FULLFSYNC) == 0) return true;
#endif
    return ::fsync(fd) == 0;
#endif
}

static void sync_parent_dir(const char* filepath) {
#if !defined(_WIN32)
    std::string path(filepath);
    size_t slash = path.find_last_of('/');
    std::string dir = (slash != std::string::npos) ? path.substr(0, slash) : ".";
    int fd = ::open(dir.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd >= 0) {
        ::fsync(fd);
        ::close(fd);
    }
#else
    (void)filepath;
#endif
}

bool file_size_64(FILE* f, uint64_t& out) {
#if defined(_WIN32)
    if (_fseeki64(f, 0, SEEK_END) != 0) return false;
    __int64 pos = _ftelli64(f);
    if (pos < 0) return false;
    out = static_cast<uint64_t>(pos);
    return _fseeki64(f, 0, SEEK_SET) == 0;
#else
    if (fseeko(f, 0, SEEK_END) != 0) return false;
    off_t pos = ftello(f);
    if (pos < 0) return false;
    out = static_cast<uint64_t>(pos);
    return fseeko(f, 0, SEEK_SET) == 0;
#endif
}

static uint32_t compute_crc32(const uint8_t* data, size_t len) {
    if (!data || len == 0) return 0;
    uint32_t crc = 0xFFFFFFFF;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int j = 0; j < 8; ++j) {
            crc = (crc >> 1) ^ (0xEDB88320 & (-(crc & 1)));
        }
    }
    return ~crc;
}

size_t find_valid_utf8_end(const std::string& s) {
    if (s.empty()) return 0;
    for (size_t back = 1; back <= std::min<size_t>(4, s.size()); ++back) {
        uint8_t b = static_cast<uint8_t>(s[s.size() - back]);
        if ((b & 0x80) == 0x00) {
            return s.size();
        }
        if ((b & 0xC0) == 0xC0) {
            size_t expected = ((b & 0xE0) == 0xC0) ? 2 : (((b & 0xF0) == 0xE0) ? 3 : (((b & 0xF8) == 0xF0) ? 4 : 1));
            if (back == expected) {
                return s.size();
            }
            if (back < expected) {
                return s.size() - back;
            }
            return s.size();
        }
    }
    return s.size();
}

bool want_bos(const struct llama_vocab* v, const char* text) {
    if (!v || !text || !llama_vocab_get_add_bos(v)) return false;
    llama_token bos_id = llama_vocab_bos(v);
    if (bos_id == LLAMA_TOKEN_NULL) return false;
    const char* bos = llama_vocab_get_text(v, bos_id);
    return !(bos && std::strncmp(text, bos, std::strlen(bos)) == 0);
}

static bool is_plain_html_tag(const char* t, size_t len) {
    static const char* const tags[] = {
        "a","b","i","u","p","br","hr","em","li","ol","ul","tr","td","th","sub","sup",
        "h1","h2","h3","h4","h5","h6","div","img","pre","span","code","table","thead",
        "tbody","tfoot","caption","strong","blockquote"
    };
    const char* name = t + 1;
    size_t n = len - 2;
    if (n > 0 && name[0] == '/') { ++name; --n; }
    for (const char* tag : tags) {
        if (n == std::strlen(tag) && std::memcmp(name, tag, n) == 0) return true;
    }
    return false;
}

static std::vector<std::string> collect_special_texts(const struct llama_vocab* v) {
    std::vector<std::string> out;
    if (!v) return out;
    const int32_t n = llama_vocab_n_tokens(v);
    for (int32_t i = 0; i < n; ++i) {
        const auto attr = llama_vocab_get_attr(v, i);
        if (!(attr & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED))) continue;
        const char* t = llama_vocab_get_text(v, i);
        if (!t) continue;
        const size_t len = std::strlen(t);
        if (len < 3) continue;

        const bool is_control  = (attr & LLAMA_TOKEN_ATTR_CONTROL) != 0;
        const bool angle       = t[0] == '<' && t[len - 1] == '>';
        const bool square      = t[0] == '[' && t[len - 1] == ']';
        const bool pipe_marker = angle && len >= 5 && t[1] == '|' && t[len - 2] == '|';

        const bool user_marker = pipe_marker || (angle && !is_plain_html_tag(t, len));
        if (is_control ? (angle || square) : user_marker) out.emplace_back(t);
    }
    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

struct SpecialTokenLookup {
    std::unordered_set<std::string_view> set;
    size_t min_len = SIZE_MAX;
    size_t max_len = 0;

    void build(const std::vector<std::string>& specials) {
        set.clear();
        min_len = SIZE_MAX;
        max_len = 0;
        for (const auto& sp : specials) {
            set.insert(sp);
            if (sp.size() < min_len) min_len = sp.size();
            if (sp.size() > max_len) max_len = sp.size();
        }
    }
};

static std::string neutralize_special_fast(const char* in, const SpecialTokenLookup& lookup) {
    if (!in || !*in) return "";
    if (lookup.set.empty()) return std::string(in);

    static const char ZWSP[] = "\xE2\x80\x8B";
    const std::string_view s(in);
    std::string out;
    out.reserve(s.size());

    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        bool hit = false;
        if (c == '<' || c == '[') {
            const size_t rem = s.size() - i;
            const size_t max_check = std::min(rem, lookup.max_len);
            for (size_t len = lookup.min_len; len <= max_check; ++len) {
                char expected_close = (c == '<') ? '>' : ']';
                if (s[i + len - 1] == expected_close) {
                    if (lookup.set.count(s.substr(i, len)) > 0) {
                        hit = true;
                        break;
                    }
                }
            }
        }
        out.push_back(c);
        if (hit) out.append(ZWSP);
    }
    return out;
}

static bool is_valid_role(const char* r) {
    if (!r) return false;
    static const char* const ok[] = {"system", "user", "assistant", "tool", "developer"};
    for (const char* o : ok) if (std::strcmp(r, o) == 0) return true;
    return false;
}

static uint64_t compute_model_fingerprint(struct llama_model* model) {
    if (!model) return 0;
    uint64_t h = 14695981039346656037ULL;
    auto add_u64 = [&h](uint64_t val) {
        for (int i = 0; i < 8; ++i) {
            h ^= (val & 0xFF);
            h *= 1099511628211ULL;
            val >>= 8;
        }
    };
    add_u64(static_cast<uint64_t>(llama_model_n_ctx_train(model)));
    add_u64(static_cast<uint64_t>(llama_model_n_embd(model)));
    add_u64(static_cast<uint64_t>(llama_model_n_layer(model)));
    add_u64(static_cast<uint64_t>(llama_model_n_head(model)));
    add_u64(static_cast<uint64_t>(llama_model_size(model)));
    add_u64(static_cast<uint64_t>(llama_model_n_params(model)));

    const struct llama_vocab* v = llama_model_get_vocab(model);
    add_u64(v ? static_cast<uint64_t>(llama_vocab_n_tokens(v)) : 0);

    char desc[256] = {};
    llama_model_desc(model, desc, sizeof(desc));
    for (const char* p = desc; *p; ++p) {
        h ^= static_cast<uint8_t>(*p);
        h *= 1099511628211ULL;
    }
    return h;
}

static bool safe_kv_seq_rm(struct llama_context* lctx, int seq_id, int p0, int p1) {
    if (!lctx) return false;
    bool ok = llama_memory_seq_rm(llama_get_memory(lctx), seq_id, p0, p1);
    if (!ok && (p0 != 0 || p1 != -1)) {
        llama_memory_clear(llama_get_memory(lctx), false);
        return false;
    }
    return ok;
}

class StreamOutputBuffer {
public:
    StreamOutputBuffer(infer_stream_cb cb, void* udata, const std::vector<std::string>& stops)
        : callback_(cb), user_data_(udata), stop_words_(stops) {
        for (const auto& w : stop_words_) {
            if (w.size() > max_stop_len_) {
                max_stop_len_ = w.size();
            }
        }
    }

    bool append_and_flush(const char* piece, size_t len) {
        if (user_aborted_) return false;
        buffer_.append(piece, len);

        size_t earliest_pos = std::string::npos;
        for (const auto& stop : stop_words_) {
            if (stop.empty()) continue;
            size_t pos = buffer_.find(stop);
            if (pos != std::string::npos) {
                if (earliest_pos == std::string::npos || pos < earliest_pos) {
                    earliest_pos = pos;
                }
            }
        }

        if (earliest_pos != std::string::npos) {
            hit_stop_ = true;
            stop_abs_ = flushed_ + earliest_pos;
            if (earliest_pos > 0) {
                std::string pre = buffer_.substr(0, earliest_pos);
                size_t valid = find_valid_utf8_end(pre);
                if (valid > 0 && !user_aborted_) {
                    if (!callback_(pre.substr(0, valid).c_str(), user_data_)) {
                        user_aborted_ = true;
                    }
                }
            }
            buffer_.clear();
            return false;
        }

        size_t keep_len = (max_stop_len_ > 1) ? (max_stop_len_ - 1) : 0;
        if (buffer_.size() <= keep_len) {
            return true;
        }

        size_t candidate = buffer_.size() - keep_len;
        while (candidate > 0 && candidate < buffer_.size() &&
               (static_cast<uint8_t>(buffer_[candidate]) & 0xC0) == 0x80) {
            candidate--;
        }

        std::string candidate_slice = buffer_.substr(0, candidate);
        size_t flushable = find_valid_utf8_end(candidate_slice);

        if (flushable > 0) {
            std::string to_emit = buffer_.substr(0, flushable);
            buffer_.erase(0, flushable);
            flushed_ += flushable;
            if (!callback_(to_emit.c_str(), user_data_)) {
                user_aborted_ = true;
                return false;
            }
        }

        return true;
    }

    bool flush_remaining() {
        if (hit_stop_ || user_aborted_) return true;

        size_t earliest_pos = std::string::npos;
        for (const auto& stop : stop_words_) {
            if (stop.empty()) continue;
            size_t pos = buffer_.find(stop);
            if (pos != std::string::npos) {
                if (earliest_pos == std::string::npos || pos < earliest_pos) {
                    earliest_pos = pos;
                }
            }
        }

        if (earliest_pos != std::string::npos) {
            hit_stop_ = true;
            stop_abs_ = flushed_ + earliest_pos;
            if (earliest_pos > 0) {
                std::string pre = buffer_.substr(0, earliest_pos);
                size_t valid = find_valid_utf8_end(pre);
                if (valid > 0 && !user_aborted_) {
                    if (!callback_(pre.substr(0, valid).c_str(), user_data_)) {
                        user_aborted_ = true;
                    }
                }
            }
            buffer_.clear();
            return true;
        }

        if (!buffer_.empty()) {
            size_t valid = find_valid_utf8_end(buffer_);
            if (valid > 0 && !user_aborted_) {
                if (!callback_(buffer_.substr(0, valid).c_str(), user_data_)) {
                    user_aborted_ = true;
                }
            }
            flushed_ += valid;
            buffer_.clear();
        }
        return true;
    }

    bool hit_stop() const { return hit_stop_; }
    size_t stop_offset() const { return stop_abs_; }
    size_t total_flushed_bytes() const { return flushed_; }
    bool user_aborted() const { return user_aborted_; }

private:
    infer_stream_cb callback_;
    void* user_data_;
    const std::vector<std::string>& stop_words_;
    size_t max_stop_len_ = 0;
    std::string buffer_;
    size_t flushed_    = 0;
    size_t stop_abs_   = 0;
    bool hit_stop_     = false;
    bool user_aborted_ = false;
};

#pragma pack(push, 1)
struct StateFileHeader {
    uint32_t magic;         // STATE_MAGIC
    uint32_t version;       // STATE_VERSION (3)
    uint32_t n_ctx;
    uint32_t n_past;
    uint32_t vocab_size;
    uint32_t n_embd;
    uint32_t n_layer;
    uint32_t n_head;
    uint64_t fingerprint;
    uint32_t payload_crc;   // CRC32 of binary KV payload
    uint32_t history_bytes; // Byte length of serialized chat history
    uint64_t data_size;
};
#pragma pack(pop)

struct FileCloser {
    void operator()(FILE* f) const noexcept {
        if (f) std::fclose(f);
    }
};
using ScopedFile = std::unique_ptr<FILE, FileCloser>;

struct OwnedMsg {
    std::string role;
    std::string content;
};

} // anonymous namespace

struct infer_ctx {
    struct llama_model*   model        = nullptr;
    struct llama_context* lctx         = nullptr;
    struct llama_sampler* sampler      = nullptr;
    struct llama_sampler* penalty_smpl = nullptr;

    infer_sampler_params     sampler_params{};
    std::string              grammar_gbnf;
    std::vector<std::string> stop_words;
    std::vector<std::string> special_texts;
    SpecialTokenLookup       special_lookup;
    uint32_t                 n_ctx  = 0;
    int                      n_keep = 0;

    std::atomic<int>      n_past{0};
    std::atomic<bool>     abort_flag{false};
    std::atomic<bool>     is_generating{false};
    std::atomic<bool>     last_success{true};
    std::atomic<uint64_t> gen_counter{0};

    std::vector<OwnedMsg>    chat_history;
    std::vector<llama_token> active_kv_tokens;

    mutable std::mutex      gen_mutex;
    std::condition_variable cv_generating;
    std::thread             async_worker;
    mutable std::mutex      worker_mutex;
    mutable std::mutex      ctx_mutex;

    mutable std::mutex err_mutex;
    std::string        last_error;

    void set_error(const std::string& err) {
        std::lock_guard<std::mutex> lock(err_mutex);
        last_error = err;
    }

    void clear_error() {
        std::lock_guard<std::mutex> lock(err_mutex);
        last_error.clear();
    }
};

namespace {

static void release_gen(infer_ctx* c) {
    if (!c) return;
    std::lock_guard<std::mutex> lk(c->gen_mutex);
    c->is_generating.store(false, std::memory_order_release);
    c->cv_generating.notify_all();
}

struct GenAcquireGuard {
    infer_ctx* ctx = nullptr;
    bool acquired = false;

    explicit GenAcquireGuard(infer_ctx* c) : ctx(c) {
        if (!ctx) return;
        bool expected = false;
        if (ctx->is_generating.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            acquired = true;
            ctx->gen_counter.fetch_add(1, std::memory_order_relaxed);
        }
    }

    ~GenAcquireGuard() {
        if (acquired && ctx) {
            release_gen(ctx);
        }
    }
};

struct PrefillRollbackGuard {
    struct llama_context* lctx = nullptr;
    int start_past = 0;
    bool completed = false;

    ~PrefillRollbackGuard() {
        if (!completed && lctx) {
            safe_kv_seq_rm(lctx, 0, start_past, -1);
        }
    }
};

static bool infer_has_real_gpu() {
    size_t count = ggml_backend_dev_count();
    for (size_t i = 0; i < count; ++i) {
        auto type = ggml_backend_dev_type(ggml_backend_dev_get(i));
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) return true;
    }
    return false;
}

static ggml_backend_dev_t infer_get_optimal_device() {
    size_t dev_count = ggml_backend_dev_count();
    ggml_backend_dev_t fallback_dev = nullptr;
    ggml_backend_dev_t igpu_dev     = nullptr;

    for (size_t i = 0; i < dev_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU) {
            return dev;
        }
        if (type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            igpu_dev = dev;
        }
    }
    return igpu_dev ? igpu_dev : fallback_dev;
}

static int32_t pick_threads(bool has_gpu, bool batch_phase) {
    if (const char* e = std::getenv("INFER_N_THREADS")) {
        int v = std::atoi(e);
        if (v > 0) return std::min(v, 256);
    }
    uint32_t hc = std::thread::hardware_concurrency();
    if (hc == 0) hc = 4;
    if (has_gpu) {
        return static_cast<int32_t>(batch_phase ? std::min<uint32_t>(hc, 8) : std::min<uint32_t>(std::max<uint32_t>(1, hc / 2), 4));
    }
    return static_cast<int32_t>(hc);
}

static uint32_t calculate_safe_n_ctx(struct llama_model* model, uint32_t requested_ctx) {
    uint32_t max_train = static_cast<uint32_t>(llama_model_n_ctx_train(model));
    uint32_t target_ctx = (requested_ctx > 0) ? requested_ctx : DEFAULT_FALLBACK_CTX;
    if (max_train > 0) {
        target_ctx = std::min<uint32_t>(target_ctx, max_train);
    }

    int32_t n_layer   = llama_model_n_layer(model);
    int32_t n_head_kv = llama_model_n_head_kv(model);
    int32_t n_head    = llama_model_n_head(model);
    int32_t n_embd    = llama_model_n_embd(model);

    if (n_head <= 0 || n_layer <= 0 || n_head_kv <= 0 || n_embd <= 0) {
        return target_ctx;
    }

    int32_t head_dim_k = n_embd / n_head;
    int32_t head_dim_v = head_dim_k;

    constexpr size_t bytes_per_element = 2; // GGML_TYPE_F16
    size_t bytes_per_token = static_cast<size_t>(n_layer) * n_head_kv * (head_dim_k + head_dim_v) * bytes_per_element;

    ggml_backend_dev_t dev = infer_get_optimal_device();
    size_t free_vram = 0, total_vram = 0;

    if (dev) {
        ggml_backend_dev_memory(dev, &free_vram, &total_vram);
    }

    if (free_vram == 0 || bytes_per_token == 0) {
        return target_ctx;
    }

    constexpr size_t headroom = 256ULL * 1024ULL * 1024ULL;
    if (free_vram <= headroom) {
        return std::min<uint32_t>(target_ctx, 512);
    }

    size_t available_vram = free_vram - headroom;
    uint32_t safe_tokens = static_cast<uint32_t>(available_vram / bytes_per_token);

    uint32_t floor_ctx = std::min<uint32_t>(target_ctx, 512);
    return std::max<uint32_t>(floor_ctx, std::min<uint32_t>(target_ctx, safe_tokens));
}

static struct llama_sampler* create_sampler(
    const infer_sampler_params& params_in,
    struct llama_model* model,
    const std::string& grammar_gbnf,
    std::string& err_out,
    struct llama_sampler** penalty_smpl_out = nullptr
) {
    if (penalty_smpl_out) *penalty_smpl_out = nullptr;

    infer_sampler_params params = params_in;
    if (std::isnan(params.temperature) || std::isinf(params.temperature) || params.temperature < 0.0f) {
        err_out = "Invalid temperature parameter.";
        return nullptr;
    }
    if (std::isnan(params.top_p) || std::isinf(params.top_p) || params.top_p < 0.0f || params.top_p > 1.0f) {
        err_out = "Invalid top_p parameter.";
        return nullptr;
    }
    if (std::isnan(params.min_p) || std::isinf(params.min_p) || params.min_p < 0.0f || params.min_p > 1.0f) {
        err_out = "Invalid min_p parameter.";
        return nullptr;
    }
    if (std::isnan(params.repeat_penalty) || std::isinf(params.repeat_penalty) || params.repeat_penalty < 0.0f) {
        err_out = "Invalid repeat_penalty parameter.";
        return nullptr;
    }

    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    struct llama_sampler* chain = llama_sampler_chain_init(sparams);
    if (!chain) {
        err_out = "Failed to allocate sampler chain.";
        return nullptr;
    }

    const struct llama_vocab* vocab = model ? llama_model_get_vocab(model) : nullptr;

    if (params.repeat_penalty > 0.0f && params.repeat_penalty != 1.0f && vocab) {
        int32_t last_n = (params.repeat_last_n > 0) ? params.repeat_last_n : 64;
        int32_t n_vocab = llama_vocab_n_tokens(vocab);
        struct llama_sampler* psmpl = llama_sampler_init_penalties(n_vocab, last_n, params.repeat_penalty, 0.0f, 0.0f);
        if (psmpl) {
            llama_sampler_chain_add(chain, psmpl);
            if (penalty_smpl_out) *penalty_smpl_out = psmpl;
        }
    }

    if (!grammar_gbnf.empty() && vocab) {
        struct llama_sampler* gsampler = llama_sampler_init_grammar(vocab, grammar_gbnf.c_str(), "root");
        if (gsampler) {
            llama_sampler_chain_add(chain, gsampler);
        } else {
            err_out = "Failed to parse grammar rules.";
            llama_sampler_free(chain);
            return nullptr;
        }
    }

    if (params.top_k > 0) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_k(params.top_k));
    }
    if (params.top_p > 0.0f && params.top_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_top_p(params.top_p, 1));
    }
    if (params.min_p > 0.0f && params.min_p < 1.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_min_p(params.min_p, 1));
    }

    if (params.temperature > 0.0f) {
        llama_sampler_chain_add(chain, llama_sampler_init_temp(params.temperature));
        llama_sampler_chain_add(chain, llama_sampler_init_dist(params.seed ? params.seed : LLAMA_DEFAULT_SEED));
    } else {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    }

    return chain;
}

} // anonymous namespace

static bool execute_decode(
    infer_ctx* ctx,
    const std::vector<llama_token>& prompt_tokens,
    bool append_to_existing_past,
    int32_t max_tokens,
    infer_stream_cb callback,
    void* user_data,
    std::string* out_generated_text = nullptr
) {
    if (out_generated_text) out_generated_text->clear();

    const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
    const int effective_batch_cap = std::min<int>(DEFAULT_BATCH_CAP, static_cast<int>(llama_n_batch(ctx->lctx)));
    size_t prefix_len = 0;

    if (append_to_existing_past) {
        while (prefix_len < ctx->active_kv_tokens.size() &&
               prefix_len < prompt_tokens.size() &&
               ctx->active_kv_tokens[prefix_len] == prompt_tokens[prefix_len]) {
            prefix_len++;
        }

        if (prefix_len == prompt_tokens.size() && prefix_len > 0) {
            prefix_len--;
        }

        if (prefix_len < ctx->active_kv_tokens.size()) {
            if (!safe_kv_seq_rm(ctx->lctx, 0, static_cast<int>(prefix_len), -1)) {
                prefix_len = 0;
            }
            ctx->active_kv_tokens.resize(prefix_len);
            ctx->n_past.store(static_cast<int>(prefix_len), std::memory_order_relaxed);
        }

        int current_past = ctx->n_past.load(std::memory_order_relaxed);
        size_t tokens_to_add = prompt_tokens.size() - prefix_len;

        while (static_cast<uint32_t>(current_past + tokens_to_add) >= ctx->n_ctx) {
            if (!llama_memory_can_shift(llama_get_memory(ctx->lctx))) {
                ctx->set_error("Context full and KV cache does not support shifting.");
                return false;
            }
            int keep_budget = (ctx->n_keep > 0) ? ctx->n_keep : 64;
            int n_keep = std::min<int>(keep_budget, static_cast<int>(ctx->n_ctx) / 4);
            if (n_keep <= 0) n_keep = 1;
            if (static_cast<int>(ctx->n_ctx) <= n_keep + 1) {
                ctx->set_error("Context capacity too small for shift management.");
                return false;
            }
            int needed = static_cast<int>(current_past + tokens_to_add - ctx->n_ctx + 16);
            int n_discard = std::max(needed, (static_cast<int>(ctx->n_ctx) - n_keep) / 2);
            if (current_past - n_discard <= n_keep) {
                n_discard = current_past - n_keep;
            }
            if (n_discard <= 0) {
                ctx->set_error("Prompt does not fit in context even after shifting.");
                return false;
            }
            if (!safe_kv_seq_rm(ctx->lctx, 0, n_keep, n_keep + n_discard)) {
                ctx->set_error("KV sequence shift refused by memory architecture.");
                return false;
            }
            llama_memory_seq_add(llama_get_memory(ctx->lctx), 0, n_keep + n_discard, current_past, -n_discard);
            current_past -= n_discard;

            if (ctx->active_kv_tokens.size() > static_cast<size_t>(n_keep + n_discard)) {
                ctx->active_kv_tokens.erase(
                    ctx->active_kv_tokens.begin() + n_keep,
                    ctx->active_kv_tokens.begin() + n_keep + n_discard
                );
            } else if (ctx->active_kv_tokens.size() > static_cast<size_t>(n_keep)) {
                ctx->active_kv_tokens.resize(n_keep);
            }

            ctx->n_past.store(current_past, std::memory_order_relaxed);
        }
    } else {
        llama_memory_clear(llama_get_memory(ctx->lctx), false);
        ctx->active_kv_tokens.clear();
        ctx->n_past.store(0, std::memory_order_relaxed);
    }

    if (ctx->sampler) {
        llama_sampler_reset(ctx->sampler);
    }

    if (ctx->penalty_smpl) {
        for (llama_token t : prompt_tokens) {
            llama_sampler_accept(ctx->penalty_smpl, t);
        }
    }

    int start_past = append_to_existing_past ? ctx->n_past.load(std::memory_order_relaxed) : 0;

    if (start_past == 0) {
        ctx->n_keep = std::min<int>(static_cast<int>(prompt_tokens.size()), static_cast<int>(ctx->n_ctx) / 4);
    }

    std::vector<llama_token> delta_prompt(prompt_tokens.begin() + prefix_len, prompt_tokens.end());

    if (static_cast<uint32_t>(start_past + delta_prompt.size()) >= ctx->n_ctx) {
        ctx->set_error("Prompt tokens exceed context capacity.");
        return false;
    }

    struct llama_batch batch = llama_batch_init(effective_batch_cap, 0, 1);
    struct BatchGuard {
        struct llama_batch& b;
        ~BatchGuard() { llama_batch_free(b); }
    } batch_guard{batch};

    PrefillRollbackGuard prefill_guard{ctx->lctx, start_past, false};

    if (!delta_prompt.empty()) {
        for (size_t i = 0; i < delta_prompt.size(); i += batch.n_tokens) {
            if (ctx->abort_flag.load(std::memory_order_relaxed)) {
                ctx->set_error("Generation aborted during prompt prefill.");
                return false;
            }

            batch.n_tokens = 0;
            int chunk = std::min<int>(effective_batch_cap, static_cast<int>(delta_prompt.size() - i));

            for (int j = 0; j < chunk; ++j) {
                size_t idx = i + j;
                batch.token[j]     = delta_prompt[idx];
                batch.pos[j]       = static_cast<llama_pos>(start_past + idx);
                batch.n_seq_id[j]  = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j]    = (idx == delta_prompt.size() - 1);
                batch.n_tokens++;
            }

            if (llama_decode(ctx->lctx, batch) != 0) {
                ctx->set_error("llama_decode failed during prompt prefill.");
                return false;
            }
        }

        ctx->active_kv_tokens.insert(ctx->active_kv_tokens.end(), delta_prompt.begin(), delta_prompt.end());
    }

    prefill_guard.completed = true;

    int n_cur = start_past + static_cast<int>(delta_prompt.size());
    ctx->n_past.store(n_cur, std::memory_order_relaxed);

    std::vector<std::string> active_stops;
    {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        active_stops = ctx->stop_words;
    }

    StreamOutputBuffer out_buf(callback, user_data, active_stops);
    std::vector<char> piece_buf(256);

    std::vector<std::pair<int, size_t>> tok_log;
    size_t cum_bytes = 0;
    const bool track_stops = !active_stops.empty();
    bool rolled_back = false;

    int32_t generated_tokens = 0;
    const int32_t hard_cap    = static_cast<int32_t>(std::min<uint32_t>(ctx->n_ctx, 1u << 28)) * 4;
    const int32_t token_limit = (max_tokens > 0) ? max_tokens : hard_cap;
    bool execution_success = true;

    while (generated_tokens < token_limit) {
        if (ctx->abort_flag.load(std::memory_order_relaxed)) {
            ctx->set_error("Generation aborted by caller.");
            execution_success = false;
            break;
        }

        if (n_cur >= static_cast<int>(ctx->n_ctx) - 4) {
            if (!llama_memory_can_shift(llama_get_memory(ctx->lctx))) {
                ctx->set_error("Context exhausted and KV cache does not support shifting.");
                execution_success = false;
                break;
            }

            int keep_budget = (ctx->n_keep > 0) ? ctx->n_keep : 64;
            int n_keep = std::min<int>(keep_budget, static_cast<int>(ctx->n_ctx) / 4);
            if (n_keep <= 0) n_keep = 1;
            int n_discard = (ctx->n_ctx - n_keep) / 2;
            if (n_discard <= 0) n_discard = 1;

            if (!safe_kv_seq_rm(ctx->lctx, 0, n_keep, n_keep + n_discard)) {
                ctx->set_error("Context shift refused by memory architecture.");
                execution_success = false;
                break;
            }
            llama_memory_seq_add(llama_get_memory(ctx->lctx), 0, n_keep + n_discard, n_cur, -n_discard);
            n_cur -= n_discard;

            if (ctx->active_kv_tokens.size() > static_cast<size_t>(n_keep + n_discard)) {
                ctx->active_kv_tokens.erase(
                    ctx->active_kv_tokens.begin() + n_keep,
                    ctx->active_kv_tokens.begin() + n_keep + n_discard
                );
            } else if (ctx->active_kv_tokens.size() > static_cast<size_t>(n_keep)) {
                ctx->active_kv_tokens.resize(n_keep);
            }

            if (track_stops) {
                std::vector<std::pair<int, size_t>> new_tok_log;
                for (const auto& e : tok_log) {
                    if (e.first < n_keep) {
                        new_tok_log.push_back(e);
                    } else if (e.first >= n_keep + n_discard) {
                        new_tok_log.emplace_back(e.first - n_discard, e.second);
                    }
                }
                tok_log = std::move(new_tok_log);
            }

            ctx->n_past.store(n_cur, std::memory_order_relaxed);
        }

        llama_token new_token_id = llama_sampler_sample(ctx->sampler, ctx->lctx, -1);

        if (llama_vocab_is_eog(vocab, new_token_id)) {
            if (append_to_existing_past) {
                batch.n_tokens = 0;
                batch.token[0]     = new_token_id;
                batch.pos[0]       = static_cast<llama_pos>(n_cur);
                batch.n_seq_id[0]  = 1;
                batch.seq_id[0][0] = 0;
                batch.logits[0]    = false;
                batch.n_tokens     = 1;
                if (llama_decode(ctx->lctx, batch) == 0) {
                    ctx->active_kv_tokens.push_back(new_token_id);
                    n_cur++;
                }
            }
            break;
        }

      int n_chars = llama_token_to_piece(vocab, new_token_id, piece_buf.data(), piece_buf.size(), 0, false);
        if (n_chars < 0) {
            piece_buf.resize(-n_chars + 1);
            n_chars = llama_token_to_piece(vocab, new_token_id, piece_buf.data(), piece_buf.size(), 0, false);
            if (n_chars < 0) {
                // Vocabulary mapping critically failed twice; abort this token safely
                ctx->set_error("Token piece translation failed repeatedly.");
                execution_success = false;
                break;
            }
        }

        if (n_chars > 0) {
            cum_bytes += static_cast<size_t>(n_chars);
            if (out_generated_text) {
                out_generated_text->append(piece_buf.data(), static_cast<size_t>(n_chars));
            }

            if (!out_buf.append_and_flush(piece_buf.data(), static_cast<size_t>(n_chars))) {
                if (out_buf.user_aborted()) {
                    ctx->abort_flag.store(true, std::memory_order_release);
                    ctx->set_error("Generation halted by stream callback.");
                    execution_success = false;
                }

                if (out_buf.hit_stop() && track_stops && !rolled_back) {
                    for (const auto& e : tok_log) {
                        if (e.second > out_buf.stop_offset()) {
                            safe_kv_seq_rm(ctx->lctx, 0, e.first, -1);
                            n_cur = e.first;
                            if (ctx->active_kv_tokens.size() > static_cast<size_t>(n_cur)) {
                                ctx->active_kv_tokens.resize(n_cur);
                            }
                            rolled_back = true;
                            break;
                        }
                    }
                }

                if (append_to_existing_past) {
                    llama_token term_token = llama_vocab_eot(vocab);
                    if (term_token == LLAMA_TOKEN_NULL) {
                        term_token = llama_vocab_eos(vocab);
                    }
                    if (term_token != LLAMA_TOKEN_NULL) {
                        batch.n_tokens = 0;
                        batch.token[0]     = term_token;
                        batch.pos[0]       = static_cast<llama_pos>(n_cur);
                        batch.n_seq_id[0]  = 1;
                        batch.seq_id[0][0] = 0;
                        batch.logits[0]    = false;
                        batch.n_tokens     = 1;
                        if (llama_decode(ctx->lctx, batch) == 0) {
                            ctx->active_kv_tokens.push_back(term_token);
                            n_cur++;
                        }
                    }
                }
                break;
            }
        }

        batch.n_tokens = 0;
        batch.token[0]     = new_token_id;
        batch.pos[0]       = static_cast<llama_pos>(n_cur);
        batch.n_seq_id[0]  = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0]    = true;
        batch.n_tokens     = 1;

        if (llama_decode(ctx->lctx, batch) != 0) {
            out_buf.flush_remaining();
            ctx->set_error("llama_decode failed during token generation.");
            ctx->n_past.store(n_cur, std::memory_order_relaxed);
            return false;
        }

        ctx->active_kv_tokens.push_back(new_token_id);
        if (track_stops) tok_log.emplace_back(n_cur, cum_bytes);
        n_cur++;
        generated_tokens++;
        ctx->n_past.store(n_cur, std::memory_order_relaxed);
    }

    if (generated_tokens >= token_limit && append_to_existing_past) {
        llama_token term_token = llama_vocab_eot(vocab);
        if (term_token == LLAMA_TOKEN_NULL) {
            term_token = llama_vocab_eos(vocab);
        }
        if (term_token != LLAMA_TOKEN_NULL) {
            batch.n_tokens = 0;
            batch.token[0]     = term_token;
            batch.pos[0]       = static_cast<llama_pos>(n_cur);
            batch.n_seq_id[0]  = 1;
            batch.seq_id[0][0] = 0;
            batch.logits[0]    = false;
            batch.n_tokens     = 1;
            if (llama_decode(ctx->lctx, batch) == 0) {
                ctx->active_kv_tokens.push_back(term_token);
                n_cur++;
            }
        }
    }

    out_buf.flush_remaining();

    if (out_buf.hit_stop() && track_stops && !rolled_back) {
        for (const auto& e : tok_log) {
            if (e.second > out_buf.stop_offset()) {
                safe_kv_seq_rm(ctx->lctx, 0, e.first, -1);
                n_cur = e.first;
                if (ctx->active_kv_tokens.size() > static_cast<size_t>(n_cur)) {
                    ctx->active_kv_tokens.resize(n_cur);
                }
                rolled_back = true;
                break;
            }
        }
    }

    if (out_generated_text) {
        if (out_buf.hit_stop()) {
            if (out_buf.stop_offset() < out_generated_text->size()) {
                out_generated_text->resize(out_buf.stop_offset());
            }
        } else if (out_buf.total_flushed_bytes() < out_generated_text->size()) {
            out_generated_text->resize(out_buf.total_flushed_bytes());
        }
    }

    if (out_buf.user_aborted() && execution_success) {
        ctx->set_error("Generation halted by stream callback.");
        execution_success = false;
    }

    ctx->n_past.store(n_cur, std::memory_order_relaxed);
    return execution_success;
}

static bool infer_generate_chat_internal(
    infer_ctx* ctx,
    const infer_chat_msg* msgs,
    size_t num_msgs,
    int32_t max_tokens,
    infer_stream_cb callback,
    void* user_data
) {
    const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
    const char* chat_template = llama_model_chat_template(ctx->model, nullptr);
    if (!chat_template) {
        ctx->set_error("Model specifies no chat template.");
        return false;
    }

    std::vector<std::string> safe_content;
    safe_content.reserve(num_msgs);
    for (size_t i = 0; i < num_msgs; ++i) {
        safe_content.push_back(neutralize_special_fast(msgs[i].content, ctx->special_lookup));
    }

    std::vector<struct llama_chat_message> llama_msgs;
    llama_msgs.reserve(num_msgs);
    for (size_t i = 0; i < num_msgs; ++i) {
        llama_msgs.push_back({ msgs[i].role, safe_content[i].c_str() });
    }

    int alloc_size = llama_chat_apply_template(chat_template, llama_msgs.data(), llama_msgs.size(), true, nullptr, 0);
    if (alloc_size <= 0) {
        ctx->set_error("Chat template formatting failed.");
        return false;
    }

    std::vector<char> formatted(alloc_size + 1, '\0');
    int written = llama_chat_apply_template(chat_template, llama_msgs.data(), llama_msgs.size(), true, formatted.data(), formatted.size());
    if (written < 0) {
        ctx->set_error("Chat template failed to render prompt.");
        return false;
    }

    bool add_bos = want_bos(vocab, formatted.data());
    int n_prompt_tokens = -llama_tokenize(vocab, formatted.data(), written, nullptr, 0, add_bos, true);
    if (n_prompt_tokens <= 0) {
        ctx->set_error("Tokenization calculation failed.");
        return false;
    }

    std::vector<llama_token> prompt_tokens(n_prompt_tokens);
    int real_tokens = llama_tokenize(vocab, formatted.data(), written, prompt_tokens.data(), prompt_tokens.size(), add_bos, true);
    if (real_tokens < 0) {
        ctx->set_error("Failed to tokenize rendered chat prompt.");
        return false;
    }
    prompt_tokens.resize(real_tokens);

    return execute_decode(ctx, prompt_tokens, false, max_tokens, callback, user_data);
}

/* ========================================================================== */
/* Public API Implementation                                                  */
/* ========================================================================== */

extern "C" {

uint32_t infer_version(void) {
    return INFER_VERSION_NUM;
}

void infer_set_log_callback(infer_log_cb callback, void* user_data) {
    {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        g_user_log_cb   = callback;
        g_user_log_data = user_data;
    }
    if (callback) {
        llama_log_set(internal_ggml_log, nullptr);
    } else {
        llama_log_set(noop_ggml_log, nullptr);
    }
}

infer_sampler_params infer_sampler_default_params(void) {
    infer_sampler_params p{};
    p.struct_size    = sizeof(infer_sampler_params);
    p.temperature    = 0.70f;
    p.top_p          = 0.90f;
    p.top_k          = 40;
    p.min_p          = 0.05f;
    p.repeat_penalty = 1.10f;
    p.repeat_last_n  = 64;
    p.seed           = 0;
    return p;
}

infer_ctx* infer_init(void) {
    bool reffed = false;
    infer_ctx* ctx = nullptr;
    try {
        if (!ref_llama_backend()) return nullptr;
        reffed = true;
        ctx = new (std::nothrow) infer_ctx();
        if (!ctx) {
            unref_llama_backend();
            return nullptr;
        }
        ctx->sampler_params = infer_sampler_default_params();
        return ctx;
    } catch (...) {
        if (ctx) delete ctx;
        if (reffed) unref_llama_backend();
        return nullptr;
    }
}

bool infer_load_model(infer_ctx* ctx, const char* path, uint32_t requested_ctx) {
    if (!ctx || !path) return false;

    if (requested_ctx != 0 && requested_ctx < 256) {
        ctx->set_error("Requested context size too small (minimum 256).");
        return false;
    }

    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Cannot load model while context is busy.");
        return false;
    }

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        ctx->clear_error();

        struct llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = 999;

        std::unique_ptr<llama_model, decltype(&llama_model_free)> new_model(
            llama_model_load_from_file(path, model_params), &llama_model_free);
        if (!new_model) {
            ctx->set_error("Failed to load model from path: " + std::string(path));
            return false;
        }

        uint32_t safe_ctx = calculate_safe_n_ctx(new_model.get(), requested_ctx);
        if (safe_ctx < 256) {
            ctx->set_error("Context size too small (minimum 256).");
            return false;
        }

        struct llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx          = safe_ctx;
        ctx_params.n_batch        = std::min<uint32_t>(safe_ctx, DEFAULT_BATCH_CAP);
        ctx_params.n_seq_max      = 2;
        ctx_params.kv_unified     = true;

        const bool has_gpu = infer_has_real_gpu();
        ctx_params.n_threads       = pick_threads(has_gpu, false);
        ctx_params.n_threads_batch = pick_threads(has_gpu, true);

        std::unique_ptr<llama_context, decltype(&llama_free)> new_lctx(
            llama_init_from_model(new_model.get(), ctx_params), &llama_free);
        if (!new_lctx) {
            ctx->set_error("Failed to initialize llama_context.");
            return false;
        }

        std::string err;
        struct llama_sampler* new_pen = nullptr;
        std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> new_sampler(
            create_sampler(ctx->sampler_params, new_model.get(), ctx->grammar_gbnf, err, &new_pen),
            &llama_sampler_free);
        if (!new_sampler) {
            ctx->set_error(err);
            return false;
        }

        auto specials = collect_special_texts(llama_model_get_vocab(new_model.get()));
        SpecialTokenLookup new_lookup;
        new_lookup.build(specials);

        const uint32_t new_n_ctx = llama_n_ctx_seq(new_lctx.get());

        if (ctx->sampler) llama_sampler_free(ctx->sampler);
        if (ctx->lctx)    llama_free(ctx->lctx);
        if (ctx->model)   llama_model_free(ctx->model);

        ctx->model          = new_model.release();
        ctx->lctx           = new_lctx.release();
        ctx->sampler        = new_sampler.release();
        ctx->penalty_smpl   = new_pen;
        ctx->special_texts  = std::move(specials);
        ctx->special_lookup = std::move(new_lookup);
        ctx->n_ctx          = new_n_ctx;
        ctx->n_past.store(0, std::memory_order_relaxed);
        ctx->n_keep         = 0;
        ctx->chat_history.clear();
        ctx->active_kv_tokens.clear();

        return true;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return false;
    } catch (...) {
        ctx->set_error("Unknown exception in infer_load_model.");
        return false;
    }
}

bool infer_set_sampler(infer_ctx* ctx, const infer_sampler_params* params) {
    if (!ctx || !params) return false;
    if (params->struct_size != 0 && params->struct_size != sizeof(infer_sampler_params)) {
        ctx->set_error("Invalid infer_sampler_params struct_size.");
        return false;
    }

    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Cannot change sampler while context is busy.");
        return false;
    }

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        std::string err;
        struct llama_sampler* new_pen = nullptr;
        struct llama_sampler* new_sampler = create_sampler(*params, ctx->model, ctx->grammar_gbnf, err, &new_pen);
        if (!new_sampler) {
            ctx->set_error(err);
            return false;
        }

      if (ctx->sampler) {
            llama_sampler_free(ctx->sampler);
        }
        ctx->sampler        = new_sampler;
        ctx->penalty_smpl   = new_pen;
        ctx->sampler_params = *params;
        return true;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return false;
    } catch (...) {
        ctx->set_error("Unknown exception in infer_set_sampler.");
        return false;
    }
}

bool infer_get_model_info(infer_ctx* ctx, infer_model_info* out_info) {
    if (!ctx || !out_info) return false;
    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        if (!ctx->model || !ctx->lctx) return false;

        const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
        out_info->struct_size  = sizeof(infer_model_info);
        out_info->n_ctx_train  = static_cast<uint32_t>(llama_model_n_ctx_train(ctx->model));
        out_info->n_embd       = static_cast<uint32_t>(llama_model_n_embd(ctx->model));
        out_info->n_layer      = static_cast<uint32_t>(llama_model_n_layer(ctx->model));
        out_info->n_head       = static_cast<uint32_t>(llama_model_n_head(ctx->model));
        out_info->vocab_size   = vocab ? static_cast<uint32_t>(llama_vocab_n_tokens(vocab)) : 0;
        out_info->n_ctx_active = ctx->n_ctx;

        std::memset(out_info->description, 0, sizeof(out_info->description));
        llama_model_desc(ctx->model, out_info->description, sizeof(out_info->description));
        return true;
    } catch (...) {
        return false;
    }
}

size_t infer_get_last_error(infer_ctx* ctx, char* out_buf, size_t buf_size) {
    if (!ctx) return 0;
    try {
        std::lock_guard<std::mutex> lock(ctx->err_mutex);
        if (!out_buf || buf_size == 0) {
            return ctx->last_error.size();
        }
        size_t copy_len = std::min(buf_size - 1, ctx->last_error.size());
        std::memcpy(out_buf, ctx->last_error.c_str(), copy_len);
        out_buf[copy_len] = '\0';
        return copy_len;
    } catch (...) {
        return 0;
    }
}

bool infer_last_success(infer_ctx* ctx) {
    return ctx ? ctx->last_success.load(std::memory_order_acquire) : false;
}

int32_t infer_tokenize_count(infer_ctx* ctx, const char* text, bool add_special) {
    if (!ctx || !text) return -1;
    size_t text_len = std::strlen(text);
    if (text_len >= static_cast<size_t>(INT32_MAX)) {
        return -1;
    }

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        if (!ctx->model) return -1;
        const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
        if (!vocab) return -1;

        int n = llama_tokenize(vocab, text, static_cast<int32_t>(text_len), nullptr, 0, add_special, true);
        return (n < 0) ? -n : n;
    } catch (...) {
        return -1;
    }
}

uint32_t infer_get_kv_cache_token_count(infer_ctx* ctx) {
    if (!ctx) return 0;
    return static_cast<uint32_t>(ctx->n_past.load(std::memory_order_relaxed));
}

bool infer_set_stop_sequences(infer_ctx* ctx, const char* const* stop_words, size_t num_words) {
    if (!ctx) return false;
    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Cannot change stop sequences while context is busy.");
        return false;
    }

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        ctx->stop_words.clear();
        if (stop_words && num_words > 0) {
            for (size_t i = 0; i < num_words; ++i) {
                if (stop_words[i]) {
                    size_t slen = std::strlen(stop_words[i]);
                    // Cap stop sequence length to 256 bytes to prevent memory bloat
                    if (slen > 0 && slen <= 256) {
                        ctx->stop_words.emplace_back(stop_words[i]);
                    }
                }
            }
        }
        return true;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return false;
    } catch (...) {
        ctx->set_error("Unknown exception in infer_set_stop_sequences.");
        return false;
    }
}

bool infer_set_grammar(infer_ctx* ctx, const char* grammar_gbnf) {
    if (!ctx) return false;
    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Cannot change grammar while context is busy.");
        return false;
    }

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        std::string new_grammar = grammar_gbnf ? grammar_gbnf : "";

        if (!ctx->model) {
            ctx->grammar_gbnf = std::move(new_grammar);
            return true;
        }

        std::string err;
        struct llama_sampler* new_pen = nullptr;
        struct llama_sampler* new_sampler = create_sampler(ctx->sampler_params, ctx->model, new_grammar, err, &new_pen);
        if (!new_sampler) {
            ctx->set_error(err);
            return false;
        }

        if (ctx->sampler) {
            llama_sampler_free(ctx->sampler);
        }
        ctx->sampler      = new_sampler;
        ctx->penalty_smpl = new_pen;
        ctx->grammar_gbnf = std::move(new_grammar);
        return true;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return false;
    } catch (...) {
        ctx->set_error("Unknown exception in infer_set_grammar.");
        return false;
    }
}

void infer_memory_clear(infer_ctx* ctx) {
    if (!ctx) return;
    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Cannot clear memory while context is busy.");
        return;
    }

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        if (ctx->lctx) {
            llama_memory_clear(llama_get_memory(ctx->lctx), false);
        }
        if (ctx->sampler) {
            llama_sampler_reset(ctx->sampler);
        }
        ctx->n_past.store(0, std::memory_order_relaxed);
        ctx->n_keep = 0;
        ctx->chat_history.clear();
        ctx->active_kv_tokens.clear();
    } catch (...) {
    }
}

bool infer_state_save(infer_ctx* ctx, const char* filepath) {
    if (!ctx || !filepath) return false;
    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Cannot save state while context is busy.");
        return false;
    }

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        if (!ctx->lctx || !ctx->model) {
            ctx->set_error("Model and context must be active to save state.");
            return false;
        }

        size_t state_size = llama_state_get_size(ctx->lctx);
        if (state_size == 0 || state_size > MAX_STATE_FILE_SIZE) {
            ctx->set_error("Invalid state size retrieved from context.");
            return false;
        }

        const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
        uint32_t vocab_size = vocab ? static_cast<uint32_t>(llama_vocab_n_tokens(vocab)) : 0;

        std::vector<uint8_t> kv_buffer(state_size);
        size_t written = llama_state_get_data(ctx->lctx, kv_buffer.data(), kv_buffer.size());
        if (written != state_size) {
            ctx->set_error("Failed to extract context state data.");
            return false;
        }

        std::vector<uint8_t> hist_buf;
        uint32_t active_tok_count = static_cast<uint32_t>(ctx->active_kv_tokens.size());
        uint32_t msg_count = static_cast<uint32_t>(ctx->chat_history.size());

        auto append_bytes = [&hist_buf](const void* src, size_t len) {
            const uint8_t* p = static_cast<const uint8_t*>(src);
            hist_buf.insert(hist_buf.end(), p, p + len);
        };

        append_bytes(&active_tok_count, sizeof(active_tok_count));
        if (active_tok_count > 0) {
            append_bytes(ctx->active_kv_tokens.data(), active_tok_count * sizeof(llama_token));
        }

        append_bytes(&msg_count, sizeof(msg_count));
        for (const auto& msg : ctx->chat_history) {
            uint32_t rlen = static_cast<uint32_t>(msg.role.size());
            uint32_t clen = static_cast<uint32_t>(msg.content.size());
            append_bytes(&rlen, sizeof(rlen));
            append_bytes(msg.role.data(), rlen);
            append_bytes(&clen, sizeof(clen));
            append_bytes(msg.content.data(), clen);
        }

        std::string temp_filepath = make_tmp_path(filepath);
        TmpGuard tmp_guard{temp_filepath, false};

        ScopedFile f(open_tmp_exclusive(temp_filepath));
        if (!f) {
            ctx->set_error("Failed to open file for state write: " + temp_filepath);
            return false;
        }
        tmp_guard.armed = true;

        StateFileHeader header{};
        header.magic         = STATE_MAGIC;
        header.version       = STATE_VERSION;
        header.n_ctx         = ctx->n_ctx;
        header.n_past        = static_cast<uint32_t>(ctx->n_past.load(std::memory_order_relaxed));
        header.vocab_size    = vocab_size;
        header.n_embd        = static_cast<uint32_t>(llama_model_n_embd(ctx->model));
        header.n_layer       = static_cast<uint32_t>(llama_model_n_layer(ctx->model));
        header.n_head        = static_cast<uint32_t>(llama_model_n_head(ctx->model));
        header.fingerprint   = compute_model_fingerprint(ctx->model);
        header.payload_crc   = kv_buffer.empty() ? 0 : compute_crc32(kv_buffer.data(), kv_buffer.size());
        header.history_bytes = static_cast<uint32_t>(hist_buf.size());
        header.data_size     = static_cast<uint64_t>(kv_buffer.size() + hist_buf.size());

        if (std::fwrite(&header, sizeof(header), 1, f.get()) != 1 ||
            (!kv_buffer.empty() && std::fwrite(kv_buffer.data(), 1, kv_buffer.size(), f.get()) != kv_buffer.size()) ||
            (!hist_buf.empty() && std::fwrite(hist_buf.data(), 1, hist_buf.size(), f.get()) != hist_buf.size())) {
            ctx->set_error("Failed to complete writing state data.");
            return false;
        }

        if (!sync_file(f.get()) || std::fclose(f.release()) != 0) {
            ctx->set_error("Failed to flush and close temporary state file.");
            return false;
        }

        bool rename_ok = false;
#if defined(_WIN32)
        int wtmp_len = MultiByteToWideChar(CP_UTF8, 0, temp_filepath.c_str(), -1, nullptr, 0);
        int wdst_len = MultiByteToWideChar(CP_UTF8, 0, filepath, -1, nullptr, 0);
        if (wtmp_len > 0 && wdst_len > 0) {
            std::vector<wchar_t> wtmp(wtmp_len);
            std::vector<wchar_t> wdst(wdst_len);
            MultiByteToWideChar(CP_UTF8, 0, temp_filepath.c_str(), -1, wtmp.data(), wtmp_len);
            MultiByteToWideChar(CP_UTF8, 0, filepath, -1, wdst.data(), wdst_len);
            rename_ok = (MoveFileExW(wtmp.data(), wdst.data(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0);
        }
#else
        rename_ok = (std::rename(temp_filepath.c_str(), filepath) == 0);
        if (rename_ok) {
            sync_parent_dir(filepath);
        }
#endif

        if (!rename_ok) {
            ctx->set_error("Atomic rename failed for saved state file.");
            return false;
        }

        tmp_guard.armed = false;
        return true;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return false;
    } catch (...) {
        ctx->set_error("Unknown exception during infer_state_save.");
        return false;
    }
}

bool infer_state_load(infer_ctx* ctx, const char* filepath) {
    if (!ctx || !filepath) return false;
    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Cannot load state while context is busy.");
        return false;
    }

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        if (!ctx->lctx || !ctx->model) {
            ctx->set_error("Model and context must be active to load state.");
            return false;
        }

        ScopedFile f(open_file_utf8(filepath, "rb"));
        if (!f) {
            ctx->set_error("Failed to open state file for reading: " + std::string(filepath));
            return false;
        }

        uint64_t file_len = 0;
        if (!file_size_64(f.get(), file_len) || file_len < sizeof(StateFileHeader)) {
            ctx->set_error("State file corrupted or smaller than header.");
            return false;
        }

        StateFileHeader header{};
        if (std::fread(&header, sizeof(header), 1, f.get()) != 1) {
            ctx->set_error("Incomplete or missing state file header.");
            return false;
        }

        const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
        uint32_t current_vocab = vocab ? static_cast<uint32_t>(llama_vocab_n_tokens(vocab)) : 0;
        uint64_t current_fp = compute_model_fingerprint(ctx->model);

        if (header.magic != STATE_MAGIC ||
            header.version != STATE_VERSION ||
            header.fingerprint != current_fp ||
            header.n_past > ctx->n_ctx ||
            header.vocab_size != current_vocab ||
            header.n_embd != static_cast<uint32_t>(llama_model_n_embd(ctx->model)) ||
            header.n_layer != static_cast<uint32_t>(llama_model_n_layer(ctx->model)) ||
            header.n_head != static_cast<uint32_t>(llama_model_n_head(ctx->model)) ||
            header.data_size == 0 ||
            header.data_size > MAX_STATE_FILE_SIZE ||
            header.history_bytes > header.data_size ||
            file_len != sizeof(StateFileHeader) + header.data_size) {
            ctx->set_error("Incompatible, corrupted, or truncated state file.");
            return false;
        }

        uint64_t kv_bytes = header.data_size - header.history_bytes;
        std::vector<uint8_t> kv_buffer(kv_bytes);
        if (kv_bytes > 0 && std::fread(kv_buffer.data(), 1, kv_bytes, f.get()) != kv_bytes) {
            ctx->set_error("Corrupted or truncated state payload in file.");
            return false;
        }

        if (kv_bytes > 0 && compute_crc32(kv_buffer.data(), kv_bytes) != header.payload_crc) {
            ctx->set_error("State payload CRC32 checksum mismatch.");
            return false;
        }

        std::vector<uint8_t> hist_buf(header.history_bytes);
        if (header.history_bytes > 0) {
            if (std::fread(hist_buf.data(), 1, header.history_bytes, f.get()) != header.history_bytes) {
                ctx->set_error("Corrupted chat history payload in state file.");
                return false;
            }
        }

        std::vector<llama_token> restored_kv;
        std::vector<OwnedMsg> restored_history;

        if (header.history_bytes > 0) {
            const uint8_t* ptr = hist_buf.data();
            const uint8_t* end = ptr + hist_buf.size();

            auto read_bytes = [&ptr, end](void* dst, size_t len) -> bool {
                if (len > static_cast<size_t>(end - ptr)) return false;
                std::memcpy(dst, ptr, len);
                ptr += len;
                return true;
            };

            uint32_t active_tok_count = 0;
            if (!read_bytes(&active_tok_count, sizeof(active_tok_count)) || active_tok_count > ctx->n_ctx) {
                ctx->set_error("Corrupted active token count in state file.");
                return false;
            }
            if (active_tok_count > 0) {
                restored_kv.resize(active_tok_count);
                if (!read_bytes(restored_kv.data(), active_tok_count * sizeof(llama_token))) {
                    ctx->set_error("Truncated active tokens in state file.");
                    return false;
                }
            }

            uint32_t msg_count = 0;
            if (!read_bytes(&msg_count, sizeof(msg_count))) {
                ctx->set_error("Corrupted message count in state file.");
                return false;
            }
            for (uint32_t i = 0; i < msg_count; ++i) {
                uint32_t rlen = 0, clen = 0;
                if (!read_bytes(&rlen, sizeof(rlen)) || rlen > static_cast<size_t>(end - ptr)) {
                    ctx->set_error("Corrupted role length in state file.");
                    return false;
                }
                std::string r(rlen, '\0');
                if (!read_bytes(&r[0], rlen)) return false;

                if (!read_bytes(&clen, sizeof(clen)) || clen > static_cast<size_t>(end - ptr)) {
                    ctx->set_error("Corrupted content length in state file.");
                    return false;
                }
                std::string c(clen, '\0');
                if (!read_bytes(&c[0], clen)) return false;

                restored_history.push_back({ std::move(r), std::move(c) });
            }
        }

        size_t loaded = llama_state_set_data(ctx->lctx, kv_buffer.data(), kv_buffer.size());
        if (loaded != kv_bytes) {
            llama_memory_clear(llama_get_memory(ctx->lctx), false);
            if (ctx->sampler) llama_sampler_reset(ctx->sampler);
            ctx->n_past.store(0, std::memory_order_relaxed);
            ctx->chat_history.clear();
            ctx->active_kv_tokens.clear();
            ctx->set_error("llama_state_set_data failed to restore context state.");
            return false;
        }

        if (ctx->sampler) {
            llama_sampler_reset(ctx->sampler);
        }
        {
            const llama_pos pmax = llama_memory_seq_pos_max(llama_get_memory(ctx->lctx), 0);
            ctx->n_past.store(pmax >= 0 ? static_cast<int>(pmax) + 1 : 0, std::memory_order_relaxed);
        }
        ctx->n_keep = 0;
        ctx->chat_history = std::move(restored_history);
        ctx->active_kv_tokens = std::move(restored_kv);
        return true;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return false;
    } catch (...) {
        ctx->set_error("Unknown exception during infer_state_load.");
        return false;
    }
}

int32_t infer_get_embedding_dim(infer_ctx* ctx) {
    if (!ctx) return 0;
    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        return ctx->model ? llama_model_n_embd(ctx->model) : 0;
    } catch (...) {
        return 0;
    }
}

bool infer_get_embeddings(infer_ctx* ctx, const char* text, float* out_vec, size_t vec_len) {
    if (!ctx || !text || !out_vec) return false;
    size_t text_len = std::strlen(text);
    if (text_len >= static_cast<size_t>(INT32_MAX)) {
        return false;
    }

    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Context is busy.");
        return false;
    }

    ctx->abort_flag.store(false, std::memory_order_release);

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        if (!ctx->model || !ctx->lctx) {
            ctx->set_error("Model and context must be active to generate embeddings.");
            return false;
        }

        const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
        if (!vocab) { ctx->set_error("Model has no vocabulary."); return false; }

        int n_embd = llama_model_n_embd(ctx->model);
        if (vec_len < static_cast<size_t>(n_embd)) {
            ctx->set_error("Output buffer smaller than model embedding dimensions.");
            return false;
        }

        bool add_bos = want_bos(vocab, text);
        int n_tokens = -llama_tokenize(vocab, text, static_cast<int32_t>(text_len), nullptr, 0, add_bos, false);
        if (n_tokens <= 0) { ctx->set_error("Embedding tokenization failed or text is empty."); return false; }

        std::vector<llama_token> tokens(n_tokens);
        int real_tokens = llama_tokenize(vocab, text, static_cast<int32_t>(text_len), tokens.data(), tokens.size(), add_bos, false);
        if (real_tokens < 0) { ctx->set_error("Embedding tokenization failed."); return false; }
        tokens.resize(real_tokens);

        const uint64_t chat_used = static_cast<uint64_t>(std::max(0, ctx->n_past.load(std::memory_order_relaxed)));
        if (static_cast<uint64_t>(tokens.size()) + chat_used > ctx->n_ctx) {
            ctx->set_error("Input text exceeds free context capacity for embedding generation "
                           "(chat KV shares the pool; call infer_memory_clear to free it).");
            return false;
        }

        const int effective_batch_cap = std::min<int>(DEFAULT_BATCH_CAP, static_cast<int>(llama_n_batch(ctx->lctx)));
        if (tokens.size() > static_cast<size_t>(effective_batch_cap) && llama_pooling_type(ctx->lctx) != LLAMA_POOLING_TYPE_NONE) {
            ctx->set_error("Input exceeds batch size; pooled embeddings cannot be chunked.");
            return false;
        }

        llama_set_embeddings(ctx->lctx, true);
        struct EmbeddingModeGuard {
            struct llama_context* l;
            ~EmbeddingModeGuard() {
                llama_set_embeddings(l, false);
                safe_kv_seq_rm(l, 1, -1, -1);
            }
        } embd_guard{ctx->lctx};

        struct llama_batch batch = llama_batch_init(effective_batch_cap, 0, 1);
        struct BatchGuard {
            struct llama_batch& b;
            ~BatchGuard() { llama_batch_free(b); }
        } guard{batch};

        for (size_t i = 0; i < tokens.size(); i += batch.n_tokens) {
            if (ctx->abort_flag.load(std::memory_order_relaxed)) {
                ctx->set_error("Embedding generation aborted.");
                return false;
            }

            batch.n_tokens = 0;
            int chunk = std::min<int>(effective_batch_cap, static_cast<int>(tokens.size() - i));

            for (int j = 0; j < chunk; ++j) {
                size_t idx = i + j;
                batch.token[j]     = tokens[idx];
                batch.pos[j]       = static_cast<llama_pos>(idx);
                batch.n_seq_id[j]  = 1;
                batch.seq_id[j][0] = 1;
                batch.logits[j]    = (idx == tokens.size() - 1);
                batch.n_tokens++;
            }

            if (llama_decode(ctx->lctx, batch) != 0) {
                ctx->set_error("Failed to decode prompt for embedding generation.");
                return false;
            }
        }

        const float* embd = llama_get_embeddings_seq(ctx->lctx, 1);
        if (!embd) embd = llama_get_embeddings_ith(ctx->lctx, -1);
        if (!embd) embd = llama_get_embeddings(ctx->lctx);

        if (!embd) {
            ctx->set_error("Failed to extract embeddings from context.");
            return false;
        }

        std::memcpy(out_vec, embd, n_embd * sizeof(float));
        return true;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return false;
    } catch (...) {
        ctx->set_error("Unknown exception during infer_get_embeddings.");
        return false;
    }
}

void infer_abort(infer_ctx* ctx) {
    if (ctx) {
        ctx->abort_flag.store(true, std::memory_order_release);
    }
}

bool infer_is_generating(infer_ctx* ctx) {
    return ctx ? ctx->is_generating.load(std::memory_order_acquire) : false;
}

void infer_wait(infer_ctx* ctx) {
    if (!ctx) return;
    try {
        std::lock_guard<std::mutex> lock(ctx->worker_mutex);
        if (ctx->async_worker.joinable()) {
            ctx->async_worker.join();
        }
    } catch (...) {
    }
}

bool infer_generate_chat(
    infer_ctx* ctx,
    const infer_chat_msg* msgs,
    size_t num_msgs,
    int32_t max_tokens,
    infer_stream_cb callback,
    void* user_data
) {
    if (!ctx || !msgs || num_msgs == 0 || !callback) return false;

    for (size_t i = 0; i < num_msgs; ++i) {
        if (!msgs[i].role || !msgs[i].content) {
            ctx->set_error("Message role or content cannot be null.");
            return false;
        }
        if (!is_valid_role(msgs[i].role)) {
            ctx->set_error("Invalid message role (allowed: system, user, assistant, tool, developer).");
            return false;
        }
    }

    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Inference context is currently busy.");
        return false;
    }

    if (!ctx->model || !ctx->lctx || !ctx->sampler) {
        ctx->set_error("Model and context are not initialized.");
        return false;
    }

    ctx->abort_flag.store(false, std::memory_order_release);
    ctx->last_success.store(false, std::memory_order_release);
    ctx->clear_error();

    try {
        ctx->chat_history.clear();
        ctx->active_kv_tokens.clear();
        bool res = infer_generate_chat_internal(ctx, msgs, num_msgs, max_tokens, callback, user_data);
        ctx->last_success.store(res, std::memory_order_release);
        return res;
    } catch (const std::exception& e) {
        ctx->last_success.store(false, std::memory_order_release);
        ctx->set_error(e.what());
        return false;
    } catch (...) {
        ctx->last_success.store(false, std::memory_order_release);
        ctx->set_error("Unknown exception during chat generation.");
        return false;
    }
}

bool infer_generate_chat_async(
    infer_ctx* ctx,
    const infer_chat_msg* msgs,
    size_t num_msgs,
    int32_t max_tokens,
    infer_stream_cb callback,
    void* user_data
) {
    if (!ctx || !msgs || num_msgs == 0 || !callback) return false;

    for (size_t i = 0; i < num_msgs; ++i) {
        if (!msgs[i].role || !msgs[i].content) {
            ctx->set_error("Message role or content cannot be null.");
            return false;
        }
        if (!is_valid_role(msgs[i].role)) {
            ctx->set_error("Invalid message role (allowed: system, user, assistant, tool, developer).");
            return false;
        }
    }

    bool expected = false;
    if (!ctx->is_generating.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
        ctx->set_error("Inference context is currently busy.");
        return false;
    }

    if (!ctx->model || !ctx->lctx || !ctx->sampler) {
        ctx->set_error("Model and context are not initialized.");
        release_gen(ctx);
        return false;
    }

    try {
        std::vector<OwnedMsg> owned;
        owned.reserve(num_msgs);
        for (size_t i = 0; i < num_msgs; ++i) {
            owned.push_back({ msgs[i].role, msgs[i].content });
        }

        std::lock_guard<std::mutex> wlock(ctx->worker_mutex);

        if (ctx->async_worker.joinable()) {
            ctx->async_worker.join();
        }

        ctx->abort_flag.store(false, std::memory_order_release);
        ctx->clear_error();
        const bool prev_success = ctx->last_success.load(std::memory_order_acquire);
        ctx->last_success.store(false, std::memory_order_release);

        try {
            ctx->async_worker = std::thread([ctx, owned_msgs = std::move(owned), max_tokens, callback, user_data]() {
                struct WorkerFlagGuard {
                    infer_ctx* c;
                    ~WorkerFlagGuard() { release_gen(c); }
                } flag_guard{ctx};

                std::vector<infer_chat_msg> view(owned_msgs.size());
                for (size_t i = 0; i < owned_msgs.size(); ++i) {
                    view[i].role    = owned_msgs[i].role.c_str();
                    view[i].content = owned_msgs[i].content.c_str();
                }

                try {
                    ctx->chat_history.clear();
                    ctx->active_kv_tokens.clear();
                    bool res = infer_generate_chat_internal(ctx, view.data(), view.size(), max_tokens, callback, user_data);
                    ctx->last_success.store(res, std::memory_order_release);
                } catch (const std::exception& e) {
                    ctx->last_success.store(false, std::memory_order_release);
                    ctx->set_error(e.what());
                } catch (...) {
                    ctx->last_success.store(false, std::memory_order_release);
                    ctx->set_error("Unknown exception during async chat generation.");
                }
            });
        } catch (...) {
            ctx->last_success.store(prev_success, std::memory_order_release);
            throw;
        }

        return true;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        release_gen(ctx);
        return false;
    } catch (...) {
        ctx->set_error("Unknown exception launching async thread.");
        release_gen(ctx);
        return false;
    }
}

bool infer_generate_chat_turn(
    infer_ctx* ctx,
    const char* role,
    const char* content,
    int32_t max_tokens,
    infer_stream_cb callback,
    void* user_data
) {
    if (!ctx || !role || !content || !callback) return false;
    if (!is_valid_role(role)) {
        ctx->set_error("Invalid message role (allowed: system, user, assistant, tool, developer).");
        return false;
    }

    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Inference context is currently busy.");
        return false;
    }

    if (!ctx->model || !ctx->lctx || !ctx->sampler) {
        ctx->set_error("Model and context are not initialized.");
        return false;
    }

    ctx->abort_flag.store(false, std::memory_order_release);
    ctx->last_success.store(false, std::memory_order_release);
    ctx->clear_error();

    struct HistoryGuard {
        std::vector<OwnedMsg>& hist_ref;
        std::vector<OwnedMsg> backup;
        bool armed = true;
        
        explicit HistoryGuard(std::vector<OwnedMsg>& h) : hist_ref(h), backup(h) {}
        ~HistoryGuard() {
            if (armed) {
                hist_ref = std::move(backup); // Restore full history state on failure
            }
        }
    } hist_guard{ ctx->chat_history };

    try {
        const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
        const char* chat_template = llama_model_chat_template(ctx->model, nullptr);
        if (!chat_template) {
            ctx->set_error("Model specifies no chat template.");
            return false;
        }

        std::string safe_content = neutralize_special_fast(content, ctx->special_lookup);
        ctx->chat_history.push_back({ role, safe_content });

        while (true) {
            std::vector<struct llama_chat_message> llama_msgs;
            llama_msgs.reserve(ctx->chat_history.size());
            for (const auto& m : ctx->chat_history) {
                llama_msgs.push_back({ m.role.c_str(), m.content.c_str() });
            }

            int alloc_size = llama_chat_apply_template(chat_template, llama_msgs.data(), llama_msgs.size(), true, nullptr, 0);
            if (alloc_size <= 0) {
                ctx->set_error("Chat template formatting failed.");
                return false;
            }

            std::vector<char> formatted(alloc_size + 1, '\0');
            int written = llama_chat_apply_template(chat_template, llama_msgs.data(), llama_msgs.size(), true, formatted.data(), formatted.size());
            if (written < 0) {
                ctx->set_error("Chat template failed to render prompt.");
                return false;
            }

            bool add_bos = want_bos(vocab, formatted.data());
            int n_tokens = -llama_tokenize(vocab, formatted.data(), written, nullptr, 0, add_bos, true);
            if (n_tokens <= 0) {
                ctx->set_error("Tokenization calculation failed.");
                return false;
            }

            const uint32_t safety_margin = 128;
            if (static_cast<uint32_t>(n_tokens) + safety_margin <= ctx->n_ctx) {
                std::vector<llama_token> prompt_tokens(n_tokens);
                int real_tokens = llama_tokenize(vocab, formatted.data(), written, prompt_tokens.data(), prompt_tokens.size(), add_bos, true);
                if (real_tokens < 0) {
                    ctx->set_error("Failed to tokenize chat turn.");
                    return false;
                }
                prompt_tokens.resize(real_tokens);

                std::string generated_response;
                bool res = execute_decode(ctx, prompt_tokens, true, max_tokens, callback, user_data, &generated_response);
                if (res && !generated_response.empty()) {
                    std::string safe_response = neutralize_special_fast(generated_response.c_str(), ctx->special_lookup);
                    ctx->chat_history.push_back({ "assistant", std::move(safe_response) });
                    hist_guard.armed = false;
                }
                ctx->last_success.store(res, std::memory_order_release);
                return res;
            }

            size_t prune_idx = (ctx->chat_history.size() > 1 && ctx->chat_history[0].role == "system") ? 1 : 0;
            if (ctx->chat_history.size() > prune_idx + 2) {
                ctx->chat_history.erase(ctx->chat_history.begin() + prune_idx, ctx->chat_history.begin() + prune_idx + 2);
            } else {
                ctx->set_error("Single chat turn exceeds context capacity.");
                return false;
            }
        }
    } catch (const std::exception& e) {
        ctx->last_success.store(false, std::memory_order_release);
        ctx->set_error(e.what());
        return false;
    } catch (...) {
        ctx->last_success.store(false, std::memory_order_release);
        ctx->set_error("Unknown exception during turn execution.");
        return false;
    }
}

bool infer_generate_completion(
    infer_ctx* ctx,
    const char* raw_prompt,
    int32_t max_tokens,
    infer_stream_cb callback,
    void* user_data
) {
    if (!ctx || !raw_prompt || !callback) return false;

    size_t prompt_len = std::strlen(raw_prompt);
    if (prompt_len >= static_cast<size_t>(INT32_MAX)) {
        return false;
    }

    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Inference context is currently busy.");
        return false;
    }

    if (!ctx->model || !ctx->lctx || !ctx->sampler) {
        ctx->set_error("Model and context are not initialized.");
        return false;
    }

    ctx->abort_flag.store(false, std::memory_order_release);
    ctx->last_success.store(false, std::memory_order_release);
    ctx->clear_error();

    try {
        ctx->chat_history.clear();
        ctx->active_kv_tokens.clear();

        const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
        int written = static_cast<int>(prompt_len);

        bool add_bos = want_bos(vocab, raw_prompt);
        int n_tokens = -llama_tokenize(vocab, raw_prompt, written, nullptr, 0, add_bos, false);
        if (n_tokens <= 0) {
            ctx->set_error("Tokenization calculation failed.");
            return false;
        }

        std::vector<llama_token> prompt_tokens(n_tokens);
        int real_tokens = llama_tokenize(vocab, raw_prompt, written, prompt_tokens.data(), prompt_tokens.size(), add_bos, false);
        if (real_tokens < 0) {
            ctx->set_error("Prompt tokenization failed.");
            return false;
        }
        prompt_tokens.resize(real_tokens);

        bool res = execute_decode(ctx, prompt_tokens, false, max_tokens, callback, user_data);
        ctx->last_success.store(res, std::memory_order_release);
        return res;
    } catch (const std::exception& e) {
        ctx->last_success.store(false, std::memory_order_release);
        ctx->set_error(e.what());
        return false;
    } catch (...) {
        ctx->last_success.store(false, std::memory_order_release);
        ctx->set_error("Unknown exception during prompt completion.");
        return false;
    }
}

void infer_free(infer_ctx* ctx) {
    if (!ctx) return;

   ctx->abort_flag.store(true, std::memory_order_release);

     try {
        std::unique_lock<std::mutex> lk(ctx->gen_mutex);
        ctx->cv_generating.wait(lk, [ctx]() {
            return !ctx->is_generating.load(std::memory_order_acquire);
        });
    } catch (...) {}

    try {
        std::lock_guard<std::mutex> wlock(ctx->worker_mutex);
        if (ctx->async_worker.joinable()) {
            ctx->async_worker.join();
        }
    } catch (...) {}

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        if (ctx->sampler) {
            llama_sampler_free(ctx->sampler);
            ctx->sampler      = nullptr;
            ctx->penalty_smpl = nullptr;
        }
        if (ctx->lctx) {
            llama_free(ctx->lctx);
            ctx->lctx = nullptr;
        }
        if (ctx->model) {
            llama_model_free(ctx->model);
            ctx->model = nullptr;
        }
    } catch (...) {}

    delete ctx;
    unref_llama_backend();
}

} // extern "C"