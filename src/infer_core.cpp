/* ========================================================================== */
/* infer_core.cpp - Production Implementation (Auto GPU/CPU Scaling)          */
/* ========================================================================== */
#include "infer.h"

#include <llama.h>
#include <ggml-backend.h>

#include <algorithm>
#include <atomic>
#include <climits>
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
#include <type_traits>
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
#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <mach/mach.h>
#endif
#endif

namespace {

constexpr uint32_t STATE_MAGIC          = 0x494E4655; // "INFU"
constexpr uint32_t STATE_VERSION        = 3;          // Version 3 header with full-file CRC & history
constexpr uint32_t DEFAULT_BATCH_CAP    = 512;
constexpr uint32_t DEFAULT_FALLBACK_CTX = 4096;
constexpr uint64_t MAX_STATE_FILE_SIZE  = 64ULL * 1024ULL * 1024ULL * 1024ULL; // 64 GiB bound

std::mutex g_backend_mutex;
int32_t    g_backend_ref_count = 0;
int32_t    g_cached_env_threads = 0;
uint32_t   g_cached_phys_cores  = 0;

infer_log_cb g_user_log_cb   = nullptr;
void*        g_user_log_data = nullptr;
std::mutex   g_log_mutex;

#if defined(_WIN32)
HANDLE g_instance_mutex = NULL;
#else
int g_instance_fd = -1;
#endif

uint32_t detect_physical_cores() {
#if defined(_WIN32)
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER && length > 0) {
        std::vector<uint8_t> buffer(length);
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data());
        if (GetLogicalProcessorInformationEx(RelationProcessorCore, info, &length)) {
            uint32_t cores = 0;
            DWORD offset = 0;
            while (offset < length) {
                auto* current = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
                if (current->Relationship == RelationProcessorCore) {
                    cores++;
                }
                if (current->Size == 0) break;
                offset += current->Size;
            }
            if (cores > 0) return cores;
        }
    }
#elif defined(__APPLE__)
    int32_t phys = 0;
    size_t sz = sizeof(phys);
    if (sysctlbyname("hw.physicalcpu", &phys, &sz, nullptr, 0) == 0 && phys > 0) {
        return static_cast<uint32_t>(phys);
    }
#elif defined(__linux__)
    FILE* f = std::fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        uint32_t cpu_cores = 0;
        while (std::fgets(line, sizeof(line), f)) {
            if (std::strncmp(line, "cpu cores", 9) == 0) {
                const char* colon = std::strchr(line, ':');
                if (colon) {
                    int val = std::atoi(colon + 1);
                    if (val > 0) {
                        cpu_cores = static_cast<uint32_t>(val);
                        break;
                    }
                }
            }
        }
        std::fclose(f);
        if (cpu_cores > 0) return cpu_cores;
    }
#endif
    uint32_t hc = std::thread::hardware_concurrency();
    if (hc > 1) return hc / 2;
    return 4;
}

uint64_t query_system_free_ram() {
#if defined(_WIN32)
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) {
        return static_cast<uint64_t>(status.ullAvailPhys);
    }
    return 0;
#elif defined(__APPLE__)
    mach_port_t host_port = mach_host_self();
    vm_size_t page_size = 4096;
    host_page_size(host_port, &page_size);
    vm_statistics64_data_t vm_stat{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(host_port, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm_stat), &count) == KERN_SUCCESS) {
        uint64_t free_pages = static_cast<uint64_t>(vm_stat.free_count) + static_cast<uint64_t>(vm_stat.inactive_count);
        return free_pages * static_cast<uint64_t>(page_size);
    }
    return 0;
#elif defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
    long pages = ::sysconf(_SC_AVPHYS_PAGES);
    long page_sz = ::sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_sz > 0) {
        return static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_sz);
    }
    return 0;
#else
    return 0;
#endif
}

bool get_module_dir_utf8(std::string& out) {
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
    out = (slash == 0) ? "/" : p.substr(0, slash);
    return true;
#endif
}

bool ref_llama_backend(std::string& err_out) {
    std::lock_guard<std::mutex> lock(g_backend_mutex);
    if (g_backend_ref_count == 0) {
        const char* single_inst = std::getenv("INFER_SINGLE_INSTANCE");
        const bool want_lock = (single_inst && std::strcmp(single_inst, "1") == 0);

        if (want_lock) {
#if defined(_WIN32)
            SetLastError(0);
            g_instance_mutex = CreateMutexA(NULL, FALSE, "Local\\InferCoreSingleInstance");
            DWORD err = GetLastError();
            if (!g_instance_mutex || err == ERROR_ALREADY_EXISTS || err == ERROR_ACCESS_DENIED) {
                if (g_instance_mutex) {
                    CloseHandle(g_instance_mutex);
                    g_instance_mutex = NULL;
                }
                err_out = "Single-instance lock already held or access denied.";
                return false;
            }
#else
            std::string lock_path;
            const char* xdg_dir = std::getenv("XDG_RUNTIME_DIR");
            if (xdg_dir && *xdg_dir) {
                lock_path = std::string(xdg_dir) + "/infer_core_single_instance.lock";
            } else {
                lock_path = "/tmp/infer_core_single_instance_" +
                            std::to_string(static_cast<unsigned long>(::getuid())) + ".lock";
            }
            g_instance_fd = ::open(lock_path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
            if (g_instance_fd < 0) {
                err_out = "Failed to open single-instance lock file: " + lock_path;
                return false;
            }
            struct flock fl{};
            fl.l_type   = F_WRLCK;
            fl.l_whence = SEEK_SET;
            if (::fcntl(g_instance_fd, F_SETLK, &fl) < 0) {
                ::close(g_instance_fd);
                g_instance_fd = -1;
                err_out = "Single-instance file lock already held: " + lock_path;
                return false;
            }
#endif
        }

        if (const char* e = std::getenv("INFER_N_THREADS")) {
            int v = std::atoi(e);
            g_cached_env_threads = (v > 0) ? std::min(v, 256) : 0;
        } else {
            g_cached_env_threads = 0;
        }
        g_cached_phys_cores = detect_physical_cores();

        std::string dir;
        if (!get_module_dir_utf8(dir)) {
#if defined(_WIN32)
            if (g_instance_mutex) {
                CloseHandle(g_instance_mutex);
                g_instance_mutex = NULL;
            }
#else
            if (g_instance_fd >= 0) {
                ::close(g_instance_fd);
                g_instance_fd = -1;
            }
#endif
            err_out = "Failed to resolve module directory for backend initialization.";
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
        if (g_instance_mutex) {
            CloseHandle(g_instance_mutex);
            g_instance_mutex = NULL;
        }
#else
        if (g_instance_fd >= 0) {
            ::close(g_instance_fd);
            g_instance_fd = -1;
        }
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

std::atomic<uint32_t> g_tmp_counter{0};

std::string make_tmp_path(const char* dst) {
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
        std::vector<wchar_t> wpath(static_cast<size_t>(wpath_len));
        std::vector<wchar_t> wmode(static_cast<size_t>(wmode_len));
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
        std::vector<wchar_t> wpath(static_cast<size_t>(wpath_len));
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

FILE* open_tmp_exclusive(const std::string& path) {
#if defined(_WIN32)
    int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    if (wlen <= 0) return nullptr;
    std::vector<wchar_t> w(static_cast<size_t>(wlen));
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, w.data(), wlen);
    return _wfopen(w.data(), L"wbx");
#else
    int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) return nullptr;
    FILE* f = ::fdopen(fd, "wb");
    if (!f) {
        ::close(fd);
        ::unlink(path.c_str());
    }
    return f;
#endif
}

bool sync_file(FILE* f) {
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

void sync_parent_dir(const char* filepath) {
#if !defined(_WIN32)
    std::string path(filepath);
    size_t slash = path.find_last_of('/');
    std::string dir = (slash == std::string::npos) ? "." : (slash == 0 ? "/" : path.substr(0, slash));
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

struct Crc32Tables {
    uint32_t tab[8][256]{};
    constexpr Crc32Tables() {
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int j = 0; j < 8; ++j) {
                c = (c >> 1) ^ (0xEDB88320u & (~(c & 1u) + 1u));
            }
            tab[0][i] = c;
        }
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = tab[0][i];
            for (int k = 1; k < 8; ++k) {
                c = tab[0][c & 0xFFu] ^ (c >> 8);
                tab[k][i] = c;
            }
        }
    }
};

constexpr Crc32Tables g_crc32_tables{};

uint32_t update_crc32(uint32_t crc_in, const void* buf, size_t len) {
    if (!buf || len == 0) return crc_in;
    const auto* data = static_cast<const uint8_t*>(buf);
    uint32_t crc = ~crc_in;

    while (len >= 8) {
        uint32_t b0 = data[0] ^ (crc & 0xFFu);
        uint32_t b1 = data[1] ^ ((crc >> 8) & 0xFFu);
        uint32_t b2 = data[2] ^ ((crc >> 16) & 0xFFu);
        uint32_t b3 = data[3] ^ ((crc >> 24) & 0xFFu);
        crc = g_crc32_tables.tab[7][b0] ^
              g_crc32_tables.tab[6][b1] ^
              g_crc32_tables.tab[5][b2] ^
              g_crc32_tables.tab[4][b3] ^
              g_crc32_tables.tab[3][data[4]] ^
              g_crc32_tables.tab[2][data[5]] ^
              g_crc32_tables.tab[1][data[6]] ^
              g_crc32_tables.tab[0][data[7]];
        data += 8;
        len -= 8;
    }

    while (len > 0) {
        crc = g_crc32_tables.tab[0][(crc ^ *data++) & 0xFFu] ^ (crc >> 8);
        --len;
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

bool tokenize_checked(
    const struct llama_vocab* vocab,
    const char* text,
    size_t text_len,
    bool add_special,
    bool parse_special,
    std::vector<llama_token>* out_tokens,
    int32_t* out_count = nullptr
) {
    if (!vocab || !text || text_len > static_cast<size_t>(INT32_MAX)) {
        return false;
    }
    const int32_t len_i32 = static_cast<int32_t>(text_len);
    const int32_t n = llama_tokenize(vocab, text, len_i32, nullptr, 0, add_special, parse_special);
    if (n == INT32_MIN) {
        return false;
    }
    int32_t needed = (n < 0) ? -n : n;
    if (needed <= 0) {
        if (out_count) *out_count = 0;
        if (out_tokens) out_tokens->clear();
        return (text_len == 0);
    }
    if (!out_tokens) {
        if (out_count) *out_count = needed;
        return true;
    }
    out_tokens->resize(static_cast<size_t>(needed));
    int32_t actual = llama_tokenize(
        vocab, text, len_i32, out_tokens->data(), static_cast<int32_t>(out_tokens->size()), add_special, parse_special);
    if (actual < 0) {
        if (actual == INT32_MIN) return false;
        needed = -actual;
        out_tokens->resize(static_cast<size_t>(needed));
        actual = llama_tokenize(
            vocab, text, len_i32, out_tokens->data(), static_cast<int32_t>(out_tokens->size()), add_special, parse_special);
        if (actual < 0) return false;
    }
    out_tokens->resize(static_cast<size_t>(actual));
    if (out_count) *out_count = actual;
    return true;
}

bool is_plain_html_tag(const char* t, size_t len) {
    static const char* const tags[] = {
        "a","b","i","u","p","br","hr","em","li","ol","ul","tr","td","th","sub","sup",
        "h1","h2","h3","h4","h5","h6","div","img","pre","span","code","table","thead",
        "tbody","tfoot","caption","strong","blockquote"
    };
    const char* name = t + 1;
    size_t n = len - 2;
    if (n > 0 && name[0] == '/') {
        ++name;
        --n;
    }
    for (const char* tag : tags) {
        if (n == std::strlen(tag) && std::memcmp(name, tag, n) == 0) return true;
    }
    return false;
}

std::vector<std::string> collect_special_texts(const struct llama_vocab* v) {
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
    std::unordered_set<std::string> set;
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

    void clear() {
        set.clear();
        min_len = SIZE_MAX;
        max_len = 0;
    }
};

std::string neutralize_special_fast(const char* in, const SpecialTokenLookup& lookup) {
    if (!in || !*in) return "";
    if (lookup.set.empty()) return std::string(in);

    static const char ZWSP[] = "\xE2\x80\x8B";
    const std::string_view s(in);
    std::string out;
    out.reserve(s.size());

    std::string probe;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        bool hit = false;
        if (c == '<' || c == '[') {
            const size_t rem = s.size() - i;
            const size_t max_check = std::min(rem, lookup.max_len);
            const char expected_close = (c == '<') ? '>' : ']';
            for (size_t len = lookup.min_len; len <= max_check; ++len) {
                if (s[i + len - 1] == expected_close) {
                    probe.assign(s.data() + i, len);
                    if (lookup.set.count(probe) > 0) {
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

bool is_valid_role(const char* r) {
    if (!r) return false;
    static const char* const ok[] = {"system", "user", "assistant", "tool", "developer"};
    for (const char* o : ok) {
        if (std::strcmp(r, o) == 0) return true;
    }
    return false;
}

uint64_t compute_model_fingerprint(struct llama_model* model) {
    if (!model) return 0;
    uint64_t h = 14695981039346656037ULL;
    auto add_bytes = [&h](const void* data, size_t len) {
        const auto* p = static_cast<const uint8_t*>(data);
        for (size_t i = 0; i < len; ++i) {
            h ^= p[i];
            h *= 1099511628211ULL;
        }
    };
    auto add_u64 = [&add_bytes](uint64_t val) {
        add_bytes(&val, sizeof(val));
    };
    auto add_cstr = [&add_bytes](const char* s) {
        if (s) add_bytes(s, std::strlen(s) + 1);
    };

    add_u64(static_cast<uint64_t>(llama_model_n_ctx_train(model)));
    add_u64(static_cast<uint64_t>(llama_model_n_embd(model)));
    add_u64(static_cast<uint64_t>(llama_model_n_layer(model)));
    add_u64(static_cast<uint64_t>(llama_model_n_head(model)));
    add_u64(static_cast<uint64_t>(llama_model_n_head_kv(model)));
    add_u64(static_cast<uint64_t>(llama_model_size(model)));
    add_u64(static_cast<uint64_t>(llama_model_n_params(model)));

    const struct llama_vocab* v = llama_model_get_vocab(model);
    const int32_t n_vocab = v ? llama_vocab_n_tokens(v) : 0;
    add_u64(static_cast<uint64_t>(n_vocab));

    char desc[256] = {};
    llama_model_desc(model, desc, sizeof(desc));
    add_cstr(desc);

    const char* chat_tmpl = llama_model_chat_template(model, nullptr);
    add_cstr(chat_tmpl ? chat_tmpl : "");

    const int32_t meta_count = llama_model_meta_count(model);
    add_u64(static_cast<uint64_t>(meta_count));
    char key_buf[256];
    char val_buf[512];
    for (int32_t i = 0; i < meta_count; ++i) {
        if (llama_model_meta_key_by_index(model, i, key_buf, sizeof(key_buf)) >= 0) {
            add_cstr(key_buf);
        }
        if (llama_model_meta_val_str_by_index(model, i, val_buf, sizeof(val_buf)) >= 0) {
            add_cstr(val_buf);
        }
    }

    if (v && n_vocab > 0) {
        const int32_t step = std::max<int32_t>(1, n_vocab / 64);
        for (int32_t tok = 0; tok < n_vocab; tok += step) {
            const char* t = llama_vocab_get_text(v, tok);
            if (t) add_cstr(t);
        }
    }

    return h;
}

bool v1_stream_adapter(const char* piece, size_t len, void* udata);

struct V1StreamThunk {
    infer_stream_cb cb = nullptr;
    void* user_data    = nullptr;
};

bool v1_stream_adapter(const char* piece, size_t len, void* udata) {
    (void)len;
    auto* thunk = static_cast<V1StreamThunk*>(udata);
    return thunk && thunk->cb ? thunk->cb(piece, thunk->user_data) : false;
}

class StreamOutputBuffer {
public:
    StreamOutputBuffer(infer_stream_cb_v2 cb, void* udata, const std::vector<std::string>& stops)
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
                    pre.resize(valid);
                    if (!callback_(pre.c_str(), valid, user_data_)) {
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
            if (!callback_(to_emit.c_str(), flushable, user_data_)) {
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
                    pre.resize(valid);
                    if (!callback_(pre.c_str(), valid, user_data_)) {
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
                std::string emit = buffer_.substr(0, valid);
                if (!callback_(emit.c_str(), valid, user_data_)) {
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
    infer_stream_cb_v2 callback_;
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
    uint32_t payload_crc;   // CRC32 over header (with payload_crc=0) + KV payload + history
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
    std::string              custom_chat_template;
    std::vector<std::string> stop_words;
    SpecialTokenLookup       special_lookup;
    uint32_t                 n_ctx  = 0;
    int                      n_keep = 0;

    std::atomic<int>             n_past{0};
    std::atomic<bool>            abort_flag{false};
    std::atomic<bool>            is_generating{false};
    std::atomic<bool>            last_success{true};
    std::atomic<std::thread::id> active_gen_thread_id{};

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

void invalidate_kv(infer_ctx* ctx) {
    if (!ctx) return;
    if (ctx->lctx) {
        llama_memory_clear(llama_get_memory(ctx->lctx), false);
    }
    ctx->active_kv_tokens.clear();
    ctx->n_past.store(0, std::memory_order_relaxed);
    ctx->n_keep = 0;
}

bool safe_kv_seq_rm(infer_ctx* ctx, int seq_id, int p0, int p1) {
    if (!ctx || !ctx->lctx) return false;
    bool ok = llama_memory_seq_rm(llama_get_memory(ctx->lctx), seq_id, p0, p1);
    if (!ok) {
        invalidate_kv(ctx);
        return false;
    }
    return true;
}

void release_gen(infer_ctx* c) {
    if (!c) return;
    std::lock_guard<std::mutex> lk(c->gen_mutex);
    c->active_gen_thread_id.store(std::thread::id{}, std::memory_order_release);
    c->is_generating.store(false, std::memory_order_release);
    c->cv_generating.notify_all();
}

struct GenAcquireGuard {
    infer_ctx* ctx = nullptr;
    bool acquired  = false;

    explicit GenAcquireGuard(infer_ctx* c) : ctx(c) {
        if (!ctx) return;
        bool expected = false;
        if (ctx->is_generating.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            ctx->active_gen_thread_id.store(std::this_thread::get_id(), std::memory_order_release);
            acquired = true;
        }
    }

    ~GenAcquireGuard() {
        if (acquired && ctx) {
            release_gen(ctx);
        }
    }
};

struct PrefillRollbackGuard {
    infer_ctx* ctx = nullptr;
    int start_past = 0;
    bool completed = false;

    ~PrefillRollbackGuard() {
        if (!completed && ctx && ctx->lctx) {
            if (safe_kv_seq_rm(ctx, 0, start_past, -1)) {
                if (ctx->active_kv_tokens.size() > static_cast<size_t>(start_past)) {
                    ctx->active_kv_tokens.resize(static_cast<size_t>(start_past));
                }
                ctx->n_past.store(start_past, std::memory_order_relaxed);
            }
        }
    }
};

bool infer_has_real_gpu() {
    size_t count = ggml_backend_dev_count();
    for (size_t i = 0; i < count; ++i) {
        auto type = ggml_backend_dev_type(ggml_backend_dev_get(i));
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            return true;
        }
    }
    return false;
}

int32_t pick_threads(bool has_gpu, bool batch_phase) {
    (void)has_gpu;
    (void)batch_phase;
    if (g_cached_env_threads > 0) {
        return g_cached_env_threads;
    }
    uint32_t phys = (g_cached_phys_cores > 0) ? g_cached_phys_cores : detect_physical_cores();
    return static_cast<int32_t>(phys > 0 ? phys : 4);
}

enum ggml_type map_kv_type(int32_t kv_type) {
    switch (kv_type) {
        case INFER_KV_TYPE_F16:     return GGML_TYPE_F16;
        case INFER_KV_TYPE_BF16:    return GGML_TYPE_BF16;
        case INFER_KV_TYPE_Q8_0:    return GGML_TYPE_Q8_0;
        case INFER_KV_TYPE_Q4_0:    return GGML_TYPE_Q4_0;
        case INFER_KV_TYPE_DEFAULT:
        default:                    return GGML_TYPE_F16;
    }
}

int32_t read_model_meta_i32(struct llama_model* model, const std::string& key, int32_t fallback) {
    char buf[128] = {};
    if (llama_model_meta_val_str(model, key.c_str(), buf, sizeof(buf)) > 0) {
        char* endp = nullptr;
        long v = std::strtol(buf, &endp, 10);
        if (endp != buf && v > 0 && v <= INT32_MAX) {
            return static_cast<int32_t>(v);
        }
    }
    return fallback;
}

 


// ============================================================================
// Replace calculate_safe_n_ctx in infer_core.cpp (inside anonymous namespace)
// ============================================================================
uint32_t calculate_safe_n_ctx(
    struct llama_model* model,
    uint32_t requested_ctx,
    int32_t n_gpu_layers,
    enum ggml_type type_k,
    enum ggml_type type_v
) {
    uint32_t max_train = static_cast<uint32_t>(llama_model_n_ctx_train(model));

    uint64_t total_vram = 0;
    uint64_t free_vram  = 0;
    size_t dev_count = ggml_backend_dev_count();
    for (size_t i = 0; i < dev_count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        auto dtype = ggml_backend_dev_type(dev);
        if (dtype == GGML_BACKEND_DEVICE_TYPE_GPU || dtype == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            size_t dev_free = 0, dev_total = 0;
            ggml_backend_dev_memory(dev, &dev_free, &dev_total);
            total_vram += static_cast<uint64_t>(dev_total);
            free_vram  += static_cast<uint64_t>(dev_free);
        }
    }

    uint64_t free_sys_ram = query_system_free_ram();

    // Auto-select target context from GPU VRAM + System RAM
    uint32_t auto_tier_ctx = DEFAULT_FALLBACK_CTX; // 4096 default (6 GB GPU)
    if (total_vram == 0) {
        // CPU-only mode: scale with System RAM (8192 if >= 16 GB free RAM, else 4096)
        auto_tier_ctx = (free_sys_ram >= (12000ULL * 1024ULL * 1024ULL)) ? 8192 : 4096;
    } else if (total_vram <= (4600ULL * 1024ULL * 1024ULL)) {
        auto_tier_ctx = 2048;  // <= 4 GB GPU
    } else if (total_vram >= (14000ULL * 1024ULL * 1024ULL)) {
        auto_tier_ctx = 16384; // >= 16 GB GPU
    } else if (total_vram >= (7500ULL * 1024ULL * 1024ULL)) {
        auto_tier_ctx = 8192;  // 8 GB - 12 GB GPU
    }

    uint32_t target_ctx = (requested_ctx > 0) ? requested_ctx : auto_tier_ctx;
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

    int32_t default_head_dim = n_embd / n_head;
    int32_t head_dim_k = default_head_dim;
    int32_t head_dim_v = default_head_dim;

    char arch_buf[64] = {};
    if (llama_model_meta_val_str(model, "general.architecture", arch_buf, sizeof(arch_buf)) > 0) {
        std::string arch(arch_buf);
        head_dim_k = read_model_meta_i32(model, arch + ".attention.key_length", default_head_dim);
        head_dim_v = read_model_meta_i32(model, arch + ".attention.value_length", head_dim_k);
    }

    const size_t blck_k = std::max<size_t>(1, ggml_blck_size(type_k));
    const size_t blck_v = std::max<size_t>(1, ggml_blck_size(type_v));
    const size_t row_bytes_k = (static_cast<size_t>(n_head_kv) * static_cast<size_t>(head_dim_k) * ggml_type_size(type_k)) / blck_k;
    const size_t row_bytes_v = (static_cast<size_t>(n_head_kv) * static_cast<size_t>(head_dim_v) * ggml_type_size(type_v)) / blck_v;
    const size_t bytes_per_token = static_cast<size_t>(n_layer) * (row_bytes_k + row_bytes_v);

    if (bytes_per_token == 0) {
        return target_ctx;
    }

    // In Hybrid GPU+RAM mode (when n_gpu_layers < n_layer), KV layers on CPU use System RAM!
    uint64_t usable_pool = (n_gpu_layers != 0) ? free_vram : free_sys_ram;
    if (n_gpu_layers > 0 && n_gpu_layers < n_layer && free_sys_ram > usable_pool) {
        // Hybrid offload: KV cache is split between VRAM and System RAM
        usable_pool += (free_sys_ram / 4);
    }
    if (usable_pool == 0) {
        usable_pool = free_sys_ram;
    }
    if (usable_pool == 0) {
        return target_ctx;
    }

    constexpr uint64_t headroom = 384ULL * 1024ULL * 1024ULL;
    if (usable_pool <= headroom) {
        return std::min<uint32_t>(target_ctx, 2048);
    }

    uint64_t available_mem = usable_pool - headroom;
    uint64_t max_tokens_64 = available_mem / static_cast<uint64_t>(bytes_per_token);
    uint32_t safe_tokens = (max_tokens_64 > UINT32_MAX) ? UINT32_MAX : static_cast<uint32_t>(max_tokens_64);

    uint32_t floor_ctx = std::min<uint32_t>(target_ctx, 2048);
    return std::max<uint32_t>(floor_ctx, std::min<uint32_t>(target_ctx, safe_tokens));
}



template <typename T, typename = void>
struct has_flash_attn_bool : std::false_type {};

template <typename T>
struct has_flash_attn_bool<T, std::void_t<decltype(std::declval<T&>().flash_attn = true)>> : std::true_type {};

template <typename T, typename = void>
struct has_flash_attn_type : std::false_type {};

template <typename T>
struct has_flash_attn_type<T, std::void_t<decltype(std::declval<T&>().flash_attn_type)>> : std::true_type {};

template <typename CtxParams = struct llama_context_params>
void apply_flash_attn_param(CtxParams& ctx_params, bool enable) {
    if constexpr (has_flash_attn_bool<CtxParams>::value) {
        ctx_params.flash_attn = enable;
    } else if constexpr (has_flash_attn_type<CtxParams>::value) {
        using FaEnum = decltype(ctx_params.flash_attn_type);
        ctx_params.flash_attn_type = static_cast<FaEnum>(enable ? 1 : 0);
    } else {
        (void)ctx_params;
        (void)enable;
    }
}

template <typename T, typename = void>
struct has_model_use_mmap : std::false_type {};

template <typename T>
struct has_model_use_mmap<T, std::void_t<decltype(std::declval<T&>().use_mmap = true)>> : std::true_type {};

template <typename T, typename = void>
struct has_model_use_mlock : std::false_type {};

template <typename T>
struct has_model_use_mlock<T, std::void_t<decltype(std::declval<T&>().use_mlock = true)>> : std::true_type {};

template <typename T, typename = void>
struct has_model_load_mode : std::false_type {};

template <typename T>
struct has_model_load_mode<T, std::void_t<decltype(std::declval<T&>().load_mode)>> : std::true_type {};

template <typename ModelParams = struct llama_model_params>
void apply_model_load_params(ModelParams& llama_mparams, bool use_mmap, bool use_mlock) {
    if constexpr (has_model_use_mmap<ModelParams>::value) {
        llama_mparams.use_mmap = use_mmap;
    }
    if constexpr (has_model_use_mlock<ModelParams>::value) {
        llama_mparams.use_mlock = use_mlock;
    }
    if constexpr (has_model_load_mode<ModelParams>::value) {
        using ModeEnum = decltype(llama_mparams.load_mode);
        if (use_mmap && use_mlock) {
            llama_mparams.load_mode = static_cast<ModeEnum>(3); // LLAMA_LOAD_MODE_MMAP_MLOCK
        } else if (use_mlock) {
            llama_mparams.load_mode = static_cast<ModeEnum>(2); // LLAMA_LOAD_MODE_MLOCK
        } else if (use_mmap) {
            llama_mparams.load_mode = static_cast<ModeEnum>(1); // LLAMA_LOAD_MODE_MMAP
        } else {
            llama_mparams.load_mode = static_cast<ModeEnum>(0); // LLAMA_LOAD_MODE_NONE
        }
    }
}

using UniqueSampler = std::unique_ptr<struct llama_sampler, decltype(&llama_sampler_free)>;

UniqueSampler create_sampler(
    const infer_sampler_params& params_in,
    struct llama_model* model,
    uint32_t n_ctx,
    const std::string& grammar_gbnf,
    std::string& err_out,
    struct llama_sampler** penalty_smpl_out = nullptr
) {
    if (penalty_smpl_out) *penalty_smpl_out = nullptr;

    infer_sampler_params params = params_in;
    if (std::isnan(params.temperature) || std::isinf(params.temperature)) {
        err_out = "Invalid temperature parameter.";
        return UniqueSampler(nullptr, &llama_sampler_free);
    }
    if (params.temperature < 0.0f) {
        params.temperature = 0.0f;
    }
    if (std::isnan(params.top_p) || std::isinf(params.top_p) || params.top_p < 0.0f || params.top_p > 1.0f) {
        err_out = "Invalid top_p parameter.";
        return UniqueSampler(nullptr, &llama_sampler_free);
    }
    if (std::isnan(params.min_p) || std::isinf(params.min_p) || params.min_p < 0.0f || params.min_p > 1.0f) {
        err_out = "Invalid min_p parameter.";
        return UniqueSampler(nullptr, &llama_sampler_free);
    }
    if (std::isnan(params.repeat_penalty) || std::isinf(params.repeat_penalty) || params.repeat_penalty < 0.0f) {
        err_out = "Invalid repeat_penalty parameter.";
        return UniqueSampler(nullptr, &llama_sampler_free);
    }

    struct llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    UniqueSampler chain(llama_sampler_chain_init(sparams), &llama_sampler_free);
    if (!chain) {
        err_out = "Failed to allocate sampler chain.";
        return UniqueSampler(nullptr, &llama_sampler_free);
    }

    const struct llama_vocab* vocab = model ? llama_model_get_vocab(model) : nullptr;

    if (params.repeat_penalty > 0.0f && params.repeat_penalty != 1.0f && vocab) {
        const int32_t max_window = (n_ctx > 0) ? static_cast<int32_t>(std::min<uint32_t>(n_ctx, 65536u)) : 4096;
        int32_t last_n = (params.repeat_last_n <= 0) ? 64 : params.repeat_last_n;
        last_n = std::min(last_n, max_window);
        int32_t n_vocab = llama_vocab_n_tokens(vocab);
        struct llama_sampler* psmpl = llama_sampler_init_penalties(n_vocab, last_n, params.repeat_penalty, 0.0f, 0.0f);
        if (psmpl) {
            llama_sampler_chain_add(chain.get(), psmpl);
            if (penalty_smpl_out) *penalty_smpl_out = psmpl;
        }
    }

    if (!grammar_gbnf.empty() && vocab) {
        struct llama_sampler* gsampler = llama_sampler_init_grammar(vocab, grammar_gbnf.c_str(), "root");
        if (gsampler) {
            llama_sampler_chain_add(chain.get(), gsampler);
        } else {
            err_out = "Failed to parse grammar rules.";
            if (penalty_smpl_out) *penalty_smpl_out = nullptr;
            return UniqueSampler(nullptr, &llama_sampler_free);
        }
    }

    if (params.top_k > 0) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_top_k(params.top_k));
    }
    if (params.top_p > 0.0f && params.top_p < 1.0f) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_top_p(params.top_p, 1));
    }
    if (params.min_p > 0.0f && params.min_p < 1.0f) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_min_p(params.min_p, 1));
    }

    if (params.temperature > 0.0f) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(params.temperature));
        llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(params.seed ? params.seed : LLAMA_DEFAULT_SEED));
    } else {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
    }

    return chain;
}

bool render_chat_template_with_fallback(
    infer_ctx* ctx,
    const std::vector<struct llama_chat_message>& llama_msgs,
    std::string& out_rendered
) {
    const char* model_tmpl = llama_model_chat_template(ctx->model, nullptr);
    const char* active_tmpl = !ctx->custom_chat_template.empty()
                                  ? ctx->custom_chat_template.c_str()
                                  : model_tmpl;

    auto try_apply = [&](const char* tmpl) -> bool {
        if (!tmpl || !*tmpl) return false;
        int alloc_size = llama_chat_apply_template(tmpl, llama_msgs.data(), llama_msgs.size(), true, nullptr, 0);
        if (alloc_size <= 0) return false;
        std::vector<char> buf(static_cast<size_t>(alloc_size) + 1, '\0');
        int written = llama_chat_apply_template(tmpl, llama_msgs.data(), llama_msgs.size(), true, buf.data(), static_cast<int32_t>(buf.size()));
        if (written < 0) return false;
        out_rendered.assign(buf.data(), static_cast<size_t>(written));
        return true;
    };

    if (try_apply(active_tmpl)) {
        return true;
    }

    if (try_apply("chatml")) {
        return true;
    }

    std::string tmpl_name = active_tmpl ? active_tmpl : "<none>";
    if (tmpl_name.size() > 64) {
        tmpl_name = tmpl_name.substr(0, 64) + "...";
    }
    ctx->set_error("Chat template formatting failed for template: '" + tmpl_name + "'");
    return false;
}

bool validate_chat_messages(infer_ctx* ctx, const infer_chat_msg* msgs, size_t num_msgs) {
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
    return true;
}

void free_model_resources_locked(infer_ctx* ctx) {
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
    ctx->special_lookup.clear();
    ctx->n_ctx  = 0;
    ctx->n_keep = 0;
    ctx->n_past.store(0, std::memory_order_relaxed);
    ctx->chat_history.clear();
    ctx->active_kv_tokens.clear();
}

bool execute_decode(
    infer_ctx* ctx,
    const std::vector<llama_token>& prompt_tokens,
    bool append_to_existing_past,
    int32_t max_tokens,
    infer_stream_cb_v2 callback,
    void* user_data,
    std::string* out_generated_text = nullptr
) {
    if (out_generated_text) out_generated_text->clear();

    const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
    const int effective_batch_cap = std::min<int>(DEFAULT_BATCH_CAP, static_cast<int>(llama_n_batch(ctx->lctx)));
    size_t prefix_len = 0;

    if (append_to_existing_past) {
        const size_t current_past_sz = static_cast<size_t>(std::max(0, ctx->n_past.load(std::memory_order_relaxed)));
        const size_t kv_bound = std::min(ctx->active_kv_tokens.size(), current_past_sz);

        while (prefix_len < kv_bound &&
               prefix_len < prompt_tokens.size() &&
               ctx->active_kv_tokens[prefix_len] == prompt_tokens[prefix_len]) {
            prefix_len++;
        }

        if (prefix_len == prompt_tokens.size() && prefix_len > 0) {
            prefix_len--;
        }

        if (prefix_len < current_past_sz || prefix_len < ctx->active_kv_tokens.size()) {
            if (!safe_kv_seq_rm(ctx, 0, static_cast<int>(prefix_len), -1)) {
                prefix_len = 0;
            } else {
                ctx->active_kv_tokens.resize(prefix_len);
                ctx->n_past.store(static_cast<int>(prefix_len), std::memory_order_relaxed);
            }
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
            if (!safe_kv_seq_rm(ctx, 0, n_keep, n_keep + n_discard)) {
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
                ctx->active_kv_tokens.resize(static_cast<size_t>(n_keep));
            }

            ctx->n_past.store(current_past, std::memory_order_relaxed);
        }
    } else {
        invalidate_kv(ctx);
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

    PrefillRollbackGuard prefill_guard{ctx, start_past, false};

    if (!delta_prompt.empty()) {
        for (size_t i = 0; i < delta_prompt.size(); ) {
            if (ctx->abort_flag.load(std::memory_order_relaxed)) {
                ctx->set_error("Generation aborted during prompt prefill.");
                return false;
            }

            batch.n_tokens = 0;
            int chunk = std::min<int>(effective_batch_cap, static_cast<int>(delta_prompt.size() - i));

            for (int j = 0; j < chunk; ++j) {
                size_t idx = i + static_cast<size_t>(j);
                batch.token[j]     = delta_prompt[idx];
                batch.pos[j]       = static_cast<llama_pos>(start_past + static_cast<int>(idx));
                batch.n_seq_id[j]  = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j]    = (idx == delta_prompt.size() - 1);
                batch.n_tokens++;
            }

            if (llama_decode(ctx->lctx, batch) != 0) {
                ctx->set_error("llama_decode failed during prompt prefill.");
                return false;
            }

            i += static_cast<size_t>(chunk);
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

    int32_t generated_tokens  = 0;
    const int32_t default_cap = static_cast<int32_t>(std::min<uint32_t>(ctx->n_ctx, static_cast<uint32_t>(INT32_MAX)));
    const int32_t token_limit = (max_tokens > 0) ? max_tokens : default_cap;
    bool execution_success = true;

    auto rollback_stop_tokens = [&](size_t stop_byte_offset) {
        for (const auto& e : tok_log) {
            if (e.second > stop_byte_offset) {
                if (safe_kv_seq_rm(ctx, 0, e.first, -1)) {
                    n_cur = e.first;
                    if (ctx->active_kv_tokens.size() > static_cast<size_t>(n_cur)) {
                        ctx->active_kv_tokens.resize(static_cast<size_t>(n_cur));
                    }
                    ctx->n_past.store(n_cur, std::memory_order_relaxed);
                } else {
                    n_cur = 0;
                }
                rolled_back = true;
                break;
            }
        }
    };

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
            int n_discard = (static_cast<int>(ctx->n_ctx) - n_keep) / 2;
            if (n_discard <= 0) n_discard = 1;

            if (!safe_kv_seq_rm(ctx, 0, n_keep, n_keep + n_discard)) {
                ctx->set_error("Context shift refused by memory architecture.");
                n_cur = 0;
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
                ctx->active_kv_tokens.resize(static_cast<size_t>(n_keep));
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
            if (append_to_existing_past && n_cur > 0) {
                batch.n_tokens     = 0;
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

        int n_chars = llama_token_to_piece(vocab, new_token_id, piece_buf.data(), static_cast<int32_t>(piece_buf.size()), 0, false);
        if (n_chars < 0) {
            if (n_chars == INT32_MIN) {
                ctx->set_error("Token piece length overflow.");
                execution_success = false;
                break;
            }
            piece_buf.resize(static_cast<size_t>(-n_chars) + 1);
            n_chars = llama_token_to_piece(vocab, new_token_id, piece_buf.data(), static_cast<int32_t>(piece_buf.size()), 0, false);
            if (n_chars < 0) {
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
                    rollback_stop_tokens(out_buf.stop_offset());
                }

                if (append_to_existing_past && n_cur > 0) {
                    llama_token term_token = llama_vocab_eot(vocab);
                    if (term_token == LLAMA_TOKEN_NULL) {
                        term_token = llama_vocab_eos(vocab);
                    }
                    if (term_token != LLAMA_TOKEN_NULL) {
                        batch.n_tokens     = 0;
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

        batch.n_tokens     = 0;
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

    if (generated_tokens >= token_limit && append_to_existing_past && n_cur > 0) {
        llama_token term_token = llama_vocab_eot(vocab);
        if (term_token == LLAMA_TOKEN_NULL) {
            term_token = llama_vocab_eos(vocab);
        }
        if (term_token != LLAMA_TOKEN_NULL) {
            batch.n_tokens     = 0;
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
        rollback_stop_tokens(out_buf.stop_offset());
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

bool infer_generate_chat_internal(
    infer_ctx* ctx,
    const infer_chat_msg* msgs,
    size_t num_msgs,
    int32_t max_tokens,
    infer_stream_cb_v2 callback,
    void* user_data
) {
    const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);

    std::vector<OwnedMsg> new_history;
    new_history.reserve(num_msgs + 1);
    for (size_t i = 0; i < num_msgs; ++i) {
        new_history.push_back({
            msgs[i].role,
            neutralize_special_fast(msgs[i].content, ctx->special_lookup)
        });
    }

    std::vector<struct llama_chat_message> llama_msgs;
    llama_msgs.reserve(num_msgs);
    for (const auto& m : new_history) {
        llama_msgs.push_back({ m.role.c_str(), m.content.c_str() });
    }

    std::string formatted;
    if (!render_chat_template_with_fallback(ctx, llama_msgs, formatted)) {
        return false;
    }

    bool add_bos = want_bos(vocab, formatted.c_str());
    std::vector<llama_token> prompt_tokens;
    if (!tokenize_checked(vocab, formatted.c_str(), formatted.size(), add_bos, true, &prompt_tokens) ||
        prompt_tokens.empty()) {
        ctx->set_error("Failed to tokenize rendered chat prompt.");
        return false;
    }

    std::string generated_response;
    bool res = execute_decode(ctx, prompt_tokens, true, max_tokens, callback, user_data, &generated_response);
    if (ctx->n_past.load(std::memory_order_relaxed) > 0) {
        if (res) {
            std::string safe_response = neutralize_special_fast(generated_response.c_str(), ctx->special_lookup);
            new_history.push_back({ "assistant", std::move(safe_response) });
        }
        ctx->chat_history = std::move(new_history);
    }
    return res;
}

} // anonymous namespace

/* ========================================================================== */
/* Public API Implementation                                                  */
/* ========================================================================== */

extern "C" {

uint32_t infer_version(void) {
    return INFER_VERSION_NUM;
}

void infer_set_log_callback(infer_log_cb callback, void* user_data) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_user_log_cb   = callback;
    g_user_log_data = user_data;
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
    p.repeat_penalty = 1.00f;
    p.repeat_last_n  = 64;
    p.seed           = 0;
    return p;
}

infer_model_params infer_model_default_params(void) {
    infer_model_params p{};
    p.struct_size         = sizeof(infer_model_params);
    p.requested_ctx       = 0;
    p.n_gpu_layers        = -1;
    p.use_mmap            = true;
    p.use_mlock           = false;
    p.flash_attn          = false;
    p.kv_type_k           = INFER_KV_TYPE_F16;
    p.kv_type_v           = INFER_KV_TYPE_F16;
    p.allow_ctx_reduction = false;
    p.chat_template       = nullptr;
    return p;
}

infer_ctx* infer_init_ex(char* err_buf, size_t err_buf_size) {
    auto write_err = [err_buf, err_buf_size](const char* msg) {
        if (err_buf && err_buf_size > 0 && msg) {
            size_t len = std::min(err_buf_size - 1, std::strlen(msg));
            std::memcpy(err_buf, msg, len);
            err_buf[len] = '\0';
        }
    };

    if (err_buf && err_buf_size > 0) {
        err_buf[0] = '\0';
    }

    bool reffed = false;
    infer_ctx* ctx = nullptr;
    try {
        std::string backend_err;
        if (!ref_llama_backend(backend_err)) {
            write_err(backend_err.c_str());
            return nullptr;
        }
        reffed = true;
        ctx = new (std::nothrow) infer_ctx();
        if (!ctx) {
            unref_llama_backend();
            write_err("Out of memory allocating infer_ctx.");
            return nullptr;
        }
        ctx->sampler_params = infer_sampler_default_params();
        return ctx;
    } catch (const std::exception& e) {
        if (ctx) delete ctx;
        if (reffed) unref_llama_backend();
        write_err(e.what());
        return nullptr;
    } catch (...) {
        if (ctx) delete ctx;
        if (reffed) unref_llama_backend();
        write_err("Unknown exception in infer_init_ex.");
        return nullptr;
    }
}

infer_ctx* infer_init(void) {
    return infer_init_ex(nullptr, 0);
}

bool infer_unload_model(infer_ctx* ctx) {
    if (!ctx) return false;
    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Cannot unload model while context is busy.");
        return false;
    }

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        ctx->clear_error();
        free_model_resources_locked(ctx);
        return true;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return false;
    } catch (...) {
        ctx->set_error("Unknown exception in infer_unload_model.");
        return false;
    }
}




bool infer_load_model_ex(infer_ctx* ctx, const char* path, const infer_model_params* params) {
    if (!ctx || !path || !params) {
        if (ctx) ctx->set_error("Null path or parameters in infer_load_model_ex.");
        return false;
    }
    if (params->struct_size == 0) {
        ctx->set_error("Invalid infer_model_params struct_size (must be non-zero).");
        return false;
    }

    infer_model_params mparams = infer_model_default_params();
    size_t copy_sz = std::min<size_t>(params->struct_size, sizeof(infer_model_params));
    std::memcpy(&mparams, params, copy_sz);
    mparams.struct_size = sizeof(infer_model_params);

    if (mparams.requested_ctx != 0 && mparams.requested_ctx < 256) {
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

        // 1. Free old model first so GPU VRAM reading is 100% accurate
        free_model_resources_locked(ctx);

        // 2. Measure GGUF file size on disk and free GPU VRAM for Hybrid GPU + RAM split
        uint64_t model_file_bytes = 0;
        {
            ScopedFile mf(open_file_utf8(path, "rb"));
            if (mf) {
                file_size_64(mf.get(), model_file_bytes);
            }
        }

        uint64_t free_vram = 0;
        size_t dev_count = ggml_backend_dev_count();
        for (size_t i = 0; i < dev_count; ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            auto dtype = ggml_backend_dev_type(dev);
            if (dtype == GGML_BACKEND_DEVICE_TYPE_GPU || dtype == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                size_t df = 0, dt = 0;
                ggml_backend_dev_memory(dev, &df, &dt);
                free_vram += static_cast<uint64_t>(df);
            }
        }

        int32_t target_gpu_layers = (mparams.n_gpu_layers < 0) ? 999 : mparams.n_gpu_layers;

        // INCREASED RESERVE: Prevents Windows WDDM from paging VRAM to system RAM (which causes the freeze)
        constexpr uint64_t KV_AND_COMPUTE_VRAM_RESERVE = 1536ULL * 1024ULL * 1024ULL; // 1.5 GB reserve
        
        if (mparams.n_gpu_layers < 0 && free_vram > 0 && model_file_bytes > 0) {
            if (free_vram <= KV_AND_COMPUTE_VRAM_RESERVE) {
                target_gpu_layers = 0; // Not enough free VRAM; safely fall back to CPU/System RAM
            } else {
                uint64_t vram_budget_for_weights = free_vram - KV_AND_COMPUTE_VRAM_RESERVE;
                if (model_file_bytes > vram_budget_for_weights) {
                    double fit_ratio = static_cast<double>(vram_budget_for_weights) / static_cast<double>(model_file_bytes);
                    // FIXED ESTIMATE: Llama 3.2 3B has 28 layers. Overestimating pushes too much to GPU.
                    int32_t est_total_layers = (model_file_bytes > 4800ULL * 1024ULL * 1024ULL) ? 32 : 28;
                    target_gpu_layers = std::max<int32_t>(1, static_cast<int32_t>(fit_ratio * est_total_layers));
                }
            }
        }

        struct llama_model_params llama_mparams = llama_model_default_params();
        llama_mparams.n_gpu_layers = target_gpu_layers;
        apply_model_load_params(llama_mparams, mparams.use_mmap, mparams.use_mlock);

        std::unique_ptr<llama_model, decltype(&llama_model_free)> new_model(
            llama_model_load_from_file(path, llama_mparams), &llama_model_free);
        if (!new_model) {
            ctx->set_error("Failed to load model from path: " + std::string(path));
            return false;
        }

        const enum ggml_type type_k = map_kv_type(mparams.kv_type_k);
        const enum ggml_type type_v = map_kv_type(mparams.kv_type_v);

        uint32_t safe_ctx = calculate_safe_n_ctx(
            new_model.get(), mparams.requested_ctx, llama_mparams.n_gpu_layers, type_k, type_v);

        if (mparams.requested_ctx > 0 && safe_ctx < mparams.requested_ctx && !mparams.allow_ctx_reduction) {
            ctx->set_error("Requested context size (" + std::to_string(mparams.requested_ctx) +
                           ") exceeds model training length or available memory budget (" +
                           std::to_string(safe_ctx) + ").");
            return false;
        }

        if (safe_ctx < 256) {
            ctx->set_error("Context size too small (minimum 256).");
            return false;
        }

        const bool has_gpu = (llama_mparams.n_gpu_layers != 0) && infer_has_real_gpu();
        const int32_t threads_gen   = pick_threads(has_gpu, false);
        const int32_t threads_batch = pick_threads(has_gpu, true);

        std::unique_ptr<llama_context, decltype(&llama_free)> new_lctx(nullptr, &llama_free);
        uint32_t candidate_ctx = safe_ctx;

        while (candidate_ctx >= 256) {
            struct llama_context_params ctx_params = llama_context_default_params();
            ctx_params.n_ctx           = candidate_ctx;
            ctx_params.n_batch         = std::min<uint32_t>(candidate_ctx, DEFAULT_BATCH_CAP);
            ctx_params.n_seq_max       = 1;
            ctx_params.kv_unified      = true;
            ctx_params.type_k          = type_k;
            ctx_params.type_v          = type_v;
            ctx_params.n_threads       = threads_gen;
            ctx_params.n_threads_batch = threads_batch;
            apply_flash_attn_param(ctx_params, mparams.flash_attn);

            new_lctx.reset(llama_init_from_model(new_model.get(), ctx_params));
            if (new_lctx) {
                break;
            }

            if (mparams.requested_ctx > 0 && !mparams.allow_ctx_reduction) {
                break;
            }
            if (candidate_ctx == 256) {
                break;
            }
            candidate_ctx = std::max<uint32_t>(256, candidate_ctx / 2);
        }

        if (!new_lctx) {
            ctx->set_error("Failed to initialize llama_context for requested context size.");
            return false;
        }

        const uint32_t new_n_ctx = llama_n_ctx_seq(new_lctx.get());
        if (mparams.requested_ctx > 0 && new_n_ctx < mparams.requested_ctx && !mparams.allow_ctx_reduction) {
            ctx->set_error("Allocated context size (" + std::to_string(new_n_ctx) +
                           ") is smaller than explicitly requested size (" +
                           std::to_string(mparams.requested_ctx) + ").");
            return false;
        }

        std::string err;
        struct llama_sampler* new_pen = nullptr;
        UniqueSampler new_sampler = create_sampler(
            ctx->sampler_params, new_model.get(), new_n_ctx, ctx->grammar_gbnf, err, &new_pen);
        if (!new_sampler) {
            ctx->set_error(err);
            return false;
        }

        auto specials = collect_special_texts(llama_model_get_vocab(new_model.get()));
        SpecialTokenLookup new_lookup;
        new_lookup.build(specials);

        ctx->model          = new_model.release();
        ctx->lctx           = new_lctx.release();
        ctx->sampler        = new_sampler.release();
        ctx->penalty_smpl   = new_pen;
        ctx->special_lookup = std::move(new_lookup);
        ctx->n_ctx          = new_n_ctx;
        ctx->n_past.store(0, std::memory_order_relaxed);
        ctx->n_keep         = 0;
        if (mparams.chat_template) {
            ctx->custom_chat_template = mparams.chat_template;
        }

        return true;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return false;
    } catch (...) {
        ctx->set_error("Unknown exception in infer_load_model_ex.");
        return false;
    }
}




bool infer_load_model(infer_ctx* ctx, const char* path, uint32_t requested_ctx) {
    infer_model_params params = infer_model_default_params();
    params.requested_ctx       = requested_ctx;
    params.allow_ctx_reduction = (requested_ctx == 0);
    return infer_load_model_ex(ctx, path, &params);
}

bool infer_set_sampler(infer_ctx* ctx, const infer_sampler_params* params) {
    if (!ctx || !params) return false;
    if (params->struct_size == 0) {
        ctx->set_error("Invalid infer_sampler_params struct_size (must be non-zero).");
        return false;
    }

    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Cannot change sampler while context is busy.");
        return false;
    }

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        infer_sampler_params safe_params{};
        size_t copy_sz = std::min<size_t>(params->struct_size, sizeof(infer_sampler_params));
        std::memcpy(&safe_params, params, copy_sz);
        safe_params.struct_size = sizeof(infer_sampler_params);

        std::string err;
        struct llama_sampler* new_pen = nullptr;
        UniqueSampler new_sampler = create_sampler(
            safe_params, ctx->model, ctx->n_ctx, ctx->grammar_gbnf, err, &new_pen);
        if (!new_sampler) {
            ctx->set_error(err);
            return false;
        }

        if (ctx->sampler) {
            llama_sampler_free(ctx->sampler);
        }
        ctx->sampler        = new_sampler.release();
        ctx->penalty_smpl   = new_pen;
        ctx->sampler_params = safe_params;
        return true;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return false;
    } catch (...) {
        ctx->set_error("Unknown exception in infer_set_sampler.");
        return false;
    }
}

bool infer_set_chat_template(infer_ctx* ctx, const char* custom_template) {
    if (!ctx) return false;
    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Cannot change chat template while context is busy.");
        return false;
    }

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        ctx->custom_chat_template = custom_template ? custom_template : "";
        return true;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return false;
    } catch (...) {
        ctx->set_error("Unknown exception in infer_set_chat_template.");
        return false;
    }
}

bool infer_get_model_info(infer_ctx* ctx, infer_model_info* out_info) {
    if (!ctx || !out_info) return false;
    if (out_info->struct_size == 0) {
        ctx->set_error("Invalid infer_model_info struct_size (must be non-zero).");
        return false;
    }

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        if (!ctx->model || !ctx->lctx) {
            ctx->set_error("No active model loaded.");
            return false;
        }

        const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
        infer_model_info tmp{};
        tmp.struct_size  = sizeof(infer_model_info);
        tmp.n_ctx_train  = static_cast<uint32_t>(llama_model_n_ctx_train(ctx->model));
        tmp.n_embd       = static_cast<uint32_t>(llama_model_n_embd(ctx->model));
        tmp.n_layer      = static_cast<uint32_t>(llama_model_n_layer(ctx->model));
        tmp.n_head       = static_cast<uint32_t>(llama_model_n_head(ctx->model));
        tmp.vocab_size   = vocab ? static_cast<uint32_t>(llama_vocab_n_tokens(vocab)) : 0;
        tmp.n_ctx_active = ctx->n_ctx;

        llama_model_desc(ctx->model, tmp.description, sizeof(tmp.description));

        size_t copy_sz = std::min<size_t>(out_info->struct_size, sizeof(infer_model_info));
        std::memcpy(out_info, &tmp, copy_sz);
        return true;
    } catch (...) {
        return false;
    }
}

size_t infer_get_last_error(infer_ctx* ctx, char* out_buf, size_t buf_size) {
    if (!ctx) return 0;
    try {
        std::lock_guard<std::mutex> lock(ctx->err_mutex);
        const size_t full_len = ctx->last_error.size();
        if (out_buf && buf_size > 0) {
            size_t copy_len = std::min(buf_size - 1, full_len);
            std::memcpy(out_buf, ctx->last_error.c_str(), copy_len);
            out_buf[copy_len] = '\0';
        }
        return full_len;
    } catch (...) {
        return 0;
    }
}

bool infer_last_success(infer_ctx* ctx) {
    return ctx ? ctx->last_success.load(std::memory_order_acquire) : false;
}

int32_t infer_tokenize_count(infer_ctx* ctx, const char* text, bool add_special) {
    if (!ctx || !text) return -1;

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        if (!ctx->model) return -1;
        const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
        if (!vocab) return -1;

        std::string safe_text = neutralize_special_fast(text, ctx->special_lookup);
        int32_t count = 0;
        if (!tokenize_checked(vocab, safe_text.c_str(), safe_text.size(), add_special, true, nullptr, &count)) {
            return -1;
        }
        return count;
    } catch (...) {
        return -1;
    }
}

uint32_t infer_get_kv_cache_token_count(infer_ctx* ctx) {
    if (!ctx) return 0;
    return static_cast<uint32_t>(std::max(0, ctx->n_past.load(std::memory_order_relaxed)));
}

bool infer_set_stop_sequences(infer_ctx* ctx, const char* const* stop_words, size_t num_words) {
    if (!ctx) return false;
    if (num_words > 0 && !stop_words) {
        ctx->set_error("stop_words pointer cannot be null when num_words > 0.");
        return false;
    }

    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Cannot change stop sequences while context is busy.");
        return false;
    }

    try {
        std::vector<std::string> validated;
        validated.reserve(num_words);
        for (size_t i = 0; i < num_words; ++i) {
            if (!stop_words[i]) {
                ctx->set_error("Stop sequence entry cannot be null.");
                return false;
            }
            size_t slen = std::strlen(stop_words[i]);
            if (slen == 0 || slen > 256) {
                ctx->set_error("Stop sequence length must be between 1 and 256 bytes.");
                return false;
            }
            validated.emplace_back(stop_words[i], slen);
        }

        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        ctx->stop_words = std::move(validated);
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
        UniqueSampler new_sampler = create_sampler(
            ctx->sampler_params, ctx->model, ctx->n_ctx, new_grammar, err, &new_pen);
        if (!new_sampler) {
            ctx->set_error(err);
            return false;
        }

        if (ctx->sampler) {
            llama_sampler_free(ctx->sampler);
        }
        ctx->sampler      = new_sampler.release();
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

bool infer_memory_clear(infer_ctx* ctx) {
    if (!ctx) return false;
    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Cannot clear memory while context is busy.");
        return false;
    }

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        ctx->clear_error();
        invalidate_kv(ctx);
        if (ctx->sampler) {
            llama_sampler_reset(ctx->sampler);
        }
        ctx->chat_history.clear();
        return true;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return false;
    } catch (...) {
        ctx->set_error("Unknown exception in infer_memory_clear.");
        return false;
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
        if (state_size == 0 || static_cast<uint64_t>(state_size) > MAX_STATE_FILE_SIZE) {
            ctx->set_error("Invalid state size retrieved from context.");
            return false;
        }

        if (ctx->active_kv_tokens.size() > UINT32_MAX || ctx->chat_history.size() > UINT32_MAX) {
            ctx->set_error("History or token count exceeds serialization limit.");
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
        uint32_t msg_count        = static_cast<uint32_t>(ctx->chat_history.size());

        auto append_bytes = [&hist_buf](const void* src, size_t len) {
            const auto* p = static_cast<const uint8_t*>(src);
            hist_buf.insert(hist_buf.end(), p, p + len);
        };

        append_bytes(&active_tok_count, sizeof(active_tok_count));
        if (active_tok_count > 0) {
            append_bytes(ctx->active_kv_tokens.data(), static_cast<size_t>(active_tok_count) * sizeof(llama_token));
        }

        append_bytes(&msg_count, sizeof(msg_count));
        for (const auto& msg : ctx->chat_history) {
            if (msg.role.size() > UINT32_MAX || msg.content.size() > UINT32_MAX) {
                ctx->set_error("Chat message length exceeds uint32_t limit.");
                return false;
            }
            uint32_t rlen = static_cast<uint32_t>(msg.role.size());
            uint32_t clen = static_cast<uint32_t>(msg.content.size());
            append_bytes(&rlen, sizeof(rlen));
            append_bytes(msg.role.data(), rlen);
            append_bytes(&clen, sizeof(clen));
            append_bytes(msg.content.data(), clen);
        }

        if (hist_buf.size() > UINT32_MAX ||
            static_cast<uint64_t>(kv_buffer.size()) + static_cast<uint64_t>(hist_buf.size()) > MAX_STATE_FILE_SIZE) {
            ctx->set_error("Serialized history size exceeds state file limits.");
            return false;
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
        header.n_past        = static_cast<uint32_t>(std::max(0, ctx->n_past.load(std::memory_order_relaxed)));
        header.vocab_size    = vocab_size;
        header.n_embd        = static_cast<uint32_t>(llama_model_n_embd(ctx->model));
        header.n_layer       = static_cast<uint32_t>(llama_model_n_layer(ctx->model));
        header.n_head        = static_cast<uint32_t>(llama_model_n_head(ctx->model));
        header.fingerprint   = compute_model_fingerprint(ctx->model);
        header.payload_crc   = 0;
        header.history_bytes = static_cast<uint32_t>(hist_buf.size());
        header.data_size     = static_cast<uint64_t>(kv_buffer.size()) + static_cast<uint64_t>(hist_buf.size());

        uint32_t crc = update_crc32(0, &header, sizeof(header));
        crc = update_crc32(crc, kv_buffer.data(), kv_buffer.size());
        crc = update_crc32(crc, hist_buf.data(), hist_buf.size());
        header.payload_crc = crc;

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
            std::vector<wchar_t> wtmp(static_cast<size_t>(wtmp_len));
            std::vector<wchar_t> wdst(static_cast<size_t>(wdst_len));
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
        if (kv_bytes > static_cast<uint64_t>(SIZE_MAX)) {
            ctx->set_error("State file KV payload exceeds addressable memory.");
            return false;
        }

        std::vector<uint8_t> kv_buffer(static_cast<size_t>(kv_bytes));
        if (kv_bytes > 0 && std::fread(kv_buffer.data(), 1, static_cast<size_t>(kv_bytes), f.get()) != static_cast<size_t>(kv_bytes)) {
            ctx->set_error("Corrupted or truncated state payload in file.");
            return false;
        }

        std::vector<uint8_t> hist_buf(header.history_bytes);
        if (header.history_bytes > 0) {
            if (std::fread(hist_buf.data(), 1, header.history_bytes, f.get()) != header.history_bytes) {
                ctx->set_error("Corrupted chat history payload in state file.");
                return false;
            }
        }

        const uint32_t expected_crc = header.payload_crc;
        StateFileHeader header_for_crc = header;
        header_for_crc.payload_crc = 0;

        uint32_t computed_crc = update_crc32(0, &header_for_crc, sizeof(header_for_crc));
        computed_crc = update_crc32(computed_crc, kv_buffer.data(), kv_buffer.size());
        computed_crc = update_crc32(computed_crc, hist_buf.data(), hist_buf.size());

        if (computed_crc != expected_crc) {
            ctx->set_error("State file CRC32 checksum mismatch.");
            return false;
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
                if (!read_bytes(restored_kv.data(), static_cast<size_t>(active_tok_count) * sizeof(llama_token))) {
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
                if (!read_bytes(&r[0], rlen)) {
                    ctx->set_error("Truncated message role in state file.");
                    return false;
                }
                if (!is_valid_role(r.c_str())) {
                    ctx->set_error("Invalid message role in state file.");
                    return false;
                }

                if (!read_bytes(&clen, sizeof(clen)) || clen > static_cast<size_t>(end - ptr)) {
                    ctx->set_error("Corrupted content length in state file.");
                    return false;
                }
                std::string c(clen, '\0');
                if (!read_bytes(&c[0], clen)) {
                    ctx->set_error("Truncated message content in state file.");
                    return false;
                }
                c = neutralize_special_fast(c.c_str(), ctx->special_lookup);

                restored_history.push_back({ std::move(r), std::move(c) });
            }
        }

        size_t loaded = llama_state_set_data(ctx->lctx, kv_buffer.data(), kv_buffer.size());
        if (loaded != static_cast<size_t>(kv_bytes)) {
            invalidate_kv(ctx);
            if (ctx->sampler) llama_sampler_reset(ctx->sampler);
            ctx->chat_history.clear();
            ctx->set_error("llama_state_set_data failed to restore context state.");
            return false;
        }

        if (ctx->sampler) {
            llama_sampler_reset(ctx->sampler);
        }

        const llama_pos pmax = llama_memory_seq_pos_max(llama_get_memory(ctx->lctx), 0);
        const int restored_n_past = (pmax >= 0) ? static_cast<int>(pmax) + 1 : 0;

        if (restored_kv.size() != static_cast<size_t>(restored_n_past)) {
            invalidate_kv(ctx);
            ctx->chat_history.clear();
            ctx->set_error("State file active_kv_tokens count does not match restored KV cache positions.");
            return false;
        }

        ctx->n_past.store(restored_n_past, std::memory_order_relaxed);
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

bool infer_get_embeddings_ex(infer_ctx* ctx, const char* text, float* out_vec, size_t vec_len, bool normalize) {
    if (!ctx) return false;
    if (!text || !out_vec) {
        ctx->set_error("Text and output buffer cannot be null.");
        ctx->last_success.store(false, std::memory_order_release);
        return false;
    }

    size_t text_len = std::strlen(text);
    if (text_len > static_cast<size_t>(INT32_MAX)) {
        ctx->set_error("Input text length exceeds maximum limit.");
        ctx->last_success.store(false, std::memory_order_release);
        return false;
    }

    GenAcquireGuard gen_guard(ctx);
    if (!gen_guard.acquired) {
        ctx->set_error("Context is busy.");
        return false;
    }

    ctx->clear_error();
    ctx->last_success.store(false, std::memory_order_release);

    try {
        std::lock_guard<std::mutex> lock(ctx->ctx_mutex);
        if (!ctx->model || !ctx->lctx) {
            ctx->set_error("Model and context must be active to generate embeddings.");
            return false;
        }

        const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
        if (!vocab) {
            ctx->set_error("Model has no vocabulary.");
            return false;
        }

        int n_embd = llama_model_n_embd(ctx->model);
        if (n_embd <= 0 || vec_len < static_cast<size_t>(n_embd)) {
            ctx->set_error("Output buffer smaller than model embedding dimensions.");
            return false;
        }

        bool add_bos = want_bos(vocab, text);
        std::vector<llama_token> tokens;
        if (!tokenize_checked(vocab, text, text_len, add_bos, false, &tokens) || tokens.empty()) {
            ctx->set_error("Embedding tokenization failed or text is empty.");
            return false;
        }

        invalidate_kv(ctx);

        const int effective_batch_cap = std::min<int>(DEFAULT_BATCH_CAP, static_cast<int>(llama_n_batch(ctx->lctx)));
        if (tokens.size() > static_cast<size_t>(effective_batch_cap) &&
            llama_pooling_type(ctx->lctx) != LLAMA_POOLING_TYPE_NONE) {
            ctx->set_error("Input exceeds batch size; pooled embeddings cannot be chunked.");
            return false;
        }

        llama_set_embeddings(ctx->lctx, true);
        struct EmbeddingModeGuard {
            infer_ctx* c;
            ~EmbeddingModeGuard() {
                llama_set_embeddings(c->lctx, false);
                invalidate_kv(c);
            }
        } embd_guard{ctx};

        struct llama_batch batch = llama_batch_init(effective_batch_cap, 0, 1);
        struct BatchGuard {
            struct llama_batch& b;
            ~BatchGuard() { llama_batch_free(b); }
        } guard{batch};

        for (size_t i = 0; i < tokens.size(); ) {
            if (ctx->abort_flag.load(std::memory_order_relaxed)) {
                ctx->set_error("Embedding generation aborted.");
                return false;
            }

            batch.n_tokens = 0;
            int chunk = std::min<int>(effective_batch_cap, static_cast<int>(tokens.size() - i));

            for (int j = 0; j < chunk; ++j) {
                size_t idx = i + static_cast<size_t>(j);
                batch.token[j]     = tokens[idx];
                batch.pos[j]       = static_cast<llama_pos>(idx);
                batch.n_seq_id[j]  = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j]    = (idx == tokens.size() - 1);
                batch.n_tokens++;
            }

            if (llama_decode(ctx->lctx, batch) != 0) {
                ctx->set_error("Failed to decode prompt for embedding generation.");
                return false;
            }

            i += static_cast<size_t>(chunk);
        }

        const float* embd = llama_get_embeddings_seq(ctx->lctx, 0);
        if (!embd) embd = llama_get_embeddings_ith(ctx->lctx, -1);
        if (!embd) embd = llama_get_embeddings(ctx->lctx);

        if (!embd) {
            ctx->set_error("Failed to extract embeddings from context.");
            return false;
        }

        std::memcpy(out_vec, embd, static_cast<size_t>(n_embd) * sizeof(float));

        if (normalize) {
            double sum_sq = 0.0;
            for (int i = 0; i < n_embd; ++i) {
                sum_sq += static_cast<double>(out_vec[i]) * static_cast<double>(out_vec[i]);
            }
            if (sum_sq > 0.0) {
                const float inv_norm = static_cast<float>(1.0 / std::sqrt(sum_sq));
                for (int i = 0; i < n_embd; ++i) {
                    out_vec[i] *= inv_norm;
                }
            }
        }

        ctx->last_success.store(true, std::memory_order_release);
        return true;
    } catch (const std::exception& e) {
        ctx->set_error(e.what());
        return false;
    } catch (...) {
        ctx->set_error("Unknown exception during infer_get_embeddings.");
        return false;
    }
}

bool infer_get_embeddings(infer_ctx* ctx, const char* text, float* out_vec, size_t vec_len) {
    return infer_get_embeddings_ex(ctx, text, out_vec, vec_len, false);
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
    if (std::this_thread::get_id() == ctx->active_gen_thread_id.load(std::memory_order_acquire)) {
        ctx->set_error("Cannot call infer_wait from inside an active generation callback.");
        return;
    }

    try {
        std::unique_lock<std::mutex> lk(ctx->gen_mutex);
        ctx->cv_generating.wait(lk, [ctx]() {
            return !ctx->is_generating.load(std::memory_order_acquire);
        });
    } catch (...) {}

    try {
        std::lock_guard<std::mutex> lock(ctx->worker_mutex);
        if (ctx->async_worker.joinable()) {
            ctx->async_worker.join();
        }
    } catch (...) {}
}

bool infer_generate_chat_v2(
    infer_ctx* ctx,
    const infer_chat_msg* msgs,
    size_t num_msgs,
    int32_t max_tokens,
    infer_stream_cb_v2 callback,
    void* user_data
) {
    if (!ctx || !msgs || num_msgs == 0 || !callback) return false;
    if (!validate_chat_messages(ctx, msgs, num_msgs)) return false;

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

bool infer_generate_chat(
    infer_ctx* ctx,
    const infer_chat_msg* msgs,
    size_t num_msgs,
    int32_t max_tokens,
    infer_stream_cb callback,
    void* user_data
) {
    if (!callback) return false;
    V1StreamThunk thunk{callback, user_data};
    return infer_generate_chat_v2(ctx, msgs, num_msgs, max_tokens, v1_stream_adapter, &thunk);
}

bool infer_generate_chat_async_v2(
    infer_ctx* ctx,
    const infer_chat_msg* msgs,
    size_t num_msgs,
    int32_t max_tokens,
    infer_stream_cb_v2 callback,
    void* user_data
) {
    if (!ctx || !msgs || num_msgs == 0 || !callback) return false;
    if (!validate_chat_messages(ctx, msgs, num_msgs)) return false;

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
                ctx->active_gen_thread_id.store(std::this_thread::get_id(), std::memory_order_release);
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

bool infer_generate_chat_async(
    infer_ctx* ctx,
    const infer_chat_msg* msgs,
    size_t num_msgs,
    int32_t max_tokens,
    infer_stream_cb callback,
    void* user_data
) {
    if (!ctx || !msgs || num_msgs == 0 || !callback) return false;
    if (!validate_chat_messages(ctx, msgs, num_msgs)) return false;

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
                ctx->active_gen_thread_id.store(std::this_thread::get_id(), std::memory_order_release);
                struct WorkerFlagGuard {
                    infer_ctx* c;
                    ~WorkerFlagGuard() { release_gen(c); }
                } flag_guard{ctx};

                std::vector<infer_chat_msg> view(owned_msgs.size());
                for (size_t i = 0; i < owned_msgs.size(); ++i) {
                    view[i].role    = owned_msgs[i].role.c_str();
                    view[i].content = owned_msgs[i].content.c_str();
                }

                V1StreamThunk thunk{callback, user_data};
                try {
                    bool res = infer_generate_chat_internal(ctx, view.data(), view.size(), max_tokens, v1_stream_adapter, &thunk);
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

bool infer_generate_chat_turn_v2(
    infer_ctx* ctx,
    const char* role,
    const char* content,
    int32_t max_tokens,
    infer_stream_cb_v2 callback,
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
                hist_ref = std::move(backup);
            }
        }
    } hist_guard{ ctx->chat_history };

    try {
        const struct llama_vocab* vocab = llama_model_get_vocab(ctx->model);
        std::string safe_content = neutralize_special_fast(content, ctx->special_lookup);
        ctx->chat_history.push_back({ role, std::move(safe_content) });

        const uint32_t safety_margin = 128;
        if (ctx->n_ctx <= safety_margin) {
            ctx->set_error("Context size too small for chat turn safety margin.");
            return false;
        }
        const uint32_t max_prompt_budget = ctx->n_ctx - safety_margin;

        auto estimate_msg_tokens = [vocab](const OwnedMsg& m) -> uint32_t {
            int32_t cnt = 0;
            if (tokenize_checked(vocab, m.content.c_str(), m.content.size(), false, true, nullptr, &cnt) && cnt > 0) {
                return static_cast<uint32_t>(cnt) + 8u;
            }
            return static_cast<uint32_t>(m.content.size() / 3) + 8u;
        };

        auto find_prune_start = [&]() -> size_t {
            size_t idx = 0;
            while (idx + 1 < ctx->chat_history.size() &&
                   (ctx->chat_history[idx].role == "system" || ctx->chat_history[idx].role == "developer")) {
                idx++;
            }
            return idx;
        };

        auto find_next_turn_boundary = [&](size_t start_idx) -> size_t {
            const size_t limit = ctx->chat_history.size() - 1;
            if (start_idx >= limit) return start_idx;
            size_t end_idx = start_idx + 1;
            while (end_idx < limit && ctx->chat_history[end_idx].role != "user") {
                end_idx++;
            }
            return end_idx;
        };

        while (true) {
            std::vector<struct llama_chat_message> llama_msgs;
            llama_msgs.reserve(ctx->chat_history.size());
            for (const auto& m : ctx->chat_history) {
                llama_msgs.push_back({ m.role.c_str(), m.content.c_str() });
            }

            std::string formatted;
            if (!render_chat_template_with_fallback(ctx, llama_msgs, formatted)) {
                return false;
            }

            bool add_bos = want_bos(vocab, formatted.c_str());
            int32_t n_tokens = 0;
            if (!tokenize_checked(vocab, formatted.c_str(), formatted.size(), add_bos, true, nullptr, &n_tokens) ||
                n_tokens <= 0) {
                ctx->set_error("Tokenization calculation failed.");
                return false;
            }

            if (static_cast<uint32_t>(n_tokens) <= max_prompt_budget) {
                std::vector<llama_token> prompt_tokens;
                if (!tokenize_checked(vocab, formatted.c_str(), formatted.size(), add_bos, true, &prompt_tokens) ||
                    prompt_tokens.empty()) {
                    ctx->set_error("Failed to tokenize chat turn.");
                    return false;
                }

                const int prev_past = ctx->n_past.load(std::memory_order_relaxed);
                std::string generated_response;
                bool res = execute_decode(ctx, prompt_tokens, true, max_tokens, callback, user_data, &generated_response);

                if (res || ctx->n_past.load(std::memory_order_relaxed) > prev_past) {
                    hist_guard.armed = false;
                }
                if (res) {
                    std::string safe_response = neutralize_special_fast(generated_response.c_str(), ctx->special_lookup);
                    ctx->chat_history.push_back({ "assistant", std::move(safe_response) });
                }
                ctx->last_success.store(res, std::memory_order_release);
                return res;
            }

            const uint32_t excess_tokens = static_cast<uint32_t>(n_tokens) - max_prompt_budget;
            const size_t prune_start = find_prune_start();
            size_t prune_end = prune_start;
            uint32_t freed_estimate = 0;

            while (freed_estimate < excess_tokens) {
                size_t next_boundary = find_next_turn_boundary(prune_end);
                if (next_boundary == prune_end) {
                    break;
                }
                for (size_t i = prune_end; i < next_boundary; ++i) {
                    freed_estimate += estimate_msg_tokens(ctx->chat_history[i]);
                }
                prune_end = next_boundary;
            }

            if (prune_end > prune_start) {
                ctx->chat_history.erase(
                    ctx->chat_history.begin() + static_cast<std::ptrdiff_t>(prune_start),
                    ctx->chat_history.begin() + static_cast<std::ptrdiff_t>(prune_end)
                );
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

bool infer_generate_chat_turn(
    infer_ctx* ctx,
    const char* role,
    const char* content,
    int32_t max_tokens,
    infer_stream_cb callback,
    void* user_data
) {
    if (!callback) return false;
    V1StreamThunk thunk{callback, user_data};
    return infer_generate_chat_turn_v2(ctx, role, content, max_tokens, v1_stream_adapter, &thunk);
}

bool infer_generate_completion_v2(
    infer_ctx* ctx,
    const char* raw_prompt,
    int32_t max_tokens,
    infer_stream_cb_v2 callback,
    void* user_data
) {
    if (!ctx || !raw_prompt || !callback) return false;

    size_t prompt_len = std::strlen(raw_prompt);
    if (prompt_len > static_cast<size_t>(INT32_MAX)) {
        ctx->set_error("Raw prompt length exceeds maximum limit.");
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
        bool add_bos = want_bos(vocab, raw_prompt);
        std::vector<llama_token> prompt_tokens;
        if (!tokenize_checked(vocab, raw_prompt, prompt_len, add_bos, false, &prompt_tokens) ||
            prompt_tokens.empty()) {
            ctx->set_error("Prompt tokenization failed.");
            return false;
        }

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

bool infer_generate_completion(
    infer_ctx* ctx,
    const char* raw_prompt,
    int32_t max_tokens,
    infer_stream_cb callback,
    void* user_data
) {
    if (!callback) return false;
    V1StreamThunk thunk{callback, user_data};
    return infer_generate_completion_v2(ctx, raw_prompt, max_tokens, v1_stream_adapter, &thunk);
}

void infer_free(infer_ctx* ctx) {
    if (!ctx) return;

    ctx->abort_flag.store(true, std::memory_order_release);

    if (std::this_thread::get_id() == ctx->active_gen_thread_id.load(std::memory_order_acquire)) {
        ctx->set_error("Cannot call infer_free from inside an active generation callback.");
        return;
    }

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
        free_model_resources_locked(ctx);
    } catch (...) {}

    delete ctx;
    unref_llama_backend();
}

} // extern "C"