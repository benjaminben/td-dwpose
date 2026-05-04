#include "hf_download.hpp"

#include "td_debug_log.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef TD_DWPOSE_HAS_CURL
#include <curl/curl.h>
#endif

namespace fs = std::filesystem;

namespace dwpose_td
{

namespace
{

std::string user_home()
{
#ifdef _WIN32
    const char* up = std::getenv("USERPROFILE");
    if(up && *up) return up;
    const char* hd = std::getenv("HOMEDRIVE");
    const char* hp = std::getenv("HOMEPATH");
    if(hd && hp) return std::string(hd) + hp;
    return "C:\\Users\\Default";
#else
    const char* h = std::getenv("HOME");
    return h ? std::string(h) : "/tmp";
#endif
}

// HF cache layout: <home>/.cache/huggingface/hub/models--<org>--<repo>/snapshots/<sha>/<filename>
std::string hf_cache_path(const HFFile& f)
{
    fs::path base = fs::path(user_home()) / ".cache" / "huggingface" / "hub";
    std::string repo_dir = "models--";
    for(char c : f.repo_id) repo_dir += (c == '/') ? '-' : c;
    if(repo_dir.find("--") == std::string::npos) repo_dir.append("--");
    fs::path repo_root = base / repo_dir / "snapshots";
    if(!fs::exists(repo_root)) return {};
    std::error_code ec;
    for(auto& sn : fs::directory_iterator(repo_root, ec))
    {
        if(!sn.is_directory()) continue;
        fs::path candidate = sn.path() / f.filename;
        if(fs::exists(candidate)) return candidate.string();
    }
    return {};
}

#ifdef TD_DWPOSE_HAS_CURL

struct CurlState
{
    std::ofstream* out;
    HFProgressCallback progress;
    uint64_t got = 0;
    uint64_t total = 0;
};

size_t write_cb(char* ptr, size_t size, size_t n, void* ud)
{
    auto* s = static_cast<CurlState*>(ud);
    const size_t bytes = size * n;
    s->out->write(ptr, bytes);
    s->got += bytes;
    if(s->progress) s->progress(s->got, s->total);
    return bytes;
}

int progress_cb(void* ud, curl_off_t dltotal, curl_off_t dlnow,
                curl_off_t, curl_off_t)
{
    auto* s = static_cast<CurlState*>(ud);
    if(dltotal > 0) s->total = static_cast<uint64_t>(dltotal);
    if(s->progress) s->progress(static_cast<uint64_t>(dlnow), s->total);
    return 0;
}

bool curl_download(const std::string& url, const std::string& dst,
                   const HFProgressCallback& cb, std::string* err_out)
{
    CURL* curl = curl_easy_init();
    if(!curl) { if(err_out) *err_out = "curl init failed"; return false; }

    fs::path tmp = dst + ".part";
    std::ofstream out(tmp.string(), std::ios::binary);
    if(!out.good())
    {
        curl_easy_cleanup(curl);
        if(err_out) *err_out = "could not open " + tmp.string() + " for write";
        return false;
    }
    CurlState st{&out, cb, 0, 0};

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &st);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_cb);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &st);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "td-dwpose/0.1");
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);

    CURLcode rc = curl_easy_perform(curl);
    out.close();
    curl_easy_cleanup(curl);

    if(rc != CURLE_OK)
    {
        std::error_code ec; fs::remove(tmp, ec);
        if(err_out) *err_out = std::string("curl: ") + curl_easy_strerror(rc);
        return false;
    }
    std::error_code ec;
    fs::rename(tmp, dst, ec);
    if(ec) { if(err_out) *err_out = "rename failed: " + ec.message(); return false; }
    return true;
}

#endif // TD_DWPOSE_HAS_CURL

} // namespace

std::string hf_find_cached(const HFFile& f, const std::string& local_dir)
{
    if(!local_dir.empty())
    {
        fs::path local = fs::path(local_dir) / f.filename;
        if(fs::exists(local)) return local.string();
    }
    return hf_cache_path(f);
}

std::string hf_resolve_or_download(
    const HFFile& f, const std::string& local_dir,
    const HFProgressCallback& progress_cb,
    std::string* err_out)
{
    std::string cached = hf_find_cached(f, local_dir);
    if(!cached.empty())
    {
        TDDBG("[hf] reused cached " << f.filename << " -> " << cached);
        return cached;
    }
#ifdef TD_DWPOSE_HAS_CURL
    if(local_dir.empty())
    {
        if(err_out) *err_out = "no local_dir set and no HF cache hit for " + f.filename;
        return {};
    }
    fs::path dst = fs::path(local_dir) / f.filename;
    fs::create_directories(dst.parent_path());
    // HF resolve URL: https://huggingface.co/<repo>/resolve/<rev>/<filename>
    std::string url = "https://huggingface.co/" + f.repo_id + "/resolve/"
                      + f.revision + "/" + f.filename;
    TDDBG("[hf] downloading " << url);
    if(!curl_download(url, dst.string(), progress_cb, err_out))
    {
        TDDBG("[hf] download failed: " << (err_out ? *err_out : std::string("?")));
        return {};
    }
    TDDBG("[hf] downloaded " << dst.string());
    return dst.string();
#else
    // Stub branch: no curl linked. Cache discovery above already ran.
    (void)progress_cb;
    if(err_out)
    {
        *err_out = "libcurl not compiled in; expected file '" + f.filename
                 + "' was not found in cache or engines folder. Either "
                   "place the file in your engines folder manually (download "
                   "from https://huggingface.co/yzd-v/DWPose), or rebuild "
                   "td-dwpose with libcurl support.";
    }
    TDDBG("[hf] stub: cannot download " << f.filename << " (built without libcurl)");
    return {};
#endif
}

} // namespace dwpose_td
