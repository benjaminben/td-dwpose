#pragma once

#include <functional>
#include <string>

namespace dwpose_td
{

// Resolve a HuggingFace file. Reuse priority:
//   1) <local_dir>/<filename>
//   2) <local_dir>/<filename>.onnx (no-op since filename includes ext)
//   3) HF cache: %USERPROFILE%/.cache/huggingface/hub/models--<repo>/snapshots/.../filename
//   4) network fetch via libcurl into <local_dir>/<filename>
// Returns the resolved on-disk path, or empty string on failure.
//
// progress_cb(downloaded_bytes, total_bytes) is invoked from within the curl
// write callback; total may be 0 if HEAD didn't supply Content-Length.
struct HFFile
{
    std::string repo_id;     // e.g. "yzd-v/DWPose"
    std::string revision;    // e.g. "main"
    std::string filename;    // e.g. "yolox_l.onnx"
};

using HFProgressCallback = std::function<void(uint64_t, uint64_t)>;

std::string hf_resolve_or_download(
    const HFFile& f, const std::string& local_dir,
    const HFProgressCallback& progress_cb,
    std::string* err_out);

// Search local + HF cache only. Returns "" if not found. No network.
std::string hf_find_cached(const HFFile& f, const std::string& local_dir);

} // namespace dwpose_td
