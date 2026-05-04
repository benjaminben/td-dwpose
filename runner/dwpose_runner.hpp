// TOP-side shim header. Mirrors the public shape of the worker's internal
// DWPoseRunner class so that plugin/dwpose_top.cpp does not change.
//
// This header MUST NOT include any TRT / nvonnxparser / worker-internal
// headers. Only opaque CUDA types (cudaArray_t, cudaStream_t) are exposed,
// and those come from <cuda_runtime.h> which the TOP itself already
// links against (CUDA::cudart).
//
// At runtime the shim's constructor calls dwpose_td::ensure_libraries_loaded()
// which LoadLibraryEx's dwpose_worker.dll from the plugin folder with
// LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32. Subsequent
// delay-loaded calls into dwpose_worker.dll then resolve transitive deps
// (nvinfer_10.dll, nvonnxparser_10.dll, cudart64_13.dll, libcurl.dll) from
// the plugin folder, NOT from TouchDesigner's bin folder.

#pragma once

#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace dwpose_td
{

// Mirrors dwpose_worker_c.h::dwpose_status_e. Numeric values match the C ABI
// so static_cast<int> on either side stays correct.
enum class Status : int
{
    Idle = 0,
    DownloadingOnnx = 1,
    BuildingEngine = 2,
    Ready = 3,
    Error = 4,
};

// Mirrors dwpose_worker_c.h::dwpose_channel_order_e. Values must match.
enum class ChannelOrder : int
{
    BGRA = 0,
    RGBA = 1,
};

struct Keypoint { float x, y, score; };

class DWPoseRunner
{
public:
    DWPoseRunner();
    ~DWPoseRunner();

    DWPoseRunner(const DWPoseRunner&) = delete;
    DWPoseRunner& operator=(const DWPoseRunner&) = delete;

    void runFrame(cudaArray_t in, int W, int H, ChannelOrder order,
                  cudaStream_t stream);

    // Render the OpenPose stick figure for the latest snapshot into
    // `out` (RGBA8 cudaArray, W x H). `src_w`/`src_h` are the coord
    // system the keypoints were decoded in (the input frame size). For
    // same-size pass-through call with src_w=W, src_h=H. `flags` is a
    // bitmask of DWPOSE_RENDER_FLAG_* values from dwpose_worker_c.h
    // (bit 0 = ORDERED_DRAW). Pass 0 for the fast / racy single-batch
    // path; ORDERED_DRAW serializes per limb_idx / keypoint_idx so
    // cross-limb overlaps match controlnet_aux byte-for-byte.
    void renderPose(cudaArray_t out, int W, int H,
                    int src_w, int src_h, cudaStream_t stream,
                    unsigned int flags);

    void requestReload(const std::string& engines_dir);

    Status status() const;
    float progress() const;
    int numPersons() const;
    float inferenceMs() const;
    const std::string& errorMessage() const;

    std::vector<Keypoint> keypoints(int personIdx) const;

private:
    // Opaque worker handle (struct dwpose_runner_t*). Kept as void* so this
    // header has zero coupling to dwpose_worker_c.h.
    void* myHandle = nullptr;

    // Cached error string -- the C ABI exposes errors via buffer-out copy.
    // Refreshed on demand by errorMessage().
    mutable std::string myErrorCache;

    // True if ensure_libraries_loaded() succeeded for this instance. When
    // false the runner is permanently in Error state and every method is a
    // no-op. The error is surfaced through errorMessage() / status().
    bool myLoaded = false;
    std::string myLoadError;
};

// dwpose_top.cpp consumes this constant; was previously defined in
// dwpose_post.hpp (a worker-internal header). Mirrored here so the TOP
// does not need to include any worker-internal header.
constexpr int OPENPOSE_BODY_COUNT = 18;
constexpr float DRAW_CONF_THRESHOLD = 0.3f;

// Mirrors dwpose_worker_c.h::DWPOSE_RENDER_FLAG_ORDERED_DRAW. Numeric
// value MUST match -- the shim passes this through to the C ABI as-is.
constexpr unsigned int RENDER_FLAG_ORDERED_DRAW = (1u << 0);

} // namespace dwpose_td
