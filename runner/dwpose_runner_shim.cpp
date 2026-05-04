// TOP-side shim implementation of dwpose_td::DWPoseRunner. Each public
// method delegates to the matching dwpose_runner_*() C ABI function in
// dwpose_worker.dll. Worker symbols are resolved via /DELAYLOAD so the
// process does NOT touch nvinfer_10.dll until the constructor's
// ensure_libraries_loaded() call has pinned dwpose_worker.dll's search
// to our plugin folder + System32.
//
// The shim must NEVER new/delete worker objects directly: the worker owns
// all heap allocations. Strings cross the boundary via buffer-out copies;
// keypoint arrays via array-out + count.

#include "dwpose_runner.hpp"

#include "dll_loader.hpp"
#include "dwpose_worker_c.h"
#include "td_debug_log.hpp"

#include <vector>

namespace dwpose_td
{

namespace
{
// Channel order values are kept aligned by hand on both sides; convert
// through the named values so a future renumbering surfaces as a build
// error rather than a silent BGRA<->RGBA swap.
inline dwpose_channel_order_e to_c_abi(ChannelOrder order)
{
    return (order == ChannelOrder::RGBA) ? DWPOSE_CHANNEL_RGBA
                                         : DWPOSE_CHANNEL_BGRA;
}

inline dwpose_runner_handle as_handle(void* p)
{
    return reinterpret_cast<dwpose_runner_handle>(p);
}
} // namespace

DWPoseRunner::DWPoseRunner()
{
    // Pre-load dwpose_worker.dll from the plugin folder BEFORE any
    // delay-loaded symbol is referenced. If this fails we never call into
    // the C ABI -- every public method becomes a no-op and surfaces the
    // load error through errorMessage() / status().
    if(!ensure_libraries_loaded(&myLoadError))
    {
        TDDBG("DWPoseRunner: dwpose_worker.dll load failed: " << myLoadError);
        return;
    }

    // First call into the worker DLL -- this is where delay-loaded import
    // resolution actually happens. ensure_libraries_loaded() above already
    // pinned dwpose_worker.dll's transitive search to the plugin folder
    // + System32, so by the time the loader resolves nvinfer / nvonnxparser
    // / cudart they should come from our staged copies, not TD's bin.
    // If that's wrong the process will crash here with a delay-load error;
    // check DbgView for the offending DLL name.
    myHandle = dwpose_runner_create();
    if(!myHandle)
    {
        myLoadError = "dwpose_runner_create returned null";
        TDDBG("!! " << myLoadError);
        return;
    }

    myLoaded = true;
}

DWPoseRunner::~DWPoseRunner()
{
    if(myHandle)
        dwpose_runner_destroy(as_handle(myHandle));
}

void DWPoseRunner::runFrame(cudaArray_t in, int W, int H, ChannelOrder order,
                            cudaStream_t stream)
{
    if(!myLoaded || !myHandle) return;
    dwpose_runner_run_frame(as_handle(myHandle), in, W, H, to_c_abi(order), stream);
}

void DWPoseRunner::renderPose(cudaArray_t out, int W, int H,
                              int src_w, int src_h, cudaStream_t stream,
                              unsigned int flags)
{
    if(!myLoaded || !myHandle) return;
    dwpose_runner_render_pose(
        as_handle(myHandle), out, W, H, src_w, src_h, stream, flags);
}

void DWPoseRunner::requestReload(const std::string& engines_dir)
{
    if(!myLoaded || !myHandle) return;
    dwpose_runner_request_reload(as_handle(myHandle), engines_dir.c_str());
}

Status DWPoseRunner::status() const
{
    if(!myLoaded || !myHandle) return Status::Error;
    return static_cast<Status>(dwpose_runner_status(as_handle(myHandle)));
}

float DWPoseRunner::progress() const
{
    if(!myLoaded || !myHandle) return 0.0f;
    return dwpose_runner_progress(as_handle(myHandle));
}

int DWPoseRunner::numPersons() const
{
    if(!myLoaded || !myHandle) return 0;
    return dwpose_runner_num_persons(as_handle(myHandle));
}

float DWPoseRunner::inferenceMs() const
{
    if(!myLoaded || !myHandle) return 0.0f;
    return dwpose_runner_inference_ms(as_handle(myHandle));
}

const std::string& DWPoseRunner::errorMessage() const
{
    // If the worker never loaded, return the load error verbatim.
    if(!myLoaded || !myHandle)
    {
        myErrorCache = myLoadError;
        return myErrorCache;
    }

    // Otherwise pull a fresh copy across the boundary. Two-call pattern:
    // first to learn the size, then resize +1 for the worker's NUL terminator
    // and fill, then trim the NUL back off so std::string size() == need.
    const size_t need = dwpose_runner_error_message(as_handle(myHandle), nullptr, 0);
    if(need == 0)
    {
        myErrorCache.clear();
        return myErrorCache;
    }
    myErrorCache.resize(need + 1);
    dwpose_runner_error_message(
        as_handle(myHandle), myErrorCache.data(), myErrorCache.size());
    myErrorCache.resize(need);
    return myErrorCache;
}

std::vector<Keypoint> DWPoseRunner::keypoints(int personIdx) const
{
    if(!myLoaded || !myHandle) return {};
    std::vector<dwpose_keypoint_t> raw(DWPOSE_KEYPOINTS_PER_PERSON);
    const int n = dwpose_runner_keypoints(
        as_handle(myHandle), personIdx, raw.data(),
        static_cast<int>(raw.size()));
    if(n <= 0) return {};
    std::vector<Keypoint> out(static_cast<size_t>(n));
    for(int i = 0; i < n; ++i)
        out[i] = {raw[i].x, raw[i].y, raw[i].score};
    return out;
}

} // namespace dwpose_td
