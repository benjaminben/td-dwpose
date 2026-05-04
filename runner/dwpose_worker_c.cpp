// C ABI exported from dwpose_worker.dll. Each function thinly wraps the
// internal C++ DWPoseRunner class (in dwpose_runner_impl.hpp). All
// allocations stay inside the worker DLL: the TOP-side shim never has to
// new/delete worker objects across the DLL boundary.

#define DWPOSE_WORKER_BUILDING 1
#include "dwpose_worker_c.h"

#include "dwpose_runner_impl.hpp"

#include <algorithm>
#include <cstring>
#include <new>

using dwpose_td::DWPoseRunner;
using dwpose_td::Status;
using dwpose_td::ChannelOrder;
using dwpose_td::Keypoint;

namespace
{
inline DWPoseRunner* as_runner(dwpose_runner_handle h)
{
    return reinterpret_cast<DWPoseRunner*>(h);
}

// Channel order: shim/C ABI uses the same numeric values (BGRA=0, RGBA=1)
// as the worker's enum class. We still cast through the named values so a
// future renumbering on either side surfaces as a compile error.
inline ChannelOrder to_internal(dwpose_channel_order_e order)
{
    switch(order)
    {
        case DWPOSE_CHANNEL_RGBA: return ChannelOrder::RGBA;
        case DWPOSE_CHANNEL_BGRA:
        default:                  return ChannelOrder::BGRA;
    }
}
} // namespace

extern "C" {

DWPOSE_API dwpose_runner_handle dwpose_runner_create(void)
{
    auto* r = new(std::nothrow) DWPoseRunner();
    return reinterpret_cast<dwpose_runner_handle>(r);
}

DWPOSE_API void dwpose_runner_destroy(dwpose_runner_handle h)
{
    delete as_runner(h);
}

DWPOSE_API void dwpose_runner_request_reload(
    dwpose_runner_handle h, const char* engines_dir)
{
    if(auto* r = as_runner(h))
        r->requestReload(engines_dir ? std::string(engines_dir) : std::string());
}

DWPOSE_API void dwpose_runner_run_frame(
    dwpose_runner_handle h,
    cudaArray_t in, int W, int H,
    dwpose_channel_order_e order,
    cudaStream_t stream)
{
    if(auto* r = as_runner(h))
        r->runFrame(in, W, H, to_internal(order), stream);
}

DWPOSE_API int dwpose_runner_status(dwpose_runner_handle h)
{
    if(auto* r = as_runner(h))
        return static_cast<int>(r->status());
    return static_cast<int>(Status::Error);
}

DWPOSE_API float dwpose_runner_progress(dwpose_runner_handle h)
{
    if(auto* r = as_runner(h))
        return r->progress();
    return 0.0f;
}

DWPOSE_API int dwpose_runner_num_persons(dwpose_runner_handle h)
{
    if(auto* r = as_runner(h))
        return r->numPersons();
    return 0;
}

DWPOSE_API float dwpose_runner_inference_ms(dwpose_runner_handle h)
{
    if(auto* r = as_runner(h))
        return r->inferenceMs();
    return 0.0f;
}

DWPOSE_API size_t dwpose_runner_error_message(
    dwpose_runner_handle h, char* buf, size_t buflen)
{
    auto* r = as_runner(h);
    if(!r)
    {
        if(buf && buflen > 0) buf[0] = '\0';
        return 0;
    }
    const std::string& msg = r->errorMessage();
    const size_t need = msg.size();
    if(buf && buflen > 0)
    {
        const size_t n = std::min(need, buflen - 1);
        std::memcpy(buf, msg.data(), n);
        buf[n] = '\0';
    }
    return need;
}

DWPOSE_API int dwpose_runner_keypoints(
    dwpose_runner_handle h, int person_idx,
    dwpose_keypoint_t* out, int max_count)
{
    auto* r = as_runner(h);
    if(!r || !out || max_count <= 0) return 0;
    const std::vector<Keypoint> kps = r->keypoints(person_idx);
    const int n = std::min(static_cast<int>(kps.size()), max_count);
    for(int i = 0; i < n; ++i)
    {
        out[i].x = kps[i].x;
        out[i].y = kps[i].y;
        out[i].score = kps[i].score;
    }
    return n;
}

DWPOSE_API void dwpose_runner_render_pose(
    dwpose_runner_handle h,
    cudaArray_t out, int W, int H,
    int src_w, int src_h,
    cudaStream_t stream,
    unsigned int flags)
{
    if(auto* r = as_runner(h))
        r->renderPose(out, W, H, src_w, src_h, stream, flags);
}

} // extern "C"
