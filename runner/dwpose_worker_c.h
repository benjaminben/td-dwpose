// C ABI for dwpose_worker.dll. The TOP-side shim consumes these symbols
// through a delay-loaded import lib so that nvinfer / nvonnxparser are
// only resolved AFTER the shim has called LoadLibraryExW on the worker
// DLL with LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32.
// That excludes TouchDesigner's bin folder from the search and ensures
// the staged pypi-tensorrt v240 nvinfer wins over TD's bundled v239.
//
// All allocation lives inside the worker DLL: the TOP never new/deletes
// across the boundary. Strings come back via buffer-out + length;
// keypoint arrays come back via array-out + count.
#ifndef DWPOSE_WORKER_C_H
#define DWPOSE_WORKER_C_H

#include <stdint.h>
#include <stddef.h>

#include <cuda_runtime.h>  // cudaArray_t, cudaStream_t -- opaque pointers

#ifdef DWPOSE_WORKER_BUILDING
  #define DWPOSE_API __declspec(dllexport)
#else
  #define DWPOSE_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct dwpose_runner_t* dwpose_runner_handle;

// Mirrors dwpose_td::Status in the original runner header.
// (Renumbered to 0..4 sequentially; values stay stable after this commit.)
typedef enum {
    DWPOSE_STATUS_IDLE = 0,
    DWPOSE_STATUS_DOWNLOADING_ONNX = 1,
    DWPOSE_STATUS_BUILDING_ENGINE = 2,
    DWPOSE_STATUS_READY = 3,
    DWPOSE_STATUS_ERROR = 4,
} dwpose_status_e;

// Mirrors dwpose_td::ChannelOrder in dwpose_pre.hpp. Values must match.
typedef enum {
    DWPOSE_CHANNEL_BGRA = 0,
    DWPOSE_CHANNEL_RGBA = 1,
} dwpose_channel_order_e;

// One keypoint as exported across the boundary.
typedef struct {
    float x;
    float y;
    float score;
} dwpose_keypoint_t;

// DWPose decode emits 134 keypoints per person (133 wholebody + neck).
// The TOP only consumes the first 18 today, but the worker may write all
// 134 if the caller's buffer is large enough.
#define DWPOSE_KEYPOINTS_PER_PERSON 134

DWPOSE_API dwpose_runner_handle dwpose_runner_create(void);
DWPOSE_API void                  dwpose_runner_destroy(dwpose_runner_handle h);

// Triggers a fresh re-check of the engines folder + reload if anything
// changed. Non-blocking; the worker prepares on its own thread.
DWPOSE_API void dwpose_runner_request_reload(
    dwpose_runner_handle h, const char* engines_dir);

// Per-frame entry point. No-op if status != READY.
DWPOSE_API void dwpose_runner_run_frame(
    dwpose_runner_handle h,
    cudaArray_t in, int W, int H,
    dwpose_channel_order_e order,
    cudaStream_t stream);

// Status / progress / metrics queries.
DWPOSE_API int   dwpose_runner_status(dwpose_runner_handle h);   // dwpose_status_e
DWPOSE_API float dwpose_runner_progress(dwpose_runner_handle h);
DWPOSE_API int   dwpose_runner_num_persons(dwpose_runner_handle h);
DWPOSE_API float dwpose_runner_inference_ms(dwpose_runner_handle h);

// Copies the latest error message into `buf`. Returns the number of bytes
// the worker WOULD write excluding the NUL terminator (so the caller can
// retry with a larger buffer). Pass buf=NULL, buflen=0 to query the size.
// `buf` is always NUL-terminated when buflen > 0.
DWPOSE_API size_t dwpose_runner_error_message(
    dwpose_runner_handle h, char* buf, size_t buflen);

// Snapshot of one person's keypoints. Writes up to `max_count` entries into
// `out`; returns the number actually written. Returns 0 if person_idx is
// out of range or if `out` is NULL.
DWPOSE_API int dwpose_runner_keypoints(
    dwpose_runner_handle h, int person_idx,
    dwpose_keypoint_t* out, int max_count);

// Render the OpenPose stick figure for the latest snapshot directly into
// `out` (RGBA8 cudaArray, W x H). Black background + colored ellipses
// for limbs + colored dots for keypoints, byte-precise match for
// controlnet_aux/dwpose/util.py::draw_bodypose() so the SD ControlNet
// OpenPose model treats this as equivalent to the python pipeline's output.
//
// `src_w` and `src_h` are the coord system the stored keypoints are in
// (the original input frame the worker last consumed); the renderer rescales
// to W x H. Pass src_w == W and src_h == H if no rescale is desired.
//
// No-op if status() != READY or if no persons have been decoded yet (canvas
// is still cleared to black, which is what the SD ControlNet expects when
// there's no subject -- "no pose" conditioning).
//
// `flags` is a bitmask of DWPOSE_RENDER_FLAG_* values. Use 0 for the
// fast/legacy single-batch dispatch. Set DWPOSE_RENDER_FLAG_ORDERED_DRAW to
// match controlnet_aux's draw_bodypose ordering (one kernel launch per
// limb_idx / keypoint_idx so cross-limb overlaps are deterministic and
// match the python reference). The flags arg is an unsigned int (not a
// bool) so future toggles can be folded in without another ABI break.
DWPOSE_API void dwpose_runner_render_pose(
    dwpose_runner_handle h,
    cudaArray_t out, int W, int H,
    int src_w, int src_h,
    cudaStream_t stream,
    unsigned int flags);

#define DWPOSE_RENDER_FLAG_ORDERED_DRAW (1u << 0)

#ifdef __cplusplus
} // extern "C"
#endif

#endif // DWPOSE_WORKER_C_H
