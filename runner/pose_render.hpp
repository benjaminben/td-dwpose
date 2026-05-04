#pragma once

// CUDA stick-figure renderer for the DWPose worker. Mirrors controlnet_aux's
// full draw_pose pipeline (draw_bodypose + draw_handpose + draw_facepose) at
// byte-precise color level so the SD ControlNet OpenPose model treats the
// C++ output as equivalent to the python pipeline's output.
//
// Drawing order (matches controlnet_aux's draw_pose exactly):
//   1. fill canvas black
//   2. for each person, for each of the first 17 body limbs: rasterize an
//      oriented ellipse polygon (length/2 major, stickwidth=4 minor) in
//      the limb's color
//   3. multiply the entire canvas by 0.6 (darkens limbs)
//   4. for each person, for each of 18 body keypoints: rasterize a filled
//      circle of radius 4 in the keypoint's color
//   5. for each hand (left + right per person), for each of 20 finger
//      edges: rasterize a thickness-2 line in the edge's HSV-cycle color
//   6. for each hand, for each of 21 keypoints: rasterize a filled circle
//      of radius 4 in solid (0,0,255) (matches util.py:142)
//   7. for each person, for each of 68 face landmarks: rasterize a filled
//      circle of radius 3 in solid white (matches util.py:155)
//
// Confidence gating: skip a limb / line if EITHER endpoint has score below
// DRAW_CONF_THRESHOLD; skip a keypoint circle on the same condition.
//
// Performance notes:
//   - Per-pixel atomic ops avoided: each limb / circle has its own kernel
//     launch over its 2D bounding box; overlap regions race but always
//     write the same color (no per-person color shift).
//   - Pre-fill is a single launch over the whole texture.
//   - The 0.6 multiply is a single launch over the whole texture, between
//     the limb passes and the keypoint passes.

#include <cuda_runtime.h>

namespace dwpose_td
{

struct PoseKp
{
    float x;      // pixel coords in the destination texture frame
    float y;
    float score;  // [0, 1]; below DRAW_CONF_THRESHOLD => skip
};

// Render `num_persons` skeletons into `out` (RGBA8 cudaArray, W x H).
// `kps` is a host-side flat array of length num_persons * stride; only
// indices [0, 18) per person are read. `stride` is the per-person stride
// (typically 134 for the worker's wholebody output).
//
// `src_w`/`src_h` are the source coord system the keypoints were decoded
// in; the renderer maps them into [0, W) x [0, H) via simple scaling
// (matches the python which multiplies normalized coords by W/H -- equivalent
// when the kp coords are in source-pixel space and we resize to dst).
//
// `flags` is a bitmask. Bit 0 (RENDER_FLAG_ORDERED_DRAW, mirrors
// DWPOSE_RENDER_FLAG_ORDERED_DRAW from the C ABI) selects the per-color_idx
// serialized dispatch that matches controlnet_aux's draw_bodypose ordering.
// Bit 0 = 0 keeps the legacy single-batch dispatch (fast, but races on
// cross-limb overlaps).
constexpr unsigned int RENDER_FLAG_ORDERED_DRAW = (1u << 0);

void render_pose(
    cudaArray_t out, int W, int H,
    const PoseKp* kps_host, int num_persons, int stride,
    int src_w, int src_h, cudaStream_t stream,
    unsigned int flags);

} // namespace dwpose_td
