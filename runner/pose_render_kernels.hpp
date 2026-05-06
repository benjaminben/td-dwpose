#pragma once

// Cross-TU launchers for the hand + face render passes. The body kernels
// stay private to pose_render.cu (it owns the orchestrator), but the hand
// line / face dot passes live in their own .cu files to keep each file
// under the 200-LOC target. The orchestrator calls these wrappers; each
// wrapper hides its kernel + grid math.

#include "pose_render_internal.hpp"

#include <cuda_runtime.h>

namespace dwpose_td
{

// Multiply the entire canvas by 0.6 (matches util.py's darken pass).
// Single launch over the whole texture.
void launch_darken(
    cudaSurfaceObject_t surf, int W, int H, cudaStream_t stream);

// Body limb ellipse pass. Dispatches `n_limbs` limbs starting at
// `d_limbs` (already at the desired offset). The OrderedDraw caller
// passes a pointer + count for one color_idx bucket at a time.
void launch_body_limbs(
    cudaSurfaceObject_t surf, int W, int H,
    const DeviceLimb* d_limbs, int n_limbs,
    int max_w, int max_h, float marker_scale, cudaStream_t stream);

// Body keypoint dot pass. Same per-bucket pattern as launch_body_limbs.
void launch_body_dots(
    cudaSurfaceObject_t surf, int W, int H,
    const DeviceDot* d_dots, int n_dots,
    int max_w, int max_h, float marker_scale, cudaStream_t stream);

// Draw all `n_lines` hand finger-chain lines into `surf` over the existing
// (already-darkened) canvas. Color comes from the per-line color_idx into
// the kHandColors[20] table (defined inside pose_render_hand.cu). Lines are
// thickness-2 Bresenham (matches cv2.line(thickness=2, lineType=LINE_8)).
void launch_hand_lines(
    cudaSurfaceObject_t surf, int W, int H,
    const DeviceLine* d_lines, int n_lines,
    int max_w, int max_h, float marker_scale, cudaStream_t stream);

// Draw all `n_dots` hand keypoint dots in solid red (0,0,255 in cv2 BGR
// terms; (255,0,0)... wait the canvas is RGB, so the (0,0,255) cv2 BGR
// value writes 255 to the BLUE plane -- in RGB that is (0,0,255) too in
// the storage order R,G,B since cv2 stores BGR but we're writing the
// canvas in RGB-cudaArray order via surf2Dwrite). See the comment inside
// pose_render_hand.cu for the exact triple stored.
void launch_hand_dots(
    cudaSurfaceObject_t surf, int W, int H,
    const DeviceDot* d_dots, int n_dots,
    int max_w, int max_h, float marker_scale, cudaStream_t stream);

// Draw all `n_dots` face landmark dots in solid white (255,255,255). Radius
// is FACE_DOT_R (=3, smaller than the body dot radius of 4).
void launch_face_dots(
    cudaSurfaceObject_t surf, int W, int H,
    const DeviceDot* d_dots, int n_dots,
    int max_w, int max_h, float marker_scale, cudaStream_t stream);

// Per-color_idx bucketing helpers (host-side). Sort `in` by color_idx in
// place (stable sort -- preserves person-order within a bucket, matching
// controlnet_aux's per-person inner loop) and return offsets + counts for
// each non-empty bucket. `n_colors` is the upper bound on color_idx + 1
// (17 for body limbs, 18 for body dots, 20 for hand lines, 21 for hand
// dots when bucketed by keypoint_idx).
struct ColorBucket { int offset; int count; };
std::vector<ColorBucket> bucket_by_color_limbs(
    std::vector<DeviceLimb>& in, int n_colors);
std::vector<ColorBucket> bucket_by_color_dots(
    std::vector<DeviceDot>& in, int n_colors);
std::vector<ColorBucket> bucket_by_color_lines(
    std::vector<DeviceLine>& in, int n_colors);

} // namespace dwpose_td
