#pragma once

// Shared between pose_render*.cu (kernels) and pose_render_tables.cpp (host-
// side table builder). Not part of the public renderer ABI.

#include "pose_render.hpp"

#include <vector>

namespace dwpose_td
{

// =============================================================================
// 134-keypoint layout (matches the worker's decoder in dwpose_post.cpp +
// the controlnet_aux _render_like_oracle slicing):
//   [ 0, 18) body  (OpenPose 18-point order; index 17 = neck)
//   [18, 24) feet  (six points; controlnet_aux's draw_pose does NOT draw
//                  these so the renderer skips them too)
//   [24, 92) face  (68 face landmarks; drawn as 3-px white dots)
//   [92,113) left hand  (21 keypoints; drawn as colored finger lines + red
//                       dots)
//   [113,134) right hand (21 keypoints; same treatment as left)
// =============================================================================

// ---- Body limbs (mirrors controlnet_aux/dwpose/util.py::draw_bodypose).
// limbSeq there is 1-indexed into an 18-keypoint array; here stored 0-indexed.
// Only the FIRST 17 entries are drawn as limbs (`for i in range(17)`); the
// python's last two (ear-shoulder cross-links at indices 17, 18) are
// intentionally skipped to match the upstream visual.
constexpr int LIMB_COUNT = 17;
struct LimbPair { int a, b; };
constexpr LimbPair kLimbs[LIMB_COUNT] = {
    {1, 2}, {1, 5}, {2, 3}, {3, 4}, {5, 6}, {6, 7}, {1, 8}, {8, 9},
    {9,10}, {1,11}, {11,12}, {12,13}, {1, 0}, {0,14}, {14,16},
    {0,15}, {15,17},
};

// ---- Hand finger chains (mirrors draw_handpose). 20 edges per hand: thumb
// (0-1-2-3-4), index (0-5-6-7-8), middle (0-9-10-11-12), ring (0-13-14-15-16),
// pinky (0-17-18-19-20). Same edge order as util.py:116-117 -- color index
// follows the loop variable `ie` which is the array position below.
constexpr int HAND_EDGE_COUNT = 20;
constexpr LimbPair kHandEdges[HAND_EDGE_COUNT] = {
    {0, 1}, {1, 2}, {2, 3}, {3, 4},
    {0, 5}, {5, 6}, {6, 7}, {7, 8},
    {0, 9}, {9,10}, {10,11}, {11,12},
    {0,13}, {13,14}, {14,15}, {15,16},
    {0,17}, {17,18}, {18,19}, {19,20},
};
constexpr int HAND_KP_PER_HAND = 21;
constexpr int FACE_KP_COUNT    = 68;

// 134-keypoint slice indices (see layout comment above).
constexpr int FACE_KP_BEGIN  = 24;
constexpr int HAND_L_BEGIN   = 92;
constexpr int HAND_R_BEGIN   = 113;

// 18 RGB colors, shared between body limbs (i-th limb uses kPoseColors[i]) and
// keypoint dots (i-th keypoint uses kPoseColors[i]). Defined in pose_render.cu
// as `__device__ const` so the kernels can read it from device global memory;
// the host-side table builder only ever stores color INDICES (DeviceLimb /
// DeviceDot::color_idx), never the RGB triples themselves, so no host-side
// copy is needed.
struct Color3 { unsigned char r, g, b; };

// Base marker sizes -- util.py hardcodes these at the 512-px training canvas
// size. The renderer multiplies each by `PoseRenderTables::marker_scale` per
// frame so HD inputs don't end up with sub-pixel markers after downstream
// resize-to-SD-target. marker_scale = 1.0 reproduces the original constants.
constexpr int STICK_W = 4;       // body limb minor axis
constexpr int DOT_R = 4;         // body keypoint radius
constexpr int HAND_LINE_W = 2;   // hand finger-line thickness (util.py:134)
constexpr int HAND_DOT_R = 4;    // hand keypoint radius (util.py:142)
constexpr int FACE_DOT_R = 3;    // face landmark radius (util.py:155)
// Renderer-side draw threshold (matches dwpose_post.hpp::DRAW_CONF_THRESHOLD).
// Duplicated here so this header has zero dependency on the worker-internal
// dwpose_post.hpp -- the .cu file should not transitively pull TRT headers.
constexpr float CONF_GATE = 0.3f;

struct DeviceLimb
{
    float mx, my;       // midpoint in dst pixels
    float ux, uy;       // unit vector along limb axis
    float half_len;     // major axis radius (length/2 in dst pixels)
    int x0, y0, x1, y1; // bounding box (inclusive-exclusive)
    int color_idx;
};
struct DeviceDot
{
    float cx, cy;
    int x0, y0, x1, y1;
    int color_idx;      // for body dots: index into kPoseColors. For hand
                        // dots / face dots: ignored (kernel uses fixed color)
};
// 1-px integer line endpoints + bbox + color index into kHandColors.
// Mirrors cv2.line(..., thickness=HAND_LINE_W, lineType=LINE_8) which is
// Bresenham + a perpendicular thickness expansion.
struct DeviceLine
{
    int ax, ay, bx, by; // integer endpoints (cv2 truncates float -> int)
    int x0, y0, x1, y1; // bounding box (inclusive-exclusive)
    int color_idx;
};

struct PoseRenderTables
{
    std::vector<DeviceLimb> limbs;       // body limb ellipses
    std::vector<DeviceDot>  dots;        // body keypoint dots
    std::vector<DeviceLine> hand_lines;  // hand finger-chain lines
    std::vector<DeviceDot>  hand_dots;   // hand keypoint dots (red, r=4)
    std::vector<DeviceDot>  face_dots;   // face landmark dots (white, r=3)
    int max_limb_w = 1, max_limb_h = 1;
    int max_dot_w = 1, max_dot_h = 1;
    int max_line_w = 1, max_line_h = 1;
    int max_hand_dot_w = 1, max_hand_dot_h = 1;
    int max_face_dot_w = 1, max_face_dot_h = 1;
    // Per-frame marker size multiplier -- applied uniformly to STICK_W,
    // DOT_R, HAND_LINE_W, HAND_DOT_R, FACE_DOT_R. Stored here so the bbox
    // padding (host) and the inside-test (device) read the same scalar.
    float marker_scale = 1.0f;
};

// Walks `num_persons * stride` keypoints and emits all device-side draw tables
// for body, hands, and face. Coord remap (src_w x src_h -> W x H) and the
// CONF_GATE filter are applied per element. Pure host-side; no CUDA calls.
PoseRenderTables build_pose_tables(
    const PoseKp* kps_host, int num_persons, int stride,
    int src_w, int src_h, int W, int H, float marker_scale);

} // namespace dwpose_td
