#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace dwpose_td
{

struct Box { float x1, y1, x2, y2, score; };

// Mirrors _yolox_postprocess + _nms in dwpose_onnx.py:
// raw [1, N, 85] FPN-grid YOLOX output -> person bboxes in original-image
// space (already divided by `letterbox_scale`).
std::vector<Box> yolox_decode(
    const float* raw, int num_anchors, int input_h, int input_w,
    float letterbox_scale, float conf_thresh, float iou_thresh);

// Mirrors _crop_warp in dwpose_onnx.py. Builds the 2x3 warp matrix M, and
// returns center (cx,cy) + scale (sx,sy) needed for SimCC inverse mapping.
// target_w x target_h are the pose model's input dims (288 x 384 for DWPose).
struct CropParams
{
    float M[6];
    float cx, cy, sx, sy;
};
CropParams build_crop(const Box& box, int target_w, int target_h);

// SimCC argmax decode + OpenPose remap. Operates on the raw pose head output
// for a single person and writes 134 keypoints + 134 scores (after neck
// insertion + mmpose<->openpose swap, matching dwpose_onnx.py).
struct Person
{
    std::array<float, 134 * 2> kp; // x0,y0,x1,y1,...
    std::array<float, 134> sc;
};

// simcc_x: [num_kp, kp_w * split_ratio]; simcc_y: [num_kp, kp_h * split_ratio]
// split_ratio fixed to 2.0 (DWPose default). Decodes 133 raw keypoints, then
// inserts the synthesized neck and applies the openpose remap to produce 134.
Person decode_simcc_to_openpose(
    const float* simcc_x, int simcc_x_len,
    const float* simcc_y, int simcc_y_len,
    int num_kp, const CropParams& crop, int target_w, int target_h);

// Renderer-side draw threshold (controlnet_aux gates keypoint rendering on
// score >= 0.3). Consumed by the stick-figure renderer.
constexpr float DRAW_CONF_THRESHOLD = 0.3f;

// OpenPose 18-body-point indices (subset of the 134 wholebody keypoints).
constexpr int OPENPOSE_BODY_COUNT = 18;

} // namespace dwpose_td
