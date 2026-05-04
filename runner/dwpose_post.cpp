#include "dwpose_post.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace dwpose_td
{

namespace
{

constexpr int YOLOX_STRIDES[3] = {8, 16, 32};

struct Detection { float cx, cy, w, h, obj; int cls_idx; float cls_score; };

} // namespace

std::vector<Box> yolox_decode(
    const float* raw, int num_anchors, int input_h, int input_w,
    float letterbox_scale, float conf_thresh, float iou_thresh)
{
    // FPN grid expansion same as _yolox_postprocess. Walk the three strides
    // and pair each anchor with its grid coords + stride.
    std::vector<Box> picks;

    int offset = 0;
    int seen = 0;
    constexpr int kFeats = 85;
    for(int stride : YOLOX_STRIDES)
    {
        const int hsize = input_h / stride;
        const int wsize = input_w / stride;
        for(int gy = 0; gy < hsize; ++gy)
        for(int gx = 0; gx < wsize; ++gx, ++offset)
        {
            if(offset >= num_anchors) break;
            const float* row = raw + offset * kFeats;
            const float cx = (row[0] + gx) * stride;
            const float cy = (row[1] + gy) * stride;
            const float bw = std::exp(row[2]) * stride;
            const float bh = std::exp(row[3]) * stride;
            const float obj = row[4];
            // Class 0 = person. Find argmax across cls (80 COCO classes).
            int cls_idx = 0;
            float cls_top = row[5];
            for(int c = 1; c < kFeats - 5; ++c)
            {
                if(row[5 + c] > cls_top) { cls_top = row[5 + c]; cls_idx = c; }
            }
            const float combined = obj * cls_top;
            if(cls_idx != 0 || combined <= conf_thresh) continue;
            Box b;
            b.x1 = (cx - bw * 0.5f) / letterbox_scale;
            b.y1 = (cy - bh * 0.5f) / letterbox_scale;
            b.x2 = (cx + bw * 0.5f) / letterbox_scale;
            b.y2 = (cy + bh * 0.5f) / letterbox_scale;
            b.score = combined;
            picks.push_back(b);
            ++seen;
        }
    }

    // NMS: sort desc by score then sweep.
    std::vector<int> order(picks.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return picks[a].score > picks[b].score; });
    std::vector<bool> killed(picks.size(), false);
    std::vector<Box> kept;
    for(size_t i = 0; i < order.size(); ++i)
    {
        const int ai = order[i];
        if(killed[ai]) continue;
        const Box& A = picks[ai];
        const float aArea = std::max(0.0f, A.x2 - A.x1) * std::max(0.0f, A.y2 - A.y1);
        kept.push_back(A);
        for(size_t j = i + 1; j < order.size(); ++j)
        {
            const int bi = order[j];
            if(killed[bi]) continue;
            const Box& B = picks[bi];
            const float xx1 = std::max(A.x1, B.x1);
            const float yy1 = std::max(A.y1, B.y1);
            const float xx2 = std::min(A.x2, B.x2);
            const float yy2 = std::min(A.y2, B.y2);
            const float w = std::max(0.0f, xx2 - xx1);
            const float h = std::max(0.0f, yy2 - yy1);
            const float inter = w * h;
            const float bArea = std::max(0.0f, B.x2 - B.x1) * std::max(0.0f, B.y2 - B.y1);
            const float iou = inter / (aArea + bArea - inter + 1e-9f);
            if(iou > iou_thresh) killed[bi] = true;
        }
    }
    (void)seen;
    return kept;
}

CropParams build_crop(const Box& box, int target_w, int target_h)
{
    const float cx = (box.x1 + box.x2) * 0.5f;
    const float cy = (box.y1 + box.y2) * 0.5f;
    float bw = (box.x2 - box.x1) * 1.25f;
    float bh = (box.y2 - box.y1) * 1.25f;
    const float target_aspect = static_cast<float>(target_w) / target_h;
    if(bw / bh > target_aspect) bh = bw / target_aspect;
    else                        bw = bh * target_aspect;
    const float sx = bw / target_w;
    const float sy = bh / target_h;
    CropParams cp;
    cp.M[0] = 1.0f / sx;
    cp.M[1] = 0.0f;
    cp.M[2] = target_w * 0.5f - cx / sx;
    cp.M[3] = 0.0f;
    cp.M[4] = 1.0f / sy;
    cp.M[5] = target_h * 0.5f - cy / sy;
    cp.cx = cx; cp.cy = cy; cp.sx = sx; cp.sy = sy;
    return cp;
}

namespace
{

// SimCC argmax with score = max(simcc).
inline void argmax_row(const float* row, int len, float& argv, float& maxv)
{
    int best = 0; float bestv = row[0];
    for(int i = 1; i < len; ++i) if(row[i] > bestv) { bestv = row[i]; best = i; }
    argv = static_cast<float>(best);
    maxv = bestv;
}

} // namespace

Person decode_simcc_to_openpose(
    const float* simcc_x, int simcc_x_len,
    const float* simcc_y, int simcc_y_len,
    int num_kp, const CropParams& crop, int target_w, int target_h)
{
    // split_ratio=2.0 hardcoded -- DWPose's RTMPose-l SimCC head divides each
    // pixel into 2 logits along x and y, so /2.0 maps argmax bin -> pixel.
    constexpr float SPLIT = 2.0f;

    std::vector<float> kp_x(num_kp), kp_y(num_kp), sc_x(num_kp), sc_y(num_kp);
    for(int k = 0; k < num_kp; ++k)
    {
        argmax_row(simcc_x + k * simcc_x_len, simcc_x_len, kp_x[k], sc_x[k]);
        argmax_row(simcc_y + k * simcc_y_len, simcc_y_len, kp_y[k], sc_y[k]);
        kp_x[k] /= SPLIT;
        kp_y[k] /= SPLIT;
    }

    // Map back to original image coords: (px - tw/2)*sx + cx
    std::vector<float> raw_x(num_kp), raw_y(num_kp), raw_sc(num_kp);
    for(int k = 0; k < num_kp; ++k)
    {
        raw_x[k] = (kp_x[k] - target_w * 0.5f) * crop.sx + crop.cx;
        raw_y[k] = (kp_y[k] - target_h * 0.5f) * crop.sy + crop.cy;
        raw_sc[k] = std::min(sc_x[k], sc_y[k]);
    }

    // Insert neck (between shoulders 5 and 6) at index 17 -> 134-long arrays.
    std::vector<float> ex_x(num_kp + 1), ex_y(num_kp + 1), ex_sc(num_kp + 1);
    const float neck_x = (raw_x[5] + raw_x[6]) * 0.5f;
    const float neck_y = (raw_y[5] + raw_y[6]) * 0.5f;
    const float neck_sc = std::min(raw_sc[5], raw_sc[6]);
    for(int i = 0; i < 17; ++i) { ex_x[i] = raw_x[i]; ex_y[i] = raw_y[i]; ex_sc[i] = raw_sc[i]; }
    ex_x[17] = neck_x; ex_y[17] = neck_y; ex_sc[17] = neck_sc;
    for(int i = 17; i < num_kp; ++i)
    {
        ex_x[i + 1] = raw_x[i]; ex_y[i + 1] = raw_y[i]; ex_sc[i + 1] = raw_sc[i];
    }

    // mmpose -> openpose remap (15 joints get repositioned). All indices into
    // the 134-long arrays. Source: dwpose_onnx.py:208-212.
    constexpr int mmpose_idx[15]   = {17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3};
    constexpr int openpose_idx[15] = { 1, 2, 3,  4, 6, 7,  8,  9, 10, 12, 13, 14, 15, 16, 17};
    std::vector<float> rx = ex_x, ry = ex_y, rsc = ex_sc;
    for(int i = 0; i < 15; ++i)
    {
        ex_x[openpose_idx[i]]  = rx[mmpose_idx[i]];
        ex_y[openpose_idx[i]]  = ry[mmpose_idx[i]];
        ex_sc[openpose_idx[i]] = rsc[mmpose_idx[i]];
    }

    Person p{};
    const int N = static_cast<int>(ex_x.size());
    for(int i = 0; i < N && i < 134; ++i)
    {
        p.kp[i * 2 + 0] = ex_x[i];
        p.kp[i * 2 + 1] = ex_y[i];
        p.sc[i] = ex_sc[i];
    }
    return p;
}

} // namespace dwpose_td
