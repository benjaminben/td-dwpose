#include "pose_render_internal.hpp"

#include <algorithm>
#include <cmath>

namespace dwpose_td
{

// Defined in pose_render_tables_body.cpp -- declared here so the orchestrator
// at the bottom of this file can call it without a separate header. Keeping
// it private to the .cpp pair (no entry in pose_render_internal.hpp) is fine
// because both translation units share that header for the data shapes.
void build_body_tables(
    const PoseKp* kps, int num_persons, int stride,
    float sx, float sy, int W, int H, PoseRenderTables& t);

namespace
{

inline int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// Append edges + dots for ONE 21-keypoint hand. `hand_offset` is the start
// index inside the per-person 134-array. Mirrors util.py::draw_handpose's
// per-hand body (the outer loop over `all_hand_peaks` is the caller's).
void build_one_hand(const PoseKp* k, int hand_offset,
                    float sx, float sy, int W, int H, PoseRenderTables& t)
{
    const PoseKp* h = k + hand_offset;
    for(int e = 0; e < HAND_EDGE_COUNT; ++e)
    {
        const PoseKp& A = h[kHandEdges[e].a];
        const PoseKp& B = h[kHandEdges[e].b];
        if(A.score < CONF_GATE || B.score < CONF_GATE) continue;
        // util.py truncates float -> int via int(...) BEFORE the eps test:
        //   x1 = int(x1 * W); ... if x1 > eps and y1 > eps and ...
        // We mirror that exactly so the rasterizer sees the same integer
        // endpoints cv2.line would (eps=0.01 means int >= 1).
        const int ax = static_cast<int>(A.x * sx);
        const int ay = static_cast<int>(A.y * sy);
        const int bx = static_cast<int>(B.x * sx);
        const int by = static_cast<int>(B.y * sy);
        if(!(ax > 0 && ay > 0 && bx > 0 && by > 0)) continue;
        DeviceLine L;
        L.ax = ax; L.ay = ay; L.bx = bx; L.by = by; L.color_idx = e;
        const int pad = (HAND_LINE_W + 1) / 2 + 1;
        L.x0 = clampi(std::min(ax, bx) - pad, 0, W);
        L.y0 = clampi(std::min(ay, by) - pad, 0, H);
        L.x1 = clampi(std::max(ax, bx) + pad, 0, W);
        L.y1 = clampi(std::max(ay, by) + pad, 0, H);
        if(L.x1 <= L.x0 || L.y1 <= L.y0) continue;
        t.max_line_w = std::max(t.max_line_w, L.x1 - L.x0);
        t.max_line_h = std::max(t.max_line_h, L.y1 - L.y0);
        t.hand_lines.push_back(L);
    }
    for(int i = 0; i < HAND_KP_PER_HAND; ++i)
    {
        const PoseKp& K = h[i];
        if(K.score < CONF_GATE) continue;
        const int icx = static_cast<int>(K.x * sx);
        const int icy = static_cast<int>(K.y * sy);
        if(!(icx > 0 && icy > 0)) continue; // util.py:141 eps gate.
        DeviceDot D;
        D.cx = static_cast<float>(icx);
        D.cy = static_cast<float>(icy);
        // Kernel ignores color_idx (hand dots are fixed-color), but we
        // store the keypoint_idx here so the OrderedDraw bucketing path
        // can group dots of the same keypoint_idx into one launch.
        D.color_idx = i;
        const int pad = HAND_DOT_R + 1;
        D.x0 = clampi(icx - pad, 0, W);
        D.y0 = clampi(icy - pad, 0, H);
        D.x1 = clampi(icx + pad, 0, W);
        D.y1 = clampi(icy + pad, 0, H);
        if(D.x1 <= D.x0 || D.y1 <= D.y0) continue;
        t.max_hand_dot_w = std::max(t.max_hand_dot_w, D.x1 - D.x0);
        t.max_hand_dot_h = std::max(t.max_hand_dot_h, D.y1 - D.y0);
        t.hand_dots.push_back(D);
    }
}

// Mirrors util.py::draw_facepose. White dots, radius FACE_DOT_R.
void build_face(const PoseKp* k,
                float sx, float sy, int W, int H, PoseRenderTables& t)
{
    for(int i = 0; i < FACE_KP_COUNT; ++i)
    {
        const PoseKp& K = k[FACE_KP_BEGIN + i];
        if(K.score < CONF_GATE) continue;
        const int icx = static_cast<int>(K.x * sx);
        const int icy = static_cast<int>(K.y * sy);
        if(!(icx > 0 && icy > 0)) continue; // util.py:154 eps gate.
        DeviceDot D;
        D.cx = static_cast<float>(icx);
        D.cy = static_cast<float>(icy);
        D.color_idx = 0; // unused (face dots are fixed-color)
        const int pad = FACE_DOT_R + 1;
        D.x0 = clampi(icx - pad, 0, W);
        D.y0 = clampi(icy - pad, 0, H);
        D.x1 = clampi(icx + pad, 0, W);
        D.y1 = clampi(icy + pad, 0, H);
        if(D.x1 <= D.x0 || D.y1 <= D.y0) continue;
        t.max_face_dot_w = std::max(t.max_face_dot_w, D.x1 - D.x0);
        t.max_face_dot_h = std::max(t.max_face_dot_h, D.y1 - D.y0);
        t.face_dots.push_back(D);
    }
}

} // namespace

PoseRenderTables build_pose_tables(
    const PoseKp* kps, int num_persons, int stride,
    int src_w, int src_h, int W, int H)
{
    PoseRenderTables t;
    if(!kps || num_persons <= 0 || stride < 18) return t;

    // Coord remap matches the python pipeline:
    //   x_norm = x / src_w (in `_render_like_oracle`); px_dst = x_norm * W
    // i.e. simple linear scale src -> dst. With src_w==W (the typical TD
    // case where the renderer writes a same-size output) this collapses to
    // identity, but supporting arbitrary scales is one extra multiply per kp.
    const float sx = (src_w > 0) ? (static_cast<float>(W) / src_w) : 1.0f;
    const float sy = (src_h > 0) ? (static_cast<float>(H) / src_h) : 1.0f;

    t.limbs.reserve(num_persons * LIMB_COUNT);
    t.dots.reserve(num_persons * 18);
    if(stride >= 134)
    {
        t.hand_lines.reserve(num_persons * HAND_EDGE_COUNT * 2);
        t.hand_dots.reserve(num_persons * HAND_KP_PER_HAND * 2);
        t.face_dots.reserve(num_persons * FACE_KP_COUNT);
    }

    build_body_tables(kps, num_persons, stride, sx, sy, W, H, t);
    if(stride >= 134)
    {
        for(int p = 0; p < num_persons; ++p)
        {
            const PoseKp* k = kps + p * stride;
            build_one_hand(k, HAND_L_BEGIN, sx, sy, W, H, t);
            build_one_hand(k, HAND_R_BEGIN, sx, sy, W, H, t);
            build_face(k, sx, sy, W, H, t);
        }
    }
    return t;
}

} // namespace dwpose_td
