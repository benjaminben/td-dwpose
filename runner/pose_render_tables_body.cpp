#include "pose_render_internal.hpp"

#include <algorithm>
#include <cmath>

// Host-side build of the body limb + body keypoint dot tables. Split out from
// pose_render_tables.cpp to keep both files under the 200-LOC target now that
// the hand and face passes also live in tables. Internal-linkage symbol --
// only the orchestrator in pose_render_tables.cpp calls it.

namespace dwpose_td
{

namespace
{

inline int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

} // namespace

void build_body_tables(
    const PoseKp* kps, int num_persons, int stride,
    float sx, float sy, int W, int H, PoseRenderTables& t)
{
    const float stick_w_eff = static_cast<float>(STICK_W) * t.marker_scale;
    const int   dot_r_pad   = static_cast<int>(std::ceil(DOT_R * t.marker_scale)) + 1;
    for(int p = 0; p < num_persons; ++p)
    {
        const PoseKp* k = kps + p * stride;
        for(int i = 0; i < LIMB_COUNT; ++i)
        {
            const PoseKp& A = k[kLimbs[i].a];
            const PoseKp& B = k[kLimbs[i].b];
            if(A.score < CONF_GATE || B.score < CONF_GATE) continue;
            const float ax = A.x * sx, ay = A.y * sy;
            const float bx = B.x * sx, by = B.y * sy;
            const float mx = 0.5f * (ax + bx);
            const float my = 0.5f * (ay + by);
            const float dx = ax - bx;
            const float dy = ay - by;
            const float len = std::sqrt(dx * dx + dy * dy);
            if(len < 1.0f) continue;
            const float half_len = 0.5f * len;
            const float ux = dx / len;
            const float uy = dy / len;
            // Bounding box of the rotated ellipse. Pad by 1 px to guarantee
            // every interior pixel falls inside the dispatched grid.
            const float bx_ext = std::fabs(half_len * ux) +
                                 std::fabs(stick_w_eff * uy) + 1.0f;
            const float by_ext = std::fabs(half_len * uy) +
                                 std::fabs(stick_w_eff * ux) + 1.0f;
            DeviceLimb L;
            L.mx = mx; L.my = my; L.ux = ux; L.uy = uy;
            L.half_len = half_len; L.color_idx = i;
            L.x0 = clampi(static_cast<int>(std::floor(mx - bx_ext)), 0, W);
            L.y0 = clampi(static_cast<int>(std::floor(my - by_ext)), 0, H);
            L.x1 = clampi(static_cast<int>(std::ceil (mx + bx_ext)), 0, W);
            L.y1 = clampi(static_cast<int>(std::ceil (my + by_ext)), 0, H);
            if(L.x1 <= L.x0 || L.y1 <= L.y0) continue;
            t.max_limb_w = std::max(t.max_limb_w, L.x1 - L.x0);
            t.max_limb_h = std::max(t.max_limb_h, L.y1 - L.y0);
            t.limbs.push_back(L);
        }
        for(int i = 0; i < 18; ++i)
        {
            const PoseKp& K = k[i];
            if(K.score < CONF_GATE) continue;
            const float cx = K.x * sx, cy = K.y * sy;
            DeviceDot D;
            D.cx = cx; D.cy = cy; D.color_idx = i;
            D.x0 = clampi(static_cast<int>(std::floor(cx)) - dot_r_pad, 0, W);
            D.y0 = clampi(static_cast<int>(std::floor(cy)) - dot_r_pad, 0, H);
            D.x1 = clampi(static_cast<int>(std::ceil (cx)) + dot_r_pad, 0, W);
            D.y1 = clampi(static_cast<int>(std::ceil (cy)) + dot_r_pad, 0, H);
            if(D.x1 <= D.x0 || D.y1 <= D.y0) continue;
            t.max_dot_w = std::max(t.max_dot_w, D.x1 - D.x0);
            t.max_dot_h = std::max(t.max_dot_h, D.y1 - D.y0);
            t.dots.push_back(D);
        }
    }
}

} // namespace dwpose_td
