// Host-side dispatch orchestrator for pose_render. Splits out of
// pose_render.cu so that .cu file stays under the 200-LOC target now that
// the renderer supports two dispatch modes (legacy single-batch vs the
// per-color_idx OrderedDraw mode that matches controlnet_aux's
// draw_bodypose ordering).
//
// This TU is pure C++ (no kernels). It uploads the host-side draw tables
// to device memory once, then calls the kernel launchers in
// pose_render_kernels.hpp -- either as one big launch per category
// (legacy, racy on overlaps) or as one launch per color_idx bucket
// (ordered, ~100 launches per frame across body limbs + body dots +
// hand edges + hand dots).

#include "pose_render.hpp"
#include "pose_render_internal.hpp"
#include "pose_render_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

namespace dwpose_td
{

namespace
{

// Stable-sort `in` by color_idx in-place; returns one ColorBucket per
// non-empty color value in [0, n_colors). The stable sort preserves the
// original person-order within each bucket -- matches controlnet_aux's
// "for each person, then for each limb" inner loop after we transpose to
// "for each limb, then for each person".
template <typename T>
std::vector<ColorBucket> bucket_impl(std::vector<T>& in, int n_colors)
{
    std::vector<ColorBucket> out;
    if(in.empty() || n_colors <= 0) return out;
    std::stable_sort(in.begin(), in.end(),
        [](const T& a, const T& b){ return a.color_idx < b.color_idx; });
    out.reserve(static_cast<size_t>(n_colors));
    int cursor = 0;
    const int n = static_cast<int>(in.size());
    for(int c = 0; c < n_colors; ++c)
    {
        const int start = cursor;
        while(cursor < n && in[cursor].color_idx == c) ++cursor;
        const int count = cursor - start;
        if(count > 0) out.push_back({start, count});
    }
    return out;
}

template <typename T>
T* upload_table(const std::vector<T>& host, cudaStream_t stream)
{
    if(host.empty()) return nullptr;
    T* dev = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&dev), sizeof(T) * host.size());
    cudaMemcpyAsync(dev, host.data(), sizeof(T) * host.size(),
                    cudaMemcpyHostToDevice, stream);
    return dev;
}

} // namespace

std::vector<ColorBucket> bucket_by_color_limbs(
    std::vector<DeviceLimb>& in, int n_colors)
{
    return bucket_impl(in, n_colors);
}

std::vector<ColorBucket> bucket_by_color_dots(
    std::vector<DeviceDot>& in, int n_colors)
{
    return bucket_impl(in, n_colors);
}

std::vector<ColorBucket> bucket_by_color_lines(
    std::vector<DeviceLine>& in, int n_colors)
{
    return bucket_impl(in, n_colors);
}

void render_pose_dispatch(
    cudaSurfaceObject_t surf, int W, int H,
    PoseRenderTables& t, unsigned int flags, cudaStream_t stream)
{
    const bool ordered = (flags & RENDER_FLAG_ORDERED_DRAW) != 0;

    // Optional per-color_idx bucketing (only computed in ordered mode --
    // the sort itself is O(n log n) but n is tiny, and the upload is a
    // single device copy of the now-sorted table). The buckets index into
    // the device-side table at the same offsets as the host vector.
    std::vector<ColorBucket> b_limbs, b_dots, b_hlines, b_hdots;
    if(ordered)
    {
        b_limbs  = bucket_by_color_limbs(t.limbs, LIMB_COUNT);
        b_dots   = bucket_by_color_dots(t.dots,  18);
        b_hlines = bucket_by_color_lines(t.hand_lines, HAND_EDGE_COUNT);
        b_hdots  = bucket_by_color_dots(t.hand_dots,   HAND_KP_PER_HAND);
    }

    DeviceLimb* d_limbs  = upload_table(t.limbs,      stream);
    DeviceDot*  d_dots   = upload_table(t.dots,       stream);
    DeviceLine* d_hlines = upload_table(t.hand_lines, stream);
    DeviceDot*  d_hdots  = upload_table(t.hand_dots,  stream);
    DeviceDot*  d_fdots  = upload_table(t.face_dots,  stream);

    const float ms = t.marker_scale;

    // Pass 2: limbs.
    if(d_limbs)
    {
        if(ordered)
            for(const auto& b : b_limbs)
                launch_body_limbs(surf, W, H, d_limbs + b.offset, b.count,
                                  t.max_limb_w, t.max_limb_h, ms, stream);
        else
            launch_body_limbs(surf, W, H, d_limbs,
                              static_cast<int>(t.limbs.size()),
                              t.max_limb_w, t.max_limb_h, ms, stream);
    }

    // Pass 3: darken (matches `canvas = (canvas * 0.6).astype(np.uint8)`).
    launch_darken(surf, W, H, stream);

    // Pass 4: body keypoint dots.
    if(d_dots)
    {
        if(ordered)
            for(const auto& b : b_dots)
                launch_body_dots(surf, W, H, d_dots + b.offset, b.count,
                                 t.max_dot_w, t.max_dot_h, ms, stream);
        else
            launch_body_dots(surf, W, H, d_dots,
                             static_cast<int>(t.dots.size()),
                             t.max_dot_w, t.max_dot_h, ms, stream);
    }

    // Pass 5: hand finger-chain lines. Bucketing by edge_idx (color_idx
    // already stores the edge_idx) groups left+right hands of the same
    // edge into one launch -- they share the same kHandColors entry so
    // drawing them together is fine.
    if(d_hlines)
    {
        if(ordered)
            for(const auto& b : b_hlines)
                launch_hand_lines(surf, W, H, d_hlines + b.offset, b.count,
                                  t.max_line_w, t.max_line_h, ms, stream);
        else
            launch_hand_lines(surf, W, H, d_hlines,
                              static_cast<int>(t.hand_lines.size()),
                              t.max_line_w, t.max_line_h, ms, stream);
    }

    // Pass 6: hand keypoint dots. Bucketed by keypoint_idx (the table
    // builder writes keypoint_idx into color_idx for hand dots).
    if(d_hdots)
    {
        if(ordered)
            for(const auto& b : b_hdots)
                launch_hand_dots(surf, W, H, d_hdots + b.offset, b.count,
                                 t.max_hand_dot_w, t.max_hand_dot_h, ms, stream);
        else
            launch_hand_dots(surf, W, H, d_hdots,
                             static_cast<int>(t.hand_dots.size()),
                             t.max_hand_dot_w, t.max_hand_dot_h, ms, stream);
    }

    // Pass 7: face landmark dots -- skip bucketing per spec (uniform
    // white, no overlap concerns).
    if(d_fdots)
        launch_face_dots(surf, W, H, d_fdots,
                         static_cast<int>(t.face_dots.size()),
                         t.max_face_dot_w, t.max_face_dot_h, ms, stream);

    // Sync before freeing because cudaFree on plain (non-pool) memory
    // cannot be deferred onto the stream the way cudaFreeAsync can.
    // Future optimization: retain a small device-side scratch buffer on
    // the runner instance to avoid the per-frame alloc/free cost.
    if(d_limbs || d_dots || d_hlines || d_hdots || d_fdots)
        cudaStreamSynchronize(stream);
    if(d_limbs)  cudaFree(d_limbs);
    if(d_dots)   cudaFree(d_dots);
    if(d_hlines) cudaFree(d_hlines);
    if(d_hdots)  cudaFree(d_hdots);
    if(d_fdots)  cudaFree(d_fdots);
}

} // namespace dwpose_td
