#include "pose_render_kernels.hpp"

#include <cuda_runtime.h>
#include <surface_types.h>
#include <surface_indirect_functions.h>

namespace dwpose_td
{

// 20 RGB triples lifted from controlnet_aux util.py:134:
//   matplotlib.colors.hsv_to_rgb([ie / 20.0, 1.0, 1.0]) * 255
// passed through cv2.line on a uint8 canvas, which applies cv::saturate_cast
// (rounding, not truncation). Recomputed offline; stored here so the kernel
// does not need a runtime HSV->RGB.
__device__ const Color3 kHandColors[20] = {
    {255,   0,   0}, // ie= 0  hue 0.000
    {255,  77,   0}, // ie= 1  hue 0.050
    {255, 153,   0}, // ie= 2  hue 0.100
    {255, 229,   0}, // ie= 3  hue 0.150
    {204, 255,   0}, // ie= 4  hue 0.200
    {128, 255,   0}, // ie= 5  hue 0.250
    { 51, 255,   0}, // ie= 6  hue 0.300
    {  0, 255,  25}, // ie= 7  hue 0.350
    {  0, 255, 102}, // ie= 8  hue 0.400
    {  0, 255, 179}, // ie= 9  hue 0.450
    {  0, 255, 255}, // ie=10  hue 0.500
    {  0, 178, 255}, // ie=11  hue 0.550
    {  0, 102, 255}, // ie=12  hue 0.600
    {  0,  25, 255}, // ie=13  hue 0.650
    { 51,   0, 255}, // ie=14  hue 0.700
    {128,   0, 255}, // ie=15  hue 0.750
    {204,   0, 255}, // ie=16  hue 0.800
    {255,   0, 230}, // ie=17  hue 0.850
    {255,   0, 153}, // ie=18  hue 0.900
    {255,   0,  77}, // ie=19  hue 0.950
};

namespace
{

// Distance from point (px,py) to the segment ((ax,ay),(bx,by)). Used to
// rasterize cv2.line with thickness > 1: cv2's thick line is the union of
// pixels whose center lies within thickness/2 of the line (orthogonal
// distance), clipped to the segment's two endpoints.
__device__ inline float segment_dist2(
    float px, float py, float ax, float ay, float bx, float by)
{
    const float vx = bx - ax;
    const float vy = by - ay;
    const float wx = px - ax;
    const float wy = py - ay;
    const float c1 = vx * wx + vy * wy;
    if(c1 <= 0.0f) return wx * wx + wy * wy;
    const float c2 = vx * vx + vy * vy;
    if(c2 <= c1)
    {
        const float dx = px - bx, dy = py - by;
        return dx * dx + dy * dy;
    }
    const float t = c1 / c2;
    const float qx = ax + t * vx;
    const float qy = ay + t * vy;
    const float dx = px - qx, dy = py - qy;
    return dx * dx + dy * dy;
}

__global__ void draw_hand_lines_kernel(
    cudaSurfaceObject_t surf, int W, int H,
    const DeviceLine* lines, int n_lines, float hand_line_w_eff)
{
    const int li = blockIdx.z;
    if(li >= n_lines) return;
    const DeviceLine L = lines[li];
    const int px = L.x0 + blockIdx.x * blockDim.x + threadIdx.x;
    const int py = L.y0 + blockIdx.y * blockDim.y + threadIdx.y;
    if(px >= L.x1 || py >= L.y1) return;
    if(px < 0 || py < 0 || px >= W || py >= H) return;
    // cv2.line(thickness=t) on a uint8 canvas paints pixel centers within
    // distance t/2 of the segment. hand_line_w_eff = HAND_LINE_W * marker_scale.
    const float r = hand_line_w_eff * 0.5f;
    const float d2 = segment_dist2(
        px + 0.5f, py + 0.5f,
        static_cast<float>(L.ax), static_cast<float>(L.ay),
        static_cast<float>(L.bx), static_cast<float>(L.by));
    if(d2 > r * r) return;
    const Color3 c = kHandColors[L.color_idx];
    uchar4 v{c.r, c.g, c.b, 255};
    // Y-flip to TD's bottom-left output convention (see pose_render.cu).
    surf2Dwrite(v, surf, px * 4, (H - 1) - py);
}

__global__ void draw_hand_dots_kernel(
    cudaSurfaceObject_t surf, int W, int H,
    const DeviceDot* dots, int n_dots, float hand_dot_r_eff)
{
    const int di = blockIdx.z;
    if(di >= n_dots) return;
    const DeviceDot D = dots[di];
    const int px = D.x0 + blockIdx.x * blockDim.x + threadIdx.x;
    const int py = D.y0 + blockIdx.y * blockDim.y + threadIdx.y;
    if(px >= D.x1 || py >= D.y1) return;
    if(px < 0 || py < 0 || px >= W || py >= H) return;
    const float dx = (px + 0.5f) - D.cx;
    const float dy = (py + 0.5f) - D.cy;
    const float r = hand_dot_r_eff > 0.5f ? hand_dot_r_eff : 0.5f;
    if(dx * dx + dy * dy > r * r) return;
    // util.py:142: cv2.circle(canvas, ..., (0, 0, 255), thickness=-1).
    // controlnet_aux passes the (B,G,R)-shaped tuple to cv2 but the rest of
    // its pipeline treats the canvas channels as straight RGB (this is the
    // same convention the body palette uses -- (255, 0, 0) is stored at
    // numpy index 0 and read back as RED by the SD ControlNet). So a literal
    // (0, 0, 255) in the source means BLUE at numpy index 2, which we mirror
    // by writing the same triple into our RGB cudaArray.
    uchar4 v{0, 0, 255, 255};
    // Y-flip to TD's bottom-left output convention (see pose_render.cu).
    surf2Dwrite(v, surf, px * 4, (H - 1) - py);
}

} // namespace

void launch_hand_lines(
    cudaSurfaceObject_t surf, int W, int H,
    const DeviceLine* d_lines, int n_lines,
    int max_w, int max_h, float marker_scale, cudaStream_t stream)
{
    if(n_lines <= 0) return;
    dim3 block(16, 16);
    dim3 grid((max_w + 15) / 16, (max_h + 15) / 16,
              static_cast<unsigned>(n_lines));
    const float hand_line_w_eff = static_cast<float>(HAND_LINE_W) * marker_scale;
    draw_hand_lines_kernel<<<grid, block, 0, stream>>>(surf, W, H, d_lines, n_lines, hand_line_w_eff);
}

void launch_hand_dots(
    cudaSurfaceObject_t surf, int W, int H,
    const DeviceDot* d_dots, int n_dots,
    int max_w, int max_h, float marker_scale, cudaStream_t stream)
{
    if(n_dots <= 0) return;
    dim3 block(16, 16);
    dim3 grid((max_w + 15) / 16, (max_h + 15) / 16,
              static_cast<unsigned>(n_dots));
    const float hand_dot_r_eff = static_cast<float>(HAND_DOT_R) * marker_scale;
    draw_hand_dots_kernel<<<grid, block, 0, stream>>>(surf, W, H, d_dots, n_dots, hand_dot_r_eff);
}

} // namespace dwpose_td
