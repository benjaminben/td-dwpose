#include "pose_render_kernels.hpp"

#include <cuda_runtime.h>
#include <surface_types.h>
#include <surface_indirect_functions.h>

namespace dwpose_td
{

namespace
{

// util.py:155 -- cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1).
// Single fixed color; no per-dot color table required for the face pass.
__global__ void draw_face_dots_kernel(
    cudaSurfaceObject_t surf, int W, int H,
    const DeviceDot* dots, int n_dots)
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
    const float r = static_cast<float>(FACE_DOT_R);
    if(dx * dx + dy * dy > r * r) return;
    uchar4 v{255, 255, 255, 255};
    // Y-flip to TD's bottom-left output convention (see pose_render.cu).
    surf2Dwrite(v, surf, px * 4, (H - 1) - py);
}

} // namespace

void launch_face_dots(
    cudaSurfaceObject_t surf, int W, int H,
    const DeviceDot* d_dots, int n_dots,
    int max_w, int max_h, cudaStream_t stream)
{
    if(n_dots <= 0) return;
    dim3 block(16, 16);
    dim3 grid((max_w + 15) / 16, (max_h + 15) / 16,
              static_cast<unsigned>(n_dots));
    draw_face_dots_kernel<<<grid, block, 0, stream>>>(surf, W, H, d_dots, n_dots);
}

} // namespace dwpose_td
