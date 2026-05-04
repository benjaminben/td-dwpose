#include "pose_render.hpp"
#include "pose_render_internal.hpp"
#include "pose_render_kernels.hpp"

#include <cuda_runtime.h>
// CUDA 13 split / removed standalone <surface_functions.h>; surf2Dwrite et al
// live in <surface_indirect_functions.h> now (and are auto-pulled by nvcc when
// compiling .cu, but include explicitly so the dependency is visible). The
// boundary-mode enum (cudaBoundaryModeZap etc.) lives separately in
// <surface_types.h> and must also be pulled in explicitly.
#include <surface_types.h>
#include <surface_indirect_functions.h>

#include <vector>

namespace dwpose_td
{

// Device-side color table (was constexpr in pose_render_internal.hpp; nvcc
// rejects host-storage constexpr arrays as "undefined in device code" inside
// kernels). Defined here once at file scope; read-only from kernels via
// kPoseColors[idx]. Values lifted from util.py:78-80 -- do NOT permute.
__device__ const Color3 kPoseColors[18] = {
    {255,   0,   0}, {255,  85,   0}, {255, 170,   0}, {255, 255,   0},
    {170, 255,   0}, { 85, 255,   0}, {  0, 255,   0}, {  0, 255,  85},
    {  0, 255, 170}, {  0, 255, 255}, {  0, 170, 255}, {  0,  85, 255},
    {  0,   0, 255}, { 85,   0, 255}, {170,   0, 255}, {255,   0, 255},
    {255,   0, 170}, {255,   0,  85},
};

namespace
{

__global__ void fill_black_kernel(cudaSurfaceObject_t surf, int W, int H)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= W || y >= H) return;
    uchar4 v{0, 0, 0, 255};
    // Y-flip: TD's CUDA TOP reads the output array with bottom-left origin
    // (OpenGL convention); CUDA arrays index top-left. Without this the
    // pose renders upside down in TD. Every renderer kernel writes (and
    // reads) the surface with (H-1-y) so the canvas stays self-consistent.
    surf2Dwrite(v, surf, x * 4, (H - 1) - y);
}

__global__ void draw_limbs_kernel(
    cudaSurfaceObject_t surf, int W, int H,
    const DeviceLimb* limbs, int n_limbs)
{
    const int li = blockIdx.z;
    if(li >= n_limbs) return;
    const DeviceLimb L = limbs[li];
    const int px = L.x0 + blockIdx.x * blockDim.x + threadIdx.x;
    const int py = L.y0 + blockIdx.y * blockDim.y + threadIdx.y;
    if(px >= L.x1 || py >= L.y1) return;
    if(px < 0 || py < 0 || px >= W || py >= H) return;

    const float dx = (px + 0.5f) - L.mx;
    const float dy = (py + 0.5f) - L.my;
    const float u =  dx * L.ux + dy * L.uy;
    const float v = -dx * L.uy + dy * L.ux;
    const float a = L.half_len > 1.0f ? L.half_len : 1.0f;
    const float b = static_cast<float>(STICK_W);
    if((u * u) / (a * a) + (v * v) / (b * b) > 1.0f) return;
    const Color3 c = kPoseColors[L.color_idx];
    uchar4 px_v{c.r, c.g, c.b, 255};
    surf2Dwrite(px_v, surf, px * 4, (H - 1) - py);
}

__global__ void darken_kernel(cudaSurfaceObject_t surf, int W, int H)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= W || y >= H) return;
    const int yf = (H - 1) - y;  // see fill_black_kernel for the Y-flip rationale
    uchar4 v{};
    surf2Dread(&v, surf, x * 4, yf);
    v.x = static_cast<unsigned char>(static_cast<int>(v.x * 0.6f));
    v.y = static_cast<unsigned char>(static_cast<int>(v.y * 0.6f));
    v.z = static_cast<unsigned char>(static_cast<int>(v.z * 0.6f));
    v.w = 255;
    surf2Dwrite(v, surf, x * 4, yf);
}

__global__ void draw_dots_kernel(
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
    const float r = static_cast<float>(DOT_R);
    if(dx * dx + dy * dy > r * r) return;
    const Color3 c = kPoseColors[D.color_idx];
    uchar4 px_v{c.r, c.g, c.b, 255};
    surf2Dwrite(px_v, surf, px * 4, (H - 1) - py);
}

cudaSurfaceObject_t make_surf(cudaArray_t arr)
{
    cudaResourceDesc rd{};
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = arr;
    cudaSurfaceObject_t s = 0;
    cudaCreateSurfaceObject(&s, &rd);
    return s;
}

} // namespace

void launch_darken(cudaSurfaceObject_t surf, int W, int H, cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    darken_kernel<<<grid, block, 0, stream>>>(surf, W, H);
}

void launch_body_limbs(
    cudaSurfaceObject_t surf, int W, int H,
    const DeviceLimb* d_limbs, int n_limbs,
    int max_w, int max_h, cudaStream_t stream)
{
    if(n_limbs <= 0) return;
    dim3 block(16, 16);
    dim3 grid((max_w + 15) / 16, (max_h + 15) / 16,
              static_cast<unsigned>(n_limbs));
    draw_limbs_kernel<<<grid, block, 0, stream>>>(surf, W, H, d_limbs, n_limbs);
}

void launch_body_dots(
    cudaSurfaceObject_t surf, int W, int H,
    const DeviceDot* d_dots, int n_dots,
    int max_w, int max_h, cudaStream_t stream)
{
    if(n_dots <= 0) return;
    dim3 block(16, 16);
    dim3 grid((max_w + 15) / 16, (max_h + 15) / 16,
              static_cast<unsigned>(n_dots));
    draw_dots_kernel<<<grid, block, 0, stream>>>(surf, W, H, d_dots, n_dots);
}

// Defined in pose_render_dispatch.cpp -- chooses between the legacy
// single-batch dispatch and the per-color_idx OrderedDraw dispatch
// based on the `flags` bit. Pure host-side: only orchestrates kernel
// launches via the launch_* wrappers above.
void render_pose_dispatch(
    cudaSurfaceObject_t surf, int W, int H,
    PoseRenderTables& t, unsigned int flags, cudaStream_t stream);

void render_pose(
    cudaArray_t out, int W, int H,
    const PoseKp* kps_host, int num_persons, int stride,
    int src_w, int src_h, cudaStream_t stream,
    unsigned int flags)
{
    if(!out || W <= 0 || H <= 0) return;

    cudaSurfaceObject_t surf = make_surf(out);
    if(!surf) return;  // Output cudaArray missing the SurfaceLoadStore flag.

    // Pass 1: fill black.
    {
        dim3 block(16, 16);
        dim3 grid((W + 15) / 16, (H + 15) / 16);
        fill_black_kernel<<<grid, block, 0, stream>>>(surf, W, H);
    }

    PoseRenderTables t = build_pose_tables(
        kps_host, num_persons, stride, src_w, src_h, W, H);

    // Pass 2: limbs.
    render_pose_dispatch(surf, W, H, t, flags, stream);

    cudaDestroySurfaceObject(surf);
}

} // namespace dwpose_td
