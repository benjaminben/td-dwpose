#include "dwpose_pre.hpp"

#include <cuda_runtime.h>

namespace dwpose_td
{

namespace
{

constexpr float DW_MEAN_B = 103.53f;
constexpr float DW_MEAN_G = 116.28f;
constexpr float DW_MEAN_R = 123.675f;
constexpr float DW_STD_B  = 57.375f;
constexpr float DW_STD_G  = 57.12f;
constexpr float DW_STD_R  = 58.395f;

// Sample TD's bottom-up surface as if top-down. dwpose_onnx.py operates on
// top-down PIL images, so we mirror the y axis here once at preprocess time.
__device__ __forceinline__ uchar4 sample_pixel(
    cudaTextureObject_t tex, int sx, int sy, int src_h)
{
    return tex2D<uchar4>(tex, sx + 0.5f, (src_h - 1 - sy) + 0.5f);
}

__device__ __forceinline__ void unpack_bgr(
    uchar4 px, int order, float& b, float& g, float& r)
{
    if(order == 0) // BGRA
    {
        b = static_cast<float>(px.x);
        g = static_cast<float>(px.y);
        r = static_cast<float>(px.z);
    }
    else // RGBA
    {
        r = static_cast<float>(px.x);
        g = static_cast<float>(px.y);
        b = static_cast<float>(px.z);
    }
}

__global__ void yolox_letterbox_kernel(
    cudaTextureObject_t tex, int src_w, int src_h, int order,
    float* dst, int dst_w, int dst_h, float scale)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= dst_w || y >= dst_h) return;

    const int nh = static_cast<int>(src_h * scale);
    const int nw = static_cast<int>(src_w * scale);

    float b, g, r;
    if(x < nw && y < nh)
    {
        const float fx = (x + 0.5f) / scale - 0.5f;
        const float fy = (y + 0.5f) / scale - 0.5f;
        const int x0 = max(0, min(src_w - 1, static_cast<int>(floorf(fx))));
        const int y0 = max(0, min(src_h - 1, static_cast<int>(floorf(fy))));
        const int x1 = min(src_w - 1, x0 + 1);
        const int y1 = min(src_h - 1, y0 + 1);
        const float ax = fx - x0;
        const float ay = fy - y0;
        uchar4 p00 = sample_pixel(tex, x0, y0, src_h);
        uchar4 p10 = sample_pixel(tex, x1, y0, src_h);
        uchar4 p01 = sample_pixel(tex, x0, y1, src_h);
        uchar4 p11 = sample_pixel(tex, x1, y1, src_h);
        float b00,g00,r00, b10,g10,r10, b01,g01,r01, b11,g11,r11;
        unpack_bgr(p00, order, b00,g00,r00);
        unpack_bgr(p10, order, b10,g10,r10);
        unpack_bgr(p01, order, b01,g01,r01);
        unpack_bgr(p11, order, b11,g11,r11);
        const float w00 = (1-ax)*(1-ay), w10 = ax*(1-ay);
        const float w01 = (1-ax)*ay,     w11 = ax*ay;
        b = b00*w00 + b10*w10 + b01*w01 + b11*w11;
        g = g00*w00 + g10*w10 + g01*w01 + g11*w11;
        r = r00*w00 + r10*w10 + r01*w01 + r11*w11;
    }
    else
    {
        b = 114.0f; g = 114.0f; r = 114.0f;
    }

    // NCHW with C order matching the python: canvas BGR -> transpose(2,0,1)
    // emits planes [B, G, R] as channel 0/1/2 (since the python reads BGR
    // straight from cv2 and never swaps).
    const int plane = dst_w * dst_h;
    const int idx = y * dst_w + x;
    dst[0 * plane + idx] = b;
    dst[1 * plane + idx] = g;
    dst[2 * plane + idx] = r;
}

__global__ void dwpose_warp_kernel(
    cudaTextureObject_t tex, int src_w, int src_h, int order,
    float m0, float m1, float m2, float m3, float m4, float m5,
    float* dst, int dst_w, int dst_h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= dst_w || y >= dst_h) return;

    // M is dst<-src inverse (warpAffine standard): src_xy = M^{-1}(dst_xy).
    // Python: M = [[1/sx, 0, tw/2 - cx/sx], [0, 1/sy, th/2 - cy/sy]] applied
    // to source coords -> dst coords. We invert: src_x = (x - m2)/m0, etc.
    const float sx = (x - m2) / m0;
    const float sy = (y - m5) / m4;

    float b = 0.0f, g = 0.0f, r = 0.0f;
    if(sx >= 0.0f && sx < src_w && sy >= 0.0f && sy < src_h)
    {
        const int x0 = max(0, min(src_w - 1, static_cast<int>(floorf(sx))));
        const int y0 = max(0, min(src_h - 1, static_cast<int>(floorf(sy))));
        const int x1 = min(src_w - 1, x0 + 1);
        const int y1 = min(src_h - 1, y0 + 1);
        const float ax = sx - x0;
        const float ay = sy - y0;
        uchar4 p00 = sample_pixel(tex, x0, y0, src_h);
        uchar4 p10 = sample_pixel(tex, x1, y0, src_h);
        uchar4 p01 = sample_pixel(tex, x0, y1, src_h);
        uchar4 p11 = sample_pixel(tex, x1, y1, src_h);
        float b00,g00,r00, b10,g10,r10, b01,g01,r01, b11,g11,r11;
        unpack_bgr(p00, order, b00,g00,r00);
        unpack_bgr(p10, order, b10,g10,r10);
        unpack_bgr(p01, order, b01,g01,r01);
        unpack_bgr(p11, order, b11,g11,r11);
        const float w00 = (1-ax)*(1-ay), w10 = ax*(1-ay);
        const float w01 = (1-ax)*ay,     w11 = ax*ay;
        b = b00*w00 + b10*w10 + b01*w01 + b11*w11;
        g = g00*w00 + g10*w10 + g01*w01 + g11*w11;
        r = r00*w00 + r10*w10 + r01*w01 + r11*w11;
    }

    // ImageNet stats applied per BGR channel (DWPose was trained with
    // to_rgb=True so the in-tensor channel order matches RGB indexing of mean
    // -- see dwpose_onnx.py:91 comment).
    b = (b - DW_MEAN_B) / DW_STD_B;
    g = (g - DW_MEAN_G) / DW_STD_G;
    r = (r - DW_MEAN_R) / DW_STD_R;

    const int plane = dst_w * dst_h;
    const int idx = y * dst_w + x;
    dst[0 * plane + idx] = b;
    dst[1 * plane + idx] = g;
    dst[2 * plane + idx] = r;
}

cudaTextureObject_t make_tex(cudaArray_t arr)
{
    cudaResourceDesc rd{};
    rd.resType = cudaResourceTypeArray;
    rd.res.array.array = arr;
    cudaTextureDesc td{};
    td.addressMode[0] = cudaAddressModeClamp;
    td.addressMode[1] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModePoint;
    td.readMode = cudaReadModeElementType;
    td.normalizedCoords = 0;
    cudaTextureObject_t tex = 0;
    cudaCreateTextureObject(&tex, &rd, &td, nullptr);
    return tex;
}

} // namespace

float launch_yolox_letterbox(
    cudaArray_t src, int src_w, int src_h, ChannelOrder src_order,
    float* dst_nchw, int dst_w, int dst_h, cudaStream_t stream)
{
    const float scale_h = static_cast<float>(dst_h) / src_h;
    const float scale_w = static_cast<float>(dst_w) / src_w;
    const float scale = scale_h < scale_w ? scale_h : scale_w;
    cudaTextureObject_t tex = make_tex(src);
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
    yolox_letterbox_kernel<<<grid, block, 0, stream>>>(
        tex, src_w, src_h, static_cast<int>(src_order),
        dst_nchw, dst_w, dst_h, scale);
    cudaDestroyTextureObject(tex);
    return scale;
}

void launch_dwpose_crop_warp(
    cudaArray_t src, int src_w, int src_h, ChannelOrder src_order,
    const float M[6], float* dst_nchw, int dst_w, int dst_h,
    cudaStream_t stream)
{
    cudaTextureObject_t tex = make_tex(src);
    dim3 block(16, 16);
    dim3 grid((dst_w + block.x - 1) / block.x, (dst_h + block.y - 1) / block.y);
    dwpose_warp_kernel<<<grid, block, 0, stream>>>(
        tex, src_w, src_h, static_cast<int>(src_order),
        M[0], M[1], M[2], M[3], M[4], M[5],
        dst_nchw, dst_w, dst_h);
    cudaDestroyTextureObject(tex);
}

} // namespace dwpose_td
