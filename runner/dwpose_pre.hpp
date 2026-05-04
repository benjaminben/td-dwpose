#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace dwpose_td
{

// 0 = BGRA byte order, 1 = RGBA byte order. Read this from
// inArray->textureDesc per-frame; hardcoding either silently swaps R/B
// when the input arrives in the opposite layout.
enum class ChannelOrder : int { BGRA = 0, RGBA = 1 };

// YOLOX letterbox: src cudaArray (uint8 BGRA/RGBA, src_w x src_h) -> dst
// float32 NCHW (1 x 3 x dst_h x dst_w), padded with 114, layout matches
// _yolox_letterbox in dwpose_onnx.py. The model was trained to_rgb=True so
// the on-device float channel order ends up B,G,R per the upstream code (see
// _yolox_letterbox writing canvas without a swap and the ONNX consuming HWC
// uint8 cast directly to float -- the python passes BGR straight through).
// Returns the scale used (min(dst_h/src_h, dst_w/src_w)).
float launch_yolox_letterbox(
    cudaArray_t src, int src_w, int src_h, ChannelOrder src_order,
    float* dst_nchw, int dst_w, int dst_h, cudaStream_t stream);

// DWPose crop+warp+normalize. src cudaArray (uint8 BGRA/RGBA) -> dst float32
// NCHW (1 x 3 x dst_h x dst_w). The 2x3 affine M[0..5] is the same matrix
// _crop_warp builds in dwpose_onnx.py (row-major, dst-from-src inverse).
// Subtracts DWPOSE_MEAN and divides by DWPOSE_STD per BGR channel.
void launch_dwpose_crop_warp(
    cudaArray_t src, int src_w, int src_h, ChannelOrder src_order,
    const float M[6], float* dst_nchw, int dst_w, int dst_h,
    cudaStream_t stream);

} // namespace dwpose_td
