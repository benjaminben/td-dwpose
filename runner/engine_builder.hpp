#pragma once

#include <array>
#include <string>
#include <vector>

namespace dwpose_td
{

struct EngineBuildSpec
{
    std::string onnx_path;
    std::string engine_path;
    std::array<int, 4> input_shape_nchw; // batch, channels, h, w
    bool fp16 = true;
    int workspace_gib = 4;
};

// Build a TRT 10 engine from ONNX. Mirrors export_dwpose.py's flags. Blocking;
// caller should run on a worker thread (the runner does this).
bool build_engine(const EngineBuildSpec& spec, std::string* err_out);

// Compatibility check for an existing engine + meta.json against the running
// machine. Returns true if the engine can be safely deserialized as-is. If
// false, err_out gets a human description so the caller can surface it.
bool engine_is_compatible(
    const std::string& engine_path,
    const std::string& meta_path,
    std::string* err_out);

// Write a meta.json with the same schema export_dwpose.py emits, so engines
// built either by the python tool or by this C++ builder are interchangeable.
struct MetaModel
{
    std::string onnx_filename;
    std::string engine_filename;
    std::array<int, 4> input_shape_nchw;
};
bool write_meta_json(
    const std::string& meta_path,
    const std::string& hf_repo, const std::string& hf_revision,
    const MetaModel& yolox, const MetaModel& dwpose,
    bool fp16, int workspace_gib);

std::string current_trt_version();
std::string current_gpu_cc();

} // namespace dwpose_td
