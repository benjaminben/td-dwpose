#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace nvinfer1 { class ICudaEngine; class IExecutionContext; class IRuntime; }

namespace dwpose_td
{

struct TensorInfo
{
    std::string name;
    std::vector<int64_t> shape;
    size_t bytes = 0; // float32 assumption
};

// Minimal TRT 10 loader. Mirrors test_dwpose_trt.py:TrtRunner but the host
// roundtrip is replaced with caller-managed device pointers via
// setTensorAddress + executeAsyncV3 on the supplied stream.
class TrtRunner
{
public:
    TrtRunner();
    ~TrtRunner();

    TrtRunner(const TrtRunner&) = delete;
    TrtRunner& operator=(const TrtRunner&) = delete;

    bool loadEngine(const std::string& path, std::string* err_out);
    bool isLoaded() const { return engine_ != nullptr; }

    const std::vector<TensorInfo>& inputs() const { return inputs_; }
    const std::vector<TensorInfo>& outputs() const { return outputs_; }

    // Caller owns device pointers. addrs are keyed by tensor name.
    bool setTensorAddress(const std::string& name, void* device_ptr);
    bool execute(cudaStream_t stream, std::string* err_out);

private:
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    std::vector<TensorInfo> inputs_;
    std::vector<TensorInfo> outputs_;
};

} // namespace dwpose_td
