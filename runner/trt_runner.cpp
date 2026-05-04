#include "trt_runner.hpp"

#include "td_debug_log.hpp"

#include <NvInfer.h>

#include <fstream>
#include <vector>

namespace dwpose_td
{

namespace
{

class TrtLogger : public nvinfer1::ILogger
{
public:
    void log(Severity sev, const char* msg) noexcept override
    {
        if(sev <= Severity::kWARNING)
            TDDBG("[trt] " << msg);
    }
};

TrtLogger& global_logger()
{
    static TrtLogger l;
    return l;
}

size_t element_size(nvinfer1::DataType dt)
{
    switch(dt)
    {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL:  return 1;
        case nvinfer1::DataType::kUINT8: return 1;
        case nvinfer1::DataType::kINT64: return 8;
        default: return 4;
    }
}

} // namespace

TrtRunner::TrtRunner() = default;

TrtRunner::~TrtRunner()
{
    if(context_) delete context_;
    if(engine_)  delete engine_;
    if(runtime_) delete runtime_;
}

bool TrtRunner::loadEngine(const std::string& path, std::string* err_out)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if(!f.good())
    {
        if(err_out) *err_out = "engine file not found: " + path;
        return false;
    }
    const std::streamsize sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(static_cast<size_t>(sz));
    if(!f.read(buf.data(), sz))
    {
        if(err_out) *err_out = "engine file read failed: " + path;
        return false;
    }

    runtime_ = nvinfer1::createInferRuntime(global_logger());
    if(!runtime_) { if(err_out) *err_out = "createInferRuntime failed"; return false; }
    engine_ = runtime_->deserializeCudaEngine(buf.data(), buf.size());
    if(!engine_) { if(err_out) *err_out = "deserializeCudaEngine failed"; return false; }
    context_ = engine_->createExecutionContext();
    if(!context_) { if(err_out) *err_out = "createExecutionContext failed"; return false; }

    const int n = engine_->getNbIOTensors();
    for(int i = 0; i < n; ++i)
    {
        const char* name = engine_->getIOTensorName(i);
        const auto dims = engine_->getTensorShape(name);
        const auto dtype = engine_->getTensorDataType(name);
        TensorInfo ti;
        ti.name = name;
        size_t numel = 1;
        for(int d = 0; d < dims.nbDims; ++d)
        {
            ti.shape.push_back(dims.d[d]);
            if(dims.d[d] > 0) numel *= static_cast<size_t>(dims.d[d]);
        }
        ti.bytes = numel * element_size(dtype);
        if(engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
            inputs_.push_back(std::move(ti));
        else
            outputs_.push_back(std::move(ti));
    }
    TDDBG("[trt] loaded " << path << " inputs=" << inputs_.size()
          << " outputs=" << outputs_.size());
    return true;
}

bool TrtRunner::setTensorAddress(const std::string& name, void* device_ptr)
{
    if(!context_) return false;
    return context_->setTensorAddress(name.c_str(), device_ptr);
}

bool TrtRunner::execute(cudaStream_t stream, std::string* err_out)
{
    if(!context_) { if(err_out) *err_out = "trt: no context"; return false; }
    if(!context_->enqueueV3(stream))
    {
        if(err_out) *err_out = "trt: enqueueV3 failed";
        return false;
    }
    return true;
}

} // namespace dwpose_td
