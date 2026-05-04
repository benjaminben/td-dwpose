#include "engine_builder.hpp"

#include "td_debug_log.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace dwpose_td
{

namespace
{

class BuildLogger : public nvinfer1::ILogger
{
public:
    void log(Severity sev, const char* msg) noexcept override
    {
        if(sev <= Severity::kWARNING) TDDBG("[trt-build] " << msg);
    }
};

BuildLogger& bl() { static BuildLogger l; return l; }

std::string read_file(const std::string& path)
{
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if(!f.good()) return {};
    const auto sz = f.tellg();
    f.seekg(0);
    std::string s(static_cast<size_t>(sz), '\0');
    f.read(&s[0], sz);
    return s;
}

std::string find_str(const std::string& s, const std::string& key)
{
    const auto p = s.find("\"" + key + "\"");
    if(p == std::string::npos) return {};
    const auto colon = s.find(':', p);
    if(colon == std::string::npos) return {};
    const auto q1 = s.find('"', colon);
    if(q1 == std::string::npos) return {};
    const auto q2 = s.find('"', q1 + 1);
    if(q2 == std::string::npos) return {};
    return s.substr(q1 + 1, q2 - q1 - 1);
}

} // namespace

std::string current_trt_version()
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%d.%d.%d",
                  NV_TENSORRT_MAJOR, NV_TENSORRT_MINOR, NV_TENSORRT_PATCH);
    return buf;
}

std::string current_gpu_cc()
{
    int dev = 0;
    if(cudaGetDevice(&dev) != cudaSuccess) return "unknown";
    cudaDeviceProp p{};
    if(cudaGetDeviceProperties(&p, dev) != cudaSuccess) return "unknown";
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%d.%d", p.major, p.minor);
    return buf;
}

bool engine_is_compatible(
    const std::string& engine_path,
    const std::string& meta_path,
    std::string* err_out)
{
    if(!fs::exists(engine_path))
    {
        if(err_out) *err_out = "engine file missing: " + engine_path;
        return false;
    }
    // meta.json may not exist if the user copied the .engine in by hand. Treat
    // that as "trust it" -- TRT will fail loudly on version mismatch when we
    // deserialize, and we surface that error elsewhere.
    if(!fs::exists(meta_path)) return true;
    std::string txt = read_file(meta_path);
    std::string trt_v = find_str(txt, "trt_version");
    std::string gpu_cc = find_str(txt, "gpu_cc");
    const std::string cur_trt = current_trt_version();
    const std::string cur_cc = current_gpu_cc();
    // TRT serialized engines are tied to the exact TRT major+minor and GPU CC
    // they were built on. Mismatch = rebuild. (Substr-based compare misreads
    // multi-digit minor versions like "10.16" as "10.1" -- split on '.'.)
    auto major_minor = [](const std::string& v) {
        auto p1 = v.find('.');
        if(p1 == std::string::npos) return v;
        auto p2 = v.find('.', p1 + 1);
        return v.substr(0, p2);
    };
    if(!trt_v.empty() && major_minor(trt_v) != major_minor(cur_trt))
    {
        if(err_out) *err_out = "engine TRT " + trt_v + " != runtime " + cur_trt;
        return false;
    }
    if(!gpu_cc.empty() && gpu_cc != cur_cc && cur_cc != "unknown")
    {
        if(err_out) *err_out = "engine gpu_cc " + gpu_cc + " != runtime " + cur_cc;
        return false;
    }
    return true;
}

bool build_engine(const EngineBuildSpec& spec, std::string* err_out)
{
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(bl());
    if(!builder) { if(err_out) *err_out = "createInferBuilder failed"; return false; }

    const auto netflag = 1U << static_cast<uint32_t>(
        nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    nvinfer1::INetworkDefinition* network = builder->createNetworkV2(netflag);
    if(!network)
    {
        delete builder;
        if(err_out) *err_out = "createNetworkV2 failed";
        return false;
    }

    nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, bl());
    std::string onnx_bytes = read_file(spec.onnx_path);
    if(onnx_bytes.empty())
    {
        delete parser; delete network; delete builder;
        if(err_out) *err_out = "could not read ONNX: " + spec.onnx_path;
        return false;
    }
    if(!parser->parse(onnx_bytes.data(), onnx_bytes.size()))
    {
        for(int i = 0; i < parser->getNbErrors(); ++i)
            TDDBG("[trt-build] onnx parse: " << parser->getError(i)->desc());
        delete parser; delete network; delete builder;
        if(err_out) *err_out = "ONNX parse failed";
        return false;
    }

    nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE,
                               static_cast<size_t>(spec.workspace_gib) << 30);
    if(spec.fp16 && builder->platformHasFastFp16())
        config->setFlag(nvinfer1::BuilderFlag::kFP16);

    // Static shape pin (matches export_dwpose.py).
    auto* in0 = network->getInput(0);
    nvinfer1::Dims4 d{spec.input_shape_nchw[0], spec.input_shape_nchw[1],
                      spec.input_shape_nchw[2], spec.input_shape_nchw[3]};
    auto* profile = builder->createOptimizationProfile();
    profile->setDimensions(in0->getName(), nvinfer1::OptProfileSelector::kMIN, d);
    profile->setDimensions(in0->getName(), nvinfer1::OptProfileSelector::kOPT, d);
    profile->setDimensions(in0->getName(), nvinfer1::OptProfileSelector::kMAX, d);
    config->addOptimizationProfile(profile);

    TDDBG("[trt-build] building " << spec.engine_path << " fp16=" << spec.fp16);
    nvinfer1::IHostMemory* mem = builder->buildSerializedNetwork(*network, *config);
    bool ok = false;
    if(mem)
    {
        fs::create_directories(fs::path(spec.engine_path).parent_path());
        std::ofstream out(spec.engine_path, std::ios::binary);
        out.write(static_cast<const char*>(mem->data()), mem->size());
        out.close();
        ok = out.good();
        delete mem;
    }
    else if(err_out) { *err_out = "buildSerializedNetwork returned null"; }
    delete config; delete parser; delete network; delete builder;
    return ok;
}

bool write_meta_json(
    const std::string& meta_path,
    const std::string& hf_repo, const std::string& hf_revision,
    const MetaModel& yolox, const MetaModel& dwpose,
    bool fp16, int workspace_gib)
{
    auto shape = [](const std::array<int, 4>& s){
        std::ostringstream o;
        o << "[" << s[0] << ", " << s[1] << ", " << s[2] << ", " << s[3] << "]";
        return o.str();
    };
    auto model = [&](const MetaModel& m){
        std::ostringstream o;
        o << "    {\n"
          << "      \"onnx_filename\": \"" << m.onnx_filename << "\",\n"
          << "      \"engine_filename\": \"" << m.engine_filename << "\",\n"
          << "      \"input_shape_nchw\": " << shape(m.input_shape_nchw) << "\n"
          << "    }";
        return o.str();
    };
    std::ostringstream o;
    o << "{\n"
      << "  \"schema\": \"dwpose-engines/v1\",\n"
      << "  \"hf_repo\": \"" << hf_repo << "\",\n"
      << "  \"hf_revision\": \"" << hf_revision << "\",\n"
      << "  \"models\": {\n"
      << "    \"yolox\": " << model(yolox) << ",\n"
      << "    \"dwpose\": " << model(dwpose) << "\n"
      << "  },\n"
      << "  \"build\": {\n"
      << "    \"fp16\": " << (fp16 ? "true" : "false") << ",\n"
      << "    \"workspace_gib\": " << workspace_gib << ",\n"
      << "    \"trt_version\": \"" << current_trt_version() << "\",\n"
      << "    \"gpu_cc\": \"" << current_gpu_cc() << "\",\n"
      << "    \"builder\": \"td-dwpose\"\n"
      << "  }\n"
      << "}\n";
    std::ofstream f(meta_path);
    if(!f.good()) return false;
    f << o.str();
    return f.good();
}

} // namespace dwpose_td
