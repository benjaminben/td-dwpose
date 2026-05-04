#include "dwpose_runner_impl.hpp"

#include "engine_builder.hpp"
#include "hf_download.hpp"
#include "pose_render.hpp"
#include "td_debug_log.hpp"
#include "trt_runner.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

namespace dwpose_td
{

namespace
{

constexpr const char* HF_REPO = "yzd-v/DWPose";
constexpr const char* HF_REVISION = "main";
constexpr const char* YOLOX_ONNX = "yolox_l.onnx";
constexpr const char* DWPOSE_ONNX = "dw-ll_ucoco_384.onnx";
constexpr const char* YOLOX_ENGINE = "yolox.engine";
constexpr const char* DWPOSE_ENGINE = "dwpose.engine";
constexpr const char* META_JSON = "meta.json";

constexpr float YOLOX_CONF_THRESH = 0.5f;
constexpr float YOLOX_IOU_THRESH = 0.45f;

void freep(void*& p)
{
    if(p) { cudaFree(p); p = nullptr; }
}

// Future optimization: pinned host memory + cudaMemcpyAsync would shave a
// couple ms here.
template <typename T>
void d2h_sync(std::vector<T>& host, void* dev, size_t bytes, cudaStream_t s)
{
    host.resize(bytes / sizeof(T));
    cudaMemcpyAsync(host.data(), dev, bytes, cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);
}

} // namespace

DWPoseRunner::DWPoseRunner() = default;

DWPoseRunner::~DWPoseRunner()
{
    if(prep_thread_.joinable()) prep_thread_.join();
    freep(det_in_dev_);
    freep(det_out_dev_);
    freep(pose_in_dev_);
    freep(pose_simcc_x_dev_);
    freep(pose_simcc_y_dev_);
}

int DWPoseRunner::numPersons() const
{
    std::lock_guard<std::mutex> g(kp_mutex_);
    return static_cast<int>(persons_.size());
}

std::vector<Keypoint> DWPoseRunner::keypoints(int personIdx) const
{
    std::lock_guard<std::mutex> g(kp_mutex_);
    if(personIdx < 0 || personIdx >= static_cast<int>(persons_.size())) return {};
    const auto& p = persons_[personIdx];
    std::vector<Keypoint> out(134);
    for(int i = 0; i < 134; ++i)
        out[i] = {p.kp[i * 2], p.kp[i * 2 + 1], p.sc[i]};
    return out;
}

void DWPoseRunner::renderPose(cudaArray_t out, int W, int H,
                              int src_w, int src_h, cudaStream_t stream,
                              unsigned int flags)
{
    // Snapshot persons under mutex, then drop the lock before kernel launches.
    std::vector<PoseKp> flat;
    int n = 0;
    {
        std::lock_guard<std::mutex> g(kp_mutex_);
        n = static_cast<int>(persons_.size());
        flat.resize(static_cast<size_t>(n) * 134);
        for(int p = 0; p < n; ++p) for(int i = 0; i < 134; ++i)
        {
            flat[p * 134 + i] = {persons_[p].kp[i * 2], persons_[p].kp[i * 2 + 1], persons_[p].sc[i]};
        }
    }
    render_pose(out, W, H, flat.data(), n, 134, src_w, src_h, stream, flags);
}

void DWPoseRunner::requestReload(const std::string& engines_dir)
{
    if(prep_running_.exchange(true)) return;
    if(prep_thread_.joinable()) prep_thread_.join();
    prep_thread_ = std::thread([this, engines_dir]{
        prepareWorker(engines_dir);
        prep_running_.store(false);
    });
}

void DWPoseRunner::prepareWorker(std::string engines_dir)
{
    if(engines_dir.empty())
    {
        // No silent default -- surface the missing param to the user via the
        // info CHOP `status` channel + last_error_.
        status_.store(Status::Error);
        last_error_ = "engines folder unset (no canonical default)";
        TDDBG("prepare: " << last_error_);
        return;
    }
    fs::create_directories(engines_dir);

    // Engine-first discovery: if both engines exist and pass compat, skip
    // ONNX entirely (no curl needed). Only resolve/build if engines missing
    // or incompatible.
    fs::path yolox_eng = fs::path(engines_dir) / YOLOX_ENGINE;
    fs::path pose_eng = fs::path(engines_dir) / DWPOSE_ENGINE;
    fs::path meta_path = fs::path(engines_dir) / META_JSON;

    auto need_build = [&](const fs::path& engf) {
        std::string werr;
        return !engine_is_compatible(engf.string(), meta_path.string(), &werr);
    };
    const bool engines_ok = !need_build(yolox_eng) && !need_build(pose_eng);

    std::string err;
    if(!engines_ok)
    {
        // Resolve ONNX inputs (local -> HF cache -> network) only when needed.
        status_.store(Status::DownloadingOnnx);
        progress_.store(0.0f);
        HFFile yolox_f{HF_REPO, HF_REVISION, YOLOX_ONNX};
        HFFile pose_f{HF_REPO, HF_REVISION, DWPOSE_ONNX};
        auto cb = [this](uint64_t got, uint64_t total){
            if(total > 0)
                progress_.store(static_cast<float>(static_cast<double>(got) / total));
        };
        std::string yolox_onnx = hf_resolve_or_download(yolox_f, engines_dir, cb, &err);
        if(yolox_onnx.empty())
        {
            status_.store(Status::Error);
            last_error_ = "yolox onnx fetch failed: " + err;
            return;
        }
        std::string pose_onnx = hf_resolve_or_download(pose_f, engines_dir, cb, &err);
        if(pose_onnx.empty())
        {
            status_.store(Status::Error);
            last_error_ = "dwpose onnx fetch failed: " + err;
            return;
        }

        status_.store(Status::BuildingEngine);
        progress_.store(0.0f);
        EngineBuildSpec yspec{yolox_onnx, yolox_eng.string(), {1,3,640,640}, true, 4};
        if(!build_engine(yspec, &err))
        {
            status_.store(Status::Error);
            last_error_ = "yolox engine build failed: " + err;
            return;
        }
        progress_.store(0.5f);
        EngineBuildSpec pspec{pose_onnx, pose_eng.string(), {1,3,384,288}, true, 4};
        if(!build_engine(pspec, &err))
        {
            status_.store(Status::Error);
            last_error_ = "dwpose engine build failed: " + err;
            return;
        }
        progress_.store(1.0f);
        write_meta_json(meta_path.string(), HF_REPO, HF_REVISION,
                        {YOLOX_ONNX, YOLOX_ENGINE, {1,3,640,640}},
                        {DWPOSE_ONNX, DWPOSE_ENGINE, {1,3,384,288}}, true, 4);
    }

    if(!loadEngines(engines_dir, &err))
    {
        status_.store(Status::Error);
        last_error_ = "engine load failed: " + err;
        return;
    }
    current_engines_dir_ = engines_dir;
    status_.store(Status::Ready);
    last_error_.clear();
    TDDBG("prepare: ready");
}

bool DWPoseRunner::loadEngines(const std::string& dir, std::string* err)
{
    auto det = std::make_unique<TrtRunner>();
    auto pose = std::make_unique<TrtRunner>();
    if(!det->loadEngine((fs::path(dir) / YOLOX_ENGINE).string(), err)) return false;
    if(!pose->loadEngine((fs::path(dir) / DWPOSE_ENGINE).string(), err)) return false;

    // Pin shapes from engine metadata. Detector input: (1,3,640,640).
    if(det->inputs().empty() || det->outputs().empty()) { *err = "det engine io"; return false; }
    if(pose->inputs().empty() || pose->outputs().size() < 2) { *err = "pose engine io"; return false; }

    const auto& dout = det->outputs()[0].shape;
    det_out_anchors_ = (dout.size() >= 2) ? static_cast<int>(dout[1]) : 8400;

    const auto& pin = pose->inputs()[0].shape;
    if(pin.size() == 4) { pose_h_ = static_cast<int>(pin[2]); pose_w_ = static_cast<int>(pin[3]); }

    const auto& sxout = pose->outputs()[0].shape; // [1, num_kp, w*split]
    const auto& syout = pose->outputs()[1].shape; // [1, num_kp, h*split]
    if(sxout.size() == 3)
    {
        pose_num_kp_ = static_cast<int>(sxout[1]);
        pose_simcc_x_len_ = static_cast<int>(sxout[2]);
    }
    if(syout.size() == 3) pose_simcc_y_len_ = static_cast<int>(syout[2]);

    // Allocate device buffers.
    cudaMalloc(&det_in_dev_, sizeof(float) * 3 * det_h_ * det_w_);
    cudaMalloc(&det_out_dev_, det->outputs()[0].bytes);
    cudaMalloc(&pose_in_dev_, sizeof(float) * 3 * pose_h_ * pose_w_);
    cudaMalloc(&pose_simcc_x_dev_, pose->outputs()[0].bytes);
    cudaMalloc(&pose_simcc_y_dev_, pose->outputs()[1].bytes);

    // Bind engine I/O addresses once; same buffers reused every frame.
    det->setTensorAddress(det->inputs()[0].name, det_in_dev_);
    det->setTensorAddress(det->outputs()[0].name, det_out_dev_);
    pose->setTensorAddress(pose->inputs()[0].name, pose_in_dev_);
    pose->setTensorAddress(pose->outputs()[0].name, pose_simcc_x_dev_);
    pose->setTensorAddress(pose->outputs()[1].name, pose_simcc_y_dev_);

    det_ = std::move(det);
    pose_ = std::move(pose);
    return true;
}

void DWPoseRunner::runFrame(cudaArray_t in, int W, int H, ChannelOrder order,
                            cudaStream_t stream)
{
    if(status_.load() != Status::Ready || !det_ || !pose_) return;
    inferLocked(in, W, H, order, stream);
}

void DWPoseRunner::inferLocked(cudaArray_t in, int W, int H, ChannelOrder order,
                               cudaStream_t stream)
{
    auto t0 = std::chrono::steady_clock::now();

    // Detector: BGRA cudaArray -> NCHW float -> TRT -> N*85 raw.
    const float scale = launch_yolox_letterbox(
        in, W, H, order, static_cast<float*>(det_in_dev_),
        det_w_, det_h_, stream);
    std::string err;
    if(!det_->execute(stream, &err)) { TDDBG("det infer: " << err); return; }

    d2h_sync(det_out_host_, det_out_dev_, det_->outputs()[0].bytes, stream);

    auto boxes = yolox_decode(
        det_out_host_.data(), det_out_anchors_, det_h_, det_w_,
        scale, YOLOX_CONF_THRESH, YOLOX_IOU_THRESH);

    std::vector<Person> new_persons;
    new_persons.reserve(boxes.size());
    for(const auto& b : boxes)
    {
        const auto crop = build_crop(b, pose_w_, pose_h_);
        launch_dwpose_crop_warp(in, W, H, order, crop.M,
                                static_cast<float*>(pose_in_dev_),
                                pose_w_, pose_h_, stream);
        if(!pose_->execute(stream, &err)) { TDDBG("pose infer: " << err); break; }
        d2h_sync(simcc_x_host_, pose_simcc_x_dev_,
                 pose_->outputs()[0].bytes, stream);
        d2h_sync(simcc_y_host_, pose_simcc_y_dev_,
                 pose_->outputs()[1].bytes, stream);
        Person p = decode_simcc_to_openpose(
            simcc_x_host_.data(), pose_simcc_x_len_,
            simcc_y_host_.data(), pose_simcc_y_len_,
            pose_num_kp_, crop, pose_w_, pose_h_);
        new_persons.push_back(std::move(p));
    }

    {
        std::lock_guard<std::mutex> g(kp_mutex_);
        persons_ = std::move(new_persons);
    }

    auto t1 = std::chrono::steady_clock::now();
    const double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
    // Rolling mean (alpha=0.1) for the info CHOP.
    const float prev = infer_ms_.load();
    const float ema = prev <= 0.0f ? static_cast<float>(dt)
                                   : prev * 0.9f + static_cast<float>(dt) * 0.1f;
    infer_ms_.store(ema);
}

} // namespace dwpose_td
