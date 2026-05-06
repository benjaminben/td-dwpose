#pragma once

#include "dwpose_post.hpp"
#include "dwpose_pre.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

namespace dwpose_td
{

class TrtRunner;

enum class Status : int
{
    Idle = 0,
    DownloadingOnnx = 1,
    BuildingEngine = 2,
    Ready = 3,
    Error = 4,
};

struct Keypoint { float x, y, score; };

class DWPoseRunner
{
public:
    DWPoseRunner();
    ~DWPoseRunner();

    DWPoseRunner(const DWPoseRunner&) = delete;
    DWPoseRunner& operator=(const DWPoseRunner&) = delete;

    // Non-blocking. If engines aren't ready, kicks the prepare thread (if not
    // already running) and returns; the next call will inherit the result.
    void runFrame(cudaArray_t in, int W, int H, ChannelOrder order,
                  cudaStream_t stream);

    // 0 disables the cap. When >0, the per-frame detection list is truncated
    // to the top-n bodies ranked by (bbox_area * detection_score). Atomic so
    // the TOP can update it from the cook thread while inferLocked() reads
    // it on the worker thread.
    void setMaxBodies(int n) { max_bodies_.store(n < 0 ? 0 : n); }

    // 0 disables. When >0, drops detections whose bbox shorter side is below
    // `px`. Applied before setMaxBodies's truncation so the cap counts only
    // viable candidates.
    void setMinBodyPx(int px) { min_body_px_.store(px < 0 ? 0 : px); }

    // Triggers a fresh re-check of the engines folder + reload if anything
    // changed. Reload pulse from the TOP routes here.
    void requestReload(const std::string& engines_dir);

    Status status() const { return status_.load(); }
    float progress() const { return progress_.load(); }
    int numPersons() const;
    float inferenceMs() const { return infer_ms_.load(); }
    const std::string& errorMessage() const { return last_error_; }

    // Thread-safe snapshot of one person's keypoints. personIdx must be in
    // [0, numPersons()). Returns empty vector if out of range.
    std::vector<Keypoint> keypoints(int personIdx) const;

    // Render the OpenPose stick figure for the latest snapshot into
    // `out` (RGBA8 cudaArray, W x H). See pose_render.hpp for the exact
    // drawing semantics. No-op if no engines loaded yet; canvas is still
    // cleared to black, which is the correct "no pose" conditioning.
    // `flags` is forwarded to render_pose; bit 0 = ORDERED_DRAW.
    void renderPose(cudaArray_t out, int W, int H,
                    int src_w, int src_h, float marker_scale,
                    cudaStream_t stream, unsigned int flags);

private:
    void prepareWorker(std::string engines_dir);
    bool loadEngines(const std::string& dir, std::string* err);
    void inferLocked(cudaArray_t in, int W, int H, ChannelOrder order,
                     cudaStream_t stream);

    std::atomic<Status> status_{Status::Idle};
    std::atomic<float> progress_{0.0f};
    std::atomic<float> infer_ms_{0.0f};
    std::atomic<int> max_bodies_{0};
    std::atomic<int> min_body_px_{0};
    std::string last_error_;
    std::string current_engines_dir_;

    std::thread prep_thread_;
    std::atomic<bool> prep_running_{false};

    std::unique_ptr<TrtRunner> det_;
    std::unique_ptr<TrtRunner> pose_;
    int det_w_ = 640, det_h_ = 640;
    int pose_w_ = 288, pose_h_ = 384;

    // Device buffers for pre/infer/post.
    void* det_in_dev_ = nullptr;       // 1*3*640*640 float
    void* det_out_dev_ = nullptr;      // 1*N*85 float
    int det_out_anchors_ = 0;
    void* pose_in_dev_ = nullptr;      // 1*3*pose_h*pose_w float
    void* pose_simcc_x_dev_ = nullptr; // 133*pose_w*2 float
    void* pose_simcc_y_dev_ = nullptr; // 133*pose_h*2 float
    int pose_simcc_x_len_ = 0;
    int pose_simcc_y_len_ = 0;
    int pose_num_kp_ = 133;

    std::vector<float> det_out_host_;
    std::vector<float> simcc_x_host_;
    std::vector<float> simcc_y_host_;

    mutable std::mutex kp_mutex_;
    std::vector<Person> persons_;

    std::chrono::steady_clock::time_point last_run_;
};

} // namespace dwpose_td
