#include "dwpose_top.hpp"

#include "dwpose_runner.hpp"
#include "td_debug_log.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

using namespace TD;

namespace dwpose_td
{

namespace
{

ChannelOrder order_for(OP_PixelFormat fmt)
{
    // Resolved per-frame from the input texture format; hardcoding either
    // side causes a silent R/B channel swap when the input arrives in the
    // opposite layout.
    return (fmt == OP_PixelFormat::BGRA8Fixed) ? ChannelOrder::BGRA
                                               : ChannelOrder::RGBA;
}

} // namespace

DWPoseTOP::DWPoseTOP(const OP_NodeInfo*, TOP_Context* context)
  : myContext{context}
  , myRunner{std::make_unique<DWPoseRunner>()}
{
}

DWPoseTOP::~DWPoseTOP() = default;

void DWPoseTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs*, void*)
{
    ginfo->cookEveryFrameIfAsked = false;
    ginfo->inputSizeIndex = 0;
}

void DWPoseTOP::pulsePressed(const char* name, void*)
{
    if(name && std::strcmp(name, kParReload) == 0) myReloadRequested = true;
}

void DWPoseTOP::execute(TOP_Output* output, const OP_Inputs* inputs, void*)
{
    const OP_TOPInput* top = inputs->getInputTOP(0);
    if(!top) return;

    const int width = top->textureDesc.width;
    const int height = top->textureDesc.height;

    const char* fp = inputs->getParFilePath(kParEnginesFolder);
    const char* sp = inputs->getParString(kParEnginesFolder);
    const std::string folder = (fp && *fp) ? fp : (sp ? sp : "");

    // Kick the prepare worker on first sight of a folder + on Reload pulse +
    // on folder change.
    if(myRunner)
    {
        const bool changed = (folder != myLastEnginesFolder);
        if(myReloadRequested || (changed && !folder.empty()))
        {
            myRunner->requestReload(folder);
            myReloadRequested = false;
            myLastEnginesFolder = folder;
        }
        else if(folder.empty() && myRunner->status() == Status::Idle)
        {
            // Surface "no folder set" via the info CHOP rather than silently doing
            // nothing. Triggers requestReload which immediately errors out.
            myRunner->requestReload(folder);
            myLastEnginesFolder = folder;
        }
    }

    cudaStream_t stream = nullptr;

    OP_CUDAAcquireInfo acquire;
    acquire.stream = stream;
    const OP_CUDAArrayInfo* inArray = top->getCUDAArray(acquire, nullptr);
    if(!inArray) return;

    TOP_CUDAOutputInfo outInfo;
    outInfo.stream = stream;
    // Force RGBA8 output: the renderer writes fresh pixels (a stick
    // figure on black), not a passthrough of the input format.
    outInfo.textureDesc = inArray->textureDesc;
    outInfo.textureDesc.pixelFormat = OP_PixelFormat::RGBA8Fixed;
    outInfo.colorBufferIndex = 0;
    const OP_CUDAArrayInfo* outArray = output->createCUDAArray(outInfo, nullptr);
    if(!outArray) return;

    // TD requires that all OP_Inputs reads happen BEFORE beginCUDAOperations;
    // the inputs object becomes invalid once we enter the CUDA section.
    const int ordered = inputs->getParInt(kParOrderedDraw);
    myLastOrdered = ordered;
    const unsigned int flags =
        ordered ? RENDER_FLAG_ORDERED_DRAW : 0u;

    // Marker auto-scale: project the canvas (W,H) onto the SD target so
    // base 4-px-at-512 markers end up ~4 px after downstream resize. Mode 0
    // (Contain / letterbox) uses the long edge, mode 1 (Fill / cover) uses
    // the short edge. Floor target at 1 to avoid divide-by-zero if a user
    // manually clears the field.
    const int target_res_raw = inputs->getParInt(kParTargetRes);
    const int target_res = target_res_raw > 0 ? target_res_raw : 512;
    const int mode = inputs->getParInt(kParScalingMode);  // 0=contain, 1=fill
    const int ref_dim = (mode == 1) ? std::min(width, height)
                                    : std::max(width, height);
    const double user_scale = inputs->getParDouble(kParMarkerScale);
    const float marker_scale =
        static_cast<float>(ref_dim) / static_cast<float>(target_res)
        * static_cast<float>(user_scale);
    myLastMarkerScale = marker_scale;

    if(myRunner)
    {
        myRunner->setMaxBodies(inputs->getParInt(kParMaxBodies));
        myRunner->setMinBodyPx(inputs->getParInt(kParMinBodyPx));
    }

    if(!myContext->beginCUDAOperations(nullptr)) return;

    do
    {
        if(!inArray->cudaArray || !outArray->cudaArray) break;

        const ChannelOrder order = order_for(inArray->textureDesc.pixelFormat);

        // Run detector + pose if engines ready (no-op otherwise).
        if(myRunner) myRunner->runFrame(inArray->cudaArray, width, height, order, stream);

        // Render the stick figure directly into the output cudaArray.
        // The renderer fills black first, so when no engines are loaded /
        // no persons detected we still get a valid black RGBA8 frame
        // (which is the correct "no pose" SD ControlNet conditioning).

        if(myRunner)
        {
            // Time the render dispatch end-to-end. The explicit
            // cudaStreamSynchronize gives a host-visible cost suitable for
            // the OrderedDraw A/B comparison, at the cost of preventing
            // the render from pipelining with downstream work. Drop the
            // sync once perf measurement isn't needed.
            const auto t0 = std::chrono::steady_clock::now();
            myRunner->renderPose(outArray->cudaArray, width, height,
                                 width, height, marker_scale, stream, flags);
            cudaStreamSynchronize(stream);
            const auto t1 = std::chrono::steady_clock::now();
            myLastRenderMs = static_cast<float>(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }

        ++myDiagCount;
    } while(false);

    myContext->endCUDAOperations(nullptr);
}

int32_t DWPoseTOP::getNumInfoCHOPChans(void*)
{
    // status, progress, num_persons, infer_ms, ordered, lastrender_ms,
    // marker_scale, plus 18 body keypoints x 2 (xy) for person 0.
    return 7 + OPENPOSE_BODY_COUNT * 2;
}

void DWPoseTOP::getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void*)
{
    if(!chan || !chan->name) return;
    if(!myRunner) return;
    auto kps = myRunner->keypoints(0);
    switch(index)
    {
        case 0: chan->name->setString("status");
                chan->value = static_cast<float>(static_cast<int>(myRunner->status()));
                return;
        case 1: chan->name->setString("progress");
                chan->value = myRunner->progress(); return;
        case 2: chan->name->setString("num_persons");
                chan->value = static_cast<float>(myRunner->numPersons()); return;
        case 3: chan->name->setString("infer_ms");
                chan->value = myRunner->inferenceMs(); return;
        case 4: chan->name->setString("ordered");
                chan->value = static_cast<float>(myLastOrdered); return;
        case 5: chan->name->setString("lastrender_ms");
                chan->value = myLastRenderMs; return;
        case 6: chan->name->setString("marker_scale");
                chan->value = myLastMarkerScale; return;
    }
    const int kp_index = (index - 7) / 2;
    const int axis = (index - 7) % 2;
    if(kp_index < 0 || kp_index >= OPENPOSE_BODY_COUNT) return;
    char name[16];
    std::snprintf(name, sizeof(name), "kp%02d%c", kp_index, axis == 0 ? 'x' : 'y');
    chan->name->setString(name);
    if(static_cast<int>(kps.size()) > kp_index)
    {
        const auto& k = kps[kp_index];
        // Confidence-gated emission: the renderer skips drawing keypoints
        // below this threshold; mirror that gate here so the CHOP stays
        // consistent with what the TOP draws.
        if(k.score >= DRAW_CONF_THRESHOLD)
            chan->value = (axis == 0) ? k.x : k.y;
        else
            chan->value = 0.0f;
    }
    else
    {
        chan->value = 0.0f;
    }
}

void DWPoseTOP::setupParameters(OP_ParameterManager* manager, void*)
{
    {
        OP_StringParameter sp;
        sp.name = kParEnginesFolder;
        sp.label = "Engines Folder";
        sp.page = "DWPose";
        sp.defaultValue = "";
        manager->appendFolder(sp);
    }
    {
        OP_NumericParameter np;
        np.name = kParReload;
        np.label = "Reload";
        np.page = "DWPose";
        manager->appendPulse(np);
    }
    {
        // A/B toggle for the ordered-vs-batched render dispatch path.
        // ON  = controlnet_aux's outer-by-limb draw order (correct
        //       cross-limb depth in multi-person scenes)
        // OFF = single-batch CUDA dispatch (faster; non-deterministic
        //       z-order on cross-person overlap)
        // The lastrender_ms info CHOP channel makes the perf delta
        // visible live.
        OP_NumericParameter np;
        np.name = kParOrderedDraw;
        np.label = "Ordered Draw";
        np.page = "DWPose";
        // Default OFF: measured 4K / 4-bodies cost was >2ms with ordered ON,
        // and the visual gain (correct cross-limb depth illusion) is invisible
        // in single-person streaming (the dominant case). Users with multi-
        // person scenes that need accurate arm overlap can flip this on.
        np.defaultValues[0] = 0.0;
        manager->appendToggle(np);
    }
    {
        // Cap detected bodies per frame. 0 disables; values >0 keep the
        // top-n ranked by bbox_area * detection_score (favors big +
        // confident subjects over tiny background ones). Live-tunable.
        OP_NumericParameter np;
        np.name = kParMaxBodies;
        np.label = "Max Bodies";
        np.page = "DWPose";
        np.defaultValues[0] = 0.0;
        np.minValues[0] = 0.0;
        np.minSliders[0] = 0.0;
        np.maxSliders[0] = 8.0;
        np.clampMins[0] = true;
        manager->appendInt(np);
    }
    {
        // Drop detections whose bbox shorter side is below this many pixels.
        // Default 40: empirically the floor below which DWPose keypoints
        // become noise (the 288x384 pose crop is upsampled too aggressively).
        // 0 disables. Live-tunable.
        OP_NumericParameter np;
        np.name = kParMinBodyPx;
        np.label = "Min Body Px";
        np.page = "DWPose";
        np.defaultValues[0] = 40.0;
        np.minValues[0] = 0.0;
        np.minSliders[0] = 0.0;
        np.maxSliders[0] = 200.0;
        np.clampMins[0] = true;
        manager->appendInt(np);
    }
    {
        // Downstream SD target resolution (square). The plugin auto-scales
        // marker sizes by ref_dim/target so a 4-px-at-512 marker survives
        // the downstream resize. Default 512 covers the SD1.5 / SD-Turbo case;
        // bump to 768/1024 for SDXL-class targets.
        OP_NumericParameter np;
        np.name = kParTargetRes;
        np.label = "Target Resolution";
        np.page = "DWPose";
        np.defaultValues[0] = 512.0;
        np.minValues[0] = 1.0;
        np.minSliders[0] = 256.0;
        np.maxSliders[0] = 2048.0;
        np.clampMins[0] = true;
        manager->appendInt(np);
    }
    {
        // Contain (letterbox-fit, long edge maps to target) vs Fill (cover,
        // short edge maps to target). Pick the one that matches the
        // downstream resize chain feeding the SD ControlNet.
        OP_StringParameter sp;
        sp.name = kParScalingMode;
        sp.label = "Scaling Mode";
        sp.page = "DWPose";
        sp.defaultValue = "contain";
        const char* names[] = {"contain", "fill"};
        const char* labels[] = {"Contain", "Fill"};
        manager->appendMenu(sp, 2, names, labels);
    }
    {
        // User multiplier on top of the auto-scaled marker size. 1.0 = auto
        // only; >1 thickens markers, <1 thins them. Useful when the SD model
        // was trained on a different marker thickness than 4-px-at-512.
        OP_NumericParameter np;
        np.name = kParMarkerScale;
        np.label = "Marker Scale";
        np.page = "DWPose";
        np.defaultValues[0] = 1.0;
        np.minValues[0] = 0.05;
        np.minSliders[0] = 0.25;
        np.maxSliders[0] = 4.0;
        np.clampMins[0] = true;
        manager->appendFloat(np);
    }
}

} // namespace dwpose_td
