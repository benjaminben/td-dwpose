#include "dwpose_top.hpp"

#include "dwpose_runner.hpp"
#include "td_debug_log.hpp"

#include <cuda_runtime.h>

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
                                 width, height, stream, flags);
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
    // plus 18 body keypoints x 2 (xy) for person 0.
    return 6 + OPENPOSE_BODY_COUNT * 2;
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
    }
    const int kp_index = (index - 6) / 2;
    const int axis = (index - 6) % 2;
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
}

} // namespace dwpose_td
