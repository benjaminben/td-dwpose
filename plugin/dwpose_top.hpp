#pragma once

#include "TOP_CPlusPlusBase.h"

#include <cstdint>
#include <memory>
#include <string>

namespace dwpose_td
{

class DWPoseRunner;

inline constexpr const char* kParEnginesFolder = "Enginesfolder";
inline constexpr const char* kParReload = "Reload";
// A/B toggle between the controlnet_aux-matching ordered draw (one
// kernel launch per limb_idx / keypoint_idx, correct cross-limb depth)
// and the legacy single-batch dispatch (faster, non-deterministic
// z-order on cross-person overlap).
inline constexpr const char* kParOrderedDraw = "Ordereddraw";
// Cap on bodies kept per frame after YOLOX detection. 0 = unlimited; >0
// keeps the top-n ranked by (bbox_area * detection_score), so the most
// prominent on-camera subjects survive even when score-only ranking would
// pick a tiny background person.
inline constexpr const char* kParMaxBodies = "Maxbodies";
// Drop detections whose bbox shorter side is below this many pixels.
// Pose keypoints from sub-threshold crops are noise; default 40 is the
// floor below which DWPose stops being reliable. 0 disables.
inline constexpr const char* kParMinBodyPx = "Minbodypx";
// Downstream SD target resolution (square). The renderer auto-scales marker
// sizes so a 4-px-at-512 marker survives the downstream resize-to-target.
inline constexpr const char* kParTargetRes = "Targetres";
// How the dwpose canvas maps to the SD target downstream:
//   "contain" -> long edge fits target (letterbox; max(W,H)/target)
//   "fill"    -> short edge fills target (cover/crop; min(W,H)/target)
inline constexpr const char* kParScalingMode = "Scalingmode";
// User multiplier on top of the auto-scaled marker size. 1.0 = auto only.
inline constexpr const char* kParMarkerScale = "Markerscale";

class DWPoseTOP : public TD::TOP_CPlusPlusBase
{
public:
    DWPoseTOP(const TD::OP_NodeInfo*, TD::TOP_Context* context);
    ~DWPoseTOP() override;

    void getGeneralInfo(TD::TOP_GeneralInfo*, const TD::OP_Inputs*, void*) override;
    void execute(TD::TOP_Output*, const TD::OP_Inputs*, void*) override;
    void setupParameters(TD::OP_ParameterManager*, void*) override;
    void pulsePressed(const char* name, void*) override;

    int32_t getNumInfoCHOPChans(void*) override;
    void getInfoCHOPChan(int32_t index, TD::OP_InfoCHOPChan*, void*) override;

private:
    TD::TOP_Context* myContext;

    std::unique_ptr<DWPoseRunner> myRunner;

    std::string myLastEnginesFolder;
    bool myReloadRequested = false;
    int myDiagCount = 0;

    // Snapshot of the most recent execute()'s render-pose wall time and
    // the toggle state, for the info CHOP. Populated inside execute()
    // each cook; read by getInfoCHOPChan(). Float ms keeps the CHOP
    // schema stable.
    float myLastRenderMs = 0.0f;
    int myLastOrdered = 1;  // matches the Toggle default (ON)
    float myLastMarkerScale = 1.0f;
};

} // namespace dwpose_td
