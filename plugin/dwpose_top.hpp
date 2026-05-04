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
};

} // namespace dwpose_td
