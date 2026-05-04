// TouchDesigner C++ TOP entry points. The class implementation lives in
// dwpose_top.cpp; this file is just the C ABI exports TD calls.

#include "TOP_CPlusPlusBase.h"

#include "dwpose_top.hpp"

using namespace TD;

extern "C"
{

DLLEXPORT void FillTOPPluginInfo(TOP_PluginInfo* info)
{
    if(!info->setAPIVersion(TOPCPlusPlusAPIVersion))
        return;
    info->executeMode = TOP_ExecuteMode::CUDA;

    OP_CustomOPInfo& customInfo = info->customOPInfo;
    customInfo.opType->setString("Dwpose");
    customInfo.opLabel->setString("DWPose");
    customInfo.authorName->setString("github.com/benjaminben");
    customInfo.authorEmail->setString("");

    customInfo.minInputs = 1;
    customInfo.maxInputs = 1;
}

DLLEXPORT TOP_CPlusPlusBase* CreateTOPInstance(const OP_NodeInfo* info, TOP_Context* context)
{
    return new dwpose_td::DWPoseTOP(info, context);
}

DLLEXPORT void DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context*)
{
    delete static_cast<dwpose_td::DWPoseTOP*>(instance);
}

} // extern "C"
