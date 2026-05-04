// Forces dwpose_worker.dll (and its TRT/CUDA dependencies) to load from
// the directory of our plugin DLL, bypassing the host application's
// directory. Without this, TouchDesigner's bundled nvinfer_10.dll wins over
// the one we staged, causing engine deserialization version mismatches.
//
// Used in conjunction with /DELAYLOAD:dwpose_worker.dll on the consumer's
// link line. ensure_libraries_loaded() must be called before any
// dwpose_worker C ABI symbol is referenced.

#pragma once

#include <string>

namespace dwpose_td
{

// Returns true if dwpose_worker.dll is loaded (idempotent).
// On first call, loads it from the directory containing the calling module
// using LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32 so
// transitive imports (nvinfer, cudart, etc.) resolve from there too.
bool ensure_libraries_loaded(std::string* err_out);

} // namespace dwpose_td
