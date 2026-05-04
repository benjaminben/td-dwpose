cmake_minimum_required(VERSION 3.18)
project(td_dwpose CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- TD plugin output goes into ./plugin/ next to td_top_plugin.cpp.
# Both the TOP DLL and the worker DLL land here so that the shim's
# LoadLibraryEx(LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR) picks dwpose_worker.dll
# from the same directory as td_dwpose_top.dll, and the worker's transitive
# imports (nvinfer_10.dll etc.) resolve from there too.
set(PLUGIN_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/plugin)
foreach(cfg "" _DEBUG _RELEASE _MINSIZEREL _RELWITHDEBINFO)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY${cfg} ${PLUGIN_OUTPUT_DIR})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY${cfg} ${PLUGIN_OUTPUT_DIR})
endforeach()

# ---- CUDA runtime + language for the pre/post .cu kernels.
find_package(CUDAToolkit REQUIRED)
# RTX 4090 = sm_89 (Ada Lovelace). Match the export_dwpose.py target. Wider
# arch list keeps the plugin usable on Turing through Blackwell with a PTX
# fallback for any future arch CUDA 13's nvcc does not yet know.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90 100 120 120-virtual)
endif()
enable_language(CUDA)

# ---- TensorRT 10. TENSORRT_ROOT must contain include/ and lib/.
set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}"
    CACHE PATH "TensorRT install root (must contain include/ and lib/)")
if(NOT TENSORRT_ROOT OR NOT EXISTS "${TENSORRT_ROOT}/include/NvInfer.h")
    message(FATAL_ERROR
        "TENSORRT_ROOT not set or invalid (need ${TENSORRT_ROOT}/include/NvInfer.h).\n"
        "Set %TENSORRT_ROOT% to the TRT install root, e.g. C:\\src\\TensorRT-10.16.1.11")
endif()
set(TENSORRT_INCLUDE "${TENSORRT_ROOT}/include")
find_library(TRT_NVINFER nvinfer_10 PATHS "${TENSORRT_ROOT}/lib" REQUIRED)
find_library(TRT_NVONNX  nvonnxparser_10 PATHS "${TENSORRT_ROOT}/lib" REQUIRED)

# ---- libcurl (optional). Try find_package first (vcpkg / system). If absent,
# fall back to a vendored third_party/libcurl/ build IF that directory exists.
# If neither is available we build without curl: HF in-TD download is stubbed
# and the user must stage the ONNX files into their engines folder manually
# (see README).
set(TD_DWPOSE_CURL_AVAILABLE FALSE)
find_package(CURL QUIET)
if(CURL_FOUND)
    set(TD_DWPOSE_CURL_AVAILABLE TRUE)
else()
    set(CURL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libcurl"
        CACHE PATH "Vendored libcurl root (override or vcpkg-install curl)")
    if(EXISTS "${CURL_ROOT}/include/curl/curl.h")
        find_path(CURL_INCLUDE_DIR curl/curl.h PATHS "${CURL_ROOT}/include")
        find_library(CURL_LIBRARY  NAMES libcurl curl PATHS "${CURL_ROOT}/lib")
        if(CURL_INCLUDE_DIR AND CURL_LIBRARY)
            set(CURL_LIBRARIES ${CURL_LIBRARY})
            set(CURL_INCLUDE_DIRS ${CURL_INCLUDE_DIR})
            set(TD_DWPOSE_CURL_AVAILABLE TRUE)
        endif()
    endif()
endif()
if(NOT TD_DWPOSE_CURL_AVAILABLE)
    message(STATUS "libcurl not found - HF download stubbed; ONNX files must be staged into the engines folder manually")
endif()

# =============================================================================
# Target 1: dwpose_worker.dll  (SHARED)
# Owns ALL nvinfer / nvonnxparser / cudart / curl linkage. The TOP never
# touches these symbols directly -- it goes through the C ABI in
# dwpose_worker_c.h, which is delay-loaded.
# =============================================================================
add_library(dwpose_worker SHARED
    runner/dwpose_runner_impl.cpp
    runner/dwpose_runner_impl.hpp
    runner/dwpose_pre.cu
    runner/dwpose_pre.hpp
    runner/dwpose_post.cpp
    runner/dwpose_post.hpp
    runner/pose_render.cu
    runner/pose_render.hpp
    runner/pose_render_internal.hpp
    runner/pose_render_kernels.hpp
    runner/pose_render_hand.cu
    runner/pose_render_face.cu
    runner/pose_render_tables.cpp
    runner/pose_render_tables_body.cpp
    runner/pose_render_dispatch.cpp
    runner/trt_runner.cpp
    runner/trt_runner.hpp
    runner/hf_download.cpp
    runner/hf_download.hpp
    runner/engine_builder.cpp
    runner/engine_builder.hpp
    runner/dwpose_worker_c.cpp
    runner/dwpose_worker_c.h
    runner/td_debug_log.hpp
)
set_target_properties(dwpose_worker PROPERTIES
    PREFIX ""
    SUFFIX ".dll"
)
# DWPOSE_WORKER_BUILDING is what flips DWPOSE_API to dllexport in the C ABI
# header. Anything that does NOT define this macro (i.e. the shim) sees
# dllimport instead and can link against the worker's import lib.
target_compile_definitions(dwpose_worker PRIVATE DWPOSE_WORKER_BUILDING)
target_include_directories(dwpose_worker PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/runner
    ${TENSORRT_INCLUDE}
)
target_link_libraries(dwpose_worker PRIVATE
    CUDA::cudart
    ${TRT_NVINFER}
    ${TRT_NVONNX}
)
if(TD_DWPOSE_CURL_AVAILABLE)
    target_compile_definitions(dwpose_worker PRIVATE TD_DWPOSE_HAS_CURL=1)
    if(TARGET CURL::libcurl)
        target_link_libraries(dwpose_worker PRIVATE CURL::libcurl)
    else()
        target_include_directories(dwpose_worker PRIVATE ${CURL_INCLUDE_DIRS})
        target_link_libraries(dwpose_worker PRIVATE ${CURL_LIBRARIES})
    endif()
endif()

# =============================================================================
# Target 2: dwpose_runner_shim  (STATIC)
# TOP-side glue. Implements dwpose_td::DWPoseRunner as a thin facade over
# the worker's C ABI. Links the worker's IMPORT lib so the shim's calls
# resolve at link time but are deferred to runtime via /DELAYLOAD.
# =============================================================================
add_library(dwpose_runner_shim STATIC
    runner/dwpose_runner.hpp
    runner/dwpose_runner_shim.cpp
    runner/dwpose_worker_c.h
    runner/dll_loader.cpp
    runner/dll_loader.hpp
    runner/td_debug_log.hpp
)
target_include_directories(dwpose_runner_shim PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/runner
)
# CUDA::cudart is linked PUBLIC for its include dirs only. The shim header
# exposes cudaArray_t / cudaStream_t (opaque pointers from cuda_runtime.h)
# so it needs the CUDA headers to compile, even though it never calls a
# CUDA function. The TOP separately links CUDA::cudart so the import lib
# being PUBLIC-propagated through here is harmless / already present.
target_link_libraries(dwpose_runner_shim PUBLIC
    dwpose_worker   # import lib only -- runtime resolution is delayed
    CUDA::cudart    # headers only (cudaArray_t / cudaStream_t)
)

# =============================================================================
# Target 3: td_dwpose_top.dll  (MODULE)
# TouchDesigner plugin entry point. Knows nothing about TRT / nvinfer.
# =============================================================================
add_library(td_dwpose_top MODULE
    plugin/td_top_plugin.cpp
    plugin/dwpose_top.cpp
    plugin/dwpose_top.hpp
)
set_target_properties(td_dwpose_top PROPERTIES
    PREFIX ""
    SUFFIX ".dll"
)
target_include_directories(td_dwpose_top PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/derivative
)
target_link_libraries(td_dwpose_top PRIVATE
    dwpose_runner_shim
    CUDA::cudart
)

if(MSVC)
    # Delay-load dwpose_worker.dll so its imports (nvinfer, nvonnxparser,
    # ...) are NOT resolved at plugin load time. ensure_libraries_loaded()
    # in the shim pre-loads dwpose_worker.dll with
    # LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32
    # so its deps resolve from our plugin folder, NOT from
    # TouchDesigner.exe's directory (which has the wrong nvinfer version).
    target_link_options(td_dwpose_top PRIVATE
        "/DELAYLOAD:dwpose_worker.dll"
    )
    target_link_libraries(td_dwpose_top PRIVATE delayimp.lib)

    # Gate MSVC-style flags to CXX so nvcc does not parse them as input paths.
    target_compile_options(td_dwpose_top PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
    )
    target_compile_options(dwpose_runner_shim PRIVATE /W3 /permissive-)
    target_compile_options(dwpose_worker PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W3>
        $<$<COMPILE_LANGUAGE:CXX>:/permissive->
    )
endif()

message(STATUS "td-dwpose will build:")
message(STATUS "  ${PLUGIN_OUTPUT_DIR}/td_dwpose_top.dll  (TD plugin entry)")
message(STATUS "  ${PLUGIN_OUTPUT_DIR}/dwpose_worker.dll  (TRT/CUDA worker)")
message(STATUS "  TENSORRT_ROOT = ${TENSORRT_ROOT}")
if(TD_DWPOSE_CURL_AVAILABLE)
    message(STATUS "  CURL          = ${CURL_LIBRARIES} (HF download enabled)")
else()
    message(STATUS "  CURL          = <none> (HF download stubbed)")
endif()
