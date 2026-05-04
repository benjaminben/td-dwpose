@echo off
REM Stage runtime DLLs into ./plugin/ so they resolve from the same directory
REM as td_dwpose_top.dll when TD loads the plugin via LoadLibraryEx +
REM LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR.
REM
REM Required environment variables (or pass positionally):
REM
REM   CUDA_BIN         Directory containing the CUDA runtime DLLs
REM                    (cudart64_13.dll). On CUDA 13+ these live under bin\x64.
REM                    Example: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64
REM
REM   TENSORRT_ROOT    Path to the TensorRT install root.
REM                    The bin\ subfolder must contain nvinfer_10.dll etc.
REM                    Example: C:\src\TensorRT-10.16.1.11
REM
REM Optional:
REM   CURL_BIN         Directory containing libcurl.dll (vcpkg installed to
REM                    %VCPKG_ROOT%\installed\x64-windows\bin, or your own
REM                    third_party/libcurl/bin). If unset and the runtime is
REM                    statically-linked curl, leave blank.
REM
REM Usage: stage [<cuda-bin> [<tensorrt-root> [<curl-bin>]]]

setlocal

set HERE=%~dp0
if "%HERE:~-1%"=="\" set HERE=%HERE:~0,-1%
set TARGET=%HERE%\plugin

if not "%~1"=="" set CUDA_BIN=%~1
if not "%~2"=="" set TENSORRT_ROOT=%~2
if not "%~3"=="" set CURL_BIN=%~3

set MISSING=
if "%CUDA_BIN%"=="" set MISSING=%MISSING% CUDA_BIN
if "%TENSORRT_ROOT%"=="" set MISSING=%MISSING% TENSORRT_ROOT

if not "%MISSING%"=="" (
    echo.
    echo ERROR: required environment variables not set:%MISSING%
    echo.
    echo Set them in your shell, e.g.:
    echo   set CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64
    echo   set TENSORRT_ROOT=C:\src\TensorRT-10.16.1.11
    echo   set CURL_BIN=C:\vcpkg\installed\x64-windows\bin     [optional]
    echo.
    echo Or pass them positionally: stage ^<cuda-bin^> ^<trt-root^> [^<curl-bin^>]
    exit /b 1
)

echo === Sources ===
echo CUDA_BIN      = %CUDA_BIN%
echo TENSORRT_ROOT = %TENSORRT_ROOT%
echo CURL_BIN      = %CURL_BIN%
echo TARGET        = %TARGET%
echo.

if not exist "%TARGET%" (
    echo ERROR: plugin folder does not exist: %TARGET%
    echo Run `build` first.
    exit /b 1
)

REM dwpose_worker.dll is built straight into %TARGET% by CMake (see
REM PLUGIN_OUTPUT_DIR in CMakeLists.txt) so no extra copy step is needed.
REM Just sanity-check it is there before staging the rest of the runtime.
if exist "%TARGET%\dwpose_worker.dll" (
    echo OK     dwpose_worker.dll  (built by CMake)
) else (
    echo MISSING %TARGET%\dwpose_worker.dll  -- run `build.cmd build` first
)

echo --- CUDA runtime DLLs ---
call :copyone "%CUDA_BIN%\cudart64_13.dll"

echo.
echo --- TensorRT *.dll from %TENSORRT_ROOT%\bin ---
if exist "%TENSORRT_ROOT%\bin" (
    copy /Y "%TENSORRT_ROOT%\bin\*.dll" "%TARGET%\" >nul && echo OK     TensorRT bin\*.dll
) else (
    echo MISSING: %TENSORRT_ROOT%\bin
)

if not "%CURL_BIN%"=="" (
    echo.
    echo --- libcurl from %CURL_BIN% ---
    call :copyone "%CURL_BIN%\libcurl.dll"
    call :copyone "%CURL_BIN%\zlib1.dll"
)

echo.
echo === Plugin folder DLLs ===
dir /B "%TARGET%\*.dll"
exit /b

:copyone
if exist %1 (
    copy /Y %1 "%TARGET%\" >nul && echo OK     %~nx1 || echo FAILED %~nx1
) else (
    echo MISSING %~nx1   (looked at %~1)
)
goto :eof
