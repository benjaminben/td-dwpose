@echo off
REM Build helper for td-dwpose.
REM
REM Usage:
REM   build           - incremental rebuild
REM   build configure - run cmake configure (first time, or after CMakeLists changes)
REM   build clean     - wipe build dir, then configure + build
REM
REM Required env vars (or pass -D... to cmake):
REM   TENSORRT_ROOT   path to TensorRT install (must contain include/ and lib/)
REM
REM MSVC environment is auto-detected via vswhere.exe (ships with VS 2017+).

setlocal EnableDelayedExpansion

set HERE=%~dp0
if "%HERE:~-1%"=="\" set HERE=%HERE:~0,-1%
set BUILD=%HERE%\build

set CMD=%1
if "%CMD%"=="" set CMD=build
if /i "%CMD%"=="help" goto :usage
if /i "%CMD%"=="-h"   goto :usage

if defined VSCMD_VER goto :have_msvc
if defined VCVARS    goto :try_vcvars

set VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe
if not exist "!VSWHERE!" goto :probe_known_paths

for /f "usebackq tokens=*" %%i in (`"!VSWHERE!" -version "[17.0^,18.0)" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
    set VS_INSTALL=%%i
)
if defined VS_INSTALL (
    set VCVARS=!VS_INSTALL!\VC\Auxiliary\Build\vcvars64.bat
    goto :try_vcvars
)

:probe_known_paths
call :check_path "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call :check_path "C:\Program Files (x86)\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
call :check_path "C:\Program Files (x86)\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
call :check_path "C:\Program Files (x86)\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
call :check_path "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call :check_path "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
call :check_path "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
call :check_path "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
if defined VCVARS goto :try_vcvars
goto :no_vs_instance

:check_path
if defined VCVARS goto :eof
if exist "%~1" set VCVARS=%~1
goto :eof

:try_vcvars
if not exist "!VCVARS!" goto :no_vcvars
echo Loading MSVC environment from !VCVARS!...
call "!VCVARS!" >nul
goto :have_msvc

:no_vs_instance
echo ERROR: no Visual Studio 2022 (17.x) installation found.
echo Install VS 2022 with the Desktop development with C++ workload, or set
echo %%VCVARS%% manually before running.
exit /b 1

:no_vcvars
echo ERROR: vcvars64.bat not found at "!VCVARS!"
exit /b 1

:have_msvc
if /i "%CMD%"=="clean" (
    if exist "!BUILD!" rmdir /S /Q "!BUILD!"
    set CMD=configure-then-build
)

if /i "%CMD%"=="configure"            goto :do_configure
if /i "%CMD%"=="configure-then-build" goto :do_configure
if /i "%CMD%"=="build"                goto :do_build

echo Unknown command: %1
goto :usage

:do_configure
echo Configuring with cmake...
cmake -S "!HERE!" -B "!BUILD!" -G Ninja -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1
if /i "%CMD%"=="configure" exit /b
goto :do_build

:do_build
if not exist "!BUILD!\build.ninja" (
    echo build dir missing -- running configure first.
    goto :do_configure
)
echo Building...
cmake --build "!BUILD!"
if errorlevel 1 exit /b 1
echo.
echo Plugin DLL: !HERE!\plugin\td_dwpose_top.dll
exit /b

:usage
echo Usage: build [configure^|build^|clean]
echo.
echo   configure  cmake configure into .\build\
echo   build      incremental cmake --build (default)
echo   clean      wipe .\build\ and reconfigure + build
exit /b 1
