# third_party

Vendored third-party headers and libraries used by the build.

## Contents

- `derivative/` — TouchDesigner Custom Operator SDK headers
  (`TOP_CPlusPlusBase.h`, `CPlusPlus_Common.h`). Property of Derivative
  Inc., redistributed under their Shared Use License. See
  `derivative/README.md` for details and upgrade notes.

- `libcurl/` — *(optional, not committed)* place a Windows x64 release of
  libcurl + headers here if you want HTTPS-enabled in-plugin model
  download without going through vcpkg. The CMake build will pick up
  `third_party/libcurl/include/curl/curl.h` + `third_party/libcurl/lib/`
  automatically. `stage.cmd` will copy the libcurl DLL into `plugin/`.

## Why vendor?

- Reproducible builds without external package managers.
- The plugin DLL load path is locked to its own folder via
  `LoadLibraryEx + LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR`; vendoring keeps
  deps alongside the plugin DLL where the loader looks.

## Licensing

- TouchDesigner SDK — Derivative Shared Use License (re-distributable
  with attribution; see `derivative/README.md`).
- libcurl — MIT/X derivative; bundle COPYING and acknowledge in the
  top-level README if you ship the libcurl DLL.
