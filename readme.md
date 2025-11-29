<div align="center">
    <img src="./images/Logo.png" alt="Logo"> <br>
    <a href="https://deepwiki.com/TianZerL/Anime4KCPP">
        <img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki">
    </a>
</div>

# Anime4KCPP v3
Anime4KCPP is a high performance anime upscaler.

Anime4KCPP v3 uses CNN based algorithm, and aims to be simple and efficient.

# Playground
Use [Anime4KCPP Playground](https://tianzerl.github.io/Anime4KCPP-Playground/) to upscale images in your browser with WebAssembly.

***For the Microsoft Edge browser, to achieve optimal performance, you need to disable the Enhanced Security for the site.***

# Build
## Dependency
Build tools:
- CMake
- A C++17 compatible compiler

Dependency handling:
- If you have internet access, CMake will automatically download and configure most required dependencies.
- You can also manually download dependencies.

Manual configuration (optional):
- Most dependencies are located via `find_package`. Others may use `pkg-config` or require explicit path specification via CMake variables.
- For certain dependencies, dedicated CMake variables (e.g., `AC_PATH_XXX`) are provided.
- Setting these variables will:
    1. Direct CMake to search in your specified paths first
    2. Override default search locations

***List of dependencies***

| Dependency                                                                                                                           | CMake option                | Module              | Acquisition | Manual Configuration                 |
| ------------------------------------------------------------------------------------------------------------------------------------ | --------------------------- | ------------------- | ----------- | ------------------------------------ |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)                                                                            | AC_CORE_WITH_CUDA           | core(CUDA)          | Manual      | find_package                         |
| [libavcodec](https://ffmpeg.org)                                                                                                     | AC_BUILD_VIDEO              | video               | Manual      | pkg-config / AC_PATH_FFMPEG          |
| [libavformat](https://ffmpeg.org)                                                                                                    | AC_BUILD_VIDEO              | video               | Manual      | pkg-config / AC_PATH_FFMPEG          |
| [libavutil](https://ffmpeg.org)                                                                                                      | AC_BUILD_VIDEO              | video               | Manual      | pkg-config / AC_PATH_FFMPEG          |
| [libswscale](https://ffmpeg.org)                                                                                                     | AC_BUILD_VIDEO              | video               | Manual      | pkg-config / AC_PATH_FFMPEG          |
| [Qt](https://www.qt.io)                                                                                                              | AC_BUILD_GUI                | gui                 | Manual      | find_package                         |
| [Avisynth SDK](https://github.com/AviSynth/AviSynthPlus/tree/master/avs_core/include)                                                | AC_BUILD_FILTER_AVISYNTH    | filter(avisynth)    | Automatic   | AC_PATH_AVISYNTH_SDK                 |
| [CLI11](https://github.com/CLIUtils/CLI11)                                                                                           | AC_BUILD_CLI                | cli                 | Automatic   | find_package                         |
| [DirectShow BaseClasses](https://github.com/microsoft/Windows-classic-samples/Samples/Win7Samples/multimedia/directshow/baseclasses) | AC_BUILD_FILTER_DIRECTSHOW  | filter(directshow)  | Automatic   | AC_PATH_DIRECTSHOW_BASECLASSES       |
| [Eigen3](https://gitlab.com/libeigen/eigen)                                                                                          | AC_CORE_WITH_EIGEN3         | core(eigen3)        | Automatic   | find_package                         |
| [OpenCL SDK](https://github.com/KhronosGroup/OpenCL-SDK)                                                                             | AC_CORE_WITH_OPENCL         | core(opencl)        | Automatic   | find_package                         |
| [pybind11](https://github.com/pybind/pybind11)                                                                                       | AC_BUILD_BINDING_PYTHON     | binding(python)     | Automatic   | find_package                         |
| [ruapu](https://github.com/nihui/ruapu)                                                                                              | N/A                         | core                | Automatic   | AC_PATH_RUAPU                        |
| [stb](https://github.com/nothings/stb)                                                                                               | N/A                         | core                | Automatic   | AC_PATH_STB                          |
| [VapourSynth SDK](https://github.com/vapoursynth/vapoursynth/tree/master/include)                                                    | AC_BUILD_FILTER_VAPOURSYNTH | filter(vapoursynth) | Automatic   | pkg-config / AC_PATH_VAPOURSYNTH_SDK |

- The minimum tested version of the CUDA Toolkit is 11.
- The minimum version of FFmpeg libraries is FFmpeg 4.
- Both Qt5 and Qt6 should be OK.
- VapourSynth SDK 4 is required.
- For non MSVC compilers, [a modified version of the DirectShow BaseClasses](https://github.com/TianZerL/DirectShow-BaseClasses-MultiCompiler) will be used.

## Platform
### Windows
Tested with MinGW-w64, Clang and MSVC.

**DirectShow filter is only available on Windows, tested compilers include MinGW-w64, ClangCL and MSVC.**

*Build with MinGW-w64:*
```powershell
mkdir build; cd build
cmake -G "MinGW Makefiles" .. -DAC_CORE_WITH_OPENCL=ON -DAC_ENABLE_STATIC_CRT=ON
cmake --build . --config Release -j8
cd bin
./ac_cli -v
```

*Build with MSVC:*
```powershell
mkdir build; cd build
cmake -G "Visual Studio 17 2022" .. -DAC_CORE_WITH_OPENCL=ON
cmake --build . --config Release -j8
cd bin/Release/
./ac_cli -v
```

To setup FFmpeg libraries for building video module on Windows, it is recommended to add an `AC_PATH_FFMPEG` variable to CMake, but you can also use `pkg-config` for Windows. `AC_PATH_FFMPEG` should be a path to the FFmpeg's root folder witch contains `lib` and `include`.

To add `AC_PATH_FFMPEG` to CMake, click `Add Entry` button in `cmake-gui` or use `-DAC_PATH_FFMPEG="path/to/ffmpeg/root"` in terminal.

You can download FFmpeg with sdk from [BtBN](https://github.com/BtbN/FFmpeg-Builds/releases) (`ffmpeg-master-latest-win64-gpl-shared.zip` or `ffmpeg-master-latest-win64-lgpl-shared.zip`) or [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (`ffmpeg-release-full-shared.7z`) for Windows.

### Linux
Tested with gcc and Clang.

```shell
# Toolchain
sudo apt install cmake build-essential
# For video module:
sudo apt install pkg-config libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
# For GUI:
sudo apt install qt6-base-dev qt6-tools-dev
# For python binding:
sudo apt install python3-dev
```

```shell
mkdir build && cd build
cmake .. -DAC_CORE_WITH_OPENCL=ON #-DAC_BUILD_VIDEO=ON -DAC_BUILD_GUI=ON
cmake --build . --config Release -j8
cd bin
./ac_cli -v
```

### Termux
To build with OpenCL support, you need install `ocl-icd` package, OpenCL SDK from Khronos seems not to be worked with termux.

```shell
pkg install cmake clang ocl-icd opencl-clhpp opencl-headers
mkdir build && cd build
cmake .. -DAC_CORE_WITH_OPENCL=ON
cmake --build . --config Release -j8
cd bin
LD_LIBRARY_PATH=/vendor/lib64:$PREFIX/lib ./ac_cli -l
```

### WASM
Tested with Emscripten. check [Anime4KCPP-Playground](https://github.com/TianZerL/Anime4KCPP-Playground).

### Mac OS
Tested with Apple Clang via github actions, `MACOSX_DEPLOYMENT_TARGET` >= 10.12 is required.

## CMake options

| Option                               | Description                                        | Default     |
| ------------------------------------ | -------------------------------------------------- | ----------- |
| AC_SHARED_LIB                        | build as a shared library                          | OFF         |
| AC_CORE_WITH_EIGEN3                  | build core with eigen3                             | OFF         |
| AC_CORE_WITH_SSE                     | build core with x86 sse                            | Auto detect |
| AC_CORE_WITH_AVX                     | build core with x86 avx                            | Auto detect |
| AC_CORE_WITH_FMA                     | build core with x86 fma and avx                    | Auto detect |
| AC_CORE_WITH_NEON                    | build core with arm neon                           | Auto detect |
| AC_CORE_WITH_WASM_SIMD128            | build core with wasm simd128                       | Auto detect |
| AC_CORE_WITH_OPENCL                  | build core with opencl                             | OFF         |
| AC_CORE_WITH_CUDA                    | build core with cuda                               | OFF         |
| AC_CORE_ENABLE_FAST_MATH             | enable fast math for core                          | OFF         |
| AC_CORE_DISABLE_IMAGE_IO             | disable image file read and write for core         | OFF         |
| AC_BUILD_CLI                         | build cli                                          | ON          |
| AC_BUILD_GUI                         | build gui                                          | OFF         |
| AC_BUILD_VIDEO                       | build video module                                 | OFF         |
| AC_BUILD_FILTER_AVISYNTH             | build avisynth filter                              | OFF         |
| AC_BUILD_FILTER_VAPOURSYNTH          | build vapoursynth filter                           | OFF         |
| AC_BUILD_FILTER_DIRECTSHOW           | build directshow filter (Windows only)             | OFF         |
| AC_BUILD_FILTER_AVISYNTH_VAPOURSYNTH | build an avisynth and vapoursynth universal filter | OFF         |
| AC_BUILD_BINDING_C                   | build c binding for core                           | OFF         |
| AC_BUILD_BINDING_PYTHON              | build python binding for core                      | OFF         |
| AC_TOOLS_BENCHMARK                   | build benchmark                                    | OFF         |
| AC_TEST_UTIL                         | build util module test                             | OFF         |
| AC_TEST_VIDEO                        | build video module test                            | OFF         |
| AC_ENABLE_LTO                        | enable LTO                                         | OFF         |
| AC_ENABLE_STATIC_CRT                 | enable static link crt                             | OFF         |
| AC_DISABLE_RTTI                      | disable rtti                                       | OFF         |
| AC_DISABLE_EXCEPTION                 | disable exception                                  | OFF         |
| AC_DISABLE_PIC                       | disable pic or pie                                 | OFF         |

There are some convenient presets:

`AC_PRESET_RELEASE`
- AC_CORE_WITH_OPENCL
- AC_CORE_WITH_CUDA
- AC_CORE_ENABLE_FAST_MATH
- AC_BUILD_CLI
- AC_BUILD_GUI
- AC_BUILD_VIDEO
- AC_BUILD_FILTER_AVISYNTH_VAPOURSYNTH
- AC_BUILD_FILTER_DIRECTSHOW (Windows only)

# LICENSE
The [video module](/video/) is under GPLv3, any module built with the video module are also under GPLv3, others under MIT.
For example, if [cli](/cli/) build with video module, it is under GPLv3, otherwise, it is under MIT.
