# Anime4KCPP v3
Anime4KCPP is a high performance anime upscaler.

Anime4KCPP v3 uses CNN based algorithm, and aims to be simple and efficient.

# Build
## Dependency
To build Anime4KCPP v3 you need CMake and a C++17 compiler, and most dependencies will be resolved automatically by CMake if you have internet.

***List of dependencies that need to be prepared by yourself:***

| Dependency                                                | CMake option      | Module     |
| --------------------------------------------------------- | ----------------- | ---------- |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) | AC_CORE_WITH_CUDA | core(CUDA) |
| [libavcodec](https://ffmpeg.org)                          | AC_BUILD_VIDEO    | video      |
| [libavformat](https://ffmpeg.org)                         | AC_BUILD_VIDEO    | video      |
| [libavutil](https://ffmpeg.org)                           | AC_BUILD_VIDEO    | video      |
| [Qt](https://www.qt.io)                                   | AC_BUILD_GUI      | gui        |

- The minimum tested version of the CUDA Toolkit is 11
- The minimum version of ffmpeg's libav is from ffmpeg 4
- Qt5 or Qt6 should be ok

***List of dependencies that can be resolved automatically:***

| Dependency                                                                                                                            | CMake option                | Module              |
| ------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- | ------------------- |
| [Avisynth SDK](https://github.com/AviSynth/AviSynthPlus/tree/master/avs_core/include)                                                 | AC_BUILD_FILTER_AVISYNTH    | filter(avisynth)    |
| [CLI11](https://github.com/CLIUtils/CLI11)                                                                                            | AC_BUILD_CLI                | cli                 |
| [DirectShow base classes](https://github.com/microsoft/Windows-classic-samples/Samples/Win7Samples/multimedia/directshow/baseclasses) | AC_BUILD_FILTER_DIRECTSHOW  | filter(directshow)  |
| [Eigen3](https://gitlab.com/libeigen/eigen)                                                                                           | AC_CORE_WITH_EIGEN3         | core(eigen3)        |
| [OpenCL SDK](https://github.com/KhronosGroup/OpenCL-SDK)                                                                              | AC_CORE_WITH_OPENCL         | core(opencl)        |
| [pybind11](https://github.com/pybind/pybind11)                                                                                        | AC_BUILD_BINDING_PYTHON     | binding(python)     |
| [ruapu](https://github.com/nihui/ruapu)                                                                                               | N/A                         | core                |
| [stb](https://github.com/nothings/stb)                                                                                                | N/A                         | core                |
| [VapourSynth SDK](https://github.com/vapoursynth/vapoursynth/tree/master/include)                                                     | AC_BUILD_FILTER_VAPOURSYNTH | filter(vapoursynth) |

## Platform specific
### Windows
Tested with MinGW64 and MSVC.

To setup ffmpeg's libav for building video module on windows, it is recommended to add an `AC_PATH_FFMPEG` variable to CMake, but you can also use `pkg-config` for windows. `AC_PATH_FFMPEG` should be a path to the ffmpeg's root folder witch contains `lib` and `include`.

To add `AC_PATH_FFMPEG` to CMake, click `Add Entry` button in `cmake-gui` or use `-DAC_PATH_FFMPEG="path/to/ffmpeg/root"` in CMake console.

For windwos, you can download ffmpeg with sdk from [BtBN](https://github.com/BtbN/FFmpeg-Builds/releases) (`ffmpeg-master-latest-win64-gpl-shared.zip` or `ffmpeg-master-latest-win64-lgpl-shared.zip`) or [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (`ffmpeg-release-full-shared.7z`).

**You need MSVC to build directshow filter, and it is only available on Windows**

### Linux

Tested with gcc and clang.

### WASM
Only tested with Emscripten. See [wasm test](test/wasm/).

### Termux
To build with opencl support, you need install `ocl-icd` package, OpenCL SDK from Khronos seems not to be worked with termux.

```shell
pkg install cmake clang ocl-icd opencl-clhpp opencl-headers
mdir build && cd build
cmake .. -DAC_CORE_WITH_OPENCL=ON
make -j8
cd bin
LD_LIBRARY_PATH=/vendor/lib64:$PREFIX/lib ./ac_cli -l
```

### Mac OS
Untested. If you do, please provide feedback.

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
| AC_CORE_ENABLE_IMAGE_IO              | enable image file read and write for core          | ON          |
| AC_CORE_ENABLE_FAST_MATH             | enable fast math for core                          | ON          |
| AC_BUILD_CLI                         | build cli                                          | ON          |
| AC_BUILD_GUI                         | build gui                                          | OFF         |
| AC_BUILD_VIDEO                       | build video module                                 | OFF         |
| AC_BUILD_FILTER_AVISYNTH             | build avisynth filter                              | OFF         |
| AC_BUILD_FILTER_VAPOURSYNTH          | build vapoursynth filter                           | OFF         |
| AC_BUILD_FILTER_DIRECTSHOW           | build directshow filter (Windows MSVC only)        | OFF         |
| AC_BUILD_FILTER_AVISYNTH_VAPOURSYNTH | build an avisynth and vapoursynth universal filter | OFF         |
| AC_BUILD_BINDING_C                   | build c binding for core                           | OFF         |
| AC_BUILD_BINDING_PYTHON              | build python binding for core                      | OFF         |
| AC_TEST_CORE                         | build core test                                    | OFF         |
| AC_TEST_UTIL                         | build util module test                             | OFF         |
| AC_TEST_VIDEO                        | build video module test                            | OFF         |
| AC_TEST_WASM                         | build wasm test (Emscripten only)                  | OFF         |
| AC_ENABLE_LTO                        | enable LTO                                         | OFF         |
| AC_ENABLE_STATIC_CRT                 | enable static link crt                             | OFF         |
| AC_DISABLE_RTTI                      | disable rtti                                       | OFF         |
| AC_DISABLE_EXCEPTION                 | disable exception                                  | OFF         |
| AC_DISABLE_PIC                       | disable pic or pie                                 | OFF         |

# LICENSE
The [video module](/video/) is under GPLv3, any module built with the video module are also under GPLv3, others under MIT.
For example, if [cli](/cli/) build with video module, it is under GPLv3, otherwise, it is under MIT.
