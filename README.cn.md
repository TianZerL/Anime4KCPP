### [**📁下载**](https://github.com/TianZerL/Anime4KCPP/releases)
### [**📖Wiki**](https://github.com/TianZerL/Anime4KCPP/wiki)

# 关于Anime4KCPP
Anime4KCPP提供一个改进后的[bloc97的Anime4K](https://github.com/bloc97/Anime4K)算法0.9版本，同时也提供自己的CNN算法[ACNet](https://github.com/TianZerL/Anime4KCPP/wiki/ACNet)。Anime4KCPP提供多种使用方式，包括预处理与实时播放，其致力于成为高性能的视频或图像处理工具。

# 关于Anime4K09算法
Anime4K算法是一种简单且高质量的动漫类图像超分辨率算法，它并不使用机器学习，因此速度非常快，可用于实时处理和预处理。    

# 关于ACNet
ACNet是一个基于卷积神经网络的超分辨率算法，旨在同时提供高质量和高性能。  
HDN模式能更好的降噪，HDN等级从1到3，越高降噪效果越好，但可能导致模糊和缺少细节。    
详情可见[wiki页面](https://github.com/TianZerL/Anime4KCPP/wiki/ACNet)

# 为什么选择Anime4KCPP
- 跨平台支持，已在Windows，Linux和macOS (感谢[NightMachinary](https://github.com/NightMachinary))上通过编译测试。
- 支持GPU加速，只需一块实现了OpenCL1.2或更高版本的GPU。
- CUDA加速同样支持。
- 高性能，低内存占用。
- 支持多种使用方式。

# 使用方式
- CLI
- GUI
- DirectShow滤镜 (仅用于Windows，支持MPC-HC/BE，Potplayer或者其他基于DirectShow的播放器)
- AviSynth+插件
- VapourSynth插件
- Android APP
- C API绑定
- [Python API绑定](https://github.com/TianZerL/pyanime4k)
- [GLSL着色器](https://github.com/TianZerL/ACNetGLSL)(支持基于MPV的播放器)

**了解如何使用和更多信息，请参阅[wiki](https://github.com/TianZerL/Anime4KCPP/wiki).**

# 效果
![examples](/images/example.png)

# 性能
单张图片 (RGB):

|处理器|类型|算法|1080p -> 4K|512p -> 1024p|性能测试分数|
-|-|-|-|-|-
|AMD Ryzen 2600|CPU|ACNet|0.630 s|0.025 s|17.0068|
|Nvidia GTX1660 Super|GPU|ACNet|0.067 s|0.005 s|250|
|AMD Ryzen 2500U|CPU|ACNet|1.304 s|0.049 s|7.59301|
|AMD Vega 8|GPU|ACNet|0.141 s|0.010 s|105.263|
|Snapdragon 820|CPU|ACNet|5.532 s|0.180 s|1.963480|
|Adreno 530|GPU|ACNet|3.133 s|0.130 s|3.292723|
|Snapdragon 855|CPU|ACNet|3.998 s|0.204 s *|3.732736|
|Adreno 640|GPU|ACNet|1.611 s|0.060 s|6.389776|
|Intel Atom N2800|CPU|ACNet|11.827 s|0.350 s|0.960984|
|Raspberry Pi Zero W|CPU|ACNet|114.94 s|3.312 s|0.101158|

*骁龙855在低负载下可能使用Cortex-A55核心, 因此性能表现可能不如骁龙820

# 编译
关于如何编译Anime4KCPP，请参阅[wiki](https://github.com/TianZerL/Anime4KCPP/wiki/Building).

# 相关项目
### pyanime4k  
[pyanime4k](https://github.com/TianZerL/pyanime4k)是对Anime4KCPP API的Python绑定，快速且简单易用。

### ACNetGLSL
[ACNetGLSL](https://github.com/TianZerL/ACNetGLSL)是ACNet(Anime4KCPP Net)的GLSL实现。

# 使用Anime4KCPP的项目
- [AaronFeng753/Waifu2x-Extension-GUI](https://github.com/AaronFeng753/Waifu2x-Extension-GUI)
- [k4yt3x/video2x](https://github.com/k4yt3x/video2x)

# 鸣谢
- [Anime4K](https://github.com/bloc97/Anime4K)
- [cmdline](https://github.com/tanakh/cmdline)
- [CUDA](https://developer.nvidia.com/cuda-zone)
- [FFmpeg](https://ffmpeg.org/)
- [OpenCL](https://www.khronos.org/opencl/)
- [OpenCV](https://opencv.org/)
- [Qt](https://www.qt.io/)

# 致谢
[semmyenator](https://github.com/semmyenator)：GUI繁体中文、日语与法语翻译

项目中引用的所有动漫图像均由我的朋友 ***King of learner*** 绘制并授权使用，请勿在未经许可的情况下使用它们。
