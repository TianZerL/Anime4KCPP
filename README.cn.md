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
### 原图
![examples](/images/example.png)
### Anime4K09
![Anime4K09](/images/Anime4K09.png)
### Anime4K09加滤镜
![Anime4K09 with filters](/images/Anime4K09Filters.png)
### ACNet
![examples](/images/ACNet.png)

# 性能
### 桌面
CPU: AMD Ryzen 3500U  
GPU: AMD Vege 8 Graphics  
RAM: 16G  
Anime4KCPP 版本 : 1.7.0  
Anime4KCPP 设置: 平衡  
算法: Anime4K09  

    CPU:
    图像:
    256x256 -> 512x512:   0.025秒  
    1080P   -> 4k:        0.650秒  

    视频(长度: 1 分 32 秒):
    480P  -> 1080P :       03 分 13 秒
    1080P -> 4K :          19 分 09 秒

    GPU:
    图像:
    256x256 -> 512x512:   0.006秒  
    1080P   -> 4k:        0.090秒  

    视频(长度: 1 分 32 秒):
    480P  -> 1080P :       00 分 29 秒
    1080P -> 4K :          02 分 55 秒

### Android
SOC: 高通骁龙855  
RAM: 8G  
Anime4KCPP 版本 : 1.7.1  
Anime4KCPP 设置: 平衡  
算法: Anime4K09  

    CPU:  
    图像:  
    256x256 -> 512x512:   0.045秒  
    1080P   -> 4k:        0.544秒 (比R5 3500U还要快，厉害！)  

    GPU:  
    图像:  
    256x256 -> 512x512:   0.008秒  
    1080P   -> 4k:        0.158秒  

    视频(Length: 1 min 32 seconds):
    480P  -> 1080P :       01 min 04 seconds

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
- [FFmpeg](https://ffmpeg.org/)
- [OpenCL](https://www.khronos.org/opencl/)
- [OpenCV](https://opencv.org/)
- [Qt](https://www.qt.io/)

# 致谢
项目中引用的所有动漫图像均由我的朋友 ***King of learner*** 绘制并授权使用，请勿在未经许可的情况下使用它们。
