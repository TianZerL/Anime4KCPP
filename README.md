<div align="center">
  <img src="./images/Logo.png">
</div>

<h1 align="center">
  Anime4KCPP
  </br>
  <a href="https://github.com/TianZerL/Anime4KCPP/releases"><img alt="Latest GitHub release" src="https://img.shields.io/github/v/release/TianZerL/Anime4KCPP?color=red&label=Latest%20release&style=flat-square"></a>
  <img alt="Platforms" src="https://img.shields.io/badge/Platforms-Windows%20%7C%20Linux%20%7C%20macOS%20%7C%20Android-blue?style=flat-square">
  <img alt="License" src="https://img.shields.io/github/license/TianZerL/Anime4KCPP?style=flat-square">
</h1>


### [**📄中文文档**](README.cn.md)
### [**📁Download**](https://github.com/TianZerL/Anime4KCPP/releases)
### [**📖Wiki**](https://github.com/TianZerL/Anime4KCPP/wiki)

# About Anime4KCPP
Anime4KCPP provides an optimized [bloc97's Anime4K](https://github.com/bloc97/Anime4K) algorithm version 0.9, and it also provides its own CNN algorithm [ACNet](https://github.com/TianZerL/Anime4KCPP/wiki/ACNet), it provides a variety of way to use, including preprocessing and real-time playback, it aims to be a high performance tools to process both image and video.  
This project is for learning and the exploration task of algorithm course in SWJTU.

# About Anime4K09
Anime4K is a simple high-quality anime upscale algorithm. The version 0.9 does not use any machine learning approaches, and can be very fast in real-time processing or pretreatment.

# About ACNet
ACNet is a CNN based anime upscale algorithm. It aims to provide both high-quality and high-performance.  
HDN mode can better denoise, HDN level is from 1 to 3, higher for better denoising but may cause blur and lack of detail.  
for detail, see [wiki page](https://github.com/TianZerL/Anime4KCPP/wiki/ACNet).

# Why Anime4KCPP
- Cross-platform, building have already tested in Windows ,Linux, and macOS (Thanks for [NightMachinary](https://github.com/NightMachinary)).
- GPU acceleration support with all GPUs that implemented OpenCL 1.2 or newer.
- High performance and low memory usage.
- Support multiple usage methods.

# Usage method
- CLI
- GUI
- DirectShow Filter (Windows only, for MPC-HC/BE, potplayer and other DirectShow based players)
- AviSynthPlus plugin
- VapourSynth plugin
- Android APP
- C API binding
- [Python API binding](https://github.com/TianZerL/pyanime4k)
- [GLSL shader](https://github.com/TianZerL/ACNetGLSL)(For MPV based players)

***For more infomation on how to use them, see [wiki](https://github.com/TianZerL/Anime4KCPP/wiki).***

# Result
![examples](/images/example.png)

# Performance
### Desktop
CPU: AMD Ryzen 3500U  
GPU: AMD Vege 8 Graphics  
RAM: 16G  
Anime4KCPP Version : 1.7.0  
Anime4KCPP Settings: balance  
algorithm: Anime4K09  

    CPU:
    Image:
    256x256 -> 512x512:   0.025s  
    1080P   -> 4k:        0.650s  

    Video(Length: 1 min 32 seconds):
    480P  -> 1080P :       03 min 13 seconds
    1080P -> 4K :          19 min 09 seconds

    GPU:
    Image:
    256x256 -> 512x512:   0.006s  
    1080P   -> 4k:        0.090s  

    Video(Length: 1 min 32 seconds):
    480P  -> 1080P :       00 min 29 seconds
    1080P -> 4K :          02 min 55 seconds

### Android
SOC: Snapdragon 855  
RAM: 8G  
Anime4KCPP Version : 1.8.1  
Anime4KCPP Settings: balance  
algorithm: Anime4K09  

    CPU:  
    Image:  
    256x256 -> 512x512:   0.045s  
    1080P   -> 4k:        0.544s (That's even faster than R5 3500U, Amazing!)  

    GPU:  
    Image:  
    256x256 -> 512x512:   0.008s  
    1080P   -> 4k:        0.158s  

    Video(Length: 1 min 32 seconds):
    480P  -> 1080P :       01 min 04 seconds

# Building
For information on how to compile Anime4KCPP, see [wiki](https://github.com/TianZerL/Anime4KCPP/wiki/Building).

# Related projects
### pyanime4k  
[pyanime4k](https://github.com/TianZerL/pyanime4k) is an Anime4KCPP API binding in Python, easy and fast. 

### ACNetGLSL
[ACNetGLSL](https://github.com/TianZerL/ACNetGLSL) is an ACNet (Anime4KCPP Net) re-implemented in GLSL for real-time anime upscaling.

# Projects that use Anime4KCPP
- [AaronFeng753/Waifu2x-Extension-GUI](https://github.com/AaronFeng753/Waifu2x-Extension-GUI)
- [k4yt3x/video2x](https://github.com/k4yt3x/video2x)

# Credits
- [Anime4K](https://github.com/bloc97/Anime4K)
- [cmdline](https://github.com/tanakh/cmdline)
- [FFmpeg](https://ffmpeg.org/)
- [OpenCL](https://www.khronos.org/opencl/)
- [OpenCV](https://opencv.org/)
- [Qt](https://www.qt.io/)

# Acknowledgement
All images are drawn by my friend ***King of learner*** and authorized to use, only for demonstration, do not use without permission.
