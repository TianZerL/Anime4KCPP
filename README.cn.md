# Anime4KCPP
Anime4KCPP是Anime4K的C++实现，它基于[bloc97的Anime4K](https://github.com/bloc97/Anime4K)算法0.9版本，并对其进行优化以提供更佳的图像质量。Anime4KCPP致力于成为高性能的视频或图像预处理工具。   

**注意: 使用CPU处理已经足够快，足以处理普通图像，其性能大约是[Go语言实现](https://github.com/TianZerL/Anime4KGo)的12倍，[Python实现](https://github.com/TianZerL/Anime4KPython)的1300倍。若使用GPU加速，一般情况下速度是CPU的10倍以上（取决于您的显卡），因此GPU加速很适合用于处理视频，尝试Anime4KCPP以获得更好的质量和高性能。**  

# 关于Anime4K算法
Anime4K算法是一种简单且高质量的动漫类图像超分辨率算法，它并不使用机器学习，因此速度非常快，可用于实时处理和预处理。    

# 为什么选择Anime4KCPP
- 跨平台支持，已在Windows和Linux上通过编译测试，MacOS同样也支持。
- 广泛的兼容性，同时支持CPU和GPU，GPU只要求支持OpenCL即可，并不限制任何品牌。
- 提供易于使用的GUI和CLI程序。
- 高性能。
- 支持GPU加速，在短时间内处理图像和视频。
- 可调节参数，尝试不同的选项以获得更佳的质量或者更快的速度。
- 滤镜支持，利用它们进行抗锯齿和降噪。  

# 效果
![examples](images/examples.png)  

# GPU加速
Anime4KCPP现已支持GPU加速，通过原生OpenCL实现，可提供高性能与跨平台性，只要您的显卡支持OpenCL 1.2或者更高版本，即可开启。在*AMD Vege 8 Graphics* (*AMD Ryzen 3500U*的核显) 上可以在0.1秒内完成1080 -> 4K图像处理。  

# 性能
CPU: AMD Ryzen 3500U  
GPU: AMD Vege 8 Graphics  
RAM: 16G  
Anime4KCPP 版本 : 1.6.0  
Anime4KCPP 设置: 平衡  

    CPU:
    图像:
    256x256 -> 512x512:   0.025秒  
    1080P   -> 4k:        0.650秒  

    视频(长度: 1 分 32 秒):
    480P  -> 1080P :       3  分 13 秒
    1080P -> 4K :          19 分 09  秒

    GPU:
    图像:
    256x256 -> 512x512:   0.006秒  
    1080P   -> 4k:        0.090秒  

    视频(长度: 1 分 32 秒):
    480P  -> 1080P :       0  分 31 秒
    1080P -> 4K :          3  分 00 秒

# GUI
Anime4KCPP支持GUI，您可以更轻松的处理您的图像与视频!  
**注意: 在处理视频前请安装 [ffmpeg](https://ffmpeg.org) 否则处理结果将没有音轨**  
**该界面已过时，请以新版为准**

![GUI](images/GUI.png)

# CLI
## 视频处理
添加参数 ```-v``` 以开启视频处理。视频处理支持多线程并默认使用所有线程，您可使用参数 ```-t``` 来手动指定使用线程的数量。

## 使用方法
### 编译
在编译之前请安装 [OpenCV](https://opencv.org)。 ( [release](https://github.com/TianZerL/Anime4KCPP/releases) 已包含OpenCV运行时库。)

您还需要一个OpenCL SDK实现，您通常可以从您的显卡提供商的网站上找到它，例如[这个](https://github.com/GPUOpen-LibrariesAndSDKs/OCL-SDK/releases)是AMD提供的一个OpenCL SDK实现。

您需要Qt开源版本以构建GUI界面。

若您想处理视频, 请先安装 [ffmpeg](https://ffmpeg.org), 否则您的视频将没有音轨。您可能还需要一个开源编解码器 [OpenH264 encoder](https://github.com/cisco/openh264/releases)。

该项目使用 [cmake](https://cmake.org) 进行构建。
### 参数

    options:
      -i, --input               待加载文件 (string [=./pic/p1.png])
      -o, --output              输出文件名 (string [=output.png])
      -p, --passes              处理次数 (int [=2])
      -n, --pushColorCount      限制边缘细化执行次数(int [=2])
      -c, --strengthColor       细化边缘，范围0-1，越大的值会越细 (double [=0.3])
      -g, --strengthGradient    锐利度，范围0-1，越大的值会越锐利 (double [=1])
      -z, --zoomFactor          缩放倍数 (double [=2])
      -t, --threads             指定处理视频时所用的线程数 (unsigned int [=8])
      -f, --fastMode            加快处理速度但可能获得低质量图像
      -v, --videoMode           视频处理模式
      -s, --preview             在处理结束后预览结果
      -b, --preProcessing       启用预处理
      -a, --postProcessing      启用后处理
      -r, --preFilters          选择预处理滤镜，仅当preProcessing开启时有效，可使用以下滤镜: Median blur=000001，Mean blur=000010Gaussian blur weak=000100, Gaussian blur=001000, Bilateral filter=010000, Bilateral filter faster=100000，使用它们对应的二进制进行自由组合，例如: Gaussian blur weak + Bilateral filter = 000100 | 010000 = 010100 = 20(D)， (unsigned int [=4])
      -e, --postFilters         选择后处理滤镜，仅当postProcessing开启时有效，可使用以下滤镜: Median blur=000001，Mean blur=000010Gaussian blur weak=000100, Gaussian blur=001000, Bilateral filter=010000, Bilateral filter faster=100000，使用它们对应的二进制进行自由组合，例如: Gaussian blur weak + Bilateral filter = 000100 | 010000 = 010100 = 20(D)，输入20即可开启Gaussian blur weak 和Bilateral filter，这也是我推荐用于小于1080P图像的设置，对于大于等于1080P的图像推荐使用24，小于1080P的视频推荐36，大于等于1080P的视频荐40 (unsigned int [=20])
      -q, --GPUMode             开启GPU加速  
      -l, --listGPUs            列出GPU平台与设备
      -h, --platformID          指定平台ID (unsigned int [=0])
      -d, --deviceID            指定设备ID (unsigned int [=0])
      -?, --help                显示帮助信息

## 滤镜
启用滤镜可以使得处理后的图像看起来更舒服，目前支持以下五种滤镜：

  - Median blur [0000001]
  - Mean blur [0000010]
  - [CAS Sharpening](https://gpuopen.com/gaming-product/fidelityfx) [0000100]
  - Gaussian blur weak [0001000]
  - Gaussian blur [0010000]
  - Bilateral filter [0100000]
  - Bilateral filter faster [1000000]

使用它们对应的二进制值进行自由组合.

例: Gaussian blur weak + Bilateral filter = 0001000 | 0100000 = 0101000(B)= 40(D)

使用 ```-b``` 开启预处理滤镜支持，```-r``` 用于指定滤镜，```-r``` 默认为4：

    Anime4KCPP -i input.png -o output.png -b -r 44


使用 ```-a``` 开启后处理滤镜支持，```-e``` 用于指定滤镜，```-e``` 默认为40：

    Anime4KCPP -i input.png -o output.png -a -e 48

对于后处理，我推荐使用 40(Gaussian blur weak + Bilateral filter) 处理小于1080P的图像，48(Gaussian blur + Bilateral filter) 处理大于等于1080P的图像，
72(Gaussian blur weak + Bilateral filter faster) 处理小于1080P的视频，80(Gaussian blur + Bilateral filter faster)处理大于等于1080P的视频。

预处理一般启用CAS即可。

CAS是AMD开源的自适应锐化技术。
# pyanime4k
[pyanime4k](https://github.com/TianZerL/pyanime4k) 是一个在python中使用Anime4K的简单方式，它基于Anime4KCPP。 


# 其它实现
- Python
  - [TianZerL/Anime4KPython](https://github.com/TianZerL/Anime4KPython)
- Go
  - [TianZerL/Anime4KGo](https://github.com/TianZerL/Anime4KGo)
- C#
  - [shadow578/Anime4kSharp](https://github.com/shadow578/Anime4kSharp)
  - [net2cn/Anime4KSharp](https://github.com/net2cn/Anime4KSharp)
- Java
  - [bloc97/Anime4K](https://github.com/bloc97/Anime4K)
- Rust
  - [andraantariksa/Anime4K-rs](https://github.com/andraantariksa/Anime4K-rs)

# 致谢
项目中引用的所有动漫图像均由我的朋友 ***King of learner*** 绘制并授权使用，请勿在未经许可的情况下使用它们。