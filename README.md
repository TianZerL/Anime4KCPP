# Anime4KCPP
This is an implementation of Anime4K in C++. It based on the [bloc97's Anime4K](https://github.com/bloc97/Anime4K) algorithm version 0.9 and some optimizations have been made, it can process both image and video.
This project is for learning and the exploration task of algorithm course in SWJTU.  

***NOTICE: Thanks for the high performance of pointer, the C++ version will be very fast. It is about 6.5 times faster than [Go version](https://github.com/TianZerL/Anime4KGo), and 650 times faster than [Python version](https://github.com/TianZerL/Anime4KPython), but please understand, this is executed in CPU, for video preprocessing, it will take a while. Therefore, if your graphic card is good enough, I recommend you to use the real-time process version by [bloc97](https://github.com/bloc97/Anime4K), if your CPU is powerful or you want higher quality, just try this.***

# pyanime4k
[pyanime4k](https://github.com/TianZerL/pyanime4k) is a simply package to use anime4k in python, easy, fast and powerful, which support both image and video processing, based on Anime4KCPP. 

# About Anime4K
Anime4K is a simple high-quality anime upscale algorithm for anime. it does not use any machine learning approaches, and can be very fast in real-time processing.

# Video processing
For video processing, all you need do is to add the argument ```-v```, and waiting. The video processing supports multithreading, and by default uses all CPU threads, but you can adjust it manually by ```-t``` to specify the number of threads for processing.  For performance, I recommend only do one pass for video processing, and don't make ```zoomFactor``` too large.

# Usage
Please install [OpenCV Library](https://opencv.org) before building, and the [release version](https://github.com/TianZerL/Anime4KCPP/releases) have already included OpenCV runtime.  

If you want to process video, please install [ffmpeg](https://ffmpeg.org) firstly, otherwise the output will be silent. And make sure you have [OpenH264 encoder V1.8.0](https://github.com/cisco/openh264/releases) for encoding.

This project uses [cmake](https://cmake.org) to build.

    options:
      -i, --input               File for loading (string [=./pic/p1.png])
      -o, --output              File for outputting (string [=output.png])
      -p, --passes              Passes for processing (int [=2])
      -c, --strengthColor       Strength for pushing color,range 0 to 1,higher for thinner (double [=0.3])
      -g, --strengthGradient    Strength for pushing gradient,range 0 to 1,higher for sharper (double [=1])
      -z, --zoomFactor          zoom factor for resizing (double [=2])
      -t, --threads             Threads count for video processing (unsigned int [=8])
      -f, --fastMode            Faster but maybe low quality
      -v, --videoMode           Video process
      -s, --preview             Preview image
      -a, --postProcessing      Enable post processing
      -e, --filters             Enhancement filter, only working when postProcessing is true,there are 5 options by binary:Median blur=000001, Mean blur=000010, Gaussian blur weak=000100, Gaussian blur=001000, Bilateral filter=010000, Bilateral filter faster=100000, you can freely combine them, eg: Gaussian blur weak + Bilateral filter = 000100 & 010000 = 010100 = 20(D), so you can put 20 to enable Gaussian blur weak and Bilateral filter, which also is what I recommend for image that < 1080P, 24 for image that >= 1080P, and for performance I recommend to use 36 for video that < 1080P, 40 for video that >=1080P (unsigned int [=20])
      -?, --help                print this message

# Filters
Enable filters can make the result like better, now Anime4kCPP support 5 filters include:

  - Median blur [000001]
  - Mean blur [000010]
  - Gaussian blur weak [000100]
  - Gaussian blur [001000]
  - Bilateral filter [010000]
  - Bilateral filter faster [100000]

You can freely combine them by their binary.  
eg: Gaussian blur weak + Bilateral filter = 000100 & 010000 = 010100(B)= 20(D)  

Easily use ```-a``` to enable filters function, and then use ```-e``` to custom your own combination, normally, if you don't specify the ```-e``` manually it will be 20. You can use command like this to enable Gaussian blur and Bilateral filter:

    Anime4KCPP -i input.png -o output.png -a -e 24

I recommend use 20(Gaussian blur weak + Bilateral filter) for image that < 1080P, 24(Gaussian blur + Bilateral filter) for image that >= 1080P, and 36(Gaussian blur weak + Bilateral filter faster) for video that < 1080P, 40(Gaussian blur + Bilateral filter faster) for video that >=1080P.


# Other implementations
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