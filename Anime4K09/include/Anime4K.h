#pragma once
#include<iostream>
#include<functional>

#include<opencv2/opencv.hpp>
#include<opencv2/core/hal/interface.h>

typedef unsigned char* RGBA;
typedef unsigned char* Line;

class Anime4K
{
public:
    Anime4K(int passes = 2, double strengthColor = 0.3, double strengthGradient = 1, bool fastMode = false);
    void loadImage(const std::string srcFile);
    void saveImage(const std::string dstFile);
    void showInfo();
    void showImg();
    void process();
    void getGray();
    void pushColor();
    void getGradient();
    void pushGradient();
private:
    void changEachPixel(cv::InputArray _src, const std::function<void(int, int, RGBA,Line)>& callBack);
    void getLightest(RGBA mc, RGBA a, RGBA b, RGBA c);
    void getAverage(RGBA mc, RGBA a, RGBA b, RGBA c);
    uint8_t max(uint8_t a, uint8_t b, uint8_t c);
    uint8_t min(uint8_t a, uint8_t b, uint8_t c);
    uint8_t unFloat(double n);
private:
    const int B, G, R, A;
    int ps;
    double sc, sg;
    bool fm;
    int orgH, orgW, H, W;
    cv::Mat orgImg, dstImg;
};

