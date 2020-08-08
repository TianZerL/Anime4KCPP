#include "filterprocessor.h"

Anime4KCPP::FilterProcessor::FilterProcessor(cv::Mat& srcImg, uint8_t _filters) :
    filters(_filters), srcImgRef(srcImg)
{
    img = srcImg;
    H = img.rows;
    W = img.cols;
}

void Anime4KCPP::FilterProcessor::process()
{
    if (filters & MEDIAN_BLUR)
        cv::medianBlur(img, img, 3);
    if (filters & MEAN_BLUR)
        cv::blur(img, img, cv::Size(3, 3));
    if (filters & CAS_SHARPENING)
        CASSharpening(img);
    if (filters & GAUSSIAN_BLUR_WEAK)
        cv::GaussianBlur(img, img, cv::Size(3, 3), 0.5);
    else if (filters & GAUSSIAN_BLUR)
        cv::GaussianBlur(img, img, cv::Size(3, 3), 1);
    if (filters & BILATERAL_FILTER)
    {
        cv::bilateralFilter(img, tmpImg, 9, 30, 30);
        srcImgRef = img;
    }
    else if (filters & BILATERAL_FILTER_FAST)
    {
        cv::bilateralFilter(img, tmpImg, 5, 35, 35);
        srcImgRef = img;
    }
}

inline void Anime4KCPP::FilterProcessor::CASSharpening(cv::Mat& img)
{
    const int lineStep = W * 3;
    changEachPixelBGR(img, [&](const int i, const int j, RGBA pixel, Line curLine) {
        const int jp = j < (W - 1) * 3 ? 3 : 0;
        const int jn = j > 3 ? -3 : 0;

        const Line pLineData = i < H - 1 ? curLine + lineStep : curLine;
        const Line cLineData = curLine;
        const Line nLineData = i > 0 ? curLine - lineStep : curLine;

        const RGBA tc = nLineData + j;
        const RGBA ml = cLineData + j + jn, mc = pixel, mr = cLineData + j + jp;
        const RGBA bc = pLineData + j;

        const uint8_t minR = MIN5(tc[R], ml[R], mc[R], mr[R], bc[R]);
        const uint8_t maxR = MAX5(tc[R], ml[R], mc[R], mr[R], bc[R]);
        const uint8_t minG = MIN5(tc[G], ml[G], mc[G], mr[G], bc[G]);
        const uint8_t maxG = MAX5(tc[G], ml[G], mc[G], mr[G], bc[G]);
        const uint8_t minB = MIN5(tc[B], ml[B], mc[B], mr[B], bc[B]);
        const uint8_t maxB = MAX5(tc[B], ml[B], mc[B], mr[B], bc[B]);

        constexpr double peak = LERP(-0.125, -0.2, 1.0);
        const double wR = peak * sqrt(MIN(minR, 255 - maxR) * REC(maxR));
        const double wG = peak * sqrt(MIN(minG, 255 - maxG) * REC(maxG));
        const double wB = peak * sqrt(MIN(minB, 255 - maxB) * REC(maxB));

        const double r = (wR * (tc[R] + ml[R] + mr[R] + bc[R]) + mc[R]) / (1.0 + 4.0 * wR);
        const double g = (wG * (tc[G] + ml[G] + mr[G] + bc[G]) + mc[G]) / (1.0 + 4.0 * wG);
        const double b = (wB * (tc[B] + ml[B] + mr[B] + bc[B]) + mc[B]) / (1.0 + 4.0 * wB);
        pixel[R] = UNFLOAT(r);
        pixel[G] = UNFLOAT(g);
        pixel[B] = UNFLOAT(b);
        });
}

inline void Anime4KCPP::FilterProcessor::changEachPixelBGR(cv::Mat& src,
    const std::function<void(const int, const int, RGBA, Line)>&& callBack)
{
    cv::Mat tmp;
    src.copyTo(tmp);

    int jMAX = W * 3;
#ifdef _MSC_VER
    Concurrency::parallel_for(0, H, [&](int i) {
        Line lineData = src.data + static_cast<size_t>(i) * static_cast<size_t>(W) * static_cast<size_t>(3);
        Line tmpLineData = tmp.data + static_cast<size_t>(i) * static_cast<size_t>(W) * static_cast<size_t>(3);
        for (int j = 0; j < jMAX; j += 3)
            callBack(i, j, tmpLineData + j, lineData);
        });
#else
#pragma omp parallel for
    for (int i = 0; i < H; i++)
    {
        Line lineData = src.data + static_cast<size_t>(i) * static_cast<size_t>(W) * static_cast<size_t>(3);
        Line tmpLineData = tmp.data + static_cast<size_t>(i) * static_cast<size_t>(W) * static_cast<size_t>(3);
        for (int j = 0; j < jMAX; j += 3)
            callBack(i, j, tmpLineData + j, lineData);
    }
#endif

    src = tmp;
}
