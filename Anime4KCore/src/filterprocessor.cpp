#include "FilterProcessor.hpp"

#define MAX5(a, b, c, d, e) std::max({a, b, c, d, e})
#define MIN5(a, b, c, d, e) std::min({a, b, c, d, e})
#define LERP(x, y, w) ((x) * (1.0 - (w)) + (y) * (w))
#define REC(n) ((n) < 1 ? 1.0 : 1.0 / (n))
#define UNFLOAT(n) ((n) >= 255 ? 255 : ((n) <= 0 ? 0 : uint8_t((n) + 0.5)))
#define CLAMP(v, lo, hi) ((v < lo) ? lo : (hi < v) ? hi : v)

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

std::vector<std::string> Anime4KCPP::FilterProcessor::filterToString(uint8_t filters)
{
    std::vector<std::string> ret;
    if (filters & MEDIAN_BLUR)
        ret.emplace_back("Median blur");
    if (filters & MEAN_BLUR)
        ret.emplace_back("Mean blur");
    if (filters & CAS_SHARPENING)
        ret.emplace_back("CAS Sharpening");
    if (filters & GAUSSIAN_BLUR_WEAK)
        ret.emplace_back("Gaussian blur weak");
    else if (filters & GAUSSIAN_BLUR)
        ret.emplace_back("Gaussian blur");
    if (filters & BILATERAL_FILTER)
        ret.emplace_back("Bilateral filter");
    else if (filters & BILATERAL_FILTER_FAST)
        ret.emplace_back("Bilateral filter faster");
    return ret;
}

inline void Anime4KCPP::FilterProcessor::CASSharpening(cv::Mat& img)
{
    const int lineStep = W * 3;
    if (img.type() == CV_8UC3)
        changEachPixelBGR(img, [&](const int i, const int j, PixelB pixel, LineB curLine) {
        const int jp = j < (W - 1) * 3 ? 3 : 0;
        const int jn = j > 3 ? -3 : 0;

        const LineB pLineData = i < H - 1 ? curLine + lineStep : curLine;
        const LineB cLineData = curLine;
        const LineB nLineData = i > 0 ? curLine - lineStep : curLine;

        const PixelB tc = nLineData + j;
        const PixelB ml = cLineData + j + jn, mc = pixel, mr = cLineData + j + jp;
        const PixelB bc = pLineData + j;

        const uint8_t minR = MIN5(tc[R], ml[R], mc[R], mr[R], bc[R]);
        const uint8_t maxR = MAX5(tc[R], ml[R], mc[R], mr[R], bc[R]);
        const uint8_t minG = MIN5(tc[G], ml[G], mc[G], mr[G], bc[G]);
        const uint8_t maxG = MAX5(tc[G], ml[G], mc[G], mr[G], bc[G]);
        const uint8_t minB = MIN5(tc[B], ml[B], mc[B], mr[B], bc[B]);
        const uint8_t maxB = MAX5(tc[B], ml[B], mc[B], mr[B], bc[B]);

        constexpr double peak = LERP(-0.125, -0.2, 1.0);
        const double wR = peak * std::sqrt(MIN(minR, 255 - maxR) * REC(maxR));
        const double wG = peak * std::sqrt(MIN(minG, 255 - maxG) * REC(maxG));
        const double wB = peak * std::sqrt(MIN(minB, 255 - maxB) * REC(maxB));

        const double r = (wR * (tc[R] + ml[R] + mr[R] + bc[R]) + mc[R]) / (1.0 + 4.0 * wR);
        const double g = (wG * (tc[G] + ml[G] + mr[G] + bc[G]) + mc[G]) / (1.0 + 4.0 * wG);
        const double b = (wB * (tc[B] + ml[B] + mr[B] + bc[B]) + mc[B]) / (1.0 + 4.0 * wB);
        pixel[R] = UNFLOAT(r);
        pixel[G] = UNFLOAT(g);
        pixel[B] = UNFLOAT(b);
        });
    else
        changEachPixelBGR(img, [&](const int i, const int j, PixelF pixel, LineF curLine) {
        const int jp = j < (W - 1) * 3 ? 3 : 0;
        const int jn = j > 3 ? -3 : 0;

        const LineF pLineData = i < H - 1 ? curLine + lineStep : curLine;
        const LineF cLineData = curLine;
        const LineF nLineData = i > 0 ? curLine - lineStep : curLine;

        const PixelF tc = nLineData + j;
        const PixelF ml = cLineData + j + jn, mc = pixel, mr = cLineData + j + jp;
        const PixelF bc = pLineData + j;

        const float minR = MIN5(tc[R], ml[R], mc[R], mr[R], bc[R]);
        const float maxR = MAX5(tc[R], ml[R], mc[R], mr[R], bc[R]);
        const float minG = MIN5(tc[G], ml[G], mc[G], mr[G], bc[G]);
        const float maxG = MAX5(tc[G], ml[G], mc[G], mr[G], bc[G]);
        const float minB = MIN5(tc[B], ml[B], mc[B], mr[B], bc[B]);
        const float maxB = MAX5(tc[B], ml[B], mc[B], mr[B], bc[B]);

        constexpr float peak = LERP(-0.125, -0.2, 1.0);
        const float wR = peak * std::sqrtf(MIN(minR, 255.0f - maxR) * REC(maxR));
        const float wG = peak * std::sqrtf(MIN(minG, 255.0f - maxG) * REC(maxG));
        const float wB = peak * std::sqrtf(MIN(minB, 255.0f - maxB) * REC(maxB));

        const float r = (wR * (tc[R] + ml[R] + mr[R] + bc[R]) + mc[R]) / (1.0f + 4.0f * wR);
        const float g = (wG * (tc[G] + ml[G] + mr[G] + bc[G]) + mc[G]) / (1.0f + 4.0f * wG);
        const float b = (wB * (tc[B] + ml[B] + mr[B] + bc[B]) + mc[B]) / (1.0f + 4.0f * wB);

        pixel[R] = CLAMP(r, 0.0f, 1.0f);
        pixel[G] = CLAMP(g, 0.0f, 1.0f);
        pixel[B] = CLAMP(b, 0.0f, 1.0f);

            });
}

inline void Anime4KCPP::FilterProcessor::changEachPixelBGR(cv::Mat& src,
    const std::function<void(const int, const int, PixelB, LineB)>&& callBack)
{
    cv::Mat tmp;
    src.copyTo(tmp);

    int jMAX = W * 3;
#if defined(_MSC_VER) || defined(USE_TBB)
    Parallel::parallel_for(0, H, [&](int i) {
        LineB lineData = src.data + static_cast<size_t>(i) * static_cast<size_t>(W) * static_cast<size_t>(3);
        LineB tmpLineData = tmp.data + static_cast<size_t>(i) * static_cast<size_t>(W) * static_cast<size_t>(3);
        for (int j = 0; j < jMAX; j += 3)
            callBack(i, j, tmpLineData + j, lineData);
        });
#else
#pragma omp parallel for
    for (int i = 0; i < H; i++)
    {
        LineB lineData = src.data + static_cast<size_t>(i) * static_cast<size_t>(W) * static_cast<size_t>(3);
        LineB tmpLineData = tmp.data + static_cast<size_t>(i) * static_cast<size_t>(W) * static_cast<size_t>(3);
        for (int j = 0; j < jMAX; j += 3)
            callBack(i, j, tmpLineData + j, lineData);
    }
#endif

    src = tmp;
}

inline void Anime4KCPP::FilterProcessor::changEachPixelBGR(cv::Mat& src,
    const std::function<void(const int, const int, PixelF, LineF)>&& callBack)
{
    cv::Mat tmp;
    src.copyTo(tmp);

    int jMAX = W * 3;
#if defined(_MSC_VER) || defined(USE_TBB)
    Parallel::parallel_for(0, H, [&](int i) {
        LineF lineData = reinterpret_cast<LineF>(src.data) + static_cast<size_t>(i) * static_cast<size_t>(W) * static_cast<size_t>(3);
        PixelF tmpLineData = reinterpret_cast<PixelF>(tmp.data) + static_cast<size_t>(i) * static_cast<size_t>(W) * static_cast<size_t>(3);
        for (int j = 0; j < jMAX; j += 3)
            callBack(i, j, tmpLineData + j, lineData);
        });
#else
#pragma omp parallel for
    for (int i = 0; i < H; i++)
    {
        LineF lineData = reinterpret_cast<LineF>(src.data) + static_cast<size_t>(i) * static_cast<size_t>(W) * static_cast<size_t>(3);
        PixelF tmpLineData = reinterpret_cast<PixelF>(tmp.data) + static_cast<size_t>(i) * static_cast<size_t>(W) * static_cast<size_t>(3);
        for (int j = 0; j < jMAX; j += 3)
            callBack(i, j, tmpLineData + j, lineData);
    }
#endif

    src = tmp;
}
