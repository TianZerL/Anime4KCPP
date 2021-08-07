#include"Parallel.hpp"
#include"AC.hpp"
#include"FilterProcessor.hpp"

#define MAX5(a, b, c, d, e) std::max({a, b, c, d, e})
#define MIN5(a, b, c, d, e) std::min({a, b, c, d, e})
#define LERP(x, y, w) ((x) * (1.0 - (w)) + (y) * (w))
#define REC(n) ((n) < 1 ? 1.0 : 1.0 / (n))
#define UNFLOATB(n) ((n) >= 255 ? 255 : ((n) <= 0 ? 0 : uint8_t((n) + 0.5)))
#define UNFLOATW(n) ((n) >= 65535 ? 65535 : ((n) <= 0 ? 0 : uint16_t((n) + 0.5)))
#define CLAMP(v, lo, hi) ((v < lo) ? lo : (hi < v) ? hi : v)

namespace Anime4KCPP
{
    namespace detail
    {
        template<typename T, typename F>
        void changEachPixel(cv::Mat& src, F&& callBack)
        {
            cv::Mat tmp;
            src.copyTo(tmp);

            const int h = src.rows, w = src.cols;
            const int channels = src.channels();
            const int jMAX = w * channels;
            const size_t step = src.step;

            Anime4KCPP::Utils::ParallelFor(0, h,
                [&](const int i) {
                    T* lineData = reinterpret_cast<T*>(src.data + static_cast<size_t>(i) * step);
                    T* tmpLineData = reinterpret_cast<T*>(tmp.data + static_cast<size_t>(i) * step);
                    for (int j = 0; j < jMAX; j += channels)
                        callBack(i, j, tmpLineData + j, lineData);
                });

            src = tmp;
        }
    }
}

Anime4KCPP::FilterProcessor::FilterProcessor(cv::Mat& srcImg, uint8_t filters)
    :filters(filters), srcImgRef(srcImg)
{
    H = srcImgRef.rows;
    W = srcImgRef.cols;
}

void Anime4KCPP::FilterProcessor::process()
{
    if (filters & MEDIAN_BLUR)
    {
        cv::Mat tmpImg;
        cv::medianBlur(srcImgRef, tmpImg, 3);
        srcImgRef = tmpImg;
    }
    if (filters & MEAN_BLUR)
    {
        cv::Mat tmpImg;
        cv::blur(srcImgRef, tmpImg, cv::Size(3, 3));
        srcImgRef = tmpImg;
    }
    if (filters & CAS_SHARPENING)
        CASSharpening(srcImgRef);
    if (filters & GAUSSIAN_BLUR_WEAK)
    {
        cv::Mat tmpImg;
        cv::GaussianBlur(srcImgRef, tmpImg, cv::Size(3, 3), 0.5);
        srcImgRef = tmpImg;
    }
    else if (filters & GAUSSIAN_BLUR)
    {
        cv::Mat tmpImg;
        cv::GaussianBlur(srcImgRef, tmpImg, cv::Size(3, 3), 1);
        srcImgRef = tmpImg;
    }
    if (filters & BILATERAL_FILTER)
    {
        cv::Mat tmpImg;
        cv::bilateralFilter(srcImgRef, tmpImg, 9, 30, 30);
        srcImgRef = tmpImg;
    }
    else if (filters & BILATERAL_FILTER_FAST)
    {
        cv::Mat tmpImg;
        cv::bilateralFilter(srcImgRef, tmpImg, 5, 35, 35);
        srcImgRef = tmpImg;
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

inline void Anime4KCPP::FilterProcessor::CASSharpening(cv::Mat& srcImgRef)
{
    const int lineStep = W * 3;
    switch (srcImgRef.depth())
    {
    case CV_8U:
        detail::changEachPixel<unsigned char>(srcImgRef, [&](const int i, const int j, PixelB pixel, LineB curLine) {
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
            pixel[R] = UNFLOATB(r);
            pixel[G] = UNFLOATB(g);
            pixel[B] = UNFLOATB(b);
            });
        break;
    case CV_16U:
        detail::changEachPixel<unsigned short>(srcImgRef, [&](const int i, const int j, PixelW pixel, LineW curLine) {
            const int jp = j < (W - 1) * 3 ? 3 : 0;
            const int jn = j > 3 ? -3 : 0;

            const LineW pLineData = i < H - 1 ? curLine + lineStep : curLine;
            const LineW cLineData = curLine;
            const LineW nLineData = i > 0 ? curLine - lineStep : curLine;

            const PixelW tc = nLineData + j;
            const PixelW ml = cLineData + j + jn, mc = pixel, mr = cLineData + j + jp;
            const PixelW bc = pLineData + j;

            const uint16_t minR = MIN5(tc[R], ml[R], mc[R], mr[R], bc[R]);
            const uint16_t maxR = MAX5(tc[R], ml[R], mc[R], mr[R], bc[R]);
            const uint16_t minG = MIN5(tc[G], ml[G], mc[G], mr[G], bc[G]);
            const uint16_t maxG = MAX5(tc[G], ml[G], mc[G], mr[G], bc[G]);
            const uint16_t minB = MIN5(tc[B], ml[B], mc[B], mr[B], bc[B]);
            const uint16_t maxB = MAX5(tc[B], ml[B], mc[B], mr[B], bc[B]);

            constexpr double peak = LERP(-0.125, -0.2, 1.0);
            const double wR = peak * std::sqrt(MIN(minR, 65535 - maxR) * REC(maxR));
            const double wG = peak * std::sqrt(MIN(minG, 65535 - maxG) * REC(maxG));
            const double wB = peak * std::sqrt(MIN(minB, 65535 - maxB) * REC(maxB));

            const double r = (wR * (tc[R] + ml[R] + mr[R] + bc[R]) + mc[R]) / (1.0 + 4.0 * wR);
            const double g = (wG * (tc[G] + ml[G] + mr[G] + bc[G]) + mc[G]) / (1.0 + 4.0 * wG);
            const double b = (wB * (tc[B] + ml[B] + mr[B] + bc[B]) + mc[B]) / (1.0 + 4.0 * wB);
            pixel[R] = UNFLOATW(r);
            pixel[G] = UNFLOATW(g);
            pixel[B] = UNFLOATW(b);
            });
        break;
    case CV_32F:
        detail::changEachPixel<float>(srcImgRef, [&](const int i, const int j, PixelF pixel, LineF curLine) {
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
            const float wR = peak * sqrtf(MIN(minR, 1.0f - maxR) * REC(maxR));
            const float wG = peak * sqrtf(MIN(minG, 1.0f - maxG) * REC(maxG));
            const float wB = peak * sqrtf(MIN(minB, 1.0f - maxB) * REC(maxB));

            const float r = (wR * (tc[R] + ml[R] + mr[R] + bc[R]) + mc[R]) / (1.0f + 4.0f * wR);
            const float g = (wG * (tc[G] + ml[G] + mr[G] + bc[G]) + mc[G]) / (1.0f + 4.0f * wG);
            const float b = (wB * (tc[B] + ml[B] + mr[B] + bc[B]) + mc[B]) / (1.0f + 4.0f * wB);

            pixel[R] = CLAMP(r, 0.0f, 1.0f);
            pixel[G] = CLAMP(g, 0.0f, 1.0f);
            pixel[B] = CLAMP(b, 0.0f, 1.0f);

            });
        break;
    }

}
