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

namespace Anime4KCPP::Filter::detail
{
    template<typename T, typename F>
    static void changEachPixel(cv::Mat& src, F&& callBack)
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

    template<typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
    static constexpr T clamp(double v)
    {
        return v > std::numeric_limits<T>::max() ?
                std::numeric_limits<T>::max() :
                (std::numeric_limits<T>::min() > v ?
                    std::numeric_limits<T>::min() :
                    static_cast<T>(std::round(v)));
    }

    template<typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
    static constexpr T clamp(double v)
    {
        return static_cast<T>(v < 0.0 ? 0.0 : (1.0 < v ? 1.0 : v));
    }

    template<typename T>
    static void CASSharpeningImpl(cv::Mat& img)
    {
        const int channels = img.channels();
        const size_t lineStep = img.step1();
        detail::changEachPixel<T>(img, [&](const int i, const int j, T* pixel, T* curLine) {
            const int jp = j < (img.cols - 1)* channels ? channels : 0;
            const int jn = j > channels ? -channels : 0;

            const T* const pLineData = i < img.rows - 1 ? curLine + lineStep : curLine;
            const T* const cLineData = curLine;
            const T* const nLineData = i > 0 ? curLine - lineStep : curLine;

            const T* const tc = nLineData + j;
            const T* const ml = cLineData + j + jn, * const mc = pixel, * const mr = cLineData + j + jp;
            const T* const bc = pLineData + j;

            const T minR = MIN5(tc[R], ml[R], mc[R], mr[R], bc[R]);
            const T maxR = MAX5(tc[R], ml[R], mc[R], mr[R], bc[R]);
            const T minG = MIN5(tc[G], ml[G], mc[G], mr[G], bc[G]);
            const T maxG = MAX5(tc[G], ml[G], mc[G], mr[G], bc[G]);
            const T minB = MIN5(tc[B], ml[B], mc[B], mr[B], bc[B]);
            const T maxB = MAX5(tc[B], ml[B], mc[B], mr[B], bc[B]);

            static constexpr double peak = LERP(-0.125, -0.2, 1.0);
            const double wR = peak * std::sqrt(MIN(minR, 255 - maxR) * REC(maxR));
            const double wG = peak * std::sqrt(MIN(minG, 255 - maxG) * REC(maxG));
            const double wB = peak * std::sqrt(MIN(minB, 255 - maxB) * REC(maxB));

            const double r = (wR * (tc[R] + ml[R] + mr[R] + bc[R]) + mc[R]) / (1.0 + 4.0 * wR);
            const double g = (wG * (tc[G] + ml[G] + mr[G] + bc[G]) + mc[G]) / (1.0 + 4.0 * wG);
            const double b = (wB * (tc[B] + ml[B] + mr[B] + bc[B]) + mc[B]) / (1.0 + 4.0 * wB);
            pixel[R] = clamp<T>(r);
            pixel[G] = clamp<T>(g);
            pixel[B] = clamp<T>(b);
            });
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

void Anime4KCPP::FilterProcessor::CASSharpening(cv::Mat& srcImgRef)
{
    const int lineStep = W * 3;
    switch (srcImgRef.depth())
    {
    case CV_8U:
        Filter::detail::CASSharpeningImpl<unsigned char>(srcImgRef);
        break;
    case CV_16U:
        Filter::detail::CASSharpeningImpl<unsigned short>(srcImgRef);
        break;
    case CV_32F:
        Filter::detail::CASSharpeningImpl<float>(srcImgRef);
        break;
    }

}
