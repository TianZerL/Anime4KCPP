#ifndef ENABLE_OPENCV_DNN

#include <limits>

#ifdef USE_RYZEN
#include<immintrin.h>
#elif defined(USE_EIGEN3)
#define EIGEN_DONT_PARALLELIZE
#include<Eigen/Core>
#endif

#include"Parallel.hpp"
#include"CPUCNNProcessor.hpp"

namespace Anime4KCPP::CPU::detail
{
    template<typename T, typename F>
    static void changEachPixel1ToN(const cv::Mat& src, F&& callBack, cv::Mat& tmpMat, int outChannels)
    {
        const int h = src.rows, w = src.cols;
        const int jMAX = w * outChannels;

        tmpMat.create(h, w, CV_32FC(outChannels));

        const std::size_t srcStep = src.step;
        const std::size_t step = tmpMat.step;

        Anime4KCPP::Utils::ParallelFor(0, h,
            [&](const int i) {
                T* lineData = reinterpret_cast<T*>(src.data + static_cast<std::size_t>(i) * srcStep);
                float* tmpLineData = reinterpret_cast<float*>(tmpMat.data + static_cast<std::size_t>(i) * step);
                for (int j = 0; j < jMAX; j += outChannels)
                    callBack(i, j, tmpLineData + j, lineData);
            });
    }

    template<typename F>
    static void changEachPixelNToN(F&& callBack, cv::Mat& tmpMat)
    {
        const int h = tmpMat.rows, w = tmpMat.cols;
        const int channels = tmpMat.channels();
        const int jMAX = w * channels;
        const std::size_t step = tmpMat.step;

        cv::Mat tmp;
        tmp.create(h, w, tmpMat.type());

        Anime4KCPP::Utils::ParallelFor(0, h,
            [&](const int i) {
                float* lineData = reinterpret_cast<float*>(tmpMat.data + static_cast<std::size_t>(i) * step);
                float* tmpLineData = reinterpret_cast<float*>(tmp.data + static_cast<std::size_t>(i) * step);
                for (int j = 0; j < jMAX; j += channels)
                    callBack(i, j, tmpLineData + j, lineData);
            });

        tmpMat = tmp;
    }

    template<typename T, typename F>
    static void changEachPixelNTo1(cv::Mat& img, F&& callBack, const cv::Mat& tmpMat)
    {
        const int h = 2 * tmpMat.rows, w = 2 * tmpMat.cols;
        img.create(h, w, cv::DataType<T>::type);

        const int jMAX = w;
        const std::size_t channels = tmpMat.channels();
        const std::size_t step = tmpMat.step;
        const std::size_t dstStep = img.step;

        Anime4KCPP::Utils::ParallelFor(0, h,
            [&](const int i) {
                float* lineData = reinterpret_cast<float*>(tmpMat.data + static_cast<std::size_t>(i >> 1) * step);
                T* tmpLineData = reinterpret_cast<T*>(img.data + static_cast<std::size_t>(i) * dstStep);
                for (int j = 0; j < jMAX; j++)
                    callBack(i, j, tmpLineData + j, lineData + static_cast<std::size_t>(j >> 1) * channels);
            });
    }

    template<typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
    static constexpr float norm(T v)
    {
        return static_cast<float>(v) / std::numeric_limits<T>::max();
    }

    template<typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
    static constexpr float norm(T v)
    {
        return static_cast<float>(v);
    }

    template<typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
    static constexpr T unnorm(float v)
    {
        return v >= 1.0f ?
            std::numeric_limits<T>::max() :
            (v <= 0.0f ?
                std::numeric_limits<T>::min() :
                static_cast<T>(std::roundf(v * std::numeric_limits<T>::max())));
    }

    template<typename T, std::enable_if_t<std::is_floating_point<T>::value>* = nullptr>
    static constexpr T unnorm(float v)
    {
        return v < 0.0f ? 0.0f : (1.0f < v ? 1.0f : v);
    }

    template<typename T>
    static void conv1To8Impl(const cv::Mat& img, const float* kernels, const float* biases, cv::Mat& tmpMat)
    {
        const int channels = 8;
        const int srcChannels = img.channels();
        const std::size_t lineStep = img.step1();
        detail::changEachPixel1ToN<T>(img, [&](const int i, const int j, float* outMat, T* curLine) {
            const int orgJ = j / channels * srcChannels;
            const int jp = orgJ < (img.cols - 1)* srcChannels ? srcChannels : 0;
            const int jn = orgJ > srcChannels ? -srcChannels : 0;

            const T* const pLineData = i < img.rows - 1 ? curLine + lineStep : curLine;
            const T* const cLineData = curLine;
            const T* const nLineData = i > 0 ? curLine - lineStep : curLine;

            const T* tl = nLineData + orgJ + jn, * const tc = nLineData + orgJ, * const tr = nLineData + orgJ + jp;
            const T* ml = cLineData + orgJ + jn, * const mc = cLineData + orgJ, * const mr = cLineData + orgJ + jp;
            const T* bl = pLineData + orgJ + jn, * const bc = pLineData + orgJ, * const br = pLineData + orgJ + jp;

            const float tln = norm<T>(tl[Y]);
            const float tcn = norm<T>(tc[Y]);
            const float trn = norm<T>(tr[Y]);
            const float mln = norm<T>(ml[Y]);
            const float mcn = norm<T>(mc[Y]);
            const float mrn = norm<T>(mr[Y]);
            const float bln = norm<T>(bl[Y]);
            const float bcn = norm<T>(bc[Y]);
            const float brn = norm<T>(br[Y]);

#ifdef USE_RYZEN
            const float* const kptr = kernels;
            const float* const bptr = biases;

            _mm256_zeroall();

            __m256 _out0 = _mm256_loadu_ps(bptr);
            __m256 _out1 = _mm256_setzero_ps();
            __m256 _out2 = _mm256_setzero_ps();

            const __m256 _r0 = _mm256_set1_ps(tln);
            const __m256 _r1 = _mm256_set1_ps(tcn);
            const __m256 _r2 = _mm256_set1_ps(trn);
            const __m256 _r3 = _mm256_set1_ps(mln);
            const __m256 _r4 = _mm256_set1_ps(mcn);
            const __m256 _r5 = _mm256_set1_ps(mrn);
            const __m256 _r6 = _mm256_set1_ps(bln);
            const __m256 _r7 = _mm256_set1_ps(bcn);
            const __m256 _r8 = _mm256_set1_ps(brn);

            const __m256 _k0 = _mm256_loadu_ps(kptr);
            const __m256 _k1 = _mm256_loadu_ps(kptr + 8);
            const __m256 _k2 = _mm256_loadu_ps(kptr + 16);
            const __m256 _k3 = _mm256_loadu_ps(kptr + 24);
            const __m256 _k4 = _mm256_loadu_ps(kptr + 32);
            const __m256 _k5 = _mm256_loadu_ps(kptr + 40);
            const __m256 _k6 = _mm256_loadu_ps(kptr + 48);
            const __m256 _k7 = _mm256_loadu_ps(kptr + 56);
            const __m256 _k8 = _mm256_loadu_ps(kptr + 64);

            _out0 = _mm256_fmadd_ps(_r0, _k0, _out0);
            _out1 = _mm256_fmadd_ps(_r1, _k1, _out1);
            _out2 = _mm256_fmadd_ps(_r2, _k2, _out2);
            _out0 = _mm256_fmadd_ps(_r3, _k3, _out0);
            _out1 = _mm256_fmadd_ps(_r4, _k4, _out1);
            _out2 = _mm256_fmadd_ps(_r5, _k5, _out2);
            _out0 = _mm256_fmadd_ps(_r6, _k6, _out0);
            _out1 = _mm256_fmadd_ps(_r7, _k7, _out1);
            _out2 = _mm256_fmadd_ps(_r8, _k8, _out2);

            _out0 = _mm256_max_ps(_mm256_add_ps(_out2, _mm256_add_ps(_out0, _out1)), _mm256_setzero_ps());

            _mm256_storeu_ps(outMat, _out0);

#elif defined(USE_EIGEN3)
            float* const kptr = const_cast<float*>(kernels);
            float* const bptr = const_cast<float*>(biases);

            Eigen::Array<float, 8, 1> out = Eigen::Map<Eigen::Array<float, 8, 1>>(bptr, 8);

            const Eigen::Map<Eigen::Array<float, 8, 1>> k0(kptr, 8);
            const Eigen::Map<Eigen::Array<float, 8, 1>> k1(kptr + 8, 8);
            const Eigen::Map<Eigen::Array<float, 8, 1>> k2(kptr + 16, 8);
            const Eigen::Map<Eigen::Array<float, 8, 1>> k3(kptr + 24, 8);
            const Eigen::Map<Eigen::Array<float, 8, 1>> k4(kptr + 32, 8);
            const Eigen::Map<Eigen::Array<float, 8, 1>> k5(kptr + 40, 8);
            const Eigen::Map<Eigen::Array<float, 8, 1>> k6(kptr + 48, 8);
            const Eigen::Map<Eigen::Array<float, 8, 1>> k7(kptr + 56, 8);
            const Eigen::Map<Eigen::Array<float, 8, 1>> k8(kptr + 64, 8);

            auto t0 = tln * k0;
            auto t1 = tcn * k1;
            auto t2 = trn * k2;
            auto t3 = mln * k3;
            auto t4 = mcn * k4;
            auto t5 = mrn * k5;
            auto t6 = bln * k6;
            auto t7 = bcn * k7;
            auto t8 = brn * k8;

            out += (t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8);

            Eigen::Map<Eigen::Array<float, 8, 1>>(outMat, 8) = out.max(0.0f);
#else
            const float* const kptr = kernels;

            const float* const k0 = kptr;
            const float* const k1 = kptr + 8;
            const float* const k2 = kptr + 16;
            const float* const k3 = kptr + 24;
            const float* const k4 = kptr + 32;
            const float* const k5 = kptr + 40;
            const float* const k6 = kptr + 48;
            const float* const k7 = kptr + 56;
            const float* const k8 = kptr + 64;

            float out[8];
            std::copy_n(biases, 8, out);

            for (std::size_t i = 0; i < 8; i++)
                out[i] += tln * k0[i];
            for (std::size_t i = 0; i < 8; i++)
                out[i] += tcn * k1[i];
            for (std::size_t i = 0; i < 8; i++)
                out[i] += trn * k2[i];
            for (std::size_t i = 0; i < 8; i++)
                out[i] += mln * k3[i];
            for (std::size_t i = 0; i < 8; i++)
                out[i] += mcn * k4[i];
            for (std::size_t i = 0; i < 8; i++)
                out[i] += mrn * k5[i];
            for (std::size_t i = 0; i < 8; i++)
                out[i] += bln * k6[i];
            for (std::size_t i = 0; i < 8; i++)
                out[i] += bcn * k7[i];
            for (std::size_t i = 0; i < 8; i++)
                out[i] += brn * k8[i];

            for (std::size_t i = 0; i < 8; i++)
                out[i] = std::max<float>(out[i], 0);

            std::copy_n(out, 8, outMat);
#endif // USE_RYZEN
            }, tmpMat, 8);
    }

    template<typename T>
    static void convTranspose8To1Impl(cv::Mat& img, const float* kernels, cv::Mat& tmpMat)
    {
        detail::changEachPixelNTo1<T>(img, [&](const int i, const int j, T* outMat, float* inMat) {
            const int index = ((i & 1) << 1) + (j & 1);

            //180 degree rotation for kernel
            //0 1  to  3 2
            //2 3      1 0

#ifdef USE_RYZEN
            const __m256 _in = _mm256_loadu_ps(inMat);
            const __m256 _k0 = _mm256_loadu_ps(kernels + index * 8);
            const __m256 _r0 = _mm256_dp_ps(_in, _k0, 0xf1);
            const __m128 _r1 = _mm256_extractf128_ps(_r0, 0x01);
            const __m128 _r2 = _mm256_castps256_ps128(_r0);
            const __m128 _r3 = _mm_add_ps(_r1, _r2);

            const float luma = _mm_cvtss_f32(_r3);

            _mm256_zeroupper();

#elif defined(USE_EIGEN3)
            float* const kptr = const_cast<float*>(kernels + index * 8);

            const float luma =
                Eigen::Map<Eigen::Matrix<float, 8, 1>>(inMat)
                .dot(Eigen::Map<Eigen::Matrix<float, 8, 1>>(kptr));
#else
            const float* const kptr = kernels + index * 8;

            float luma = 0;
            for (std::size_t i = 0; i < 8; i++)
                luma += kptr[i] * inMat[i];
#endif

            *outMat = unnorm<T>(luma);
            }, tmpMat);
    }
}

void Anime4KCPP::CPU::CNNProcessor::conv1To8(const cv::Mat& img, const float* kernels, const float* biases, cv::Mat& tmpMat)
{
    switch (img.depth())
    {
    case CV_8U:
        detail::conv1To8Impl<std::uint8_t>(img, kernels, biases, tmpMat);
        break;
    case CV_16U:
        detail::conv1To8Impl<std::uint16_t>(img, kernels, biases, tmpMat);
        break;
    case CV_32F:
        detail::conv1To8Impl<float>(img, kernels, biases, tmpMat);
        break;
    default:
        throw ACException<ExceptionType::RunTimeError>("Unsupported image data type");
    }
}

void Anime4KCPP::CPU::CNNProcessor::conv8To8(const float* kernels, const float* biases, cv::Mat& tmpMat)
{
    const int channels = 8;
    const std::size_t lineStep = tmpMat.step1();
    detail::changEachPixelNToN([&](const int i, const int j, float* outMat, float* curLine) {
        const int jp = j < (tmpMat.cols - 1)* channels ? channels : 0;
        const int jn = j > channels ? -channels : 0;

        const float* const pLineData = i < tmpMat.rows - 1 ? curLine + lineStep : curLine;
        const float* const cLineData = curLine;
        const float* const nLineData = i > 0 ? curLine - lineStep : curLine;

        const float* const tl = nLineData + j + jn, * const tc = nLineData + j, * const tr = nLineData + j + jp;
        const float* const ml = cLineData + j + jn, * const mc = cLineData + j, * const mr = cLineData + j + jp;
        const float* const bl = pLineData + j + jn, * const bc = pLineData + j, * const br = pLineData + j + jp;

#ifdef USE_RYZEN
        const float* const kptr = kernels;
        const float* const bptr = biases;

        __m256 _out0 = _mm256_loadu_ps(bptr);
        __m256 _out1 = _mm256_setzero_ps();
        __m256 _out2 = _mm256_setzero_ps();

        for (std::size_t i = 0; i < 8; i += 2)
        {
            const __m256 _r00 = _mm256_broadcast_ss(tl + i);
            const __m256 _r01 = _mm256_broadcast_ss(tc + i);
            const __m256 _r02 = _mm256_broadcast_ss(tr + i);
            const __m256 _r03 = _mm256_broadcast_ss(ml + i);
            const __m256 _r04 = _mm256_broadcast_ss(mc + i);
            const __m256 _r05 = _mm256_broadcast_ss(mr + i);
            const __m256 _r06 = _mm256_broadcast_ss(bl + i);
            const __m256 _r07 = _mm256_broadcast_ss(bc + i);
            const __m256 _r08 = _mm256_broadcast_ss(br + i);

            const __m256 _k00 = _mm256_loadu_ps(kptr + i * 72);
            const __m256 _k01 = _mm256_loadu_ps(kptr + i * 72 + 8);
            const __m256 _k02 = _mm256_loadu_ps(kptr + i * 72 + 16);
            const __m256 _k03 = _mm256_loadu_ps(kptr + i * 72 + 24);
            const __m256 _k04 = _mm256_loadu_ps(kptr + i * 72 + 32);
            const __m256 _k05 = _mm256_loadu_ps(kptr + i * 72 + 40);
            const __m256 _k06 = _mm256_loadu_ps(kptr + i * 72 + 48);
            const __m256 _k07 = _mm256_loadu_ps(kptr + i * 72 + 56);
            const __m256 _k08 = _mm256_loadu_ps(kptr + i * 72 + 64);

            _out0 = _mm256_fmadd_ps(_r00, _k00, _out0);
            _out1 = _mm256_fmadd_ps(_r01, _k01, _out1);
            _out2 = _mm256_fmadd_ps(_r02, _k02, _out2);
            _out0 = _mm256_fmadd_ps(_r03, _k03, _out0);
            _out1 = _mm256_fmadd_ps(_r04, _k04, _out1);
            _out2 = _mm256_fmadd_ps(_r05, _k05, _out2);
            _out0 = _mm256_fmadd_ps(_r06, _k06, _out0);
            _out1 = _mm256_fmadd_ps(_r07, _k07, _out1);
            _out2 = _mm256_fmadd_ps(_r08, _k08, _out2);

            const __m256 _r10 = _mm256_broadcast_ss(tl + i + 1);
            const __m256 _r11 = _mm256_broadcast_ss(tc + i + 1);
            const __m256 _r12 = _mm256_broadcast_ss(tr + i + 1);
            const __m256 _r13 = _mm256_broadcast_ss(ml + i + 1);
            const __m256 _r14 = _mm256_broadcast_ss(mc + i + 1);
            const __m256 _r15 = _mm256_broadcast_ss(mr + i + 1);
            const __m256 _r16 = _mm256_broadcast_ss(bl + i + 1);
            const __m256 _r17 = _mm256_broadcast_ss(bc + i + 1);
            const __m256 _r18 = _mm256_broadcast_ss(br + i + 1);

            const __m256 _k10 = _mm256_loadu_ps(kptr + (i + 1) * 72);
            const __m256 _k11 = _mm256_loadu_ps(kptr + (i + 1) * 72 + 8);
            const __m256 _k12 = _mm256_loadu_ps(kptr + (i + 1) * 72 + 16);
            const __m256 _k13 = _mm256_loadu_ps(kptr + (i + 1) * 72 + 24);
            const __m256 _k14 = _mm256_loadu_ps(kptr + (i + 1) * 72 + 32);
            const __m256 _k15 = _mm256_loadu_ps(kptr + (i + 1) * 72 + 40);
            const __m256 _k16 = _mm256_loadu_ps(kptr + (i + 1) * 72 + 48);
            const __m256 _k17 = _mm256_loadu_ps(kptr + (i + 1) * 72 + 56);
            const __m256 _k18 = _mm256_loadu_ps(kptr + (i + 1) * 72 + 64);

            _out0 = _mm256_fmadd_ps(_r10, _k10, _out0);
            _out1 = _mm256_fmadd_ps(_r11, _k11, _out1);
            _out2 = _mm256_fmadd_ps(_r12, _k12, _out2);
            _out0 = _mm256_fmadd_ps(_r13, _k13, _out0);
            _out1 = _mm256_fmadd_ps(_r14, _k14, _out1);
            _out2 = _mm256_fmadd_ps(_r15, _k15, _out2);
            _out0 = _mm256_fmadd_ps(_r16, _k16, _out0);
            _out1 = _mm256_fmadd_ps(_r17, _k17, _out1);
            _out2 = _mm256_fmadd_ps(_r18, _k18, _out2);
        }
        _out0 = _mm256_max_ps(_mm256_add_ps(_out2, _mm256_add_ps(_out0, _out1)), _mm256_setzero_ps());

        _mm256_storeu_ps(outMat, _out0);
#elif defined(USE_EIGEN3)
        float* const kptr = const_cast<float*>(kernels);
        float* const bptr = const_cast<float*>(biases);

        Eigen::Array<float, 8, 1> out = Eigen::Map<Eigen::Array<float, 8, 1>>(bptr, 8);

        for (std::size_t i = 0; i < 8; i++)
        {
            const Eigen::Map<Eigen::Array<float, 8, 1>> k0(kptr + i * 72);
            const Eigen::Map<Eigen::Array<float, 8, 1>> k1(kptr + i * 72 + 8);
            const Eigen::Map<Eigen::Array<float, 8, 1>> k2(kptr + i * 72 + 16);
            const Eigen::Map<Eigen::Array<float, 8, 1>> k3(kptr + i * 72 + 24);
            const Eigen::Map<Eigen::Array<float, 8, 1>> k4(kptr + i * 72 + 32);
            const Eigen::Map<Eigen::Array<float, 8, 1>> k5(kptr + i * 72 + 40);
            const Eigen::Map<Eigen::Array<float, 8, 1>> k6(kptr + i * 72 + 48);
            const Eigen::Map<Eigen::Array<float, 8, 1>> k7(kptr + i * 72 + 56);
            const Eigen::Map<Eigen::Array<float, 8, 1>> k8(kptr + i * 72 + 64);

            auto t0 = tl[i] * k0;
            auto t1 = tc[i] * k1;
            auto t2 = tr[i] * k2;
            auto t3 = ml[i] * k3;
            auto t4 = mc[i] * k4;
            auto t5 = mr[i] * k5;
            auto t6 = bl[i] * k6;
            auto t7 = bc[i] * k7;
            auto t8 = br[i] * k8;

            out += (t0 + t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8);
        }

        Eigen::Map<Eigen::Array<float, 8, 1>>(outMat, 8) = out.max(0.0f);
#else
        float out[8];
        std::copy_n(biases, 8, out);

        const float* const kptr = kernels;
        for (std::size_t c = 0; c < 8; c++)
        {
            const float* const k0 = kptr + c * 72;
            const float* const k1 = kptr + c * 72 + 8;
            const float* const k2 = kptr + c * 72 + 16;
            const float* const k3 = kptr + c * 72 + 24;
            const float* const k4 = kptr + c * 72 + 32;
            const float* const k5 = kptr + c * 72 + 40;
            const float* const k6 = kptr + c * 72 + 48;
            const float* const k7 = kptr + c * 72 + 56;
            const float* const k8 = kptr + c * 72 + 64;

            for (std::size_t i = 0; i < 8; i++)
                out[i] += tl[c] * k0[i];
            for (std::size_t i = 0; i < 8; i++)
                out[i] += tc[c] * k1[i];
            for (std::size_t i = 0; i < 8; i++)
                out[i] += tr[c] * k2[i];
            for (std::size_t i = 0; i < 8; i++)
                out[i] += ml[c] * k3[i];
            for (std::size_t i = 0; i < 8; i++)
                out[i] += mc[c] * k4[i];
            for (std::size_t i = 0; i < 8; i++)
                out[i] += mr[c] * k5[i];
            for (std::size_t i = 0; i < 8; i++)
                out[i] += bl[c] * k6[i];
            for (std::size_t i = 0; i < 8; i++)
                out[i] += bc[c] * k7[i];
            for (std::size_t i = 0; i < 8; i++)
                out[i] += br[c] * k8[i];
        }

        for (std::size_t i = 0; i < 8; i++)
            out[i] = std::max<float>(out[i], 0);

        std::copy_n(out, 8, outMat);
#endif // USE_RYZEN

        }, tmpMat);
}

void Anime4KCPP::CPU::CNNProcessor::convTranspose8To1(cv::Mat& img, const float* kernels, cv::Mat& tmpMat)
{
    switch (img.depth())
    {
    case CV_8U:
        detail::convTranspose8To1Impl<std::uint8_t>(img, kernels, tmpMat);
        break;
    case CV_16U:
        detail::convTranspose8To1Impl<std::uint16_t>(img, kernels, tmpMat);
        break;
    case CV_32F:
        detail::convTranspose8To1Impl<float>(img, kernels, tmpMat);
        break;
    default:
        throw ACException<ExceptionType::RunTimeError>("Unsupported image data type");
    }
}

#endif
