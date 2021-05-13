#ifndef ENABLE_OPENCV_DNN

#ifdef USE_RYZEN
#include<immintrin.h>
#elif defined(USE_EIGEN3)
#include<Eigen/Core>
#endif

#include"Parallel.hpp"
#include"CPUCNNProcessor.hpp"

#define NORMB(X) (static_cast<float>(X) / static_cast<float>(255.0))
#define NORMW(X) (static_cast<float>(X) / static_cast<float>(65535.0))
#define UNNORMB(n) ((n) >= static_cast<float>(1.0)? static_cast<uint8_t>(255) : ((n) <= static_cast<float>(0.0) ? static_cast<uint8_t>(0) : static_cast<uint8_t>(n * static_cast<float>(255.0) + static_cast<float>(0.5))))
#define UNNORMW(n) ((n) >= static_cast<float>(1.0)? static_cast<uint16_t>(65535) : ((n) <= static_cast<float>(0.0) ? static_cast<uint16_t>(0) : static_cast<uint16_t>(n * static_cast<float>(65535.0) + static_cast<float>(0.5))))
#define CLAMP(v, lo, hi) ((v < lo) ? lo : (hi < v) ? hi : v)

namespace Anime4KCPP
{
    namespace detail
    {
        template<typename T, typename F>
        void changEachPixel1ToN(const cv::Mat& src, F&& callBack, cv::Mat& tmpMat, int outChannels)
        {
            const int h = src.rows, w = src.cols;
            const int jMAX = w * outChannels;

            tmpMat.create(h, w, CV_32FC(outChannels));

            const size_t srcStep = src.step;
            const size_t step = tmpMat.step;

            Anime4KCPP::Utils::ParallelFor(0, h, 
                [&](const int i) {
                    T* lineData = reinterpret_cast<T*>(src.data + static_cast<size_t>(i) * srcStep);
                    float* tmpLineData = reinterpret_cast<float*>(tmpMat.data + static_cast<size_t>(i) * step);
                    for (int j = 0; j < jMAX; j += outChannels)
                        callBack(i, j, tmpLineData + j, lineData);
                });
        }

        template<typename F>
        void changEachPixelNToN(F && callBack, cv::Mat & tmpMat)
        {
            const int h = tmpMat.rows, w = tmpMat.cols;
            const int channels = tmpMat.channels();
            const int jMAX = w * channels;
            const size_t step = tmpMat.step;

            cv::Mat tmp;
            tmp.create(h, w, tmpMat.type());

            Anime4KCPP::Utils::ParallelFor(0, h, 
                [&](const int i) {
                    float* lineData = reinterpret_cast<float*>(tmpMat.data + static_cast<size_t>(i) * step);
                    float* tmpLineData = reinterpret_cast<float*>(tmp.data + static_cast<size_t>(i) * step);
                    for (int j = 0; j < jMAX; j += channels)
                        callBack(i, j, tmpLineData + j, lineData);
                });

            tmpMat = tmp;
        }

        template<typename T, typename F>
        void changEachPixelNTo1(cv::Mat& img, F&& callBack, const cv::Mat& tmpMat)
        {
            const int h = 2 * tmpMat.rows, w = 2 * tmpMat.cols;
            img.create(h, w, cv::DataType<T>::type);

            const int jMAX = w;
            const size_t channels = tmpMat.channels();
            const size_t step = tmpMat.step;
            const size_t dstStep = img.step;

            Anime4KCPP::Utils::ParallelFor(0, h, 
                [&](const int i) {
                    float* lineData = reinterpret_cast<float*>(tmpMat.data + static_cast<size_t>(i >> 1) * step);
                    T* tmpLineData = reinterpret_cast<T*>(img.data + static_cast<size_t>(i) * dstStep);
                    for (int j = 0; j < jMAX; j++)
                        callBack(i, j, tmpLineData + j, lineData + static_cast<size_t>(j >> 1) * channels);
                });
        }
    }
}

void Anime4KCPP::CPU::CNNProcessor::conv1To8B(const cv::Mat& img, const float* kernels, const float* biases, cv::Mat& tmpMat)
{
    const int channels = 8;
    const int srcChannels = img.channels();
    const size_t lineStep = img.step;
    detail::changEachPixel1ToN<unsigned char>(img, [&](const int i, const int j, ChanFP outMat, LineB curLine) {
        const int orgJ = j / channels * srcChannels;
        const int jp = orgJ < (img.cols - 1)* srcChannels ? srcChannels : 0;
        const int jn = orgJ > srcChannels ? -srcChannels : 0;

        const LineB pLineData = i < img.rows - 1 ? curLine + lineStep : curLine;
        const LineB cLineData = curLine;
        const LineB nLineData = i > 0 ? curLine - lineStep : curLine;

        const PixelB tl = nLineData + orgJ + jn, tc = nLineData + orgJ, tr = nLineData + orgJ + jp;
        const PixelB ml = cLineData + orgJ + jn, mc = cLineData + orgJ, mr = cLineData + orgJ + jp;
        const PixelB bl = pLineData + orgJ + jn, bc = pLineData + orgJ, br = pLineData + orgJ + jp;

        const float tln = NORMB(tl[Y]);
        const float tcn = NORMB(tc[Y]);
        const float trn = NORMB(tr[Y]);
        const float mln = NORMB(ml[Y]);
        const float mcn = NORMB(mc[Y]);
        const float mrn = NORMB(mr[Y]);
        const float bln = NORMB(bl[Y]);
        const float bcn = NORMB(bc[Y]);
        const float brn = NORMB(br[Y]);

#ifdef USE_RYZEN
        const float* kptr = kernels;
        const float* bptr = biases;

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
        float* kptr = const_cast<float*>(kernels);
        float* bptr = const_cast<float*>(biases);

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

        out += tln * k0;
        out += tcn * k1;
        out += trn * k2;
        out += mln * k3;
        out += mcn * k4;
        out += mrn * k5;
        out += bln * k6;
        out += bcn * k7;
        out += brn * k8;

        Eigen::Map<Eigen::Array<float, 8, 1>>(outMat, 8) = out.max(0.0f);
#else
        const float* kptr = kernels;

        const float* k0 = kptr;
        const float* k1 = kptr + 8;
        const float* k2 = kptr + 16;
        const float* k3 = kptr + 24;
        const float* k4 = kptr + 32;
        const float* k5 = kptr + 40;
        const float* k6 = kptr + 48;
        const float* k7 = kptr + 56;
        const float* k8 = kptr + 64;

        float out[8];
        std::copy_n(biases, 8, out);

        for (size_t i = 0; i < 8; i++)
            out[i] += tln * k0[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += tcn * k1[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += trn * k2[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += mln * k3[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += mcn * k4[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += mrn * k5[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += bln * k6[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += bcn * k7[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += brn * k8[i];

        for (size_t i = 0; i < 8; i++)
            out[i] = std::max<float>(out[i], 0);

        std::copy_n(out, 8, outMat);
#endif // USE_RYZEN
        }, tmpMat, 8);
}

void Anime4KCPP::CPU::CNNProcessor::conv1To8W(const cv::Mat& img, const float* kernels, const float* biases, cv::Mat& tmpMat)
{
    const int channels = 8;
    const int srcChannels = img.channels();
    const size_t lineStep = img.step;
    detail::changEachPixel1ToN<unsigned short>(img, [&](const int i, const int j, ChanFP outMat, LineW curLine) {
        const int orgJ = j / channels * srcChannels;
        const int jp = orgJ < (img.cols - 1)* srcChannels ? srcChannels : 0;
        const int jn = orgJ > srcChannels ? -srcChannels : 0;

        const LineB tempLine = reinterpret_cast<LineB>(curLine);
        const LineW pLineData = i < img.rows - 1 ? reinterpret_cast<LineW>(tempLine + lineStep) : curLine;
        const LineW cLineData = curLine;
        const LineW nLineData = i > 0 ? reinterpret_cast<LineW>(tempLine - lineStep) : curLine;

        const PixelW tl = nLineData + orgJ + jn, tc = nLineData + orgJ, tr = nLineData + orgJ + jp;
        const PixelW ml = cLineData + orgJ + jn, mc = cLineData + orgJ, mr = cLineData + orgJ + jp;
        const PixelW bl = pLineData + orgJ + jn, bc = pLineData + orgJ, br = pLineData + orgJ + jp;

        const float tln = NORMW(tl[Y]);
        const float tcn = NORMW(tc[Y]);
        const float trn = NORMW(tr[Y]);
        const float mln = NORMW(ml[Y]);
        const float mcn = NORMW(mc[Y]);
        const float mrn = NORMW(mr[Y]);
        const float bln = NORMW(bl[Y]);
        const float bcn = NORMW(bc[Y]);
        const float brn = NORMW(br[Y]);

#ifdef USE_RYZEN
        const float* kptr = kernels;
        const float* bptr = biases;

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
        float* kptr = const_cast<float*>(kernels);
        float* bptr = const_cast<float*>(biases);

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

        out += tln * k0;
        out += tcn * k1;
        out += trn * k2;
        out += mln * k3;
        out += mcn * k4;
        out += mrn * k5;
        out += bln * k6;
        out += bcn * k7;
        out += brn * k8;

        Eigen::Map<Eigen::Array<float, 8, 1>>(outMat, 8) = out.max(0.0f);
#else
        const float* kptr = kernels;

        const float* k0 = kptr;
        const float* k1 = kptr + 8;
        const float* k2 = kptr + 16;
        const float* k3 = kptr + 24;
        const float* k4 = kptr + 32;
        const float* k5 = kptr + 40;
        const float* k6 = kptr + 48;
        const float* k7 = kptr + 56;
        const float* k8 = kptr + 64;

        float out[8];
        std::copy_n(biases, 8, out);

        for (size_t i = 0; i < 8; i++)
            out[i] += tln * k0[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += tcn * k1[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += trn * k2[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += mln * k3[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += mcn * k4[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += mrn * k5[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += bln * k6[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += bcn * k7[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += brn * k8[i];

        for (size_t i = 0; i < 8; i++)
            out[i] = std::max<float>(out[i], 0);

        std::copy_n(out, 8, outMat);
#endif // USE_RYZEN

        }, tmpMat, 8);
}

void Anime4KCPP::CPU::CNNProcessor::conv1To8F(const cv::Mat& img, const float* kernels, const float* biases, cv::Mat& tmpMat)
{
    const int channels = 8;
    const int srcChannels = img.channels();
    const size_t lineStep = img.step;
    detail::changEachPixel1ToN<float>(img, [&](const int i, const int j, ChanFP outMat, LineF curLine) {
        const int orgJ = j / channels * srcChannels;
        const int jp = orgJ < (img.cols - 1)* srcChannels ? srcChannels : 0;
        const int jn = orgJ > srcChannels ? -srcChannels : 0;

        const LineB tempLine = reinterpret_cast<LineB>(curLine);
        const LineF pLineData = i < img.rows - 1 ? reinterpret_cast<LineF>(tempLine + lineStep) : curLine;
        const LineF cLineData = curLine;
        const LineF nLineData = i > 0 ? reinterpret_cast<LineF>(tempLine - lineStep) : curLine;

        const PixelF tl = nLineData + orgJ + jn, tc = nLineData + orgJ, tr = nLineData + orgJ + jp;
        const PixelF ml = cLineData + orgJ + jn, mc = cLineData + orgJ, mr = cLineData + orgJ + jp;
        const PixelF bl = pLineData + orgJ + jn, bc = pLineData + orgJ, br = pLineData + orgJ + jp;

#ifdef USE_RYZEN
        const float* kptr = kernels;
        const float* bptr = biases;

        _mm256_zeroall();

        __m256 _out0 = _mm256_loadu_ps(bptr);
        __m256 _out1 = _mm256_setzero_ps();
        __m256 _out2 = _mm256_setzero_ps();

        const __m256 _r0 = _mm256_broadcast_ss(tl);
        const __m256 _r1 = _mm256_broadcast_ss(tc);
        const __m256 _r2 = _mm256_broadcast_ss(tr);
        const __m256 _r3 = _mm256_broadcast_ss(ml);
        const __m256 _r4 = _mm256_broadcast_ss(mc);
        const __m256 _r5 = _mm256_broadcast_ss(mr);
        const __m256 _r6 = _mm256_broadcast_ss(bl);
        const __m256 _r7 = _mm256_broadcast_ss(bc);
        const __m256 _r8 = _mm256_broadcast_ss(br);

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
        float* kptr = const_cast<float*>(kernels);
        float* bptr = const_cast<float*>(biases);

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

        out += *tl * k0;
        out += *tc * k1;
        out += *tr * k2;
        out += *ml * k3;
        out += *mc * k4;
        out += *mr * k5;
        out += *bl * k6;
        out += *bc * k7;
        out += *br * k8;

        Eigen::Map<Eigen::Array<float, 8, 1>>(outMat, 8) = out.max(0.0f);
#else
        const float* kptr = kernels;

        const float* k0 = kptr;
        const float* k1 = kptr + 8;
        const float* k2 = kptr + 16;
        const float* k3 = kptr + 24;
        const float* k4 = kptr + 32;
        const float* k5 = kptr + 40;
        const float* k6 = kptr + 48;
        const float* k7 = kptr + 56;
        const float* k8 = kptr + 64;

        float out[8];
        std::copy_n(biases, 8, out);

        for (size_t i = 0; i < 8; i++)
            out[i] += *tl * k0[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += *tc * k1[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += *tr * k2[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += *ml * k3[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += *mc * k4[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += *mr * k5[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += *bl * k6[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += *bc * k7[i];
        for (size_t i = 0; i < 8; i++)
            out[i] += *br * k8[i];

        for (size_t i = 0; i < 8; i++)
            out[i] = std::max<float>(out[i], 0);

        std::copy_n(out, 8, outMat);
#endif // USE_RYZEN

        }, tmpMat, 8);
}

void Anime4KCPP::CPU::CNNProcessor::conv8To8(const float* kernels, const float* biases, cv::Mat& tmpMat)
{
    const int channels = 8;
    const size_t lineStep = tmpMat.step1();
    detail::changEachPixelNToN([&](const int i, const int j, ChanFP outMat, LineFP curLine) {
        const int jp = j < (tmpMat.cols - 1)* channels ? channels : 0;
        const int jn = j > channels ? -channels : 0;

        const LineFP pLineData = i < tmpMat.rows - 1 ? curLine + lineStep : curLine;
        const LineFP cLineData = curLine;
        const LineFP nLineData = i > 0 ? curLine - lineStep : curLine;

        const PixelFP tl = nLineData + j + jn, tc = nLineData + j, tr = nLineData + j + jp;
        const PixelFP ml = cLineData + j + jn, mc = cLineData + j, mr = cLineData + j + jp;
        const PixelFP bl = pLineData + j + jn, bc = pLineData + j, br = pLineData + j + jp;

#ifdef USE_RYZEN
        const float* kptr = kernels;
        const float* bptr = biases;

        __m256 _out0 = _mm256_loadu_ps(bptr);
        __m256 _out1 = _mm256_setzero_ps();
        __m256 _out2 = _mm256_setzero_ps();

        for (size_t i = 0; i < 8; i += 2)
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
        float* kptr = const_cast<float*>(kernels);
        float* bptr = const_cast<float*>(biases);

        Eigen::Array<float, 8, 1> out = Eigen::Map<Eigen::Array<float, 8, 1>>(bptr, 8);

        for (size_t i = 0; i < 8; i++)
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

            out += tl[i] * k0;
            out += tc[i] * k1;
            out += tr[i] * k2;
            out += ml[i] * k3;
            out += mc[i] * k4;
            out += mr[i] * k5;
            out += bl[i] * k6;
            out += bc[i] * k7;
            out += br[i] * k8;
        }

        Eigen::Map<Eigen::Array<float, 8, 1>>(outMat, 8) = out.max(0.0f);
#else
        float out[8];
        std::copy_n(biases, 8, out);

        const float* kptr = kernels;
        for (size_t c = 0; c < 8; c++)
        {
            const float* k0 = kptr + c * 72;
            const float* k1 = kptr + c * 72 + 8;
            const float* k2 = kptr + c * 72 + 16;
            const float* k3 = kptr + c * 72 + 24;
            const float* k4 = kptr + c * 72 + 32;
            const float* k5 = kptr + c * 72 + 40;
            const float* k6 = kptr + c * 72 + 48;
            const float* k7 = kptr + c * 72 + 56;
            const float* k8 = kptr + c * 72 + 64;

            for (size_t i = 0; i < 8; i++)
                out[i] += tl[c] * k0[i];
            for (size_t i = 0; i < 8; i++)
                out[i] += tc[c] * k1[i];
            for (size_t i = 0; i < 8; i++)
                out[i] += tr[c] * k2[i];
            for (size_t i = 0; i < 8; i++)
                out[i] += ml[c] * k3[i];
            for (size_t i = 0; i < 8; i++)
                out[i] += mc[c] * k4[i];
            for (size_t i = 0; i < 8; i++)
                out[i] += mr[c] * k5[i];
            for (size_t i = 0; i < 8; i++)
                out[i] += bl[c] * k6[i];
            for (size_t i = 0; i < 8; i++)
                out[i] += bc[c] * k7[i];
            for (size_t i = 0; i < 8; i++)
                out[i] += br[c] * k8[i];
        }

        for (size_t i = 0; i < 8; i++)
            out[i] = std::max<float>(out[i], 0);

        std::copy_n(out, 8, outMat);
#endif // USE_RYZEN

        }, tmpMat);
}

void Anime4KCPP::CPU::CNNProcessor::convTranspose8To1B(cv::Mat& img, const float* kernels, cv::Mat& tmpMat)
{
    detail::changEachPixelNTo1<unsigned char>(img, [&](const int i, const int j, ChanB outMat, LineFP inMat) {
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

        *outMat = UNNORMB(luma);
#elif defined(USE_EIGEN3)
        float* kptr = const_cast<float*>(kernels + index * 8);

        const float luma =
            Eigen::Map<Eigen::Vector<float, 8>>(inMat, 8)
            .dot(Eigen::Map<Eigen::Vector<float, 8>>(kptr, 8));

        *outMat = UNNORMB(luma);
#else
        const float* kptr = kernels + index * 8;

        float luma = 0;
        for (size_t i = 0; i < 8; i++)
            luma += kptr[i] * inMat[i];

        *outMat = UNNORMB(luma);
#endif
        }, tmpMat);
}

void Anime4KCPP::CPU::CNNProcessor::convTranspose8To1W(cv::Mat& img, const float* kernels, cv::Mat& tmpMat)
{
    detail::changEachPixelNTo1<unsigned short>(img, [&](const int i, const int j, ChanW outMat, LineFP inMat) {
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

        *outMat = UNNORMW(luma);
#elif defined(USE_EIGEN3)
        float* kptr = const_cast<float*>(kernels + index * 8);

        const float luma =
            Eigen::Map<Eigen::Vector<float, 8>>(inMat, 8)
            .dot(Eigen::Map<Eigen::Vector<float, 8>>(kptr, 8));

        *outMat = UNNORMW(luma);
#else
        const float* kptr = kernels + index * 8;

        float luma = 0;
        for (size_t i = 0; i < 8; i++)
            luma += kptr[i] * inMat[i];

        *outMat = UNNORMW(luma);
#endif
        }, tmpMat);
}

void Anime4KCPP::CPU::CNNProcessor::convTranspose8To1F(cv::Mat& img, const float* kernels, cv::Mat& tmpMat)
{
    detail::changEachPixelNTo1<float>(img, [&](const int i, const int j, ChanF outMat, LineFP inMat) {
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

        *outMat = CLAMP(luma, 0.0f, 1.0f);
#elif defined(USE_EIGEN3)
        float* kptr = const_cast<float*>(kernels + index * 8);

        const float luma =
            Eigen::Map<Eigen::Vector<float, 8>>(inMat, 8)
            .dot(Eigen::Map<Eigen::Vector<float, 8>>(kptr, 8));

        *outMat = CLAMP(luma, 0.0f, 1.0f);
#else
        const float* kptr = kernels + index * 8;

        float luma = 0;
        for (size_t i = 0; i < 8; i++)
            luma += kptr[i] * inMat[i];

        *outMat = static_cast<float>(CLAMP(luma, 0.0f, 1.0f));
#endif
        }, tmpMat);
        }

#endif
