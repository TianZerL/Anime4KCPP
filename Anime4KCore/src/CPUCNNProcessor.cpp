#ifdef ENABLE_AVX
#include <immintrin.h>
#endif

#include "CPUCNNProcessor.hpp"

#define RELU(x) std::max(x, static_cast<FP>(0.0))
#define NORMB(X) (static_cast<FP>(X) / static_cast<FP>(255.0))
#define NORMW(X) (static_cast<FP>(X) / static_cast<FP>(65535.0))
#define UNNORMB(n) ((n) >= static_cast<FP>(1.0)? static_cast<uint8_t>(255) : ((n) <= static_cast<FP>(0.0) ? static_cast<uint8_t>(0) : static_cast<uint8_t>(n * static_cast<FP>(255.0) + static_cast<FP>(0.5))))
#define UNNORMW(n) ((n) >= static_cast<FP>(1.0)? static_cast<uint16_t>(65535) : ((n) <= static_cast<FP>(0.0) ? static_cast<uint16_t>(0) : static_cast<uint16_t>(n * static_cast<FP>(65535.0) + static_cast<FP>(0.5))))
#define CLAMP(v, lo, hi) ((v < lo) ? lo : (hi < v) ? hi : v)

void Anime4KCPP::CPU::CNNProcessor::conv1To8B(const cv::Mat& img, const FP* kernels, const FP* biases, cv::Mat& tmpMat)
{
    const int channels = 8;
    const int srcChannels = img.channels();
    const size_t lineStep = img.step;
    changEachPixel1ToN(img, [&](const int i, const int j, ChanFP outMat, LineB curLine) {
        const int orgJ = j / channels * srcChannels;
        const int jp = orgJ < (img.cols - 1)* srcChannels ? srcChannels : 0;
        const int jn = orgJ > srcChannels ? -srcChannels : 0;

        const LineB pLineData = i < img.rows - 1 ? curLine + lineStep : curLine;
        const LineB cLineData = curLine;
        const LineB nLineData = i > 0 ? curLine - lineStep : curLine;

        const PixelB tl = nLineData + orgJ + jn, tc = nLineData + orgJ, tr = nLineData + orgJ + jp;
        const PixelB ml = cLineData + orgJ + jn, mc = cLineData + orgJ, mr = cLineData + orgJ + jp;
        const PixelB bl = pLineData + orgJ + jn, bc = pLineData + orgJ, br = pLineData + orgJ + jp;

        const FP tln = NORMB(tl[Y]);
        const FP tcn = NORMB(tc[Y]);
        const FP trn = NORMB(tr[Y]);
        const FP mln = NORMB(ml[Y]);
        const FP mcn = NORMB(mc[Y]);
        const FP mrn = NORMB(mr[Y]);
        const FP bln = NORMB(bl[Y]);
        const FP bcn = NORMB(bc[Y]);
        const FP brn = NORMB(br[Y]);

#ifdef ENABLE_AVX
        const float* kptr = kernels;
        const float* bptr = biases;

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

        _out0 = _mm256_max_ps(_mm256_add_ps(_out0, _mm256_add_ps(_out1, _out2)), _mm256_setzero_ps());

        _mm256_storeu_ps(outMat, _out0);
#else
        outMat[0] =
            RELU(
                tln * kernels[0 * 9 + 0] + tcn * kernels[0 * 9 + 1] + trn * kernels[0 * 9 + 2] +
                mln * kernels[0 * 9 + 3] + mcn * kernels[0 * 9 + 4] + mrn * kernels[0 * 9 + 5] +
                bln * kernels[0 * 9 + 6] + bcn * kernels[0 * 9 + 7] + brn * kernels[0 * 9 + 8] + biases[0]);
        outMat[1] =
            RELU(
                tln * kernels[1 * 9 + 0] + tcn * kernels[1 * 9 + 1] + trn * kernels[1 * 9 + 2] +
                mln * kernels[1 * 9 + 3] + mcn * kernels[1 * 9 + 4] + mrn * kernels[1 * 9 + 5] +
                bln * kernels[1 * 9 + 6] + bcn * kernels[1 * 9 + 7] + brn * kernels[1 * 9 + 8] + biases[1]);
        outMat[2] =
            RELU(
                tln * kernels[2 * 9 + 0] + tcn * kernels[2 * 9 + 1] + trn * kernels[2 * 9 + 2] +
                mln * kernels[2 * 9 + 3] + mcn * kernels[2 * 9 + 4] + mrn * kernels[2 * 9 + 5] +
                bln * kernels[2 * 9 + 6] + bcn * kernels[2 * 9 + 7] + brn * kernels[2 * 9 + 8] + biases[2]);
        outMat[3] =
            RELU(
                tln * kernels[3 * 9 + 0] + tcn * kernels[3 * 9 + 1] + trn * kernels[3 * 9 + 2] +
                mln * kernels[3 * 9 + 3] + mcn * kernels[3 * 9 + 4] + mrn * kernels[3 * 9 + 5] +
                bln * kernels[3 * 9 + 6] + bcn * kernels[3 * 9 + 7] + brn * kernels[3 * 9 + 8] + biases[3]);
        outMat[4] =
            RELU(
                tln * kernels[4 * 9 + 0] + tcn * kernels[4 * 9 + 1] + trn * kernels[4 * 9 + 2] +
                mln * kernels[4 * 9 + 3] + mcn * kernels[4 * 9 + 4] + mrn * kernels[4 * 9 + 5] +
                bln * kernels[4 * 9 + 6] + bcn * kernels[4 * 9 + 7] + brn * kernels[4 * 9 + 8] + biases[4]);
        outMat[5] =
            RELU(
                tln * kernels[5 * 9 + 0] + tcn * kernels[5 * 9 + 1] + trn * kernels[5 * 9 + 2] +
                mln * kernels[5 * 9 + 3] + mcn * kernels[5 * 9 + 4] + mrn * kernels[5 * 9 + 5] +
                bln * kernels[5 * 9 + 6] + bcn * kernels[5 * 9 + 7] + brn * kernels[5 * 9 + 8] + biases[5]);
        outMat[6] =
            RELU(
                tln * kernels[6 * 9 + 0] + tcn * kernels[6 * 9 + 1] + trn * kernels[6 * 9 + 2] +
                mln * kernels[6 * 9 + 3] + mcn * kernels[6 * 9 + 4] + mrn * kernels[6 * 9 + 5] +
                bln * kernels[6 * 9 + 6] + bcn * kernels[6 * 9 + 7] + brn * kernels[6 * 9 + 8] + biases[6]);
        outMat[7] =
            RELU(
                tln * kernels[7 * 9 + 0] + tcn * kernels[7 * 9 + 1] + trn * kernels[7 * 9 + 2] +
                mln * kernels[7 * 9 + 3] + mcn * kernels[7 * 9 + 4] + mrn * kernels[7 * 9 + 5] +
                bln * kernels[7 * 9 + 6] + bcn * kernels[7 * 9 + 7] + brn * kernels[7 * 9 + 8] + biases[7]);
#endif // ENABLE_AVX
        }, tmpMat, 8);
}

void Anime4KCPP::CPU::CNNProcessor::conv1To8W(const cv::Mat& img, const FP* kernels, const FP* biases, cv::Mat& tmpMat)
{
    const int channels = 8;
    const int srcChannels = img.channels();
    const size_t lineStep = img.step;
    changEachPixel1ToN(img, [&](const int i, const int j, ChanFP outMat, LineW curLine) {
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

        const FP tln = NORMW(tl[Y]);
        const FP tcn = NORMW(tc[Y]);
        const FP trn = NORMW(tr[Y]);
        const FP mln = NORMW(ml[Y]);
        const FP mcn = NORMW(mc[Y]);
        const FP mrn = NORMW(mr[Y]);
        const FP bln = NORMW(bl[Y]);
        const FP bcn = NORMW(bc[Y]);
        const FP brn = NORMW(br[Y]);

#ifdef ENABLE_AVX
        const float* kptr = kernels;
        const float* bptr = biases;

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

        _out0 = _mm256_max_ps(_mm256_add_ps(_out0, _mm256_add_ps(_out1, _out2)), _mm256_setzero_ps());

        _mm256_storeu_ps(outMat, _out0);
#else
        outMat[0] =
            RELU(
                tln * kernels[0 * 9 + 0] + tcn * kernels[0 * 9 + 1] + trn * kernels[0 * 9 + 2] +
                mln * kernels[0 * 9 + 3] + mcn * kernels[0 * 9 + 4] + mrn * kernels[0 * 9 + 5] +
                bln * kernels[0 * 9 + 6] + bcn * kernels[0 * 9 + 7] + brn * kernels[0 * 9 + 8] + biases[0]);
        outMat[1] =
            RELU(
                tln * kernels[1 * 9 + 0] + tcn * kernels[1 * 9 + 1] + trn * kernels[1 * 9 + 2] +
                mln * kernels[1 * 9 + 3] + mcn * kernels[1 * 9 + 4] + mrn * kernels[1 * 9 + 5] +
                bln * kernels[1 * 9 + 6] + bcn * kernels[1 * 9 + 7] + brn * kernels[1 * 9 + 8] + biases[1]);
        outMat[2] =
            RELU(
                tln * kernels[2 * 9 + 0] + tcn * kernels[2 * 9 + 1] + trn * kernels[2 * 9 + 2] +
                mln * kernels[2 * 9 + 3] + mcn * kernels[2 * 9 + 4] + mrn * kernels[2 * 9 + 5] +
                bln * kernels[2 * 9 + 6] + bcn * kernels[2 * 9 + 7] + brn * kernels[2 * 9 + 8] + biases[2]);
        outMat[3] =
            RELU(
                tln * kernels[3 * 9 + 0] + tcn * kernels[3 * 9 + 1] + trn * kernels[3 * 9 + 2] +
                mln * kernels[3 * 9 + 3] + mcn * kernels[3 * 9 + 4] + mrn * kernels[3 * 9 + 5] +
                bln * kernels[3 * 9 + 6] + bcn * kernels[3 * 9 + 7] + brn * kernels[3 * 9 + 8] + biases[3]);
        outMat[4] =
            RELU(
                tln * kernels[4 * 9 + 0] + tcn * kernels[4 * 9 + 1] + trn * kernels[4 * 9 + 2] +
                mln * kernels[4 * 9 + 3] + mcn * kernels[4 * 9 + 4] + mrn * kernels[4 * 9 + 5] +
                bln * kernels[4 * 9 + 6] + bcn * kernels[4 * 9 + 7] + brn * kernels[4 * 9 + 8] + biases[4]);
        outMat[5] =
            RELU(
                tln * kernels[5 * 9 + 0] + tcn * kernels[5 * 9 + 1] + trn * kernels[5 * 9 + 2] +
                mln * kernels[5 * 9 + 3] + mcn * kernels[5 * 9 + 4] + mrn * kernels[5 * 9 + 5] +
                bln * kernels[5 * 9 + 6] + bcn * kernels[5 * 9 + 7] + brn * kernels[5 * 9 + 8] + biases[5]);
        outMat[6] =
            RELU(
                tln * kernels[6 * 9 + 0] + tcn * kernels[6 * 9 + 1] + trn * kernels[6 * 9 + 2] +
                mln * kernels[6 * 9 + 3] + mcn * kernels[6 * 9 + 4] + mrn * kernels[6 * 9 + 5] +
                bln * kernels[6 * 9 + 6] + bcn * kernels[6 * 9 + 7] + brn * kernels[6 * 9 + 8] + biases[6]);
        outMat[7] =
            RELU(
                tln * kernels[7 * 9 + 0] + tcn * kernels[7 * 9 + 1] + trn * kernels[7 * 9 + 2] +
                mln * kernels[7 * 9 + 3] + mcn * kernels[7 * 9 + 4] + mrn * kernels[7 * 9 + 5] +
                bln * kernels[7 * 9 + 6] + bcn * kernels[7 * 9 + 7] + brn * kernels[7 * 9 + 8] + biases[7]);
#endif // ENABLE_AVX

        }, tmpMat, 8);
}

void Anime4KCPP::CPU::CNNProcessor::conv1To8F(const cv::Mat& img, const FP* kernels, const FP* biases, cv::Mat& tmpMat)
{
    const int channels = 8;
    const int srcChannels = img.channels();
    const size_t lineStep = img.step;
    changEachPixel1ToN(img, [&](const int i, const int j, ChanFP outMat, LineF curLine) {
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

#ifdef ENABLE_AVX
        const float* kptr = kernels;
        const float* bptr = biases;

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

        _out0 = _mm256_max_ps(_mm256_add_ps(_out0, _mm256_add_ps(_out1, _out2)), _mm256_setzero_ps());

        _mm256_storeu_ps(outMat, _out0);
#else
        const FP tln = tl[Y];
        const FP tcn = tc[Y];
        const FP trn = tr[Y];
        const FP mln = ml[Y];
        const FP mcn = mc[Y];
        const FP mrn = mr[Y];
        const FP bln = bl[Y];
        const FP bcn = bc[Y];
        const FP brn = br[Y];

        outMat[0] =
            RELU(
                tln * kernels[0 * 9 + 0] + tcn * kernels[0 * 9 + 1] + trn * kernels[0 * 9 + 2] +
                mln * kernels[0 * 9 + 3] + mcn * kernels[0 * 9 + 4] + mrn * kernels[0 * 9 + 5] +
                bln * kernels[0 * 9 + 6] + bcn * kernels[0 * 9 + 7] + brn * kernels[0 * 9 + 8] + biases[0]);
        outMat[1] =
            RELU(
                tln * kernels[1 * 9 + 0] + tcn * kernels[1 * 9 + 1] + trn * kernels[1 * 9 + 2] +
                mln * kernels[1 * 9 + 3] + mcn * kernels[1 * 9 + 4] + mrn * kernels[1 * 9 + 5] +
                bln * kernels[1 * 9 + 6] + bcn * kernels[1 * 9 + 7] + brn * kernels[1 * 9 + 8] + biases[1]);
        outMat[2] =
            RELU(
                tln * kernels[2 * 9 + 0] + tcn * kernels[2 * 9 + 1] + trn * kernels[2 * 9 + 2] +
                mln * kernels[2 * 9 + 3] + mcn * kernels[2 * 9 + 4] + mrn * kernels[2 * 9 + 5] +
                bln * kernels[2 * 9 + 6] + bcn * kernels[2 * 9 + 7] + brn * kernels[2 * 9 + 8] + biases[2]);
        outMat[3] =
            RELU(
                tln * kernels[3 * 9 + 0] + tcn * kernels[3 * 9 + 1] + trn * kernels[3 * 9 + 2] +
                mln * kernels[3 * 9 + 3] + mcn * kernels[3 * 9 + 4] + mrn * kernels[3 * 9 + 5] +
                bln * kernels[3 * 9 + 6] + bcn * kernels[3 * 9 + 7] + brn * kernels[3 * 9 + 8] + biases[3]);
        outMat[4] =
            RELU(
                tln * kernels[4 * 9 + 0] + tcn * kernels[4 * 9 + 1] + trn * kernels[4 * 9 + 2] +
                mln * kernels[4 * 9 + 3] + mcn * kernels[4 * 9 + 4] + mrn * kernels[4 * 9 + 5] +
                bln * kernels[4 * 9 + 6] + bcn * kernels[4 * 9 + 7] + brn * kernels[4 * 9 + 8] + biases[4]);
        outMat[5] =
            RELU(
                tln * kernels[5 * 9 + 0] + tcn * kernels[5 * 9 + 1] + trn * kernels[5 * 9 + 2] +
                mln * kernels[5 * 9 + 3] + mcn * kernels[5 * 9 + 4] + mrn * kernels[5 * 9 + 5] +
                bln * kernels[5 * 9 + 6] + bcn * kernels[5 * 9 + 7] + brn * kernels[5 * 9 + 8] + biases[5]);
        outMat[6] =
            RELU(
                tln * kernels[6 * 9 + 0] + tcn * kernels[6 * 9 + 1] + trn * kernels[6 * 9 + 2] +
                mln * kernels[6 * 9 + 3] + mcn * kernels[6 * 9 + 4] + mrn * kernels[6 * 9 + 5] +
                bln * kernels[6 * 9 + 6] + bcn * kernels[6 * 9 + 7] + brn * kernels[6 * 9 + 8] + biases[6]);
        outMat[7] =
            RELU(
                tln * kernels[7 * 9 + 0] + tcn * kernels[7 * 9 + 1] + trn * kernels[7 * 9 + 2] +
                mln * kernels[7 * 9 + 3] + mcn * kernels[7 * 9 + 4] + mrn * kernels[7 * 9 + 5] +
                bln * kernels[7 * 9 + 6] + bcn * kernels[7 * 9 + 7] + brn * kernels[7 * 9 + 8] + biases[7]);
#endif // ENABLE_AVX

        }, tmpMat, 8);
}

void Anime4KCPP::CPU::CNNProcessor::conv8To8(const FP* kernels, const FP* biases, cv::Mat& tmpMat)
{
    const int channels = 8;
    const size_t lineStep = tmpMat.step1();
    changEachPixelNToN([&](const int i, const int j, ChanFP outMat, LineFP curLine) {
        const int jp = j < (tmpMat.cols - 1)* channels ? channels : 0;
        const int jn = j > channels ? -channels : 0;

        const LineFP pLineData = i < tmpMat.rows - 1 ? curLine + lineStep : curLine;
        const LineFP cLineData = curLine;
        const LineFP nLineData = i > 0 ? curLine - lineStep : curLine;

        const PixelFP tl = nLineData + j + jn, tc = nLineData + j, tr = nLineData + j + jp;
        const PixelFP ml = cLineData + j + jn, mc = cLineData + j, mr = cLineData + j + jp;
        const PixelFP bl = pLineData + j + jn, bc = pLineData + j, br = pLineData + j + jp;

#ifdef ENABLE_AVX
        const float* kptr = kernels;
        const float* bptr = biases;

        __m256 _out0 = _mm256_loadu_ps(bptr);
        __m256 _out1 = _mm256_setzero_ps();
        __m256 _out2 = _mm256_setzero_ps();

        for (int i = 0; i < 8; i++)
        {
            const __m256 _r0 = _mm256_broadcast_ss(tl + i);
            const __m256 _r1 = _mm256_broadcast_ss(tc + i);
            const __m256 _r2 = _mm256_broadcast_ss(tr + i);
            const __m256 _r3 = _mm256_broadcast_ss(ml + i);
            const __m256 _r4 = _mm256_broadcast_ss(mc + i);
            const __m256 _r5 = _mm256_broadcast_ss(mr + i);
            const __m256 _r6 = _mm256_broadcast_ss(bl + i);
            const __m256 _r7 = _mm256_broadcast_ss(bc + i);
            const __m256 _r8 = _mm256_broadcast_ss(br + i);

            const __m256 _k0 = _mm256_loadu_ps(kptr + i * 72);
            const __m256 _k1 = _mm256_loadu_ps(kptr + i * 72 + 8);
            const __m256 _k2 = _mm256_loadu_ps(kptr + i * 72 + 16);
            const __m256 _k3 = _mm256_loadu_ps(kptr + i * 72 + 24);
            const __m256 _k4 = _mm256_loadu_ps(kptr + i * 72 + 32);
            const __m256 _k5 = _mm256_loadu_ps(kptr + i * 72 + 40);
            const __m256 _k6 = _mm256_loadu_ps(kptr + i * 72 + 48);
            const __m256 _k7 = _mm256_loadu_ps(kptr + i * 72 + 56);
            const __m256 _k8 = _mm256_loadu_ps(kptr + i * 72 + 64);

            _out0 = _mm256_fmadd_ps(_r0, _k0, _out0);
            _out1 = _mm256_fmadd_ps(_r1, _k1, _out1);
            _out2 = _mm256_fmadd_ps(_r2, _k2, _out2);
            _out0 = _mm256_fmadd_ps(_r3, _k3, _out0);
            _out1 = _mm256_fmadd_ps(_r4, _k4, _out1);
            _out2 = _mm256_fmadd_ps(_r5, _k5, _out2);
            _out0 = _mm256_fmadd_ps(_r6, _k6, _out0);
            _out1 = _mm256_fmadd_ps(_r7, _k7, _out1);
            _out2 = _mm256_fmadd_ps(_r8, _k8, _out2);
        }
        _out0 = _mm256_max_ps(_mm256_add_ps(_out0, _mm256_add_ps(_out1, _out2)), _mm256_setzero_ps());

        _mm256_storeu_ps(outMat, _out0);
#else
        FP c1 =
            tl[0] * kernels[0 * 72 + 0 * 9 + 0] + tc[0] * kernels[0 * 72 + 0 * 9 + 1] + tr[0] * kernels[0 * 72 + 0 * 9 + 2] +
            ml[0] * kernels[0 * 72 + 0 * 9 + 3] + mc[0] * kernels[0 * 72 + 0 * 9 + 4] + mr[0] * kernels[0 * 72 + 0 * 9 + 5] +
            bl[0] * kernels[0 * 72 + 0 * 9 + 6] + bc[0] * kernels[0 * 72 + 0 * 9 + 7] + br[0] * kernels[0 * 72 + 0 * 9 + 8];

        FP c2 =
            tl[1] * kernels[0 * 72 + 1 * 9 + 0] + tc[1] * kernels[0 * 72 + 1 * 9 + 1] + tr[1] * kernels[0 * 72 + 1 * 9 + 2] +
            ml[1] * kernels[0 * 72 + 1 * 9 + 3] + mc[1] * kernels[0 * 72 + 1 * 9 + 4] + mr[1] * kernels[0 * 72 + 1 * 9 + 5] +
            bl[1] * kernels[0 * 72 + 1 * 9 + 6] + bc[1] * kernels[0 * 72 + 1 * 9 + 7] + br[1] * kernels[0 * 72 + 1 * 9 + 8];

        FP c3 =
            tl[2] * kernels[0 * 72 + 2 * 9 + 0] + tc[2] * kernels[0 * 72 + 2 * 9 + 1] + tr[2] * kernels[0 * 72 + 2 * 9 + 2] +
            ml[2] * kernels[0 * 72 + 2 * 9 + 3] + mc[2] * kernels[0 * 72 + 2 * 9 + 4] + mr[2] * kernels[0 * 72 + 2 * 9 + 5] +
            bl[2] * kernels[0 * 72 + 2 * 9 + 6] + bc[2] * kernels[0 * 72 + 2 * 9 + 7] + br[2] * kernels[0 * 72 + 2 * 9 + 8];

        FP c4 =
            tl[3] * kernels[0 * 72 + 3 * 9 + 0] + tc[3] * kernels[0 * 72 + 3 * 9 + 1] + tr[3] * kernels[0 * 72 + 3 * 9 + 2] +
            ml[3] * kernels[0 * 72 + 3 * 9 + 3] + mc[3] * kernels[0 * 72 + 3 * 9 + 4] + mr[3] * kernels[0 * 72 + 3 * 9 + 5] +
            bl[3] * kernels[0 * 72 + 3 * 9 + 6] + bc[3] * kernels[0 * 72 + 3 * 9 + 7] + br[3] * kernels[0 * 72 + 3 * 9 + 8];

        FP c5 =
            tl[4] * kernels[0 * 72 + 4 * 9 + 0] + tc[4] * kernels[0 * 72 + 4 * 9 + 1] + tr[4] * kernels[0 * 72 + 4 * 9 + 2] +
            ml[4] * kernels[0 * 72 + 4 * 9 + 3] + mc[4] * kernels[0 * 72 + 4 * 9 + 4] + mr[4] * kernels[0 * 72 + 4 * 9 + 5] +
            bl[4] * kernels[0 * 72 + 4 * 9 + 6] + bc[4] * kernels[0 * 72 + 4 * 9 + 7] + br[4] * kernels[0 * 72 + 4 * 9 + 8];

        FP c6 =
            tl[5] * kernels[0 * 72 + 5 * 9 + 0] + tc[5] * kernels[0 * 72 + 5 * 9 + 1] + tr[5] * kernels[0 * 72 + 5 * 9 + 2] +
            ml[5] * kernels[0 * 72 + 5 * 9 + 3] + mc[5] * kernels[0 * 72 + 5 * 9 + 4] + mr[5] * kernels[0 * 72 + 5 * 9 + 5] +
            bl[5] * kernels[0 * 72 + 5 * 9 + 6] + bc[5] * kernels[0 * 72 + 5 * 9 + 7] + br[5] * kernels[0 * 72 + 5 * 9 + 8];

        FP c7 =
            tl[6] * kernels[0 * 72 + 6 * 9 + 0] + tc[6] * kernels[0 * 72 + 6 * 9 + 1] + tr[6] * kernels[0 * 72 + 6 * 9 + 2] +
            ml[6] * kernels[0 * 72 + 6 * 9 + 3] + mc[6] * kernels[0 * 72 + 6 * 9 + 4] + mr[6] * kernels[0 * 72 + 6 * 9 + 5] +
            bl[6] * kernels[0 * 72 + 6 * 9 + 6] + bc[6] * kernels[0 * 72 + 6 * 9 + 7] + br[6] * kernels[0 * 72 + 6 * 9 + 8];

        FP c8 =
            tl[7] * kernels[0 * 72 + 7 * 9 + 0] + tc[7] * kernels[0 * 72 + 7 * 9 + 1] + tr[7] * kernels[0 * 72 + 7 * 9 + 2] +
            ml[7] * kernels[0 * 72 + 7 * 9 + 3] + mc[7] * kernels[0 * 72 + 7 * 9 + 4] + mr[7] * kernels[0 * 72 + 7 * 9 + 5] +
            bl[7] * kernels[0 * 72 + 7 * 9 + 6] + bc[7] * kernels[0 * 72 + 7 * 9 + 7] + br[7] * kernels[0 * 72 + 7 * 9 + 8];

        outMat[0] = RELU(c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + biases[0]);

        c1 =
            tl[0] * kernels[1 * 72 + 0 * 9 + 0] + tc[0] * kernels[1 * 72 + 0 * 9 + 1] + tr[0] * kernels[1 * 72 + 0 * 9 + 2] +
            ml[0] * kernels[1 * 72 + 0 * 9 + 3] + mc[0] * kernels[1 * 72 + 0 * 9 + 4] + mr[0] * kernels[1 * 72 + 0 * 9 + 5] +
            bl[0] * kernels[1 * 72 + 0 * 9 + 6] + bc[0] * kernels[1 * 72 + 0 * 9 + 7] + br[0] * kernels[1 * 72 + 0 * 9 + 8];

        c2 =
            tl[1] * kernels[1 * 72 + 1 * 9 + 0] + tc[1] * kernels[1 * 72 + 1 * 9 + 1] + tr[1] * kernels[1 * 72 + 1 * 9 + 2] +
            ml[1] * kernels[1 * 72 + 1 * 9 + 3] + mc[1] * kernels[1 * 72 + 1 * 9 + 4] + mr[1] * kernels[1 * 72 + 1 * 9 + 5] +
            bl[1] * kernels[1 * 72 + 1 * 9 + 6] + bc[1] * kernels[1 * 72 + 1 * 9 + 7] + br[1] * kernels[1 * 72 + 1 * 9 + 8];

        c3 =
            tl[2] * kernels[1 * 72 + 2 * 9 + 0] + tc[2] * kernels[1 * 72 + 2 * 9 + 1] + tr[2] * kernels[1 * 72 + 2 * 9 + 2] +
            ml[2] * kernels[1 * 72 + 2 * 9 + 3] + mc[2] * kernels[1 * 72 + 2 * 9 + 4] + mr[2] * kernels[1 * 72 + 2 * 9 + 5] +
            bl[2] * kernels[1 * 72 + 2 * 9 + 6] + bc[2] * kernels[1 * 72 + 2 * 9 + 7] + br[2] * kernels[1 * 72 + 2 * 9 + 8];

        c4 =
            tl[3] * kernels[1 * 72 + 3 * 9 + 0] + tc[3] * kernels[1 * 72 + 3 * 9 + 1] + tr[3] * kernels[1 * 72 + 3 * 9 + 2] +
            ml[3] * kernels[1 * 72 + 3 * 9 + 3] + mc[3] * kernels[1 * 72 + 3 * 9 + 4] + mr[3] * kernels[1 * 72 + 3 * 9 + 5] +
            bl[3] * kernels[1 * 72 + 3 * 9 + 6] + bc[3] * kernels[1 * 72 + 3 * 9 + 7] + br[3] * kernels[1 * 72 + 3 * 9 + 8];

        c5 =
            tl[4] * kernels[1 * 72 + 4 * 9 + 0] + tc[4] * kernels[1 * 72 + 4 * 9 + 1] + tr[4] * kernels[1 * 72 + 4 * 9 + 2] +
            ml[4] * kernels[1 * 72 + 4 * 9 + 3] + mc[4] * kernels[1 * 72 + 4 * 9 + 4] + mr[4] * kernels[1 * 72 + 4 * 9 + 5] +
            bl[4] * kernels[1 * 72 + 4 * 9 + 6] + bc[4] * kernels[1 * 72 + 4 * 9 + 7] + br[4] * kernels[1 * 72 + 4 * 9 + 8];

        c6 =
            tl[5] * kernels[1 * 72 + 5 * 9 + 0] + tc[5] * kernels[1 * 72 + 5 * 9 + 1] + tr[5] * kernels[1 * 72 + 5 * 9 + 2] +
            ml[5] * kernels[1 * 72 + 5 * 9 + 3] + mc[5] * kernels[1 * 72 + 5 * 9 + 4] + mr[5] * kernels[1 * 72 + 5 * 9 + 5] +
            bl[5] * kernels[1 * 72 + 5 * 9 + 6] + bc[5] * kernels[1 * 72 + 5 * 9 + 7] + br[5] * kernels[1 * 72 + 5 * 9 + 8];

        c7 =
            tl[6] * kernels[1 * 72 + 6 * 9 + 0] + tc[6] * kernels[1 * 72 + 6 * 9 + 1] + tr[6] * kernels[1 * 72 + 6 * 9 + 2] +
            ml[6] * kernels[1 * 72 + 6 * 9 + 3] + mc[6] * kernels[1 * 72 + 6 * 9 + 4] + mr[6] * kernels[1 * 72 + 6 * 9 + 5] +
            bl[6] * kernels[1 * 72 + 6 * 9 + 6] + bc[6] * kernels[1 * 72 + 6 * 9 + 7] + br[6] * kernels[1 * 72 + 6 * 9 + 8];

        c8 =
            tl[7] * kernels[1 * 72 + 7 * 9 + 0] + tc[7] * kernels[1 * 72 + 7 * 9 + 1] + tr[7] * kernels[1 * 72 + 7 * 9 + 2] +
            ml[7] * kernels[1 * 72 + 7 * 9 + 3] + mc[7] * kernels[1 * 72 + 7 * 9 + 4] + mr[7] * kernels[1 * 72 + 7 * 9 + 5] +
            bl[7] * kernels[1 * 72 + 7 * 9 + 6] + bc[7] * kernels[1 * 72 + 7 * 9 + 7] + br[7] * kernels[1 * 72 + 7 * 9 + 8];

        outMat[1] = RELU(c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + biases[1]);

        c1 =
            tl[0] * kernels[2 * 72 + 0 * 9 + 0] + tc[0] * kernels[2 * 72 + 0 * 9 + 1] + tr[0] * kernels[2 * 72 + 0 * 9 + 2] +
            ml[0] * kernels[2 * 72 + 0 * 9 + 3] + mc[0] * kernels[2 * 72 + 0 * 9 + 4] + mr[0] * kernels[2 * 72 + 0 * 9 + 5] +
            bl[0] * kernels[2 * 72 + 0 * 9 + 6] + bc[0] * kernels[2 * 72 + 0 * 9 + 7] + br[0] * kernels[2 * 72 + 0 * 9 + 8];

        c2 =
            tl[1] * kernels[2 * 72 + 1 * 9 + 0] + tc[1] * kernels[2 * 72 + 1 * 9 + 1] + tr[1] * kernels[2 * 72 + 1 * 9 + 2] +
            ml[1] * kernels[2 * 72 + 1 * 9 + 3] + mc[1] * kernels[2 * 72 + 1 * 9 + 4] + mr[1] * kernels[2 * 72 + 1 * 9 + 5] +
            bl[1] * kernels[2 * 72 + 1 * 9 + 6] + bc[1] * kernels[2 * 72 + 1 * 9 + 7] + br[1] * kernels[2 * 72 + 1 * 9 + 8];

        c3 =
            tl[2] * kernels[2 * 72 + 2 * 9 + 0] + tc[2] * kernels[2 * 72 + 2 * 9 + 1] + tr[2] * kernels[2 * 72 + 2 * 9 + 2] +
            ml[2] * kernels[2 * 72 + 2 * 9 + 3] + mc[2] * kernels[2 * 72 + 2 * 9 + 4] + mr[2] * kernels[2 * 72 + 2 * 9 + 5] +
            bl[2] * kernels[2 * 72 + 2 * 9 + 6] + bc[2] * kernels[2 * 72 + 2 * 9 + 7] + br[2] * kernels[2 * 72 + 2 * 9 + 8];

        c4 =
            tl[3] * kernels[2 * 72 + 3 * 9 + 0] + tc[3] * kernels[2 * 72 + 3 * 9 + 1] + tr[3] * kernels[2 * 72 + 3 * 9 + 2] +
            ml[3] * kernels[2 * 72 + 3 * 9 + 3] + mc[3] * kernels[2 * 72 + 3 * 9 + 4] + mr[3] * kernels[2 * 72 + 3 * 9 + 5] +
            bl[3] * kernels[2 * 72 + 3 * 9 + 6] + bc[3] * kernels[2 * 72 + 3 * 9 + 7] + br[3] * kernels[2 * 72 + 3 * 9 + 8];

        c5 =
            tl[4] * kernels[2 * 72 + 4 * 9 + 0] + tc[4] * kernels[2 * 72 + 4 * 9 + 1] + tr[4] * kernels[2 * 72 + 4 * 9 + 2] +
            ml[4] * kernels[2 * 72 + 4 * 9 + 3] + mc[4] * kernels[2 * 72 + 4 * 9 + 4] + mr[4] * kernels[2 * 72 + 4 * 9 + 5] +
            bl[4] * kernels[2 * 72 + 4 * 9 + 6] + bc[4] * kernels[2 * 72 + 4 * 9 + 7] + br[4] * kernels[2 * 72 + 4 * 9 + 8];

        c6 =
            tl[5] * kernels[2 * 72 + 5 * 9 + 0] + tc[5] * kernels[2 * 72 + 5 * 9 + 1] + tr[5] * kernels[2 * 72 + 5 * 9 + 2] +
            ml[5] * kernels[2 * 72 + 5 * 9 + 3] + mc[5] * kernels[2 * 72 + 5 * 9 + 4] + mr[5] * kernels[2 * 72 + 5 * 9 + 5] +
            bl[5] * kernels[2 * 72 + 5 * 9 + 6] + bc[5] * kernels[2 * 72 + 5 * 9 + 7] + br[5] * kernels[2 * 72 + 5 * 9 + 8];

        c7 =
            tl[6] * kernels[2 * 72 + 6 * 9 + 0] + tc[6] * kernels[2 * 72 + 6 * 9 + 1] + tr[6] * kernels[2 * 72 + 6 * 9 + 2] +
            ml[6] * kernels[2 * 72 + 6 * 9 + 3] + mc[6] * kernels[2 * 72 + 6 * 9 + 4] + mr[6] * kernels[2 * 72 + 6 * 9 + 5] +
            bl[6] * kernels[2 * 72 + 6 * 9 + 6] + bc[6] * kernels[2 * 72 + 6 * 9 + 7] + br[6] * kernels[2 * 72 + 6 * 9 + 8];

        c8 =
            tl[7] * kernels[2 * 72 + 7 * 9 + 0] + tc[7] * kernels[2 * 72 + 7 * 9 + 1] + tr[7] * kernels[2 * 72 + 7 * 9 + 2] +
            ml[7] * kernels[2 * 72 + 7 * 9 + 3] + mc[7] * kernels[2 * 72 + 7 * 9 + 4] + mr[7] * kernels[2 * 72 + 7 * 9 + 5] +
            bl[7] * kernels[2 * 72 + 7 * 9 + 6] + bc[7] * kernels[2 * 72 + 7 * 9 + 7] + br[7] * kernels[2 * 72 + 7 * 9 + 8];

        outMat[2] = RELU(c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + biases[2]);

        c1 =
            tl[0] * kernels[3 * 72 + 0 * 9 + 0] + tc[0] * kernels[3 * 72 + 0 * 9 + 1] + tr[0] * kernels[3 * 72 + 0 * 9 + 2] +
            ml[0] * kernels[3 * 72 + 0 * 9 + 3] + mc[0] * kernels[3 * 72 + 0 * 9 + 4] + mr[0] * kernels[3 * 72 + 0 * 9 + 5] +
            bl[0] * kernels[3 * 72 + 0 * 9 + 6] + bc[0] * kernels[3 * 72 + 0 * 9 + 7] + br[0] * kernels[3 * 72 + 0 * 9 + 8];

        c2 =
            tl[1] * kernels[3 * 72 + 1 * 9 + 0] + tc[1] * kernels[3 * 72 + 1 * 9 + 1] + tr[1] * kernels[3 * 72 + 1 * 9 + 2] +
            ml[1] * kernels[3 * 72 + 1 * 9 + 3] + mc[1] * kernels[3 * 72 + 1 * 9 + 4] + mr[1] * kernels[3 * 72 + 1 * 9 + 5] +
            bl[1] * kernels[3 * 72 + 1 * 9 + 6] + bc[1] * kernels[3 * 72 + 1 * 9 + 7] + br[1] * kernels[3 * 72 + 1 * 9 + 8];

        c3 =
            tl[2] * kernels[3 * 72 + 2 * 9 + 0] + tc[2] * kernels[3 * 72 + 2 * 9 + 1] + tr[2] * kernels[3 * 72 + 2 * 9 + 2] +
            ml[2] * kernels[3 * 72 + 2 * 9 + 3] + mc[2] * kernels[3 * 72 + 2 * 9 + 4] + mr[2] * kernels[3 * 72 + 2 * 9 + 5] +
            bl[2] * kernels[3 * 72 + 2 * 9 + 6] + bc[2] * kernels[3 * 72 + 2 * 9 + 7] + br[2] * kernels[3 * 72 + 2 * 9 + 8];

        c4 =
            tl[3] * kernels[3 * 72 + 3 * 9 + 0] + tc[3] * kernels[3 * 72 + 3 * 9 + 1] + tr[3] * kernels[3 * 72 + 3 * 9 + 2] +
            ml[3] * kernels[3 * 72 + 3 * 9 + 3] + mc[3] * kernels[3 * 72 + 3 * 9 + 4] + mr[3] * kernels[3 * 72 + 3 * 9 + 5] +
            bl[3] * kernels[3 * 72 + 3 * 9 + 6] + bc[3] * kernels[3 * 72 + 3 * 9 + 7] + br[3] * kernels[3 * 72 + 3 * 9 + 8];

        c5 =
            tl[4] * kernels[3 * 72 + 4 * 9 + 0] + tc[4] * kernels[3 * 72 + 4 * 9 + 1] + tr[4] * kernels[3 * 72 + 4 * 9 + 2] +
            ml[4] * kernels[3 * 72 + 4 * 9 + 3] + mc[4] * kernels[3 * 72 + 4 * 9 + 4] + mr[4] * kernels[3 * 72 + 4 * 9 + 5] +
            bl[4] * kernels[3 * 72 + 4 * 9 + 6] + bc[4] * kernels[3 * 72 + 4 * 9 + 7] + br[4] * kernels[3 * 72 + 4 * 9 + 8];

        c6 =
            tl[5] * kernels[3 * 72 + 5 * 9 + 0] + tc[5] * kernels[3 * 72 + 5 * 9 + 1] + tr[5] * kernels[3 * 72 + 5 * 9 + 2] +
            ml[5] * kernels[3 * 72 + 5 * 9 + 3] + mc[5] * kernels[3 * 72 + 5 * 9 + 4] + mr[5] * kernels[3 * 72 + 5 * 9 + 5] +
            bl[5] * kernels[3 * 72 + 5 * 9 + 6] + bc[5] * kernels[3 * 72 + 5 * 9 + 7] + br[5] * kernels[3 * 72 + 5 * 9 + 8];

        c7 =
            tl[6] * kernels[3 * 72 + 6 * 9 + 0] + tc[6] * kernels[3 * 72 + 6 * 9 + 1] + tr[6] * kernels[3 * 72 + 6 * 9 + 2] +
            ml[6] * kernels[3 * 72 + 6 * 9 + 3] + mc[6] * kernels[3 * 72 + 6 * 9 + 4] + mr[6] * kernels[3 * 72 + 6 * 9 + 5] +
            bl[6] * kernels[3 * 72 + 6 * 9 + 6] + bc[6] * kernels[3 * 72 + 6 * 9 + 7] + br[6] * kernels[3 * 72 + 6 * 9 + 8];

        c8 =
            tl[7] * kernels[3 * 72 + 7 * 9 + 0] + tc[7] * kernels[3 * 72 + 7 * 9 + 1] + tr[7] * kernels[3 * 72 + 7 * 9 + 2] +
            ml[7] * kernels[3 * 72 + 7 * 9 + 3] + mc[7] * kernels[3 * 72 + 7 * 9 + 4] + mr[7] * kernels[3 * 72 + 7 * 9 + 5] +
            bl[7] * kernels[3 * 72 + 7 * 9 + 6] + bc[7] * kernels[3 * 72 + 7 * 9 + 7] + br[7] * kernels[3 * 72 + 7 * 9 + 8];

        outMat[3] = RELU(c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + biases[3]);

        c1 =
            tl[0] * kernels[4 * 72 + 0 * 9 + 0] + tc[0] * kernels[4 * 72 + 0 * 9 + 1] + tr[0] * kernels[4 * 72 + 0 * 9 + 2] +
            ml[0] * kernels[4 * 72 + 0 * 9 + 3] + mc[0] * kernels[4 * 72 + 0 * 9 + 4] + mr[0] * kernels[4 * 72 + 0 * 9 + 5] +
            bl[0] * kernels[4 * 72 + 0 * 9 + 6] + bc[0] * kernels[4 * 72 + 0 * 9 + 7] + br[0] * kernels[4 * 72 + 0 * 9 + 8];

        c2 =
            tl[1] * kernels[4 * 72 + 1 * 9 + 0] + tc[1] * kernels[4 * 72 + 1 * 9 + 1] + tr[1] * kernels[4 * 72 + 1 * 9 + 2] +
            ml[1] * kernels[4 * 72 + 1 * 9 + 3] + mc[1] * kernels[4 * 72 + 1 * 9 + 4] + mr[1] * kernels[4 * 72 + 1 * 9 + 5] +
            bl[1] * kernels[4 * 72 + 1 * 9 + 6] + bc[1] * kernels[4 * 72 + 1 * 9 + 7] + br[1] * kernels[4 * 72 + 1 * 9 + 8];

        c3 =
            tl[2] * kernels[4 * 72 + 2 * 9 + 0] + tc[2] * kernels[4 * 72 + 2 * 9 + 1] + tr[2] * kernels[4 * 72 + 2 * 9 + 2] +
            ml[2] * kernels[4 * 72 + 2 * 9 + 3] + mc[2] * kernels[4 * 72 + 2 * 9 + 4] + mr[2] * kernels[4 * 72 + 2 * 9 + 5] +
            bl[2] * kernels[4 * 72 + 2 * 9 + 6] + bc[2] * kernels[4 * 72 + 2 * 9 + 7] + br[2] * kernels[4 * 72 + 2 * 9 + 8];

        c4 =
            tl[3] * kernels[4 * 72 + 3 * 9 + 0] + tc[3] * kernels[4 * 72 + 3 * 9 + 1] + tr[3] * kernels[4 * 72 + 3 * 9 + 2] +
            ml[3] * kernels[4 * 72 + 3 * 9 + 3] + mc[3] * kernels[4 * 72 + 3 * 9 + 4] + mr[3] * kernels[4 * 72 + 3 * 9 + 5] +
            bl[3] * kernels[4 * 72 + 3 * 9 + 6] + bc[3] * kernels[4 * 72 + 3 * 9 + 7] + br[3] * kernels[4 * 72 + 3 * 9 + 8];

        c5 =
            tl[4] * kernels[4 * 72 + 4 * 9 + 0] + tc[4] * kernels[4 * 72 + 4 * 9 + 1] + tr[4] * kernels[4 * 72 + 4 * 9 + 2] +
            ml[4] * kernels[4 * 72 + 4 * 9 + 3] + mc[4] * kernels[4 * 72 + 4 * 9 + 4] + mr[4] * kernels[4 * 72 + 4 * 9 + 5] +
            bl[4] * kernels[4 * 72 + 4 * 9 + 6] + bc[4] * kernels[4 * 72 + 4 * 9 + 7] + br[4] * kernels[4 * 72 + 4 * 9 + 8];

        c6 =
            tl[5] * kernels[4 * 72 + 5 * 9 + 0] + tc[5] * kernels[4 * 72 + 5 * 9 + 1] + tr[5] * kernels[4 * 72 + 5 * 9 + 2] +
            ml[5] * kernels[4 * 72 + 5 * 9 + 3] + mc[5] * kernels[4 * 72 + 5 * 9 + 4] + mr[5] * kernels[4 * 72 + 5 * 9 + 5] +
            bl[5] * kernels[4 * 72 + 5 * 9 + 6] + bc[5] * kernels[4 * 72 + 5 * 9 + 7] + br[5] * kernels[4 * 72 + 5 * 9 + 8];

        c7 =
            tl[6] * kernels[4 * 72 + 6 * 9 + 0] + tc[6] * kernels[4 * 72 + 6 * 9 + 1] + tr[6] * kernels[4 * 72 + 6 * 9 + 2] +
            ml[6] * kernels[4 * 72 + 6 * 9 + 3] + mc[6] * kernels[4 * 72 + 6 * 9 + 4] + mr[6] * kernels[4 * 72 + 6 * 9 + 5] +
            bl[6] * kernels[4 * 72 + 6 * 9 + 6] + bc[6] * kernels[4 * 72 + 6 * 9 + 7] + br[6] * kernels[4 * 72 + 6 * 9 + 8];

        c8 =
            tl[7] * kernels[4 * 72 + 7 * 9 + 0] + tc[7] * kernels[4 * 72 + 7 * 9 + 1] + tr[7] * kernels[4 * 72 + 7 * 9 + 2] +
            ml[7] * kernels[4 * 72 + 7 * 9 + 3] + mc[7] * kernels[4 * 72 + 7 * 9 + 4] + mr[7] * kernels[4 * 72 + 7 * 9 + 5] +
            bl[7] * kernels[4 * 72 + 7 * 9 + 6] + bc[7] * kernels[4 * 72 + 7 * 9 + 7] + br[7] * kernels[4 * 72 + 7 * 9 + 8];

        outMat[4] = RELU(c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + biases[4]);

        c1 =
            tl[0] * kernels[5 * 72 + 0 * 9 + 0] + tc[0] * kernels[5 * 72 + 0 * 9 + 1] + tr[0] * kernels[5 * 72 + 0 * 9 + 2] +
            ml[0] * kernels[5 * 72 + 0 * 9 + 3] + mc[0] * kernels[5 * 72 + 0 * 9 + 4] + mr[0] * kernels[5 * 72 + 0 * 9 + 5] +
            bl[0] * kernels[5 * 72 + 0 * 9 + 6] + bc[0] * kernels[5 * 72 + 0 * 9 + 7] + br[0] * kernels[5 * 72 + 0 * 9 + 8];

        c2 =
            tl[1] * kernels[5 * 72 + 1 * 9 + 0] + tc[1] * kernels[5 * 72 + 1 * 9 + 1] + tr[1] * kernels[5 * 72 + 1 * 9 + 2] +
            ml[1] * kernels[5 * 72 + 1 * 9 + 3] + mc[1] * kernels[5 * 72 + 1 * 9 + 4] + mr[1] * kernels[5 * 72 + 1 * 9 + 5] +
            bl[1] * kernels[5 * 72 + 1 * 9 + 6] + bc[1] * kernels[5 * 72 + 1 * 9 + 7] + br[1] * kernels[5 * 72 + 1 * 9 + 8];

        c3 =
            tl[2] * kernels[5 * 72 + 2 * 9 + 0] + tc[2] * kernels[5 * 72 + 2 * 9 + 1] + tr[2] * kernels[5 * 72 + 2 * 9 + 2] +
            ml[2] * kernels[5 * 72 + 2 * 9 + 3] + mc[2] * kernels[5 * 72 + 2 * 9 + 4] + mr[2] * kernels[5 * 72 + 2 * 9 + 5] +
            bl[2] * kernels[5 * 72 + 2 * 9 + 6] + bc[2] * kernels[5 * 72 + 2 * 9 + 7] + br[2] * kernels[5 * 72 + 2 * 9 + 8];

        c4 =
            tl[3] * kernels[5 * 72 + 3 * 9 + 0] + tc[3] * kernels[5 * 72 + 3 * 9 + 1] + tr[3] * kernels[5 * 72 + 3 * 9 + 2] +
            ml[3] * kernels[5 * 72 + 3 * 9 + 3] + mc[3] * kernels[5 * 72 + 3 * 9 + 4] + mr[3] * kernels[5 * 72 + 3 * 9 + 5] +
            bl[3] * kernels[5 * 72 + 3 * 9 + 6] + bc[3] * kernels[5 * 72 + 3 * 9 + 7] + br[3] * kernels[5 * 72 + 3 * 9 + 8];

        c5 =
            tl[4] * kernels[5 * 72 + 4 * 9 + 0] + tc[4] * kernels[5 * 72 + 4 * 9 + 1] + tr[4] * kernels[5 * 72 + 4 * 9 + 2] +
            ml[4] * kernels[5 * 72 + 4 * 9 + 3] + mc[4] * kernels[5 * 72 + 4 * 9 + 4] + mr[4] * kernels[5 * 72 + 4 * 9 + 5] +
            bl[4] * kernels[5 * 72 + 4 * 9 + 6] + bc[4] * kernels[5 * 72 + 4 * 9 + 7] + br[4] * kernels[5 * 72 + 4 * 9 + 8];

        c6 =
            tl[5] * kernels[5 * 72 + 5 * 9 + 0] + tc[5] * kernels[5 * 72 + 5 * 9 + 1] + tr[5] * kernels[5 * 72 + 5 * 9 + 2] +
            ml[5] * kernels[5 * 72 + 5 * 9 + 3] + mc[5] * kernels[5 * 72 + 5 * 9 + 4] + mr[5] * kernels[5 * 72 + 5 * 9 + 5] +
            bl[5] * kernels[5 * 72 + 5 * 9 + 6] + bc[5] * kernels[5 * 72 + 5 * 9 + 7] + br[5] * kernels[5 * 72 + 5 * 9 + 8];

        c7 =
            tl[6] * kernels[5 * 72 + 6 * 9 + 0] + tc[6] * kernels[5 * 72 + 6 * 9 + 1] + tr[6] * kernels[5 * 72 + 6 * 9 + 2] +
            ml[6] * kernels[5 * 72 + 6 * 9 + 3] + mc[6] * kernels[5 * 72 + 6 * 9 + 4] + mr[6] * kernels[5 * 72 + 6 * 9 + 5] +
            bl[6] * kernels[5 * 72 + 6 * 9 + 6] + bc[6] * kernels[5 * 72 + 6 * 9 + 7] + br[6] * kernels[5 * 72 + 6 * 9 + 8];

        c8 =
            tl[7] * kernels[5 * 72 + 7 * 9 + 0] + tc[7] * kernels[5 * 72 + 7 * 9 + 1] + tr[7] * kernels[5 * 72 + 7 * 9 + 2] +
            ml[7] * kernels[5 * 72 + 7 * 9 + 3] + mc[7] * kernels[5 * 72 + 7 * 9 + 4] + mr[7] * kernels[5 * 72 + 7 * 9 + 5] +
            bl[7] * kernels[5 * 72 + 7 * 9 + 6] + bc[7] * kernels[5 * 72 + 7 * 9 + 7] + br[7] * kernels[5 * 72 + 7 * 9 + 8];

        outMat[5] = RELU(c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + biases[5]);

        c1 =
            tl[0] * kernels[6 * 72 + 0 * 9 + 0] + tc[0] * kernels[6 * 72 + 0 * 9 + 1] + tr[0] * kernels[6 * 72 + 0 * 9 + 2] +
            ml[0] * kernels[6 * 72 + 0 * 9 + 3] + mc[0] * kernels[6 * 72 + 0 * 9 + 4] + mr[0] * kernels[6 * 72 + 0 * 9 + 5] +
            bl[0] * kernels[6 * 72 + 0 * 9 + 6] + bc[0] * kernels[6 * 72 + 0 * 9 + 7] + br[0] * kernels[6 * 72 + 0 * 9 + 8];

        c2 =
            tl[1] * kernels[6 * 72 + 1 * 9 + 0] + tc[1] * kernels[6 * 72 + 1 * 9 + 1] + tr[1] * kernels[6 * 72 + 1 * 9 + 2] +
            ml[1] * kernels[6 * 72 + 1 * 9 + 3] + mc[1] * kernels[6 * 72 + 1 * 9 + 4] + mr[1] * kernels[6 * 72 + 1 * 9 + 5] +
            bl[1] * kernels[6 * 72 + 1 * 9 + 6] + bc[1] * kernels[6 * 72 + 1 * 9 + 7] + br[1] * kernels[6 * 72 + 1 * 9 + 8];

        c3 =
            tl[2] * kernels[6 * 72 + 2 * 9 + 0] + tc[2] * kernels[6 * 72 + 2 * 9 + 1] + tr[2] * kernels[6 * 72 + 2 * 9 + 2] +
            ml[2] * kernels[6 * 72 + 2 * 9 + 3] + mc[2] * kernels[6 * 72 + 2 * 9 + 4] + mr[2] * kernels[6 * 72 + 2 * 9 + 5] +
            bl[2] * kernels[6 * 72 + 2 * 9 + 6] + bc[2] * kernels[6 * 72 + 2 * 9 + 7] + br[2] * kernels[6 * 72 + 2 * 9 + 8];

        c4 =
            tl[3] * kernels[6 * 72 + 3 * 9 + 0] + tc[3] * kernels[6 * 72 + 3 * 9 + 1] + tr[3] * kernels[6 * 72 + 3 * 9 + 2] +
            ml[3] * kernels[6 * 72 + 3 * 9 + 3] + mc[3] * kernels[6 * 72 + 3 * 9 + 4] + mr[3] * kernels[6 * 72 + 3 * 9 + 5] +
            bl[3] * kernels[6 * 72 + 3 * 9 + 6] + bc[3] * kernels[6 * 72 + 3 * 9 + 7] + br[3] * kernels[6 * 72 + 3 * 9 + 8];

        c5 =
            tl[4] * kernels[6 * 72 + 4 * 9 + 0] + tc[4] * kernels[6 * 72 + 4 * 9 + 1] + tr[4] * kernels[6 * 72 + 4 * 9 + 2] +
            ml[4] * kernels[6 * 72 + 4 * 9 + 3] + mc[4] * kernels[6 * 72 + 4 * 9 + 4] + mr[4] * kernels[6 * 72 + 4 * 9 + 5] +
            bl[4] * kernels[6 * 72 + 4 * 9 + 6] + bc[4] * kernels[6 * 72 + 4 * 9 + 7] + br[4] * kernels[6 * 72 + 4 * 9 + 8];

        c6 =
            tl[5] * kernels[6 * 72 + 5 * 9 + 0] + tc[5] * kernels[6 * 72 + 5 * 9 + 1] + tr[5] * kernels[6 * 72 + 5 * 9 + 2] +
            ml[5] * kernels[6 * 72 + 5 * 9 + 3] + mc[5] * kernels[6 * 72 + 5 * 9 + 4] + mr[5] * kernels[6 * 72 + 5 * 9 + 5] +
            bl[5] * kernels[6 * 72 + 5 * 9 + 6] + bc[5] * kernels[6 * 72 + 5 * 9 + 7] + br[5] * kernels[6 * 72 + 5 * 9 + 8];

        c7 =
            tl[6] * kernels[6 * 72 + 6 * 9 + 0] + tc[6] * kernels[6 * 72 + 6 * 9 + 1] + tr[6] * kernels[6 * 72 + 6 * 9 + 2] +
            ml[6] * kernels[6 * 72 + 6 * 9 + 3] + mc[6] * kernels[6 * 72 + 6 * 9 + 4] + mr[6] * kernels[6 * 72 + 6 * 9 + 5] +
            bl[6] * kernels[6 * 72 + 6 * 9 + 6] + bc[6] * kernels[6 * 72 + 6 * 9 + 7] + br[6] * kernels[6 * 72 + 6 * 9 + 8];

        c8 =
            tl[7] * kernels[6 * 72 + 7 * 9 + 0] + tc[7] * kernels[6 * 72 + 7 * 9 + 1] + tr[7] * kernels[6 * 72 + 7 * 9 + 2] +
            ml[7] * kernels[6 * 72 + 7 * 9 + 3] + mc[7] * kernels[6 * 72 + 7 * 9 + 4] + mr[7] * kernels[6 * 72 + 7 * 9 + 5] +
            bl[7] * kernels[6 * 72 + 7 * 9 + 6] + bc[7] * kernels[6 * 72 + 7 * 9 + 7] + br[7] * kernels[6 * 72 + 7 * 9 + 8];

        outMat[6] = RELU(c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + biases[6]);

        c1 =
            tl[0] * kernels[7 * 72 + 0 * 9 + 0] + tc[0] * kernels[7 * 72 + 0 * 9 + 1] + tr[0] * kernels[7 * 72 + 0 * 9 + 2] +
            ml[0] * kernels[7 * 72 + 0 * 9 + 3] + mc[0] * kernels[7 * 72 + 0 * 9 + 4] + mr[0] * kernels[7 * 72 + 0 * 9 + 5] +
            bl[0] * kernels[7 * 72 + 0 * 9 + 6] + bc[0] * kernels[7 * 72 + 0 * 9 + 7] + br[0] * kernels[7 * 72 + 0 * 9 + 8];

        c2 =
            tl[1] * kernels[7 * 72 + 1 * 9 + 0] + tc[1] * kernels[7 * 72 + 1 * 9 + 1] + tr[1] * kernels[7 * 72 + 1 * 9 + 2] +
            ml[1] * kernels[7 * 72 + 1 * 9 + 3] + mc[1] * kernels[7 * 72 + 1 * 9 + 4] + mr[1] * kernels[7 * 72 + 1 * 9 + 5] +
            bl[1] * kernels[7 * 72 + 1 * 9 + 6] + bc[1] * kernels[7 * 72 + 1 * 9 + 7] + br[1] * kernels[7 * 72 + 1 * 9 + 8];

        c3 =
            tl[2] * kernels[7 * 72 + 2 * 9 + 0] + tc[2] * kernels[7 * 72 + 2 * 9 + 1] + tr[2] * kernels[7 * 72 + 2 * 9 + 2] +
            ml[2] * kernels[7 * 72 + 2 * 9 + 3] + mc[2] * kernels[7 * 72 + 2 * 9 + 4] + mr[2] * kernels[7 * 72 + 2 * 9 + 5] +
            bl[2] * kernels[7 * 72 + 2 * 9 + 6] + bc[2] * kernels[7 * 72 + 2 * 9 + 7] + br[2] * kernels[7 * 72 + 2 * 9 + 8];

        c4 =
            tl[3] * kernels[7 * 72 + 3 * 9 + 0] + tc[3] * kernels[7 * 72 + 3 * 9 + 1] + tr[3] * kernels[7 * 72 + 3 * 9 + 2] +
            ml[3] * kernels[7 * 72 + 3 * 9 + 3] + mc[3] * kernels[7 * 72 + 3 * 9 + 4] + mr[3] * kernels[7 * 72 + 3 * 9 + 5] +
            bl[3] * kernels[7 * 72 + 3 * 9 + 6] + bc[3] * kernels[7 * 72 + 3 * 9 + 7] + br[3] * kernels[7 * 72 + 3 * 9 + 8];

        c5 =
            tl[4] * kernels[7 * 72 + 4 * 9 + 0] + tc[4] * kernels[7 * 72 + 4 * 9 + 1] + tr[4] * kernels[7 * 72 + 4 * 9 + 2] +
            ml[4] * kernels[7 * 72 + 4 * 9 + 3] + mc[4] * kernels[7 * 72 + 4 * 9 + 4] + mr[4] * kernels[7 * 72 + 4 * 9 + 5] +
            bl[4] * kernels[7 * 72 + 4 * 9 + 6] + bc[4] * kernels[7 * 72 + 4 * 9 + 7] + br[4] * kernels[7 * 72 + 4 * 9 + 8];

        c6 =
            tl[5] * kernels[7 * 72 + 5 * 9 + 0] + tc[5] * kernels[7 * 72 + 5 * 9 + 1] + tr[5] * kernels[7 * 72 + 5 * 9 + 2] +
            ml[5] * kernels[7 * 72 + 5 * 9 + 3] + mc[5] * kernels[7 * 72 + 5 * 9 + 4] + mr[5] * kernels[7 * 72 + 5 * 9 + 5] +
            bl[5] * kernels[7 * 72 + 5 * 9 + 6] + bc[5] * kernels[7 * 72 + 5 * 9 + 7] + br[5] * kernels[7 * 72 + 5 * 9 + 8];

        c7 =
            tl[6] * kernels[7 * 72 + 6 * 9 + 0] + tc[6] * kernels[7 * 72 + 6 * 9 + 1] + tr[6] * kernels[7 * 72 + 6 * 9 + 2] +
            ml[6] * kernels[7 * 72 + 6 * 9 + 3] + mc[6] * kernels[7 * 72 + 6 * 9 + 4] + mr[6] * kernels[7 * 72 + 6 * 9 + 5] +
            bl[6] * kernels[7 * 72 + 6 * 9 + 6] + bc[6] * kernels[7 * 72 + 6 * 9 + 7] + br[6] * kernels[7 * 72 + 6 * 9 + 8];

        c8 =
            tl[7] * kernels[7 * 72 + 7 * 9 + 0] + tc[7] * kernels[7 * 72 + 7 * 9 + 1] + tr[7] * kernels[7 * 72 + 7 * 9 + 2] +
            ml[7] * kernels[7 * 72 + 7 * 9 + 3] + mc[7] * kernels[7 * 72 + 7 * 9 + 4] + mr[7] * kernels[7 * 72 + 7 * 9 + 5] +
            bl[7] * kernels[7 * 72 + 7 * 9 + 6] + bc[7] * kernels[7 * 72 + 7 * 9 + 7] + br[7] * kernels[7 * 72 + 7 * 9 + 8];

        outMat[7] = RELU(c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + biases[7]);

#endif // ENABLE_AVX

        }, tmpMat);
}

void Anime4KCPP::CPU::CNNProcessor::convTranspose8To1B(cv::Mat& img, const FP* kernels, cv::Mat& tmpMat)
{
    changEachPixelNTo1(img, [&](const int i, const int j, ChanB outMat, LineFP inMat) {
        const int index = ((i & 1) << 1) + (j & 1);

        //180 degree rotation for kernel
        //0 1  to  3 2
        //2 3      1 0

#ifdef ENABLE_AVX
        const __m256 _in = _mm256_loadu_ps(inMat);
        const __m256 _k0 = _mm256_loadu_ps(kernels + index * 8);
        const __m256 _r0 = _mm256_dp_ps(_in, _k0, 0xf1);
        const __m128 _r1 = _mm256_extractf128_ps(_r0, 0x01);
        const __m128 _r2 = _mm256_castps256_ps128(_r0);
        const __m128 _r3 = _mm_add_ps(_r1, _r2);
       
        const FP luma = _mm_cvtss_f32(_r3);

        *outMat = UNNORMB(luma);
#else
        const FP luma = (
            inMat[0] * kernels[0 + index] +
            inMat[1] * kernels[4 + index] +
            inMat[2] * kernels[8 + index] +
            inMat[3] * kernels[12 + index] +
            inMat[4] * kernels[16 + index] +
            inMat[5] * kernels[20 + index] +
            inMat[6] * kernels[24 + index] +
            inMat[7] * kernels[28 + index]);

        *outMat = UNNORMB(luma);
#endif
        }, tmpMat);
}

void Anime4KCPP::CPU::CNNProcessor::convTranspose8To1W(cv::Mat& img, const FP* kernels, cv::Mat& tmpMat)
{
    changEachPixelNTo1(img, [&](const int i, const int j, ChanW outMat, LineFP inMat) {
        const int index = ((i & 1) << 1) + (j & 1);

        //180 degree rotation for kernel
        //0 1  to  3 2
        //2 3      1 0

#ifdef ENABLE_AVX
        const __m256 _in = _mm256_loadu_ps(inMat);
        const __m256 _k0 = _mm256_loadu_ps(kernels + index * 8);
        const __m256 _r0 = _mm256_dp_ps(_in, _k0, 0xf1);
        const __m128 _r1 = _mm256_extractf128_ps(_r0, 0x01);
        const __m128 _r2 = _mm256_castps256_ps128(_r0);
        const __m128 _r3 = _mm_add_ps(_r1, _r2);

        const FP luma = _mm_cvtss_f32(_r3);

        *outMat = UNNORMW(luma);
#else
        const FP luma = (
            inMat[0] * kernels[0 + index] +
            inMat[1] * kernels[4 + index] +
            inMat[2] * kernels[8 + index] +
            inMat[3] * kernels[12 + index] +
            inMat[4] * kernels[16 + index] +
            inMat[5] * kernels[20 + index] +
            inMat[6] * kernels[24 + index] +
            inMat[7] * kernels[28 + index]);

        *outMat = UNNORMW(luma);
#endif // ENABLE_AVX


        }, tmpMat);
}

void Anime4KCPP::CPU::CNNProcessor::convTranspose8To1F(cv::Mat& img, const FP* kernels, cv::Mat& tmpMat)
{
    changEachPixelNTo1(img, [&](const int i, const int j, ChanF outMat, LineFP inMat) {
        const int index = ((i & 1) << 1) + (j & 1);

        //180 degree rotation for kernel
        //0 1  to  3 2
        //2 3      1 0
#ifdef ENABLE_AVX
        const __m256 _in = _mm256_loadu_ps(inMat);
        const __m256 _k0 = _mm256_loadu_ps(kernels + index * 8);
        const __m256 _r0 = _mm256_dp_ps(_in, _k0, 0xf1);
        const __m128 _r1 = _mm256_extractf128_ps(_r0, 0x01);
        const __m128 _r2 = _mm256_castps256_ps128(_r0);
        const __m128 _r3 = _mm_add_ps(_r1, _r2);

        const FP luma = _mm_cvtss_f32(_r3);

        *outMat = CLAMP(luma, 0.0f, 1.0f);
#else
        const FP luma = (
            inMat[0] * kernels[0 + index] +
            inMat[1] * kernels[4 + index] +
            inMat[2] * kernels[8 + index] +
            inMat[3] * kernels[12 + index] +
            inMat[4] * kernels[16 + index] +
            inMat[5] * kernels[20 + index] +
            inMat[6] * kernels[24 + index] +
            inMat[7] * kernels[28 + index]);

        *outMat = static_cast<float>(CLAMP(luma, 0.0, 1.0));
#endif


        }, tmpMat);
}

void Anime4KCPP::CPU::CNNProcessor::changEachPixel1ToN(const cv::Mat& src,
    const std::function<void(int, int, ChanFP, LineB)>&& callBack,
    cv::Mat& tmpMat, int outChannels)
{
    const int h = src.rows, w = src.cols;
    const int jMAX = w * outChannels;

    tmpMat.create(h, w, AC_CV_FPC(outChannels));

    const size_t srcStep = src.step;
    const size_t step = tmpMat.step;

#if defined(_MSC_VER) || defined(USE_TBB)
    Parallel::parallel_for(0, h, [&](int i) {
        LineB lineData = src.data + static_cast<size_t>(i) * srcStep;
        ChanFP tmpLineData = reinterpret_cast<ChanFP>(tmpMat.data + static_cast<size_t>(i) * step);
        for (int j = 0; j < jMAX; j += outChannels)
            callBack(i, j, tmpLineData + j, lineData);
        });
#else
#pragma omp parallel for
    for (int i = 0; i < h; i++)
    {
        LineB lineData = src.data + static_cast<size_t>(i) * srcStep;
        ChanFP tmpLineData = reinterpret_cast<ChanFP>(tmpMat.data + static_cast<size_t>(i) * step);
        for (int j = 0; j < jMAX; j += outChannels)
            callBack(i, j, tmpLineData + j, lineData);
    }
#endif
}

void Anime4KCPP::CPU::CNNProcessor::changEachPixel1ToN(const cv::Mat& src,
    const std::function<void(int, int, ChanFP, LineW)>&& callBack,
    cv::Mat& tmpMat, int outChannels)
{
    const int h = src.rows, w = src.cols;
    const int jMAX = w * outChannels;

    tmpMat.create(h, w, AC_CV_FPC(outChannels));

    const size_t srcStep = src.step;
    const size_t step = tmpMat.step;

#if defined(_MSC_VER) || defined(USE_TBB)
    Parallel::parallel_for(0, h, [&](int i) {
        LineW lineData = reinterpret_cast<LineW>(src.data + static_cast<size_t>(i) * srcStep);
        ChanFP tmpLineData = reinterpret_cast<ChanFP>(tmpMat.data + static_cast<size_t>(i) * step);
        for (int j = 0; j < jMAX; j += outChannels)
            callBack(i, j, tmpLineData + j, lineData);
        });
#else
#pragma omp parallel for
    for (int i = 0; i < h; i++)
    {
        LineW lineData = reinterpret_cast<LineW>(src.data + static_cast<size_t>(i) * srcStep);
        ChanFP tmpLineData = reinterpret_cast<ChanFP>(tmpMat.data + static_cast<size_t>(i) * step);
        for (int j = 0; j < jMAX; j += outChannels)
            callBack(i, j, tmpLineData + j, lineData);
    }
#endif
}

void Anime4KCPP::CPU::CNNProcessor::changEachPixel1ToN(const cv::Mat& src,
    const std::function<void(int, int, ChanFP, LineF)>&& callBack,
    cv::Mat& tmpMat, int outChannels)
{
    const int h = src.rows, w = src.cols;
    const int jMAX = w * outChannels;

    tmpMat.create(h, w, AC_CV_FPC(outChannels));

    const size_t srcStep = src.step;
    const size_t step = tmpMat.step;

#if defined(_MSC_VER) || defined(USE_TBB)
    Parallel::parallel_for(0, h, [&](int i) {
        LineF lineData = reinterpret_cast<LineF>(src.data + static_cast<size_t>(i) * srcStep);
        ChanFP tmpLineData = reinterpret_cast<ChanFP>(tmpMat.data + static_cast<size_t>(i) * step);
        for (int j = 0; j < jMAX; j += outChannels)
            callBack(i, j, tmpLineData + j, lineData);
        });
#else
#pragma omp parallel for
    for (int i = 0; i < h; i++)
    {
        LineF lineData = reinterpret_cast<LineF>(src.data + static_cast<size_t>(i) * srcStep);
        ChanFP tmpLineData = reinterpret_cast<ChanFP>(tmpMat.data + static_cast<size_t>(i) * step);
        for (int j = 0; j < jMAX; j += outChannels)
            callBack(i, j, tmpLineData + j, lineData);
    }
#endif
}

void Anime4KCPP::CPU::CNNProcessor::changEachPixelNToN(
    const std::function<void(int, int, ChanFP, LineFP)>&& callBack,
    cv::Mat& tmpMat)
{
    const int h = tmpMat.rows, w = tmpMat.cols;
    const int channels = tmpMat.channels();
    const int jMAX = w * channels;
    const size_t step = tmpMat.step;

    cv::Mat tmp;
    tmp.create(h, w, tmpMat.type());

#if defined(_MSC_VER) || defined(USE_TBB)
    Parallel::parallel_for(0, h, [&](int i) {
        LineFP lineData = reinterpret_cast<LineFP>(tmpMat.data + static_cast<size_t>(i) * step);
        ChanFP tmpLineData = reinterpret_cast<ChanFP>(tmp.data + static_cast<size_t>(i) * step);
        for (int j = 0; j < jMAX; j += channels)
            callBack(i, j, tmpLineData + j, lineData);
        });
#else
#pragma omp parallel for
    for (int i = 0; i < h; i++)
    {
        LineFP lineData = reinterpret_cast<LineFP>(tmpMat.data + static_cast<size_t>(i) * step);
        ChanFP tmpLineData = reinterpret_cast<ChanFP>(tmp.data + static_cast<size_t>(i) * step);
        for (int j = 0; j < jMAX; j += channels)
            callBack(i, j, tmpLineData + j, lineData);
    }
#endif

    tmpMat = tmp;
}

void Anime4KCPP::CPU::CNNProcessor::changEachPixelNTo1(cv::Mat& img,
    const std::function<void(int, int, ChanB, LineFP)>&& callBack,
    const cv::Mat& tmpMat)
{
    const int h = 2 * tmpMat.rows, w = 2 * tmpMat.cols;
    img.create(h, w, CV_8UC1);

    const int jMAX = w;
    const size_t channels = tmpMat.channels();
    const size_t step = tmpMat.step;
    const size_t dstStep = img.step;

#if defined(_MSC_VER) || defined(USE_TBB)
    Parallel::parallel_for(0, h, [&](int i) {
        LineFP lineData = reinterpret_cast<LineFP>(tmpMat.data + static_cast<size_t>(i >> 1) * step);
        ChanB tmpLineData = img.data + static_cast<size_t>(i) * dstStep;
        for (int j = 0; j < jMAX; j++)
            callBack(i, j, tmpLineData + j, lineData + static_cast<size_t>(j >> 1) * channels);
        });
#else
#pragma omp parallel for
    for (int i = 0; i < h; i++)
    {
        LineFP lineData = reinterpret_cast<LineFP>(tmpMat.data + static_cast<size_t>(i >> 1) * step);
        ChanB tmpLineData = img.data + static_cast<size_t>(i) * dstStep;
        for (int j = 0; j < jMAX; j++)
            callBack(i, j, tmpLineData + j, lineData + static_cast<size_t>(j >> 1) * channels);
    }
#endif
}

void Anime4KCPP::CPU::CNNProcessor::changEachPixelNTo1(cv::Mat& img,
    const std::function<void(int, int, ChanW, LineFP)>&& callBack,
    const cv::Mat& tmpMat)
{
    const int h = 2 * tmpMat.rows, w = 2 * tmpMat.cols;
    img.create(h, w, CV_16UC1);

    const int jMAX = w;
    const size_t channels = tmpMat.channels();
    const size_t step = tmpMat.step;
    const size_t dstStep = img.step;

#if defined(_MSC_VER) || defined(USE_TBB)
    Parallel::parallel_for(0, h, [&](int i) {
        LineFP lineData = reinterpret_cast<LineFP>(tmpMat.data + static_cast<size_t>(i >> 1) * step);
        ChanW tmpLineData = reinterpret_cast<ChanW>(img.data + static_cast<size_t>(i) * dstStep);
        for (int j = 0; j < jMAX; j++)
            callBack(i, j, tmpLineData + j, lineData + static_cast<size_t>(j >> 1) * channels);
        });
#else
#pragma omp parallel for
    for (int i = 0; i < h; i++)
    {
        LineFP lineData = reinterpret_cast<LineFP>(tmpMat.data + static_cast<size_t>(i >> 1) * step);
        ChanW tmpLineData = reinterpret_cast<ChanW>(img.data + static_cast<size_t>(i) * dstStep);
        for (int j = 0; j < jMAX; j++)
            callBack(i, j, tmpLineData + j, lineData + static_cast<size_t>(j >> 1) * channels);
    }
#endif
}

void Anime4KCPP::CPU::CNNProcessor::changEachPixelNTo1(cv::Mat& img,
    const std::function<void(int, int, ChanF, LineFP)>&& callBack,
    const cv::Mat& tmpMat)
{
    const int h = 2 * tmpMat.rows, w = 2 * tmpMat.cols;
    img.create(h, w, CV_32FC1);

    const int jMAX = w;
    const size_t channels = tmpMat.channels();
    const size_t step = tmpMat.step;
    const size_t dstStep = img.step;

#if defined(_MSC_VER) || defined(USE_TBB)
    Parallel::parallel_for(0, h, [&](int i) {
        LineFP lineData = reinterpret_cast<LineFP>(tmpMat.data + static_cast<size_t>(i >> 1) * step);
        ChanF tmpLineData = reinterpret_cast<ChanF>(img.data + static_cast<size_t>(i) * dstStep);
        for (int j = 0; j < jMAX; j++)
            callBack(i, j, tmpLineData + j, lineData + static_cast<size_t>(j >> 1) * channels);
        });
#else
#pragma omp parallel for
    for (int i = 0; i < h; i++)
    {
        LineFP lineData = reinterpret_cast<LineFP>(tmpMat.data + static_cast<size_t>(i >> 1) * step);
        ChanF tmpLineData = reinterpret_cast<ChanF>(img.data + static_cast<size_t>(i) * dstStep);
        for (int j = 0; j < jMAX; j++)
            callBack(i, j, tmpLineData + j, lineData + static_cast<size_t>(j >> 1) * channels);
    }
#endif
}
