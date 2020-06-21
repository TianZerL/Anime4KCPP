#include "CNNProcessor.h"

void Anime4KCPP::CNNProcessor::conv1To8(cv::InputArray img, const double* kernels, const double* biases, std::pair<cv::Mat, cv::Mat>& tmpMats)
{
    const int srcChannels = img.channels();
    const int lineStep = img.cols() * srcChannels;
    changEachPixel1To8(img, [&](const int i, const int j, Chan tmpMat1, Chan tmpMat2, LineC curLine) {
        const int orgJ = j / 4 * srcChannels;
        const int jp = orgJ < (img.cols() - 1) * srcChannels ? srcChannels : 0;
        const int jn = orgJ > srcChannels ? -srcChannels : 0;
        const LineC pLineData = i < img.rows() - 1 ? curLine + lineStep : curLine;
        const LineC cLineData = curLine;
        const LineC nLineData = i > 0 ? curLine - lineStep : curLine;

        const PIXEL tl = nLineData + orgJ + jn, tc = nLineData + orgJ, tr = nLineData + orgJ + jp;
        const PIXEL ml = cLineData + orgJ + jn, mc = cLineData + orgJ, mr = cLineData + orgJ + jp;
        const PIXEL bl = pLineData + orgJ + jn, bc = pLineData + orgJ, br = pLineData + orgJ + jp;

        double tln = NORM(tl[Y]);
        double tcn = NORM(tc[Y]);
        double trn = NORM(tr[Y]);
        double mln = NORM(ml[Y]);
        double mcn = NORM(mc[Y]);
        double mrn = NORM(mr[Y]);
        double bln = NORM(bl[Y]);
        double bcn = NORM(bc[Y]);
        double brn = NORM(br[Y]);

        tmpMat1[0] =
            RULE(
                tln * kernels[0 * 9 + 0] + tcn * kernels[0 * 9 + 1] + trn * kernels[0 * 9 + 2] +
                mln * kernels[0 * 9 + 3] + mcn * kernels[0 * 9 + 4] + mrn * kernels[0 * 9 + 5] +
                bln * kernels[0 * 9 + 6] + bcn * kernels[0 * 9 + 7] + brn * kernels[0 * 9 + 8] + biases[0]);
        tmpMat1[1] =
            RULE(
                tln * kernels[1 * 9 + 0] + tcn * kernels[1 * 9 + 1] + trn * kernels[1 * 9 + 2] +
                mln * kernels[1 * 9 + 3] + mcn * kernels[1 * 9 + 4] + mrn * kernels[1 * 9 + 5] +
                bln * kernels[1 * 9 + 6] + bcn * kernels[1 * 9 + 7] + brn * kernels[1 * 9 + 8] + biases[1]);
        tmpMat1[2] =
            RULE(
                tln * kernels[2 * 9 + 0] + tcn * kernels[2 * 9 + 1] + trn * kernels[2 * 9 + 2] +
                mln * kernels[2 * 9 + 3] + mcn * kernels[2 * 9 + 4] + mrn * kernels[2 * 9 + 5] +
                bln * kernels[2 * 9 + 6] + bcn * kernels[2 * 9 + 7] + brn * kernels[2 * 9 + 8] + biases[2]);
        tmpMat1[3] =
            RULE(
                tln * kernels[3 * 9 + 0] + tcn * kernels[3 * 9 + 1] + trn * kernels[3 * 9 + 2] +
                mln * kernels[3 * 9 + 3] + mcn * kernels[3 * 9 + 4] + mrn * kernels[3 * 9 + 5] +
                bln * kernels[3 * 9 + 6] + bcn * kernels[3 * 9 + 7] + brn * kernels[3 * 9 + 8] + biases[3]);
        tmpMat2[0] =
            RULE(
                tln * kernels[4 * 9 + 0] + tcn * kernels[4 * 9 + 1] + trn * kernels[4 * 9 + 2] +
                mln * kernels[4 * 9 + 3] + mcn * kernels[4 * 9 + 4] + mrn * kernels[4 * 9 + 5] +
                bln * kernels[4 * 9 + 6] + bcn * kernels[4 * 9 + 7] + brn * kernels[4 * 9 + 8] + biases[4]);
        tmpMat2[1] =
            RULE(
                tln * kernels[5 * 9 + 0] + tcn * kernels[5 * 9 + 1] + trn * kernels[5 * 9 + 2] +
                mln * kernels[5 * 9 + 3] + mcn * kernels[5 * 9 + 4] + mrn * kernels[5 * 9 + 5] +
                bln * kernels[5 * 9 + 6] + bcn * kernels[5 * 9 + 7] + brn * kernels[5 * 9 + 8] + biases[5]);
        tmpMat2[2] =
            RULE(
                tln * kernels[6 * 9 + 0] + tcn * kernels[6 * 9 + 1] + trn * kernels[6 * 9 + 2] +
                mln * kernels[6 * 9 + 3] + mcn * kernels[6 * 9 + 4] + mrn * kernels[6 * 9 + 5] +
                bln * kernels[6 * 9 + 6] + bcn * kernels[6 * 9 + 7] + brn * kernels[6 * 9 + 8] + biases[6]);
        tmpMat2[3] =
            RULE(
                tln * kernels[7 * 9 + 0] + tcn * kernels[7 * 9 + 1] + trn * kernels[7 * 9 + 2] +
                mln * kernels[7 * 9 + 3] + mcn * kernels[7 * 9 + 4] + mrn * kernels[7 * 9 + 5] +
                bln * kernels[7 * 9 + 6] + bcn * kernels[7 * 9 + 7] + brn * kernels[7 * 9 + 8] + biases[7]);

        }, tmpMats);
}

void Anime4KCPP::CNNProcessor::conv8To8(const double* kernels, const double* biases, std::pair<cv::Mat, cv::Mat>& tmpMats)
{
    const int lineStep = tmpMats.first.cols * 4;
    changEachPixel8To8([&](const int i, const int j, Chan tmpMat1, Chan tmpMat2, LineF curLine1, LineF curLine2) {
        const int jp = j < (tmpMats.first.cols - 1) * 4 ? 4 : 0;
        const int jn = j > 4 ? -4 : 0;
        const LineF pLineData1 = i < tmpMats.first.rows - 1 ? curLine1 + lineStep : curLine1;
        const LineF cLineData1 = curLine1;
        const LineF nLineData1 = i > 0 ? curLine1 - lineStep : curLine1;

        const LineF pLineData2 = i < tmpMats.first.rows - 1 ? curLine2 + lineStep : curLine2;
        const LineF cLineData2 = curLine2;
        const LineF nLineData2 = i > 0 ? curLine2 - lineStep : curLine2;

        const Chan tl1 = nLineData1 + j + jn, tc1 = nLineData1 + j, tr1 = nLineData1 + j + jp;
        const Chan ml1 = cLineData1 + j + jn, mc1 = cLineData1 + j, mr1 = cLineData1 + j + jp;
        const Chan bl1 = pLineData1 + j + jn, bc1 = pLineData1 + j, br1 = pLineData1 + j + jp;

        const Chan tl2 = nLineData2 + j + jn, tc2 = nLineData2 + j, tr2 = nLineData2 + j + jp;
        const Chan ml2 = cLineData2 + j + jn, mc2 = cLineData2 + j, mr2 = cLineData2 + j + jp;
        const Chan bl2 = pLineData2 + j + jn, bc2 = pLineData2 + j, br2 = pLineData2 + j + jp;

        double c1 =
            tl1[0] * kernels[0 * 72 + 0 * 9 + 0] + tc1[0] * kernels[0 * 72 + 0 * 9 + 1] + tr1[0] * kernels[0 * 72 + 0 * 9 + 2] +
            ml1[0] * kernels[0 * 72 + 0 * 9 + 3] + mc1[0] * kernels[0 * 72 + 0 * 9 + 4] + mr1[0] * kernels[0 * 72 + 0 * 9 + 5] +
            bl1[0] * kernels[0 * 72 + 0 * 9 + 6] + bc1[0] * kernels[0 * 72 + 0 * 9 + 7] + br1[0] * kernels[0 * 72 + 0 * 9 + 8];

        double c2 =
            tl1[1] * kernels[0 * 72 + 1 * 9 + 0] + tc1[1] * kernels[0 * 72 + 1 * 9 + 1] + tr1[1] * kernels[0 * 72 + 1 * 9 + 2] +
            ml1[1] * kernels[0 * 72 + 1 * 9 + 3] + mc1[1] * kernels[0 * 72 + 1 * 9 + 4] + mr1[1] * kernels[0 * 72 + 1 * 9 + 5] +
            bl1[1] * kernels[0 * 72 + 1 * 9 + 6] + bc1[1] * kernels[0 * 72 + 1 * 9 + 7] + br1[1] * kernels[0 * 72 + 1 * 9 + 8];

        double c3 =
            tl1[2] * kernels[0 * 72 + 2 * 9 + 0] + tc1[2] * kernels[0 * 72 + 2 * 9 + 1] + tr1[2] * kernels[0 * 72 + 2 * 9 + 2] +
            ml1[2] * kernels[0 * 72 + 2 * 9 + 3] + mc1[2] * kernels[0 * 72 + 2 * 9 + 4] + mr1[2] * kernels[0 * 72 + 2 * 9 + 5] +
            bl1[2] * kernels[0 * 72 + 2 * 9 + 6] + bc1[2] * kernels[0 * 72 + 2 * 9 + 7] + br1[2] * kernels[0 * 72 + 2 * 9 + 8];

        double c4 =
            tl1[3] * kernels[0 * 72 + 3 * 9 + 0] + tc1[3] * kernels[0 * 72 + 3 * 9 + 1] + tr1[3] * kernels[0 * 72 + 3 * 9 + 2] +
            ml1[3] * kernels[0 * 72 + 3 * 9 + 3] + mc1[3] * kernels[0 * 72 + 3 * 9 + 4] + mr1[3] * kernels[0 * 72 + 3 * 9 + 5] +
            bl1[3] * kernels[0 * 72 + 3 * 9 + 6] + bc1[3] * kernels[0 * 72 + 3 * 9 + 7] + br1[3] * kernels[0 * 72 + 3 * 9 + 8];

        double c5 =
            tl2[0] * kernels[0 * 72 + 4 * 9 + 0] + tc2[0] * kernels[0 * 72 + 4 * 9 + 1] + tr2[0] * kernels[0 * 72 + 4 * 9 + 2] +
            ml2[0] * kernels[0 * 72 + 4 * 9 + 3] + mc2[0] * kernels[0 * 72 + 4 * 9 + 4] + mr2[0] * kernels[0 * 72 + 4 * 9 + 5] +
            bl2[0] * kernels[0 * 72 + 4 * 9 + 6] + bc2[0] * kernels[0 * 72 + 4 * 9 + 7] + br2[0] * kernels[0 * 72 + 4 * 9 + 8];

        double c6 =
            tl2[1] * kernels[0 * 72 + 5 * 9 + 0] + tc2[1] * kernels[0 * 72 + 5 * 9 + 1] + tr2[1] * kernels[0 * 72 + 5 * 9 + 2] +
            ml2[1] * kernels[0 * 72 + 5 * 9 + 3] + mc2[1] * kernels[0 * 72 + 5 * 9 + 4] + mr2[1] * kernels[0 * 72 + 5 * 9 + 5] +
            bl2[1] * kernels[0 * 72 + 5 * 9 + 6] + bc2[1] * kernels[0 * 72 + 5 * 9 + 7] + br2[1] * kernels[0 * 72 + 5 * 9 + 8];

        double c7 =
            tl2[2] * kernels[0 * 72 + 6 * 9 + 0] + tc2[2] * kernels[0 * 72 + 6 * 9 + 1] + tr2[2] * kernels[0 * 72 + 6 * 9 + 2] +
            ml2[2] * kernels[0 * 72 + 6 * 9 + 3] + mc2[2] * kernels[0 * 72 + 6 * 9 + 4] + mr2[2] * kernels[0 * 72 + 6 * 9 + 5] +
            bl2[2] * kernels[0 * 72 + 6 * 9 + 6] + bc2[2] * kernels[0 * 72 + 6 * 9 + 7] + br2[2] * kernels[0 * 72 + 6 * 9 + 8];

        double c8 =
            tl2[3] * kernels[0 * 72 + 7 * 9 + 0] + tc2[3] * kernels[0 * 72 + 7 * 9 + 1] + tr2[3] * kernels[0 * 72 + 7 * 9 + 2] +
            ml2[3] * kernels[0 * 72 + 7 * 9 + 3] + mc2[3] * kernels[0 * 72 + 7 * 9 + 4] + mr2[3] * kernels[0 * 72 + 7 * 9 + 5] +
            bl2[3] * kernels[0 * 72 + 7 * 9 + 6] + bc2[3] * kernels[0 * 72 + 7 * 9 + 7] + br2[3] * kernels[0 * 72 + 7 * 9 + 8];

        tmpMat1[0] = RULE(c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + biases[0]);

        c1 =
            tl1[0] * kernels[1 * 72 + 0 * 9 + 0] + tc1[0] * kernels[1 * 72 + 0 * 9 + 1] + tr1[0] * kernels[1 * 72 + 0 * 9 + 2] +
            ml1[0] * kernels[1 * 72 + 0 * 9 + 3] + mc1[0] * kernels[1 * 72 + 0 * 9 + 4] + mr1[0] * kernels[1 * 72 + 0 * 9 + 5] +
            bl1[0] * kernels[1 * 72 + 0 * 9 + 6] + bc1[0] * kernels[1 * 72 + 0 * 9 + 7] + br1[0] * kernels[1 * 72 + 0 * 9 + 8];

        c2 =
            tl1[1] * kernels[1 * 72 + 1 * 9 + 0] + tc1[1] * kernels[1 * 72 + 1 * 9 + 1] + tr1[1] * kernels[1 * 72 + 1 * 9 + 2] +
            ml1[1] * kernels[1 * 72 + 1 * 9 + 3] + mc1[1] * kernels[1 * 72 + 1 * 9 + 4] + mr1[1] * kernels[1 * 72 + 1 * 9 + 5] +
            bl1[1] * kernels[1 * 72 + 1 * 9 + 6] + bc1[1] * kernels[1 * 72 + 1 * 9 + 7] + br1[1] * kernels[1 * 72 + 1 * 9 + 8];

        c3 =
            tl1[2] * kernels[1 * 72 + 2 * 9 + 0] + tc1[2] * kernels[1 * 72 + 2 * 9 + 1] + tr1[2] * kernels[1 * 72 + 2 * 9 + 2] +
            ml1[2] * kernels[1 * 72 + 2 * 9 + 3] + mc1[2] * kernels[1 * 72 + 2 * 9 + 4] + mr1[2] * kernels[1 * 72 + 2 * 9 + 5] +
            bl1[2] * kernels[1 * 72 + 2 * 9 + 6] + bc1[2] * kernels[1 * 72 + 2 * 9 + 7] + br1[2] * kernels[1 * 72 + 2 * 9 + 8];

        c4 =
            tl1[3] * kernels[1 * 72 + 3 * 9 + 0] + tc1[3] * kernels[1 * 72 + 3 * 9 + 1] + tr1[3] * kernels[1 * 72 + 3 * 9 + 2] +
            ml1[3] * kernels[1 * 72 + 3 * 9 + 3] + mc1[3] * kernels[1 * 72 + 3 * 9 + 4] + mr1[3] * kernels[1 * 72 + 3 * 9 + 5] +
            bl1[3] * kernels[1 * 72 + 3 * 9 + 6] + bc1[3] * kernels[1 * 72 + 3 * 9 + 7] + br1[3] * kernels[1 * 72 + 3 * 9 + 8];

        c5 =
            tl2[0] * kernels[1 * 72 + 4 * 9 + 0] + tc2[0] * kernels[1 * 72 + 4 * 9 + 1] + tr2[0] * kernels[1 * 72 + 4 * 9 + 2] +
            ml2[0] * kernels[1 * 72 + 4 * 9 + 3] + mc2[0] * kernels[1 * 72 + 4 * 9 + 4] + mr2[0] * kernels[1 * 72 + 4 * 9 + 5] +
            bl2[0] * kernels[1 * 72 + 4 * 9 + 6] + bc2[0] * kernels[1 * 72 + 4 * 9 + 7] + br2[0] * kernels[1 * 72 + 4 * 9 + 8];

        c6 =
            tl2[1] * kernels[1 * 72 + 5 * 9 + 0] + tc2[1] * kernels[1 * 72 + 5 * 9 + 1] + tr2[1] * kernels[1 * 72 + 5 * 9 + 2] +
            ml2[1] * kernels[1 * 72 + 5 * 9 + 3] + mc2[1] * kernels[1 * 72 + 5 * 9 + 4] + mr2[1] * kernels[1 * 72 + 5 * 9 + 5] +
            bl2[1] * kernels[1 * 72 + 5 * 9 + 6] + bc2[1] * kernels[1 * 72 + 5 * 9 + 7] + br2[1] * kernels[1 * 72 + 5 * 9 + 8];

        c7 =
            tl2[2] * kernels[1 * 72 + 6 * 9 + 0] + tc2[2] * kernels[1 * 72 + 6 * 9 + 1] + tr2[2] * kernels[1 * 72 + 6 * 9 + 2] +
            ml2[2] * kernels[1 * 72 + 6 * 9 + 3] + mc2[2] * kernels[1 * 72 + 6 * 9 + 4] + mr2[2] * kernels[1 * 72 + 6 * 9 + 5] +
            bl2[2] * kernels[1 * 72 + 6 * 9 + 6] + bc2[2] * kernels[1 * 72 + 6 * 9 + 7] + br2[2] * kernels[1 * 72 + 6 * 9 + 8];

        c8 =
            tl2[3] * kernels[1 * 72 + 7 * 9 + 0] + tc2[3] * kernels[1 * 72 + 7 * 9 + 1] + tr2[3] * kernels[1 * 72 + 7 * 9 + 2] +
            ml2[3] * kernels[1 * 72 + 7 * 9 + 3] + mc2[3] * kernels[1 * 72 + 7 * 9 + 4] + mr2[3] * kernels[1 * 72 + 7 * 9 + 5] +
            bl2[3] * kernels[1 * 72 + 7 * 9 + 6] + bc2[3] * kernels[1 * 72 + 7 * 9 + 7] + br2[3] * kernels[1 * 72 + 7 * 9 + 8];

        tmpMat1[1] = RULE(c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + biases[1]);

        c1 =
            tl1[0] * kernels[2 * 72 + 0 * 9 + 0] + tc1[0] * kernels[2 * 72 + 0 * 9 + 1] + tr1[0] * kernels[2 * 72 + 0 * 9 + 2] +
            ml1[0] * kernels[2 * 72 + 0 * 9 + 3] + mc1[0] * kernels[2 * 72 + 0 * 9 + 4] + mr1[0] * kernels[2 * 72 + 0 * 9 + 5] +
            bl1[0] * kernels[2 * 72 + 0 * 9 + 6] + bc1[0] * kernels[2 * 72 + 0 * 9 + 7] + br1[0] * kernels[2 * 72 + 0 * 9 + 8];

        c2 =
            tl1[1] * kernels[2 * 72 + 1 * 9 + 0] + tc1[1] * kernels[2 * 72 + 1 * 9 + 1] + tr1[1] * kernels[2 * 72 + 1 * 9 + 2] +
            ml1[1] * kernels[2 * 72 + 1 * 9 + 3] + mc1[1] * kernels[2 * 72 + 1 * 9 + 4] + mr1[1] * kernels[2 * 72 + 1 * 9 + 5] +
            bl1[1] * kernels[2 * 72 + 1 * 9 + 6] + bc1[1] * kernels[2 * 72 + 1 * 9 + 7] + br1[1] * kernels[2 * 72 + 1 * 9 + 8];

        c3 =
            tl1[2] * kernels[2 * 72 + 2 * 9 + 0] + tc1[2] * kernels[2 * 72 + 2 * 9 + 1] + tr1[2] * kernels[2 * 72 + 2 * 9 + 2] +
            ml1[2] * kernels[2 * 72 + 2 * 9 + 3] + mc1[2] * kernels[2 * 72 + 2 * 9 + 4] + mr1[2] * kernels[2 * 72 + 2 * 9 + 5] +
            bl1[2] * kernels[2 * 72 + 2 * 9 + 6] + bc1[2] * kernels[2 * 72 + 2 * 9 + 7] + br1[2] * kernels[2 * 72 + 2 * 9 + 8];

        c4 =
            tl1[3] * kernels[2 * 72 + 3 * 9 + 0] + tc1[3] * kernels[2 * 72 + 3 * 9 + 1] + tr1[3] * kernels[2 * 72 + 3 * 9 + 2] +
            ml1[3] * kernels[2 * 72 + 3 * 9 + 3] + mc1[3] * kernels[2 * 72 + 3 * 9 + 4] + mr1[3] * kernels[2 * 72 + 3 * 9 + 5] +
            bl1[3] * kernels[2 * 72 + 3 * 9 + 6] + bc1[3] * kernels[2 * 72 + 3 * 9 + 7] + br1[3] * kernels[2 * 72 + 3 * 9 + 8];

        c5 =
            tl2[0] * kernels[2 * 72 + 4 * 9 + 0] + tc2[0] * kernels[2 * 72 + 4 * 9 + 1] + tr2[0] * kernels[2 * 72 + 4 * 9 + 2] +
            ml2[0] * kernels[2 * 72 + 4 * 9 + 3] + mc2[0] * kernels[2 * 72 + 4 * 9 + 4] + mr2[0] * kernels[2 * 72 + 4 * 9 + 5] +
            bl2[0] * kernels[2 * 72 + 4 * 9 + 6] + bc2[0] * kernels[2 * 72 + 4 * 9 + 7] + br2[0] * kernels[2 * 72 + 4 * 9 + 8];

        c6 =
            tl2[1] * kernels[2 * 72 + 5 * 9 + 0] + tc2[1] * kernels[2 * 72 + 5 * 9 + 1] + tr2[1] * kernels[2 * 72 + 5 * 9 + 2] +
            ml2[1] * kernels[2 * 72 + 5 * 9 + 3] + mc2[1] * kernels[2 * 72 + 5 * 9 + 4] + mr2[1] * kernels[2 * 72 + 5 * 9 + 5] +
            bl2[1] * kernels[2 * 72 + 5 * 9 + 6] + bc2[1] * kernels[2 * 72 + 5 * 9 + 7] + br2[1] * kernels[2 * 72 + 5 * 9 + 8];

        c7 =
            tl2[2] * kernels[2 * 72 + 6 * 9 + 0] + tc2[2] * kernels[2 * 72 + 6 * 9 + 1] + tr2[2] * kernels[2 * 72 + 6 * 9 + 2] +
            ml2[2] * kernels[2 * 72 + 6 * 9 + 3] + mc2[2] * kernels[2 * 72 + 6 * 9 + 4] + mr2[2] * kernels[2 * 72 + 6 * 9 + 5] +
            bl2[2] * kernels[2 * 72 + 6 * 9 + 6] + bc2[2] * kernels[2 * 72 + 6 * 9 + 7] + br2[2] * kernels[2 * 72 + 6 * 9 + 8];

        c8 =
            tl2[3] * kernels[2 * 72 + 7 * 9 + 0] + tc2[3] * kernels[2 * 72 + 7 * 9 + 1] + tr2[3] * kernels[2 * 72 + 7 * 9 + 2] +
            ml2[3] * kernels[2 * 72 + 7 * 9 + 3] + mc2[3] * kernels[2 * 72 + 7 * 9 + 4] + mr2[3] * kernels[2 * 72 + 7 * 9 + 5] +
            bl2[3] * kernels[2 * 72 + 7 * 9 + 6] + bc2[3] * kernels[2 * 72 + 7 * 9 + 7] + br2[3] * kernels[2 * 72 + 7 * 9 + 8];

        tmpMat1[2] = RULE(c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + biases[2]);

        c1 =
            tl1[0] * kernels[3 * 72 + 0 * 9 + 0] + tc1[0] * kernels[3 * 72 + 0 * 9 + 1] + tr1[0] * kernels[3 * 72 + 0 * 9 + 2] +
            ml1[0] * kernels[3 * 72 + 0 * 9 + 3] + mc1[0] * kernels[3 * 72 + 0 * 9 + 4] + mr1[0] * kernels[3 * 72 + 0 * 9 + 5] +
            bl1[0] * kernels[3 * 72 + 0 * 9 + 6] + bc1[0] * kernels[3 * 72 + 0 * 9 + 7] + br1[0] * kernels[3 * 72 + 0 * 9 + 8];

        c2 =
            tl1[1] * kernels[3 * 72 + 1 * 9 + 0] + tc1[1] * kernels[3 * 72 + 1 * 9 + 1] + tr1[1] * kernels[3 * 72 + 1 * 9 + 2] +
            ml1[1] * kernels[3 * 72 + 1 * 9 + 3] + mc1[1] * kernels[3 * 72 + 1 * 9 + 4] + mr1[1] * kernels[3 * 72 + 1 * 9 + 5] +
            bl1[1] * kernels[3 * 72 + 1 * 9 + 6] + bc1[1] * kernels[3 * 72 + 1 * 9 + 7] + br1[1] * kernels[3 * 72 + 1 * 9 + 8];

        c3 =
            tl1[2] * kernels[3 * 72 + 2 * 9 + 0] + tc1[2] * kernels[3 * 72 + 2 * 9 + 1] + tr1[2] * kernels[3 * 72 + 2 * 9 + 2] +
            ml1[2] * kernels[3 * 72 + 2 * 9 + 3] + mc1[2] * kernels[3 * 72 + 2 * 9 + 4] + mr1[2] * kernels[3 * 72 + 2 * 9 + 5] +
            bl1[2] * kernels[3 * 72 + 2 * 9 + 6] + bc1[2] * kernels[3 * 72 + 2 * 9 + 7] + br1[2] * kernels[3 * 72 + 2 * 9 + 8];

        c4 =
            tl1[3] * kernels[3 * 72 + 3 * 9 + 0] + tc1[3] * kernels[3 * 72 + 3 * 9 + 1] + tr1[3] * kernels[3 * 72 + 3 * 9 + 2] +
            ml1[3] * kernels[3 * 72 + 3 * 9 + 3] + mc1[3] * kernels[3 * 72 + 3 * 9 + 4] + mr1[3] * kernels[3 * 72 + 3 * 9 + 5] +
            bl1[3] * kernels[3 * 72 + 3 * 9 + 6] + bc1[3] * kernels[3 * 72 + 3 * 9 + 7] + br1[3] * kernels[3 * 72 + 3 * 9 + 8];

        c5 =
            tl2[0] * kernels[3 * 72 + 4 * 9 + 0] + tc2[0] * kernels[3 * 72 + 4 * 9 + 1] + tr2[0] * kernels[3 * 72 + 4 * 9 + 2] +
            ml2[0] * kernels[3 * 72 + 4 * 9 + 3] + mc2[0] * kernels[3 * 72 + 4 * 9 + 4] + mr2[0] * kernels[3 * 72 + 4 * 9 + 5] +
            bl2[0] * kernels[3 * 72 + 4 * 9 + 6] + bc2[0] * kernels[3 * 72 + 4 * 9 + 7] + br2[0] * kernels[3 * 72 + 4 * 9 + 8];

        c6 =
            tl2[1] * kernels[3 * 72 + 5 * 9 + 0] + tc2[1] * kernels[3 * 72 + 5 * 9 + 1] + tr2[1] * kernels[3 * 72 + 5 * 9 + 2] +
            ml2[1] * kernels[3 * 72 + 5 * 9 + 3] + mc2[1] * kernels[3 * 72 + 5 * 9 + 4] + mr2[1] * kernels[3 * 72 + 5 * 9 + 5] +
            bl2[1] * kernels[3 * 72 + 5 * 9 + 6] + bc2[1] * kernels[3 * 72 + 5 * 9 + 7] + br2[1] * kernels[3 * 72 + 5 * 9 + 8];

        c7 =
            tl2[2] * kernels[3 * 72 + 6 * 9 + 0] + tc2[2] * kernels[3 * 72 + 6 * 9 + 1] + tr2[2] * kernels[3 * 72 + 6 * 9 + 2] +
            ml2[2] * kernels[3 * 72 + 6 * 9 + 3] + mc2[2] * kernels[3 * 72 + 6 * 9 + 4] + mr2[2] * kernels[3 * 72 + 6 * 9 + 5] +
            bl2[2] * kernels[3 * 72 + 6 * 9 + 6] + bc2[2] * kernels[3 * 72 + 6 * 9 + 7] + br2[2] * kernels[3 * 72 + 6 * 9 + 8];

        c8 =
            tl2[3] * kernels[3 * 72 + 7 * 9 + 0] + tc2[3] * kernels[3 * 72 + 7 * 9 + 1] + tr2[3] * kernels[3 * 72 + 7 * 9 + 2] +
            ml2[3] * kernels[3 * 72 + 7 * 9 + 3] + mc2[3] * kernels[3 * 72 + 7 * 9 + 4] + mr2[3] * kernels[3 * 72 + 7 * 9 + 5] +
            bl2[3] * kernels[3 * 72 + 7 * 9 + 6] + bc2[3] * kernels[3 * 72 + 7 * 9 + 7] + br2[3] * kernels[3 * 72 + 7 * 9 + 8];

        tmpMat1[3] = RULE(c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + biases[3]);

        c1 =
            tl1[0] * kernels[4 * 72 + 0 * 9 + 0] + tc1[0] * kernels[4 * 72 + 0 * 9 + 1] + tr1[0] * kernels[4 * 72 + 0 * 9 + 2] +
            ml1[0] * kernels[4 * 72 + 0 * 9 + 3] + mc1[0] * kernels[4 * 72 + 0 * 9 + 4] + mr1[0] * kernels[4 * 72 + 0 * 9 + 5] +
            bl1[0] * kernels[4 * 72 + 0 * 9 + 6] + bc1[0] * kernels[4 * 72 + 0 * 9 + 7] + br1[0] * kernels[4 * 72 + 0 * 9 + 8];

        c2 =
            tl1[1] * kernels[4 * 72 + 1 * 9 + 0] + tc1[1] * kernels[4 * 72 + 1 * 9 + 1] + tr1[1] * kernels[4 * 72 + 1 * 9 + 2] +
            ml1[1] * kernels[4 * 72 + 1 * 9 + 3] + mc1[1] * kernels[4 * 72 + 1 * 9 + 4] + mr1[1] * kernels[4 * 72 + 1 * 9 + 5] +
            bl1[1] * kernels[4 * 72 + 1 * 9 + 6] + bc1[1] * kernels[4 * 72 + 1 * 9 + 7] + br1[1] * kernels[4 * 72 + 1 * 9 + 8];

        c3 =
            tl1[2] * kernels[4 * 72 + 2 * 9 + 0] + tc1[2] * kernels[4 * 72 + 2 * 9 + 1] + tr1[2] * kernels[4 * 72 + 2 * 9 + 2] +
            ml1[2] * kernels[4 * 72 + 2 * 9 + 3] + mc1[2] * kernels[4 * 72 + 2 * 9 + 4] + mr1[2] * kernels[4 * 72 + 2 * 9 + 5] +
            bl1[2] * kernels[4 * 72 + 2 * 9 + 6] + bc1[2] * kernels[4 * 72 + 2 * 9 + 7] + br1[2] * kernels[4 * 72 + 2 * 9 + 8];

        c4 =
            tl1[3] * kernels[4 * 72 + 3 * 9 + 0] + tc1[3] * kernels[4 * 72 + 3 * 9 + 1] + tr1[3] * kernels[4 * 72 + 3 * 9 + 2] +
            ml1[3] * kernels[4 * 72 + 3 * 9 + 3] + mc1[3] * kernels[4 * 72 + 3 * 9 + 4] + mr1[3] * kernels[4 * 72 + 3 * 9 + 5] +
            bl1[3] * kernels[4 * 72 + 3 * 9 + 6] + bc1[3] * kernels[4 * 72 + 3 * 9 + 7] + br1[3] * kernels[4 * 72 + 3 * 9 + 8];

        c5 =
            tl2[0] * kernels[4 * 72 + 4 * 9 + 0] + tc2[0] * kernels[4 * 72 + 4 * 9 + 1] + tr2[0] * kernels[4 * 72 + 4 * 9 + 2] +
            ml2[0] * kernels[4 * 72 + 4 * 9 + 3] + mc2[0] * kernels[4 * 72 + 4 * 9 + 4] + mr2[0] * kernels[4 * 72 + 4 * 9 + 5] +
            bl2[0] * kernels[4 * 72 + 4 * 9 + 6] + bc2[0] * kernels[4 * 72 + 4 * 9 + 7] + br2[0] * kernels[4 * 72 + 4 * 9 + 8];

        c6 =
            tl2[1] * kernels[4 * 72 + 5 * 9 + 0] + tc2[1] * kernels[4 * 72 + 5 * 9 + 1] + tr2[1] * kernels[4 * 72 + 5 * 9 + 2] +
            ml2[1] * kernels[4 * 72 + 5 * 9 + 3] + mc2[1] * kernels[4 * 72 + 5 * 9 + 4] + mr2[1] * kernels[4 * 72 + 5 * 9 + 5] +
            bl2[1] * kernels[4 * 72 + 5 * 9 + 6] + bc2[1] * kernels[4 * 72 + 5 * 9 + 7] + br2[1] * kernels[4 * 72 + 5 * 9 + 8];

        c7 =
            tl2[2] * kernels[4 * 72 + 6 * 9 + 0] + tc2[2] * kernels[4 * 72 + 6 * 9 + 1] + tr2[2] * kernels[4 * 72 + 6 * 9 + 2] +
            ml2[2] * kernels[4 * 72 + 6 * 9 + 3] + mc2[2] * kernels[4 * 72 + 6 * 9 + 4] + mr2[2] * kernels[4 * 72 + 6 * 9 + 5] +
            bl2[2] * kernels[4 * 72 + 6 * 9 + 6] + bc2[2] * kernels[4 * 72 + 6 * 9 + 7] + br2[2] * kernels[4 * 72 + 6 * 9 + 8];

        c8 =
            tl2[3] * kernels[4 * 72 + 7 * 9 + 0] + tc2[3] * kernels[4 * 72 + 7 * 9 + 1] + tr2[3] * kernels[4 * 72 + 7 * 9 + 2] +
            ml2[3] * kernels[4 * 72 + 7 * 9 + 3] + mc2[3] * kernels[4 * 72 + 7 * 9 + 4] + mr2[3] * kernels[4 * 72 + 7 * 9 + 5] +
            bl2[3] * kernels[4 * 72 + 7 * 9 + 6] + bc2[3] * kernels[4 * 72 + 7 * 9 + 7] + br2[3] * kernels[4 * 72 + 7 * 9 + 8];

        tmpMat2[0] = RULE(c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + biases[4]);

        c1 =
            tl1[0] * kernels[5 * 72 + 0 * 9 + 0] + tc1[0] * kernels[5 * 72 + 0 * 9 + 1] + tr1[0] * kernels[5 * 72 + 0 * 9 + 2] +
            ml1[0] * kernels[5 * 72 + 0 * 9 + 3] + mc1[0] * kernels[5 * 72 + 0 * 9 + 4] + mr1[0] * kernels[5 * 72 + 0 * 9 + 5] +
            bl1[0] * kernels[5 * 72 + 0 * 9 + 6] + bc1[0] * kernels[5 * 72 + 0 * 9 + 7] + br1[0] * kernels[5 * 72 + 0 * 9 + 8];

        c2 =
            tl1[1] * kernels[5 * 72 + 1 * 9 + 0] + tc1[1] * kernels[5 * 72 + 1 * 9 + 1] + tr1[1] * kernels[5 * 72 + 1 * 9 + 2] +
            ml1[1] * kernels[5 * 72 + 1 * 9 + 3] + mc1[1] * kernels[5 * 72 + 1 * 9 + 4] + mr1[1] * kernels[5 * 72 + 1 * 9 + 5] +
            bl1[1] * kernels[5 * 72 + 1 * 9 + 6] + bc1[1] * kernels[5 * 72 + 1 * 9 + 7] + br1[1] * kernels[5 * 72 + 1 * 9 + 8];

        c3 =
            tl1[2] * kernels[5 * 72 + 2 * 9 + 0] + tc1[2] * kernels[5 * 72 + 2 * 9 + 1] + tr1[2] * kernels[5 * 72 + 2 * 9 + 2] +
            ml1[2] * kernels[5 * 72 + 2 * 9 + 3] + mc1[2] * kernels[5 * 72 + 2 * 9 + 4] + mr1[2] * kernels[5 * 72 + 2 * 9 + 5] +
            bl1[2] * kernels[5 * 72 + 2 * 9 + 6] + bc1[2] * kernels[5 * 72 + 2 * 9 + 7] + br1[2] * kernels[5 * 72 + 2 * 9 + 8];

        c4 =
            tl1[3] * kernels[5 * 72 + 3 * 9 + 0] + tc1[3] * kernels[5 * 72 + 3 * 9 + 1] + tr1[3] * kernels[5 * 72 + 3 * 9 + 2] +
            ml1[3] * kernels[5 * 72 + 3 * 9 + 3] + mc1[3] * kernels[5 * 72 + 3 * 9 + 4] + mr1[3] * kernels[5 * 72 + 3 * 9 + 5] +
            bl1[3] * kernels[5 * 72 + 3 * 9 + 6] + bc1[3] * kernels[5 * 72 + 3 * 9 + 7] + br1[3] * kernels[5 * 72 + 3 * 9 + 8];

        c5 =
            tl2[0] * kernels[5 * 72 + 4 * 9 + 0] + tc2[0] * kernels[5 * 72 + 4 * 9 + 1] + tr2[0] * kernels[5 * 72 + 4 * 9 + 2] +
            ml2[0] * kernels[5 * 72 + 4 * 9 + 3] + mc2[0] * kernels[5 * 72 + 4 * 9 + 4] + mr2[0] * kernels[5 * 72 + 4 * 9 + 5] +
            bl2[0] * kernels[5 * 72 + 4 * 9 + 6] + bc2[0] * kernels[5 * 72 + 4 * 9 + 7] + br2[0] * kernels[5 * 72 + 4 * 9 + 8];

        c6 =
            tl2[1] * kernels[5 * 72 + 5 * 9 + 0] + tc2[1] * kernels[5 * 72 + 5 * 9 + 1] + tr2[1] * kernels[5 * 72 + 5 * 9 + 2] +
            ml2[1] * kernels[5 * 72 + 5 * 9 + 3] + mc2[1] * kernels[5 * 72 + 5 * 9 + 4] + mr2[1] * kernels[5 * 72 + 5 * 9 + 5] +
            bl2[1] * kernels[5 * 72 + 5 * 9 + 6] + bc2[1] * kernels[5 * 72 + 5 * 9 + 7] + br2[1] * kernels[5 * 72 + 5 * 9 + 8];

        c7 =
            tl2[2] * kernels[5 * 72 + 6 * 9 + 0] + tc2[2] * kernels[5 * 72 + 6 * 9 + 1] + tr2[2] * kernels[5 * 72 + 6 * 9 + 2] +
            ml2[2] * kernels[5 * 72 + 6 * 9 + 3] + mc2[2] * kernels[5 * 72 + 6 * 9 + 4] + mr2[2] * kernels[5 * 72 + 6 * 9 + 5] +
            bl2[2] * kernels[5 * 72 + 6 * 9 + 6] + bc2[2] * kernels[5 * 72 + 6 * 9 + 7] + br2[2] * kernels[5 * 72 + 6 * 9 + 8];

        c8 =
            tl2[3] * kernels[5 * 72 + 7 * 9 + 0] + tc2[3] * kernels[5 * 72 + 7 * 9 + 1] + tr2[3] * kernels[5 * 72 + 7 * 9 + 2] +
            ml2[3] * kernels[5 * 72 + 7 * 9 + 3] + mc2[3] * kernels[5 * 72 + 7 * 9 + 4] + mr2[3] * kernels[5 * 72 + 7 * 9 + 5] +
            bl2[3] * kernels[5 * 72 + 7 * 9 + 6] + bc2[3] * kernels[5 * 72 + 7 * 9 + 7] + br2[3] * kernels[5 * 72 + 7 * 9 + 8];

        tmpMat2[1] = RULE(c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + biases[5]);

        c1 =
            tl1[0] * kernels[6 * 72 + 0 * 9 + 0] + tc1[0] * kernels[6 * 72 + 0 * 9 + 1] + tr1[0] * kernels[6 * 72 + 0 * 9 + 2] +
            ml1[0] * kernels[6 * 72 + 0 * 9 + 3] + mc1[0] * kernels[6 * 72 + 0 * 9 + 4] + mr1[0] * kernels[6 * 72 + 0 * 9 + 5] +
            bl1[0] * kernels[6 * 72 + 0 * 9 + 6] + bc1[0] * kernels[6 * 72 + 0 * 9 + 7] + br1[0] * kernels[6 * 72 + 0 * 9 + 8];

        c2 =
            tl1[1] * kernels[6 * 72 + 1 * 9 + 0] + tc1[1] * kernels[6 * 72 + 1 * 9 + 1] + tr1[1] * kernels[6 * 72 + 1 * 9 + 2] +
            ml1[1] * kernels[6 * 72 + 1 * 9 + 3] + mc1[1] * kernels[6 * 72 + 1 * 9 + 4] + mr1[1] * kernels[6 * 72 + 1 * 9 + 5] +
            bl1[1] * kernels[6 * 72 + 1 * 9 + 6] + bc1[1] * kernels[6 * 72 + 1 * 9 + 7] + br1[1] * kernels[6 * 72 + 1 * 9 + 8];

        c3 =
            tl1[2] * kernels[6 * 72 + 2 * 9 + 0] + tc1[2] * kernels[6 * 72 + 2 * 9 + 1] + tr1[2] * kernels[6 * 72 + 2 * 9 + 2] +
            ml1[2] * kernels[6 * 72 + 2 * 9 + 3] + mc1[2] * kernels[6 * 72 + 2 * 9 + 4] + mr1[2] * kernels[6 * 72 + 2 * 9 + 5] +
            bl1[2] * kernels[6 * 72 + 2 * 9 + 6] + bc1[2] * kernels[6 * 72 + 2 * 9 + 7] + br1[2] * kernels[6 * 72 + 2 * 9 + 8];

        c4 =
            tl1[3] * kernels[6 * 72 + 3 * 9 + 0] + tc1[3] * kernels[6 * 72 + 3 * 9 + 1] + tr1[3] * kernels[6 * 72 + 3 * 9 + 2] +
            ml1[3] * kernels[6 * 72 + 3 * 9 + 3] + mc1[3] * kernels[6 * 72 + 3 * 9 + 4] + mr1[3] * kernels[6 * 72 + 3 * 9 + 5] +
            bl1[3] * kernels[6 * 72 + 3 * 9 + 6] + bc1[3] * kernels[6 * 72 + 3 * 9 + 7] + br1[3] * kernels[6 * 72 + 3 * 9 + 8];

        c5 =
            tl2[0] * kernels[6 * 72 + 4 * 9 + 0] + tc2[0] * kernels[6 * 72 + 4 * 9 + 1] + tr2[0] * kernels[6 * 72 + 4 * 9 + 2] +
            ml2[0] * kernels[6 * 72 + 4 * 9 + 3] + mc2[0] * kernels[6 * 72 + 4 * 9 + 4] + mr2[0] * kernels[6 * 72 + 4 * 9 + 5] +
            bl2[0] * kernels[6 * 72 + 4 * 9 + 6] + bc2[0] * kernels[6 * 72 + 4 * 9 + 7] + br2[0] * kernels[6 * 72 + 4 * 9 + 8];

        c6 =
            tl2[1] * kernels[6 * 72 + 5 * 9 + 0] + tc2[1] * kernels[6 * 72 + 5 * 9 + 1] + tr2[1] * kernels[6 * 72 + 5 * 9 + 2] +
            ml2[1] * kernels[6 * 72 + 5 * 9 + 3] + mc2[1] * kernels[6 * 72 + 5 * 9 + 4] + mr2[1] * kernels[6 * 72 + 5 * 9 + 5] +
            bl2[1] * kernels[6 * 72 + 5 * 9 + 6] + bc2[1] * kernels[6 * 72 + 5 * 9 + 7] + br2[1] * kernels[6 * 72 + 5 * 9 + 8];

        c7 =
            tl2[2] * kernels[6 * 72 + 6 * 9 + 0] + tc2[2] * kernels[6 * 72 + 6 * 9 + 1] + tr2[2] * kernels[6 * 72 + 6 * 9 + 2] +
            ml2[2] * kernels[6 * 72 + 6 * 9 + 3] + mc2[2] * kernels[6 * 72 + 6 * 9 + 4] + mr2[2] * kernels[6 * 72 + 6 * 9 + 5] +
            bl2[2] * kernels[6 * 72 + 6 * 9 + 6] + bc2[2] * kernels[6 * 72 + 6 * 9 + 7] + br2[2] * kernels[6 * 72 + 6 * 9 + 8];

        c8 =
            tl2[3] * kernels[6 * 72 + 7 * 9 + 0] + tc2[3] * kernels[6 * 72 + 7 * 9 + 1] + tr2[3] * kernels[6 * 72 + 7 * 9 + 2] +
            ml2[3] * kernels[6 * 72 + 7 * 9 + 3] + mc2[3] * kernels[6 * 72 + 7 * 9 + 4] + mr2[3] * kernels[6 * 72 + 7 * 9 + 5] +
            bl2[3] * kernels[6 * 72 + 7 * 9 + 6] + bc2[3] * kernels[6 * 72 + 7 * 9 + 7] + br2[3] * kernels[6 * 72 + 7 * 9 + 8];

        tmpMat2[2] = RULE(c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + biases[6]);

        c1 =
            tl1[0] * kernels[7 * 72 + 0 * 9 + 0] + tc1[0] * kernels[7 * 72 + 0 * 9 + 1] + tr1[0] * kernels[7 * 72 + 0 * 9 + 2] +
            ml1[0] * kernels[7 * 72 + 0 * 9 + 3] + mc1[0] * kernels[7 * 72 + 0 * 9 + 4] + mr1[0] * kernels[7 * 72 + 0 * 9 + 5] +
            bl1[0] * kernels[7 * 72 + 0 * 9 + 6] + bc1[0] * kernels[7 * 72 + 0 * 9 + 7] + br1[0] * kernels[7 * 72 + 0 * 9 + 8];

        c2 =
            tl1[1] * kernels[7 * 72 + 1 * 9 + 0] + tc1[1] * kernels[7 * 72 + 1 * 9 + 1] + tr1[1] * kernels[7 * 72 + 1 * 9 + 2] +
            ml1[1] * kernels[7 * 72 + 1 * 9 + 3] + mc1[1] * kernels[7 * 72 + 1 * 9 + 4] + mr1[1] * kernels[7 * 72 + 1 * 9 + 5] +
            bl1[1] * kernels[7 * 72 + 1 * 9 + 6] + bc1[1] * kernels[7 * 72 + 1 * 9 + 7] + br1[1] * kernels[7 * 72 + 1 * 9 + 8];

        c3 =
            tl1[2] * kernels[7 * 72 + 2 * 9 + 0] + tc1[2] * kernels[7 * 72 + 2 * 9 + 1] + tr1[2] * kernels[7 * 72 + 2 * 9 + 2] +
            ml1[2] * kernels[7 * 72 + 2 * 9 + 3] + mc1[2] * kernels[7 * 72 + 2 * 9 + 4] + mr1[2] * kernels[7 * 72 + 2 * 9 + 5] +
            bl1[2] * kernels[7 * 72 + 2 * 9 + 6] + bc1[2] * kernels[7 * 72 + 2 * 9 + 7] + br1[2] * kernels[7 * 72 + 2 * 9 + 8];

        c4 =
            tl1[3] * kernels[7 * 72 + 3 * 9 + 0] + tc1[3] * kernels[7 * 72 + 3 * 9 + 1] + tr1[3] * kernels[7 * 72 + 3 * 9 + 2] +
            ml1[3] * kernels[7 * 72 + 3 * 9 + 3] + mc1[3] * kernels[7 * 72 + 3 * 9 + 4] + mr1[3] * kernels[7 * 72 + 3 * 9 + 5] +
            bl1[3] * kernels[7 * 72 + 3 * 9 + 6] + bc1[3] * kernels[7 * 72 + 3 * 9 + 7] + br1[3] * kernels[7 * 72 + 3 * 9 + 8];

        c5 =
            tl2[0] * kernels[7 * 72 + 4 * 9 + 0] + tc2[0] * kernels[7 * 72 + 4 * 9 + 1] + tr2[0] * kernels[7 * 72 + 4 * 9 + 2] +
            ml2[0] * kernels[7 * 72 + 4 * 9 + 3] + mc2[0] * kernels[7 * 72 + 4 * 9 + 4] + mr2[0] * kernels[7 * 72 + 4 * 9 + 5] +
            bl2[0] * kernels[7 * 72 + 4 * 9 + 6] + bc2[0] * kernels[7 * 72 + 4 * 9 + 7] + br2[0] * kernels[7 * 72 + 4 * 9 + 8];

        c6 =
            tl2[1] * kernels[7 * 72 + 5 * 9 + 0] + tc2[1] * kernels[7 * 72 + 5 * 9 + 1] + tr2[1] * kernels[7 * 72 + 5 * 9 + 2] +
            ml2[1] * kernels[7 * 72 + 5 * 9 + 3] + mc2[1] * kernels[7 * 72 + 5 * 9 + 4] + mr2[1] * kernels[7 * 72 + 5 * 9 + 5] +
            bl2[1] * kernels[7 * 72 + 5 * 9 + 6] + bc2[1] * kernels[7 * 72 + 5 * 9 + 7] + br2[1] * kernels[7 * 72 + 5 * 9 + 8];

        c7 =
            tl2[2] * kernels[7 * 72 + 6 * 9 + 0] + tc2[2] * kernels[7 * 72 + 6 * 9 + 1] + tr2[2] * kernels[7 * 72 + 6 * 9 + 2] +
            ml2[2] * kernels[7 * 72 + 6 * 9 + 3] + mc2[2] * kernels[7 * 72 + 6 * 9 + 4] + mr2[2] * kernels[7 * 72 + 6 * 9 + 5] +
            bl2[2] * kernels[7 * 72 + 6 * 9 + 6] + bc2[2] * kernels[7 * 72 + 6 * 9 + 7] + br2[2] * kernels[7 * 72 + 6 * 9 + 8];

        c8 =
            tl2[3] * kernels[7 * 72 + 7 * 9 + 0] + tc2[3] * kernels[7 * 72 + 7 * 9 + 1] + tr2[3] * kernels[7 * 72 + 7 * 9 + 2] +
            ml2[3] * kernels[7 * 72 + 7 * 9 + 3] + mc2[3] * kernels[7 * 72 + 7 * 9 + 4] + mr2[3] * kernels[7 * 72 + 7 * 9 + 5] +
            bl2[3] * kernels[7 * 72 + 7 * 9 + 6] + bc2[3] * kernels[7 * 72 + 7 * 9 + 7] + br2[3] * kernels[7 * 72 + 7 * 9 + 8];

        tmpMat2[3] = RULE(c1 + c2 + c3 + c4 + c5 + c6 + c7 + c8 + biases[7]);

        }, tmpMats);
}

void Anime4KCPP::CNNProcessor::convTranspose8To1(cv::Mat& img, const double* kernels,
    std::pair<cv::Mat, cv::Mat>& tmpMats)
{
    changEachPixel8To1(img, [&](const int i, const int j, PIXEL tmpMat, LineF tmpMat1, LineF tmpMat2) {
        int flag = 0;
        int r = i & 1;
        int c = j & 1;
        if (r != 0 && c == 0)
            flag = 0;
        //0 x
        //0 0
        else if (r == 0 && c == 0)
            flag = 1;
        //0 0
        //0 x
        else if (r == 0 && c != 0)
            flag = 2;
        //0 0
        //x 0

        else if (r != 0 && c != 0)
            flag = 3;
        //x 0
        //0 0

        //180 degree rotation for kernel
        //0 1  to  3 2
        //2 3      1 0

        double tmp = 0;
        switch (flag)
        {
        case 0:
            tmp = (
                tmpMat1[0] * kernels[0 * 4 + 2] +
                tmpMat1[1] * kernels[1 * 4 + 2] +
                tmpMat1[2] * kernels[2 * 4 + 2] +
                tmpMat1[3] * kernels[3 * 4 + 2] +
                tmpMat2[0] * kernels[4 * 4 + 2] +
                tmpMat2[1] * kernels[5 * 4 + 2] +
                tmpMat2[2] * kernels[6 * 4 + 2] +
                tmpMat2[3] * kernels[7 * 4 + 2]) * 255.0;
            *tmpMat = UNNORM(tmp);
            break;
        case 1:
            tmp = (
                tmpMat1[0] * kernels[0 * 4 + 0] +
                tmpMat1[1] * kernels[1 * 4 + 0] +
                tmpMat1[2] * kernels[2 * 4 + 0] +
                tmpMat1[3] * kernels[3 * 4 + 0] +
                tmpMat2[0] * kernels[4 * 4 + 0] +
                tmpMat2[1] * kernels[5 * 4 + 0] +
                tmpMat2[2] * kernels[6 * 4 + 0] +
                tmpMat2[3] * kernels[7 * 4 + 0]) * 255.0;
            *tmpMat = UNNORM(tmp);
            break;
        case 2:
            tmp = (
                tmpMat1[0] * kernels[0 * 4 + 1] +
                tmpMat1[1] * kernels[1 * 4 + 1] +
                tmpMat1[2] * kernels[2 * 4 + 1] +
                tmpMat1[3] * kernels[3 * 4 + 1] +
                tmpMat2[0] * kernels[4 * 4 + 1] +
                tmpMat2[1] * kernels[5 * 4 + 1] +
                tmpMat2[2] * kernels[6 * 4 + 1] +
                tmpMat2[3] * kernels[7 * 4 + 1]) * 255.0;
            *tmpMat = UNNORM(tmp);
            break;
        case 3:
            tmp = (
                tmpMat1[0] * kernels[0 * 4 + 3] +
                tmpMat1[1] * kernels[1 * 4 + 3] +
                tmpMat1[2] * kernels[2 * 4 + 3] +
                tmpMat1[3] * kernels[3 * 4 + 3] +
                tmpMat2[0] * kernels[4 * 4 + 3] +
                tmpMat2[1] * kernels[5 * 4 + 3] +
                tmpMat2[2] * kernels[6 * 4 + 3] +
                tmpMat2[3] * kernels[7 * 4 + 3]) * 255.0;
            *tmpMat = UNNORM(tmp);
            break;
        }

        }, tmpMats);
}

void Anime4KCPP::CNNProcessor::changEachPixel1To8(cv::InputArray _src,
    const std::function<void(int, int, Chan, Chan, LineC)>&& callBack,
    std::pair<cv::Mat, cv::Mat>& tmpMats)
{
    cv::Mat src = _src.getMat();
    tmpMats.first.create(src.size(), CV_64FC4);
    tmpMats.second.create(src.size(), CV_64FC4);

    const size_t srcChannels = src.channels();

    const int h = src.rows, w = src.cols;

    const int jMAX = w * 4;
#ifdef _MSC_VER
    Concurrency::parallel_for(0, h, [&](int i) {
        LineC lineData = src.data + static_cast<size_t>(i) * static_cast<size_t>(w) * srcChannels;
        LineF tmpLineData1 = reinterpret_cast<double*>(tmpMats.first.data) + static_cast<size_t>(i) * static_cast<size_t>(w) * static_cast<size_t>(4);
        LineF tmpLineData2 = reinterpret_cast<double*>(tmpMats.second.data) + static_cast<size_t>(i) * static_cast<size_t>(w) * static_cast<size_t>(4);
        for (int j = 0; j < jMAX; j += 4)
            callBack(i, j, tmpLineData1 + j, tmpLineData2 + j, lineData);
        });
#else
#pragma omp parallel for
    for (int i = 0; i < h; i++)
    {
        LineC lineData = src.data + static_cast<size_t>(i) * static_cast<size_t>(w) * srcChannels;
        LineF tmpLineData1 = reinterpret_cast<double*>(tmpMats.first.data) + static_cast<size_t>(i) * static_cast<size_t>(w) * static_cast<size_t>(4);
        LineF tmpLineData2 = reinterpret_cast<double*>(tmpMats.second.data) + static_cast<size_t>(i) * static_cast<size_t>(w) * static_cast<size_t>(4);
        for (int j = 0; j < jMAX; j += 4)
            callBack(i, j, tmpLineData1 + j, tmpLineData2 + j, lineData);
    }
#endif
}

void Anime4KCPP::CNNProcessor::changEachPixel8To8(
    const std::function<void(int, int, Chan, Chan, LineF, LineF)>&& callBack,
    std::pair<cv::Mat, cv::Mat>& tmpMats)
{
    cv::Mat tmp1, tmp2;
    tmp1.create(tmpMats.first.size(), tmpMats.first.type());
    tmp2.create(tmpMats.second.size(), tmpMats.second.type());

    const int h = tmpMats.first.rows, w = tmpMats.first.cols;

    const int jMAX = w * 4;
#ifdef _MSC_VER
    Concurrency::parallel_for(0, h, [&](int i) {
        LineF lineData1 = reinterpret_cast<double*>(tmpMats.first.data) + static_cast<size_t>(i) * static_cast<size_t>(w) * static_cast<size_t>(4);
        LineF lineData2 = reinterpret_cast<double*>(tmpMats.second.data) + static_cast<size_t>(i) * static_cast<size_t>(w) * static_cast<size_t>(4);
        LineF tmpLineData1 = reinterpret_cast<double*>(tmp1.data) + static_cast<size_t>(i) * static_cast<size_t>(w) * static_cast<size_t>(4);
        LineF tmpLineData2 = reinterpret_cast<double*>(tmp2.data) + static_cast<size_t>(i) * static_cast<size_t>(w) * static_cast<size_t>(4);
        for (int j = 0; j < jMAX; j += 4)
            callBack(i, j, tmpLineData1 + j, tmpLineData2 + j, lineData1, lineData2);
        });
#else
#pragma omp parallel for
    for (int i = 0; i < h; i++)
    {
        LineF lineData1 = reinterpret_cast<double*>(tmpMats.first.data) + static_cast<size_t>(i) * static_cast<size_t>(w) * static_cast<size_t>(4);
        LineF lineData2 = reinterpret_cast<double*>(tmpMats.second.data) + static_cast<size_t>(i) * static_cast<size_t>(w) * static_cast<size_t>(4);
        LineF tmpLineData1 = reinterpret_cast<double*>(tmp1.data) + static_cast<size_t>(i) * static_cast<size_t>(w) * static_cast<size_t>(4);
        LineF tmpLineData2 = reinterpret_cast<double*>(tmp2.data) + static_cast<size_t>(i) * static_cast<size_t>(w) * static_cast<size_t>(4);
        for (int j = 0; j < jMAX; j += 4)
            callBack(i, j, tmpLineData1 + j, tmpLineData2 + j, lineData1, lineData2);
    }
#endif

    tmp1.copyTo(tmpMats.first);
    tmp2.copyTo(tmpMats.second);
}

void Anime4KCPP::CNNProcessor::changEachPixel8To1(cv::Mat& img,
    const std::function<void(int, int, PIXEL, LineF, LineF)>&& callBack,
    std::pair<cv::Mat, cv::Mat>& tmpMats)
{
    cv::Mat tmp;
    const int h = 2 * tmpMats.first.rows, w = 2 * tmpMats.first.cols;
    tmp.create(h, w, CV_8UC1);

    const int jMAX = w;
#ifdef _MSC_VER
    Concurrency::parallel_for(0, h, [&](int i) {
        LineF lineData1 = reinterpret_cast<double*>(tmpMats.first.data) + static_cast<size_t>(i / 2) * static_cast<size_t>(w / 2) * static_cast<size_t>(4);
        LineF lineData2 = reinterpret_cast<double*>(tmpMats.second.data) + static_cast<size_t>(i / 2) * static_cast<size_t>(w / 2) * static_cast<size_t>(4);
        LineC tmpLineData = tmp.data + static_cast<size_t>(i) * static_cast<size_t>(w);
        for (int j = 0; j < jMAX; j++)
            callBack(i, j, tmpLineData + j, lineData1 + static_cast<size_t>((j / 2)) * static_cast<size_t>(4), lineData2 + static_cast<size_t>((j / 2)) * static_cast<size_t>(4)
            );
        });
#else
#pragma omp parallel for
    for (int i = 0; i < h; i++)
    {
        LineF lineData1 = reinterpret_cast<double*>(tmpMats.first.data) + static_cast<size_t>(i / 2) * static_cast<size_t>(w / 2) * static_cast<size_t>(4);
        LineF lineData2 = reinterpret_cast<double*>(tmpMats.second.data) + static_cast<size_t>(i / 2) * static_cast<size_t>(w / 2) * static_cast<size_t>(4);
        LineC tmpLineData = tmp.data + static_cast<size_t>(i) * static_cast<size_t>(w);
        for (int j = 0; j < jMAX; j++)
            callBack(i, j, tmpLineData + j, lineData1 + static_cast<size_t>((j / 2)) * static_cast<size_t>(4), lineData2 + static_cast<size_t>((j / 2)) * static_cast<size_t>(4)
            );
    }
#endif

    img = tmp;
}
