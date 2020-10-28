#define DLL

#include "Anime4KCPU.h"

Anime4KCPP::Anime4KCPU::Anime4KCPU(const Parameters& parameters) :
    Anime4K(parameters) {}

void Anime4KCPP::Anime4KCPU::process()
{
    if (!param.videoMode)
    {
        if (inputYUV)
        {
            cv::merge(std::vector<cv::Mat>{ orgY, orgU, orgV }, orgImg);
            cv::cvtColor(orgImg, orgImg, cv::COLOR_YUV2BGR);
        }
        int tmpPcc = param.pushColorCount;
        if (param.zoomFactor == 2.0F)
            cv::resize(orgImg, dstImg, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_LINEAR);
        else
            cv::resize(orgImg, dstImg, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
        if (param.preprocessing)
            FilterProcessor(dstImg, param.preFilters).process();
        cv::cvtColor(dstImg, dstImg, cv::COLOR_BGR2BGRA);
        for (int i = 0; i < param.passes; i++)
        {
            getGray(dstImg);
            if (param.strengthColor && (tmpPcc-- > 0))
                pushColor(dstImg);
            getGradient(dstImg);
            pushGradient(dstImg);
        }
        cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
        if (param.postprocessing)//PostProcessing
            FilterProcessor(dstImg, param.postFilters).process();
        if (inputYUV)
        {
            cv::cvtColor(dstImg, dstImg, cv::COLOR_BGR2YUV);
            std::vector<cv::Mat> yuv(3);
            cv::split(dstImg, yuv);
            dstY = yuv[Y];
            dstU = yuv[U];
            dstV = yuv[V];
        }
    }
    else
    {
        videoIO->init(
            [this]()
            {
                Frame frame = videoIO->read();
                cv::Mat orgFrame = frame.first;
                cv::Mat dstFrame(H, W, CV_8UC4);
                int tmpPcc = param.pushColorCount;
                if (param.preprocessing)
                    FilterProcessor(orgFrame, param.preFilters).process();
                cv::cvtColor(orgFrame, orgFrame, cv::COLOR_BGR2BGRA);
                if (param.zoomFactor == 2.0F)
                    cv::resize(orgFrame, dstFrame, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_LINEAR);
                else
                    cv::resize(orgFrame, dstFrame, cv::Size(0, 0), param.zoomFactor, param.zoomFactor, cv::INTER_CUBIC);
                for (int i = 0; i < param.passes; i++)
                {
                    getGray(dstFrame);
                    if (param.strengthColor && (tmpPcc-- > 0))
                        pushColor(dstFrame);
                    getGradient(dstFrame);
                    pushGradient(dstFrame);
                }
                cv::cvtColor(dstFrame, dstFrame, cv::COLOR_BGRA2BGR);
                if (param.postprocessing)//PostProcessing
                    FilterProcessor(dstFrame, param.postFilters).process();
                frame.first = dstFrame;
                videoIO->write(frame);
            }
            , param.maxThreads
                ).process();
    }
}

inline void Anime4KCPP::Anime4KCPU::getGray(cv::Mat& img)
{
    changEachPixelBGRA(img, [](const int i, const int j, PixelB pixel, LineB curLine) {
        pixel[A] = (pixel[R] >> 2) + (pixel[R] >> 4) + (pixel[G] >> 1) + (pixel[G] >> 4) + (pixel[B] >> 3);
        });
}

inline void Anime4KCPP::Anime4KCPU::pushColor(cv::Mat& img)
{
    const int lineStep = W * 4;
    changEachPixelBGRA(img, [&](const int i, const int j, PixelB pixel, LineB curLine) {
        const int jp = j < (W - 1) * 4 ? 4 : 0;
        const int jn = j > 4 ? -4 : 0;
        const LineB pLineData = i < H - 1 ? curLine + lineStep : curLine;
        const LineB cLineData = curLine;
        const LineB nLineData = i > 0 ? curLine - lineStep : curLine;

        const PixelB tl = nLineData + j + jn, tc = nLineData + j, tr = nLineData + j + jp;
        const PixelB ml = cLineData + j + jn, mc = pixel, mr = cLineData + j + jp;
        const PixelB bl = pLineData + j + jn, bc = pLineData + j, br = pLineData + j + jp;

        uint8_t maxD, minL;

        //top and bottom
        maxD = MAX3(bl[A], bc[A], br[A]);
        minL = MIN3(tl[A], tc[A], tr[A]);
        if (minL > mc[A] && mc[A] > maxD)
            getLightest(mc, tl, tc, tr);
        else
        {
            maxD = MAX3(tl[A], tc[A], tr[A]);
            minL = MIN3(bl[A], bc[A], br[A]);
            if (minL > mc[A] && mc[A] > maxD)
                getLightest(mc, bl, bc, br);
        }

        //sundiagonal
        maxD = MAX3(ml[A], mc[A], bc[A]);
        minL = MIN3(tc[A], tr[A], mr[A]);
        if (minL > maxD)
            getLightest(mc, tc, tr, mr);
        else
        {
            maxD = MAX3(tc[A], mc[A], mr[A]);
            minL = MIN3(ml[A], bl[A], bc[A]);
            if (minL > maxD)
                getLightest(mc, ml, bl, bc);
        }

        //left and right
        maxD = MAX3(tl[A], ml[A], bl[A]);
        minL = MIN3(tr[A], mr[A], br[A]);
        if (minL > mc[A] && mc[A] > maxD)
            getLightest(mc, tr, mr, br);
        else
        {
            maxD = MAX3(tr[A], mr[A], br[A]);
            minL = MIN3(tl[A], ml[A], bl[A]);
            if (minL > mc[A] && mc[A] > maxD)
                getLightest(mc, tl, ml, bl);
        }

        //diagonal
        maxD = MAX3(tc[A], mc[A], ml[A]);
        minL = MIN3(mr[A], br[A], bc[A]);
        if (minL > maxD)
            getLightest(mc, mr, br, bc);
        else
        {
            maxD = MAX3(bc[A], mc[A], mr[A]);
            minL = MIN3(ml[A], tl[A], tc[A]);
            if (minL > maxD)
                getLightest(mc, ml, tl, tc);
        }
        });
}

inline void Anime4KCPP::Anime4KCPU::getGradient(cv::Mat& img)
{
    if (!param.fastMode)
    {
        const int lineStep = W * 4;
        changEachPixelBGRA(img, [&](const int i, const int j, PixelB pixel, LineB curLine) {
            if (i == 0 || j == 0 || i == H - 1 || j == (W - 1) * 4)
                return;
            const LineB pLineData = curLine + lineStep;
            const LineB cLineData = curLine;
            const LineB nLineData = curLine - lineStep;
            const int jp = 4, jn = -4;

            int gradX =
                (pLineData + j + jn)[A] + (pLineData + j)[A] + (pLineData + j)[A] + (pLineData + j + jp)[A] -
                (nLineData + j + jn)[A] - (nLineData + j)[A] - (nLineData + j)[A] - (nLineData + j + jp)[A];
            int gradY =
                (nLineData + j + jn)[A] + (cLineData + j + jn)[A] + (cLineData + j + jn)[A] + (pLineData + j + jn)[A] -
                (nLineData + j + jp)[A] - (cLineData + j + jp)[A] - (cLineData + j + jp)[A] - (pLineData + j + jp)[A];
            double grad = sqrt(gradX * gradX + gradY * gradY);

            pixel[A] = 255 - UNFLOAT(grad);
            });
    }
    else
    {
        cv::Mat tmpGradX(H, W, CV_16SC1), tmpGradY(H, W, CV_16SC1);
        cv::Mat gradX(H, W, CV_8UC1), gradY(H, W, CV_8UC1), alpha(H, W, CV_8UC1);

        constexpr int fromTo_get[] = { A,0 };
        cv::mixChannels(img, alpha, fromTo_get, 1);

        cv::Sobel(alpha, tmpGradX, CV_16SC1, 1, 0);
        cv::Sobel(alpha, tmpGradY, CV_16SC1, 0, 1);
        cv::convertScaleAbs(tmpGradX, gradX);
        cv::convertScaleAbs(tmpGradY, gradY);
        cv::addWeighted(gradX, 0.5, gradY, 0.5, 0, alpha);

        constexpr int fromTo_set[] = { 0,A };
        cv::mixChannels(255 - alpha, img, fromTo_set, 1);
    }
}

inline void Anime4KCPP::Anime4KCPU::pushGradient(cv::Mat& img)
{
    const int lineStep = W * 4;
    changEachPixelBGRA(img, [&](const int i, const int j, PixelB pixel, LineB curLine) {
        const int jp = j < (W - 1) * 4 ? 4 : 0;
        const int jn = j > 4 ? -4 : 0;

        const LineB pLineData = i < H - 1 ? curLine + lineStep : curLine;
        const LineB cLineData = curLine;
        const LineB nLineData = i > 0 ? curLine - lineStep : curLine;

        const PixelB tl = nLineData + j + jn, tc = nLineData + j, tr = nLineData + j + jp;
        const PixelB ml = cLineData + j + jn, mc = pixel, mr = cLineData + j + jp;
        const PixelB bl = pLineData + j + jn, bc = pLineData + j, br = pLineData + j + jp;

        uint8_t maxD, minL;

        //top and bottom
        maxD = MAX3(bl[A], bc[A], br[A]);
        minL = MIN3(tl[A], tc[A], tr[A]);
        if (minL > mc[A] && mc[A] > maxD)
            return getAverage(mc, tl, tc, tr);

        maxD = MAX3(tl[A], tc[A], tr[A]);
        minL = MIN3(bl[A], bc[A], br[A]);
        if (minL > mc[A] && mc[A] > maxD)
            return getAverage(mc, bl, bc, br);

        //sundiagonal
        maxD = MAX3(ml[A], mc[A], bc[A]);
        minL = MIN3(tc[A], tr[A], mr[A]);
        if (minL > maxD)
            return getAverage(mc, tc, tr, mr);

        maxD = MAX3(tc[A], mc[A], mr[A]);
        minL = MIN3(ml[A], bl[A], bc[A]);
        if (minL > maxD)
            return getAverage(mc, ml, bl, bc);

        //left and right
        maxD = MAX3(tl[A], ml[A], bl[A]);
        minL = MIN3(tr[A], mr[A], br[A]);
        if (minL > mc[A] && mc[A] > maxD)
            return getAverage(mc, tr, mr, br);

        maxD = MAX3(tr[A], mr[A], br[A]);
        minL = MIN3(tl[A], ml[A], bl[A]);
        if (minL > mc[A] && mc[A] > maxD)
            return getAverage(mc, tl, ml, bl);

        //diagonal
        maxD = MAX3(tc[A], mc[A], ml[A]);
        minL = MIN3(mr[A], br[A], bc[A]);
        if (minL > maxD)
            return getAverage(mc, mr, br, bc);

        maxD = MAX3(bc[A], mc[A], mr[A]);
        minL = MIN3(ml[A], tl[A], tc[A]);
        if (minL > maxD)
            return getAverage(mc, ml, tl, tc);

        pixel[A] = 255;
        });
}

inline void Anime4KCPP::Anime4KCPU::changEachPixelBGRA(cv::Mat& src,
    const std::function<void(const int, const int, PixelB, LineB)>&& callBack)
{
    cv::Mat tmp;
    src.copyTo(tmp);

    const int jMAX = W * 4;
    const size_t step = jMAX;

#if defined(_MSC_VER) || defined(USE_TBB) //let's do something crazy
    Parallel::parallel_for(0, H, [&](int i) {
        LineB lineData = src.data + static_cast<size_t>(i) * step;
        LineB tmpLineData = tmp.data + static_cast<size_t>(i) * step;
        for (int j = 0; j < jMAX; j += 4)
            callBack(i, j, tmpLineData + j, lineData);
        });
#else //for gcc and others
#pragma omp parallel for
    for (int i = 0; i < H; i++)
    {
        LineB lineData = src.data + static_cast<size_t>(i) * step;
        LineB tmpLineData = tmp.data + static_cast<size_t>(i) * step;
        for (int j = 0; j < jMAX; j += 4)
            callBack(i, j, tmpLineData + j, lineData);
    }
#endif //something crazy

    src = tmp;
}

inline void Anime4KCPP::Anime4KCPU::getLightest(PixelB mc, const PixelB a, const PixelB b, const PixelB c) noexcept
{
    //RGBA
    for (int i = 0; i <= 3; i++)
        mc[i] = mc[i] * (1.0 - param.strengthColor) + ((a[i] + b[i] + c[i]) / 3.0) * param.strengthColor + 0.5;
}

inline void Anime4KCPP::Anime4KCPU::getAverage(PixelB mc, const PixelB a, const PixelB b, const PixelB c) noexcept
{
    //RGB
    for (int i = 0; i <= 2; i++)
        mc[i] = mc[i] * (1.0 - param.strengthGradient) + ((a[i] + b[i] + c[i]) / 3.0) * param.strengthGradient + 0.5;

    mc[A] = 255;
}

Anime4KCPP::ProcessorType Anime4KCPP::Anime4KCPU::getProcessorType() noexcept
{
    return ProcessorType::CPU_Anime4K09;
}
