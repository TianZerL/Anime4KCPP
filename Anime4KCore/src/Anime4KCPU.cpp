#define DLL

#include "Anime4KCPU.h"

Anime4KCPP::Anime4KCPU::Anime4KCPU(const Parameters& parameters) : 
    Anime4K(parameters) {}

void Anime4KCPP::Anime4KCPU::process()
{
    if (!vm)
    {
        int tmpPcc = this->pcc;
        if (zf == 2.0F)
            cv::resize(orgImg, dstImg, cv::Size(0, 0), zf, zf, cv::INTER_LINEAR);
        else
            cv::resize(orgImg, dstImg, cv::Size(0, 0), zf, zf, cv::INTER_CUBIC);
        if (pre)
            FilterProcessor(dstImg, pref).process();
        cv::cvtColor(dstImg, dstImg, cv::COLOR_BGR2BGRA);
        for (int i = 0; i < ps; i++)
        {
            getGray(dstImg);
            if (sc && (tmpPcc-- > 0))
                pushColor(dstImg);
            getGradient(dstImg);
            pushGradient(dstImg);
        }
        cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
        if (post)//PostProcessing
            FilterProcessor(dstImg, postf).process();
    }
    else
    {
        uint64_t count = mt;
        cv::Mat orgFrame, dstFrame;
        ThreadPool pool(mt);
        uint64_t curFrame = 0,doneFrameCount = 0;
        frameCount = 0;
        while (true)
        {
            curFrame = video.get(cv::CAP_PROP_POS_FRAMES);
            if (!video.read(orgFrame))
            {
                while (frameCount < totalFrameCount)
                    std::this_thread::yield();
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                break;
            }

            pool.exec<std::function<void()>>([orgFrame = orgFrame.clone(), dstFrame = dstFrame.clone(), this, curFrame, tmpPcc = this->pcc]()mutable
            {
                if (zf == 2.0)
                    cv::resize(orgFrame, dstFrame, cv::Size(0, 0), zf, zf, cv::INTER_LINEAR);
                else
                    cv::resize(orgFrame, dstFrame, cv::Size(0, 0), zf, zf, cv::INTER_CUBIC);
                if (pre)
                    FilterProcessor(dstFrame, pref).process();
                cv::cvtColor(dstFrame, dstFrame, cv::COLOR_BGR2BGRA);
                for (int i = 0; i < ps; i++)
                {
                    getGray(dstFrame);
                    if (sc && (tmpPcc-- > 0))
                        pushColor(dstFrame);
                    getGradient(dstFrame);
                    pushGradient(dstFrame);
                }
                cv::cvtColor(dstFrame, dstFrame, cv::COLOR_BGRA2BGR);
                if (post)//PostProcessing
                    FilterProcessor(dstFrame, postf).process();
                {
                    std::unique_lock<std::mutex> lock(videoMtx);
                    while (true)
                    {
                        if (curFrame == frameCount)
                        {
                            videoWriter.write(dstFrame);
                            dstFrame.release();
                            frameCount++;
                            break;
                        }
                        else
                        {
                            cnd.wait(lock);
                        }
                    }
                }
                cnd.notify_all();
            });
            //limit RAM usage
            if (!(--count))
            {
                while (frameCount == doneFrameCount)
                    std::this_thread::yield();
                count = frameCount - doneFrameCount;
                doneFrameCount = frameCount;
            }
        }
    }
}

inline void Anime4KCPP::Anime4KCPU::getGray(cv::InputArray img)
{
    changEachPixelBGRA(img, [](const int i, const int j, RGBA pixel, Line curLine) {
        pixel[A] = (pixel[R] >> 2) + (pixel[R] >> 4) + (pixel[G] >> 1) + (pixel[G] >> 4) + (pixel[B] >> 3);
        });
}

inline void Anime4KCPP::Anime4KCPU::pushColor(cv::InputArray img)
{
    const int lineStep = W * 4;
    changEachPixelBGRA(img, [&](const int i, const int j, RGBA pixel, Line curLine) {
        const int jp = j < (W - 1) * 4 ? 4 : 0;
        const int jn = j > 4 ? -4 : 0;
        const Line pLineData = i < H - 1 ? curLine + lineStep : curLine;
        const Line cLineData = curLine;
        const Line nLineData = i > 0 ? curLine - lineStep : curLine;

        const RGBA tl = nLineData + j + jn, tc = nLineData + j, tr = nLineData + j + jp;
        const RGBA ml = cLineData + j + jn, mc = pixel, mr = cLineData + j + jp;
        const RGBA bl = pLineData + j + jn, bc = pLineData + j, br = pLineData + j + jp;

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

inline void Anime4KCPP::Anime4KCPU::getGradient(cv::InputArray img)
{
    if (!fm)
    {
        const int lineStep = W * 4;
        changEachPixelBGRA(img, [&](const int i, const int j, RGBA pixel, Line curLine) {
            if (i == 0 || j == 0 || i == H - 1 || j == (W - 1) * 4)
                return;
            const Line pLineData = curLine + lineStep;
            const Line cLineData = curLine;
            const Line nLineData = curLine - lineStep;
            const int jp = 4, jn = -4;

            int gradX =
                (pLineData + j + jn)[A] + (pLineData + j)[A] + (pLineData + j)[A] + (pLineData + j + jp)[A] -
                (nLineData + j + jn)[A] - (nLineData + j)[A] - (nLineData + j)[A] - (nLineData + j + jp)[A];
            int gradY =
                (nLineData + j + jn)[A] + (cLineData + j + jn)[A] + (cLineData + j + jn)[A] + (pLineData + j + jn)[A] -
                (nLineData + j + jp)[A] - (cLineData + j + jp)[A] - (cLineData + j + jp)[A] - (pLineData + j + jp)[A];
            float Grad = sqrt(gradX * gradX + gradY * gradY);

            pixel[A] = 255 - UNFLOAT(Grad);
            });
    }
    else
    {
        cv::Mat tmpGradX(H, W, CV_16SC1), tmpGradY(H, W, CV_16SC1);
        cv::Mat gradX(H, W, CV_8UC1), gradY(H, W, CV_8UC1), alpha(H, W, CV_8UC1);

        int fromTo_get[] = { A,0 };
        cv::mixChannels(img, alpha, fromTo_get, 1);

        cv::Sobel(alpha, tmpGradX, CV_16SC1, 1, 0);
        cv::Sobel(alpha, tmpGradY, CV_16SC1, 0, 1);
        cv::convertScaleAbs(tmpGradX, gradX);
        cv::convertScaleAbs(tmpGradY, gradY);
        cv::addWeighted(gradX, 0.5, gradY, 0.5, 0, alpha);

        int fromTo_set[] = { 0,A };
        cv::mixChannels(255 - alpha, img.getMat(), fromTo_set, 1);
    }
}

inline void Anime4KCPP::Anime4KCPU::pushGradient(cv::InputArray img)
{
    const int lineStep = W * 4;
    changEachPixelBGRA(img, [&](const int i, const int j, RGBA pixel, Line curLine) {
        const int jp = j < (W - 1) * 4 ? 4 : 0;
        const int jn = j > 4 ? -4 : 0;

        const Line pLineData = i < H - 1 ? curLine + lineStep : curLine;
        const Line cLineData = curLine;
        const Line nLineData = i > 0 ? curLine - lineStep : curLine;

        const RGBA tl = nLineData + j + jn, tc = nLineData + j, tr = nLineData + j + jp;
        const RGBA ml = cLineData + j + jn, mc = pixel, mr = cLineData + j + jp;
        const RGBA bl = pLineData + j + jn, bc = pLineData + j, br = pLineData + j + jp;

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

inline void Anime4KCPP::Anime4KCPU::changEachPixelBGRA(cv::InputArray _src,
    const std::function<void(const int, const int, RGBA, Line)>&& callBack)
{
    cv::Mat src = _src.getMat();
    cv::Mat tmp;
    src.copyTo(tmp);

    int jMAX = W * 4;
#ifdef _MSC_VER //let's do something crazy
    Concurrency::parallel_for(0, H, [&](int i) {
        Line lineData = src.data + static_cast<size_t>(i) * static_cast<size_t>(W) * static_cast<size_t>(4);
        Line tmpLineData = tmp.data + static_cast<size_t>(i) * static_cast<size_t>(W) * static_cast<size_t>(4);
        for (int j = 0; j < jMAX; j += 4)
            callBack(i, j, tmpLineData + j, lineData);
        });
#else //for gcc and others
#pragma omp parallel for
    for (int i = 0; i < H; i++)
    {
        Line lineData = src.data + static_cast<size_t>(i) * static_cast<size_t>(W) * static_cast<size_t>(4);
        Line tmpLineData = tmp.data + static_cast<size_t>(i) * static_cast<size_t>(W) * static_cast<size_t>(4);
        for (int j = 0; j < jMAX; j += 4)
            callBack(i, j, tmpLineData + j, lineData);
    }
#endif //something crazy

    tmp.copyTo(src);
}

inline void Anime4KCPP::Anime4KCPU::getLightest(RGBA mc, const RGBA a, const RGBA b, const RGBA c)
{
    //RGBA
    for (int i = 0; i <= 3; i++)
        mc[i] = mc[i] * (1 - sc) + (static_cast<float>(a[i] + b[i] + c[i]) / 3.0F) * sc + 0.5F;
}

inline void Anime4KCPP::Anime4KCPU::getAverage(RGBA mc, const RGBA a, const RGBA b, const RGBA c)
{
    //RGB
    for (int i = 0; i <= 2; i++)
        mc[i] = mc[i] * (1 - sg) + (static_cast<float>(a[i] + b[i] + c[i]) / 3.0F) * sg + 0.5F;

    mc[A] = 255;
}
