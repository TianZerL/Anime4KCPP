#include "Anime4K.h"

Anime4K::Anime4K(
    int passes,
    int pushColorCount,
    double strengthColor,
    double strengthGradient,
    double zoomFactor,
    bool fastMode,
    bool videoMode,
    bool PreProcessing,
    bool postProcessing,
    uint8_t preFilters,
    uint8_t postFilters,
    unsigned int maxThreads
) :
    ps(passes), pcc(pushColorCount),
    sc(strengthColor), sg(strengthGradient),
    zf(zoomFactor), fm(fastMode), vm(videoMode), 
    pre(PreProcessing), post(postProcessing), pref(preFilters), 
    postf(postFilters), mt(maxThreads)
{
    orgH = orgW = H = W = 0;
    frameCount = totalFrameCount = fps = 0;
}

void Anime4K::loadVideo(const std::string& dstFile)
{
    video.open(dstFile);
    if (!video.isOpened())
        throw "Fail to load file, file may not exist or decoder did'n been installed.";
    orgH = video.get(cv::CAP_PROP_FRAME_HEIGHT);
    orgW = video.get(cv::CAP_PROP_FRAME_WIDTH);
    fps = video.get(cv::CAP_PROP_FPS);
    totalFrameCount = video.get(cv::CAP_PROP_FRAME_COUNT);
    H = zf * orgH;
    W = zf * orgW;
}

void Anime4K::loadImage(const std::string& srcFile)
{
    orgImg = cv::imread(srcFile, cv::IMREAD_UNCHANGED);
    if (orgImg.empty())
        throw "Fail to load file, file may not exist.";
    orgH = orgImg.rows;
    orgW = orgImg.cols;
    H = zf * orgH;
    W = zf * orgW;
}

void Anime4K::setVideoSaveInfo(const std::string& dstFile)
{
    if (!videoWriter.open(dstFile, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, cv::Size(W, H)))
        throw "Fail to initial video writer.";
}

void Anime4K::saveImage(const std::string& dstFile)
{
    cv::imwrite(dstFile, dstImg);
}

void Anime4K::saveVideo()
{
    videoWriter.release();
    video.release();
}

void Anime4K::showInfo()
{
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "Welcome to use Anime4KCPP" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    if (vm)
    {
        std::cout << "Threads: " << mt << std::endl;
        std::cout << "Total frame: " << totalFrameCount << std::endl;
    }
    std::cout << orgW << "x" << orgH << " to " << W << "x" << H << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "Passes: " << ps << std::endl
        << "pushColorCount: " << pcc << std::endl
        << "Zoom Factor: " << zf << std::endl
        << "Video Mode: " << std::boolalpha << vm << std::endl
        << "Fast Mode: " << std::boolalpha << fm << std::endl
        << "Strength Color: " << sc << std::endl
        << "Strength Gradient: " << sg << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
}

void Anime4K::showFiltersInfo()
{
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "Pre processing filters list:" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    if (!pre)
    {
        std::cout << "Pre processing disable" << std::endl;
    }
    else
    {
        if (pref & MEDIAN_BLUR)
            std::cout << "Median blur" << std::endl;
        if (pref & MEAN_BLUR)
            std::cout << "Mean blur" << std::endl;
        if (pref & CAS_SHARPENING)
            std::cout << "CAS Sharpening" << std::endl;
        if (pref & GAUSSIAN_BLUR_WEAK)
            std::cout << "Gaussian blur weak" << std::endl;
        else if (pref & GAUSSIAN_BLUR)
            std::cout << "Gaussian blur" << std::endl;
        if (pref & BILATERAL_FILTER)
            std::cout << "Bilateral filter" << std::endl;
        else if (pref & BILATERAL_FILTER_FAST)
            std::cout << "Bilateral filter faster" << std::endl;
    }
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "Post processing filters list:" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    if (!post)
    {
        std::cout << "Post processing disable" << std::endl;
    }
    else
    {
        if (postf & MEDIAN_BLUR)
            std::cout << "Median blur" << std::endl;
        if (postf & MEAN_BLUR)
            std::cout << "Mean blur" << std::endl;
        if (postf & CAS_SHARPENING)
            std::cout << "CAS Sharpening" << std::endl;
        if (postf & GAUSSIAN_BLUR_WEAK)
            std::cout << "Gaussian blur weak" << std::endl;
        else if (postf & GAUSSIAN_BLUR)
            std::cout << "Gaussian blur" << std::endl;
        if (postf & BILATERAL_FILTER)
            std::cout << "Bilateral filter" << std::endl;
        else if (postf & BILATERAL_FILTER_FAST)
            std::cout << "Bilateral filter faster" << std::endl;
    }
    std::cout << "----------------------------------------------" << std::endl;
}

void Anime4K::showImage()
{
    cv::imshow("dstImg", dstImg);
    cv::waitKey();
}

void Anime4K::process()
{
    if (!vm)
    {
        cv::resize(orgImg, dstImg, cv::Size(0, 0), zf, zf, cv::INTER_CUBIC);
        if (pre)
        {
            if (dstImg.channels() == 4)
                cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
            FilterProcessor(dstImg, pref).process();
        }
        if (dstImg.channels() == 3)
            cv::cvtColor(dstImg, dstImg, cv::COLOR_BGR2BGRA);
        for (int i = 0; i < ps; i++)
        {
            getGray(dstImg);
            if (sc && (pcc-- > 0))
                pushColor(dstImg);
            getGradient(dstImg);
            pushGradient(dstImg);
        }
        if (post)//PostProcessing
        {
            cv::cvtColor(dstImg, dstImg, cv::COLOR_BGRA2BGR);
            FilterProcessor(dstImg, postf).process();
        }
    }
    else
    {
        cv::Mat orgFrame, dstFrame;
        ThreadPool pool(mt);
        size_t curFrame = 0;
        while (true)
        {
            curFrame = video.get(cv::CAP_PROP_POS_FRAMES);
            if (!video.read(orgFrame))
            {
                while (frameCount < totalFrameCount)
                    std::this_thread::yield();
                break;
            }

            pool.exec<std::function<void()>>([orgFrame = orgFrame.clone(), dstFrame = dstFrame.clone(), this, curFrame, pcc = this->pcc]()mutable
            {
                cv::resize(orgFrame, dstFrame, cv::Size(0, 0), zf, zf, cv::INTER_CUBIC);
                if (pre)
                {
                    if (dstFrame.channels() == 4)
                        cv::cvtColor(dstFrame, dstFrame, cv::COLOR_BGRA2BGR);
                    FilterProcessor(dstFrame, pref).process();
                }
                if (dstFrame.channels() == 3)
                    cv::cvtColor(dstFrame, dstFrame, cv::COLOR_BGR2BGRA);
                for (int i = 0; i < ps; i++)
                {
                    getGray(dstFrame);
                    if (sc && (pcc-- > 0))
                        pushColor(dstFrame);
                    getGradient(dstFrame);
                    pushGradient(dstFrame);
                }
                cv::cvtColor(dstFrame, dstFrame, cv::COLOR_BGRA2BGR);
                if (post)//PostProcessing
                    FilterProcessor(dstFrame, postf).process();
                std::unique_lock<std::mutex> lock(videoMtx);
                while (true)
                {
                    if (curFrame == frameCount)
                    {
                        videoWriter.write(dstFrame);
                        frameCount++;
                        break;
                    }
                    else
                    {
                        cnd.wait(lock);
                    }
                }
                cnd.notify_all();
            });
        }
    }
}

inline void Anime4K::getGray(cv::InputArray img)
{
    changEachPixelBGRA(img, [&](int i, int j, RGBA pixel, Line curLine) {
        pixel[A] = 0.299 * pixel[R] + 0.587 * pixel[G] + 0.114 * pixel[B];
        });
}

inline void Anime4K::pushColor(cv::InputArray img)
{
    int lineStep = W * 4;
    changEachPixelBGRA(img, [&](int i, int j, RGBA pixel, Line curLine) {
        int jp = j < (W - 1) * 4 ? 4 : 0;
        int jn = j > 4 ? -4 : 0;
        Line pLineData = i < H - 1 ? curLine + lineStep : curLine;
        Line cLineData = curLine;
        Line nLineData = i > 0 ? curLine - lineStep : curLine;

        RGBA tl = nLineData + j + jn, tc = nLineData + j, tr = nLineData + j + jp;
        RGBA ml = cLineData + j + jn, mc = pixel, mr = cLineData + j + jp;
        RGBA bl = pLineData + j + jn, bc = pLineData + j, br = pLineData + j + jp;

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

inline void Anime4K::getGradient(cv::InputArray img)
{
    if (!fm)
    {
        int lineStep = W * 4;
        changEachPixelBGRA(img, [&](int i, int j, RGBA pixel, Line curLine) {
            if (i == 0 || j == 0 || i == H - 1 || j == (W - 1) * 4)
                return;
            Line pLineData = curLine + lineStep;
            Line cLineData = curLine;
            Line nLineData = curLine - lineStep;
            int jp = 4, jn = -4;
            float GradX =
                (pLineData + j + jn)[A] + (pLineData + j)[A] + (pLineData + j)[A] + (pLineData + j + jp)[A] -
                (nLineData + j + jn)[A] - (nLineData + j)[A] - (nLineData + j)[A] - (nLineData + j + jp)[A];
            float GradY =
                (nLineData + j + jn)[A] + (cLineData + j + jn)[A] + (cLineData + j + jn)[A] + (pLineData + j + jn)[A] -
                (nLineData + j + jp)[A] - (cLineData + j + jp)[A] - (cLineData + j + jp)[A] - (pLineData + j + jp)[A];
            float Grad = sqrt(GradX * GradX + GradY * GradY);

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

inline void Anime4K::pushGradient(cv::InputArray img)
{
    int lineStep = W * 4;
    changEachPixelBGRA(img, [&](int i, int j, RGBA pixel, Line curLine) {
        int jp = j < (W - 1) * 4 ? 4 : 0;
        int jn = j > 4 ? -4 : 0;

        Line pLineData = i < H - 1 ? curLine + lineStep : curLine;
        Line cLineData = curLine;
        Line nLineData = i > 0 ? curLine - lineStep : curLine;

        RGBA tl = nLineData + j + jn, tc = nLineData + j, tr = nLineData + j + jp;
        RGBA ml = cLineData + j + jn, mc = pixel, mr = cLineData + j + jp;
        RGBA bl = pLineData + j + jn, bc = pLineData + j, br = pLineData + j + jp;

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

inline void Anime4K::changEachPixelBGRA(cv::InputArray _src,
    const std::function<void(int, int, RGBA, Line)>&& callBack)
{
    cv::Mat src = _src.getMat();
    cv::Mat tmp;
    src.copyTo(tmp);

    int jMAX = W * 4;
#ifdef _MSC_VER //let's do something crazy
    Concurrency::parallel_for(0, H, [&](int i) {
        Line lineData = src.data + i * W * 4;
        Line tmpLineData = tmp.data + i * W * 4;
        for (int j = 0; j < jMAX; j += 4)
            callBack(i, j, tmpLineData + j, lineData);
        });
#else //for gcc and others
#pragma omp parallel for
    for (int i = 0; i < H; i++)
    {
        Line lineData = src.data + i * W * 4;
        Line tmpLineData = tmp.data + i * W * 4;
        for (int j = 0; j < jMAX; j += 4)
            callBack(i, j, tmpLineData + j, lineData);
    }
#endif //something crazy

    tmp.copyTo(src);
}

inline void Anime4K::getLightest(RGBA mc, RGBA a, RGBA b, RGBA c)
{
    mc[R] = mc[R] * (1 - sc) + ((a[R] + b[R] + c[R]) / 3.0) * sc;
    mc[G] = mc[G] * (1 - sc) + ((a[G] + b[G] + c[G]) / 3.0) * sc;
    mc[B] = mc[B] * (1 - sc) + ((a[B] + b[B] + c[B]) / 3.0) * sc;
    mc[A] = mc[A] * (1 - sc) + ((a[A] + b[A] + c[A]) / 3.0) * sc;
}

inline void Anime4K::getAverage(RGBA mc, RGBA a, RGBA b, RGBA c)
{
    mc[R] = mc[R] * (1 - sg) + ((a[R] + b[R] + c[R]) / 3.0) * sg;
    mc[G] = mc[G] * (1 - sg) + ((a[G] + b[G] + c[G]) / 3.0) * sg;
    mc[B] = mc[B] * (1 - sg) + ((a[B] + b[B] + c[B]) / 3.0) * sg;
    mc[A] = 255;
}
