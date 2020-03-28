#include "Anime4K.h"

Anime4K::Anime4K(int passes, double strengthColor, double strengthGradient, double zoomFactor, bool fastMode, bool videoMode, unsigned int maxThreads) :
    ps(passes), sc(strengthColor), sg(strengthGradient), zf(zoomFactor),
    fm(fastMode), vm(videoMode),
    mt(maxThreads)
{
    orgH = orgW = H = W = 0;
    frameCount = totalFrameCount = fps = 0;
}

void Anime4K::loadVideo(const std::string &dstFile)
{
    video.open(dstFile);
    if(!video.isOpened())
        throw "Fail to load file, file may not exist or decoder did'n been installed.";
    orgH = video.get(cv::CAP_PROP_FRAME_HEIGHT);
    orgW = video.get(cv::CAP_PROP_FRAME_WIDTH);
    fps = video.get(cv::CAP_PROP_FPS);
    totalFrameCount = video.get(cv::CAP_PROP_FRAME_COUNT);
    H = zf * orgH;
    W = zf * orgW;
}

void Anime4K::loadImage(const std::string &srcFile)
{
    orgImg = cv::imread(srcFile, cv::IMREAD_UNCHANGED);
    if (orgImg.empty())
        throw "Fail to load file, file may not exist.";
    orgH = orgImg.rows;
    orgW = orgImg.cols;

    cv::resize(orgImg, dstImg, cv::Size(0, 0), zf, zf, cv::INTER_CUBIC);
    if (dstImg.channels() == 3)
        cv::cvtColor(dstImg, dstImg, cv::COLOR_BGR2BGRA);
    H = dstImg.rows;
    W = dstImg.cols;
}

void Anime4K::setVideiSaveInfo(const std::string &dstFile)
{
    if(!videoWriter.open(dstFile, cv::VideoWriter::fourcc('a','v','c','1'), fps, cv::Size(W, H)))
        throw "Fail to initial video writer.";
}

void Anime4K::saveImage(const std::string &dstFile)
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
    if (vm)
    {
        std::cout << "Threads: " << mt << std::endl;
        std::cout << "Total frame: " << totalFrameCount << std::endl;
    }  
    std::cout << orgW << "x" << orgH << " to " << W << "x" << H << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
    std::cout << "Passes: " << ps << std::endl
        << "zoom Factor: " << zf << std::endl
        << "Video Mode: " << std::boolalpha << vm << std::endl
        << "Fast Mode: " << std::boolalpha << fm << std::endl
        << "Strength Color: " << sc << std::endl
        << "Strength Gradient: " << sg << std::endl;
    std::cout << "----------------------------------------------" << std::endl;
}

void Anime4K::showImg()
{
    cv::imshow("dstImg", dstImg);
    cv::waitKey();
}

void Anime4K::process()
{
    if (!vm)
    {
        for (int i = 0; i < ps; i++)
        {
            getGray(dstImg);
            pushColor(dstImg);
            getGradient(dstImg);
            pushGradient(dstImg);
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
                    std::this_thread::sleep_for(std::chrono::milliseconds(500));
                break;
            }
            
            pool.exec<std::function<void()>>([orgFrame = orgFrame.clone(), dstFrame = dstFrame.clone(), this, curFrame]()mutable
            {
                cv::resize(orgFrame, dstFrame, cv::Size(0, 0), zf, zf, cv::INTER_CUBIC);
                if (dstFrame.channels() == 3)
                    cv::cvtColor(dstFrame, dstFrame, cv::COLOR_BGR2BGRA);
                for (int i = 0; i < ps; i++)
                {
                    getGray(dstFrame);
                    pushColor(dstFrame);
                    getGradient(dstFrame);
                    pushGradient(dstFrame);
                }
                cv::cvtColor(dstFrame, dstFrame, cv::COLOR_BGRA2BGR);
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

void Anime4K::getGray(cv::InputArray img)
{
    changEachPixel(img, [&](int i, int j, RGBA pixel, Line curLine) {
        pixel[A] = 0.299 * pixel[R] + 0.587 * pixel[G] + 0.114 * pixel[B];
        });
}

void Anime4K::pushColor(cv::InputArray img)
{
    int lineStep = W * 4;
    changEachPixel(img, [&](int i, int j, RGBA pixel, Line curLine) {
        int jp = j < (W - 1) * 4 ? 4 : 0;;
        int jn = j > 4 ? -4 : 0;
        Line pLineData = i < H - 1? curLine + lineStep : curLine;
        Line cLineData = curLine;
        Line nLineData = i > 0 ? curLine - lineStep : curLine;

        RGBA tl = nLineData + j + jn, tc = nLineData + j, tr = nLineData + j + jp;
        RGBA ml = cLineData + j + jn, mc = pixel, mr = cLineData + j + jp;
        RGBA bl = pLineData + j + jn, bc = pLineData + j, br = pLineData + j + jp;

        uint8_t maxD, minL;

        //top and bottom
        maxD = max(bl[A], bc[A], br[A]);
        minL = min(tl[A], tc[A], tr[A]);
        if (minL > mc[A] && mc[A] > maxD)
            getLightest(mc, tl, tc, tr);
        else
        {
            maxD = max(tl[A], tc[A], tr[A]);
            minL = min(bl[A], bc[A], br[A]);
            if (minL > mc[A] && mc[A] > maxD)
                getLightest(mc, bl, bc, br);
        }

        //sundiagonal
        maxD = max(ml[A], mc[A], bc[A]);
        minL = min(tc[A], tr[A], mr[A]);
        if (minL > maxD)
            getLightest(mc, tc, tr, mr);
        else
        {
            maxD = max(tc[A], mc[A], mr[A]);
            minL = min(ml[A], bl[A], bc[A]);
            if (minL > maxD)
                getLightest(mc, ml, bl, bc);
        }

        //left and right
        maxD = max(tl[A], ml[A], bl[A]);
        minL = min(tr[A], mr[A], br[A]);
        if (minL > mc[A] && mc[A] > maxD)
            getLightest(mc, tr, mr, br);
        else
        {
            maxD = max(tr[A], mr[A], br[A]);
            minL = min(tl[A], ml[A], bl[A]);
            if (minL > mc[A] && mc[A] > maxD)
                getLightest(mc, tl, ml, bl);
        }

        //diagonal
        maxD = max(tc[A], mc[A], ml[A]);
        minL = min(mr[A], br[A], bc[A]);
        if (minL > maxD)
            getLightest(mc, mr, br, bc);
        else
        {
            maxD = max(bc[A], mc[A], mr[A]);
            minL = min(ml[A], tl[A], tc[A]);
            if (minL > maxD)
                getLightest(mc, ml, tl, tc);
        }
        });
}

void Anime4K::getGradient(cv::InputArray img)
{
    if (!fm) 
    {
        int lineStep = W * 4;
        changEachPixel(img, [&](int i, int j, RGBA pixel, Line curLine) {
            if (i == 0 || j == 0 || i == H - 1 || j == (W - 1) * 4)
                return;
            Line pLineData = curLine + lineStep;
            Line cLineData = curLine;
            Line nLineData = curLine - lineStep;
            int jp = 4, jn = -4;
            double GradX = 
                (pLineData + j + jn)[A] + (pLineData + j)[A] + (pLineData + j)[A] + (pLineData + j + jp)[A] -
                (nLineData + j + jn)[A] - (nLineData + j)[A] - (nLineData + j)[A] - (nLineData + j + jp)[A];
            double GradY = 
                (nLineData + j + jn)[A] + (cLineData + j + jn)[A] + (cLineData + j + jn)[A] + (pLineData + j + jn)[A] -
                (nLineData + j + jp)[A] - (cLineData + j + jp)[A] - (cLineData + j + jp)[A] - (pLineData + j + jp)[A];
            double Grad = sqrt(GradX * GradX + GradY * GradY);

            pixel[A] = 255 - unFloat(Grad);
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

void Anime4K::pushGradient(cv::InputArray img)
{
    int lineStep = W*4;
    changEachPixel(img, [&](int i, int j, RGBA pixel, Line curLine) {
        int jp = j < (W - 1) * 4 ? 4 : 0;;
        int jn = j > 4 ? -4 : 0;

        Line pLineData = i < H - 1 ? curLine + lineStep : curLine;
        Line cLineData = curLine;
        Line nLineData = i > 0 ? curLine - lineStep : curLine;

        RGBA tl = nLineData + j + jn, tc = nLineData + j, tr = nLineData + j + jp;
        RGBA ml = cLineData + j + jn, mc = pixel, mr = cLineData + j + jp;
        RGBA bl = pLineData + j + jn, bc = pLineData + j, br = pLineData + j + jp;

        uint8_t maxD, minL;

        //top and bottom
        maxD = max(bl[A], bc[A], br[A]);
        minL = min(tl[A], tc[A], tr[A]);
        if (minL > mc[A] && mc[A] > maxD)
            return getAverage(mc, tl, tc, tr);

        maxD = max(tl[A], tc[A], tr[A]);
        minL = min(bl[A], bc[A], br[A]);
        if (minL > mc[A] && mc[A] > maxD)
            return getAverage(mc, bl, bc, br);

        //sundiagonal
        maxD = max(ml[A], mc[A], bc[A]);
        minL = min(tc[A], tr[A], mr[A]);
        if (minL > maxD)
            return getAverage(mc, tc, tr, mr);

        maxD = max(tc[A], mc[A], mr[A]);
        minL = min(ml[A], bl[A], bc[A]);
        if (minL > maxD)
            return getAverage(mc, ml, bl, bc);
 

        //left and right
        maxD = max(tl[A], ml[A], bl[A]);
        minL = min(tr[A], mr[A], br[A]);
        if (minL > mc[A] && mc[A] > maxD)
            return getAverage(mc, tr, mr, br);
  
        maxD = max(tr[A], mr[A], br[A]);
        minL = min(tl[A], ml[A], bl[A]);
        if (minL > mc[A] && mc[A] > maxD)
            return getAverage(mc, tl, ml, bl);
 

        //diagonal
        maxD = max(tc[A], mc[A], ml[A]);
        minL = min(mr[A], br[A], bc[A]);
        if (minL > maxD)
            return getAverage(mc, mr, br, bc);

        maxD = max(bc[A], mc[A], mr[A]);
        minL = min(ml[A], tl[A], tc[A]);
        if (minL > maxD)
            return getAverage(mc, ml, tl, tc);

        pixel[A] = 255;
        });
}

void Anime4K::changEachPixel(cv::InputArray _src,
    const std::function<void(int, int, RGBA, Line)>&& callBack)
{
    cv::Mat src = _src.getMat();
    cv::Mat tmp;
    src.copyTo(tmp);

    Line lineData,tmpLineData;
    int jMAX = W * 4;

    for (int i = 0; i < H; i++)
    {
        lineData = src.data + i * W*4;
        tmpLineData = tmp.data + i * W*4;
        for (int j = 0; j < jMAX; j += 4)
            callBack(i, j, tmpLineData + j, lineData);
    } 

    tmp.copyTo(src);
}

void Anime4K::getLightest(RGBA mc, RGBA a, RGBA b, RGBA c)
{
    mc[R] = mc[R] * (1 - sc) + ((a[R] + b[R] + c[R]) / 3.0) * sc;
    mc[G] = mc[G] * (1 - sc) + ((a[G] + b[G] + c[G]) / 3.0) * sc;
    mc[B] = mc[B] * (1 - sc) + ((a[B] + b[B] + c[B]) / 3.0) * sc;
    mc[A] = mc[A] * (1 - sc) + ((a[A] + b[A] + c[A]) / 3.0) * sc;
}

void Anime4K::getAverage(RGBA mc, RGBA a, RGBA b, RGBA c)
{
    mc[R] = mc[R] * (1 - sg) + ((a[R] + b[R] + c[R]) / 3.0) * sg;
    mc[G] = mc[G] * (1 - sg) + ((a[G] + b[G] + c[G]) / 3.0) * sg;
    mc[B] = mc[B] * (1 - sg) + ((a[B] + b[B] + c[B]) / 3.0) * sg;
    mc[A] = 255;
}

uint8_t Anime4K::max(uint8_t a, uint8_t b, uint8_t c)
{
    return a > b && a > c ? a : (b > c ? b : c);
}

uint8_t Anime4K::min(uint8_t a, uint8_t b, uint8_t c)
{
    return a < b && a < c ? a : (b < c ? b : c);
}

uint8_t Anime4K::unFloat(double n)
{
    n += 0.5;
    if (n >= 255)
        return 255;
    else if (n <= 0)
        return 0;
    return uint8_t(n);
}
