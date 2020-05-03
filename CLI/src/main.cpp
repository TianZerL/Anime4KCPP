#include "Anime4K.h"
#include "Anime4KGPU.h"
#include<cmdline.h>

#include<iostream>


bool checkFFmpeg()
{
    if (!system("ffmpeg -version"))
        return true;
    return false;
}

bool mergrAudio2Video(const std::string& output, const std::string& srcFile)
{
    std::string command("ffmpeg -i \"tmp_out.mp4\" -i \"" + srcFile + "\" -c copy -map 0 -map 1:1 -y \"" + output + "\"");
    std::cout << command << std::endl;

    if (!system(command.data()))
        return true;
    return false;
}

CODEC string2Codec(const std::string& codec)
{
    if (codec == "mp4v")
        return MP4V;
    else if (codec == "dxva")
        return DXVA;
    else if (codec == "avc1")
        return AVC1;
    else if (codec == "vp09")
        return VP09;
    else if (codec == "hevc")
        return HEVC;
    else if (codec == "av01")
        return AV01;
    else if (codec == "other")
        return OTHER;
    else
        return MP4V;
}

int main(int argc, char* argv[])
{
    //Options
    cmdline::parser opt;
    opt.add<std::string>("input", 'i', "File for loading", false, "./pic/p1.png");
    opt.add<std::string>("output", 'o', "File for outputting", false, "output.png");
    opt.add<int>("passes", 'p', "Passes for processing", false, 2);
    opt.add<int>("pushColorCount", 'n', "Limit the number of color pushes", false, 2);
    opt.add<double>("strengthColor", 'c', "Strength for pushing color,range 0 to 1,higher for thinner", false, 0.3, cmdline::range(0.0, 1.0));
    opt.add<double>("strengthGradient", 'g', "Strength for pushing gradient,range 0 to 1,higher for sharper", false, 1.0, cmdline::range(0.0, 1.0));
    opt.add<double>("zoomFactor", 'z', "zoom factor for resizing", false, 2.0);
    opt.add<unsigned int>("threads", 't', "Threads count for video processing", false, std::thread::hardware_concurrency(), cmdline::range(1, int(32 * std::thread::hardware_concurrency())));
    opt.add("fastMode", 'f', "Faster but maybe low quality");
    opt.add("videoMode", 'v', "Video process");
    opt.add("preview", 's', "Preview image");
    opt.add("preProcessing", 'b', "Enable pre processing");
    opt.add("postProcessing", 'a', "Enable post processing");
    opt.add<unsigned int>("preFilters", 'r',
        "Enhancement filter, only working when preProcessing is true,there are 5 options by binary:\
Median blur=0000001, Mean blur=0000010, CAS Sharpening=0000100, Gaussian blur weak=0001000, Gaussian blur=0010000, Bilateral filter=0100000, Bilateral filter faster=1000000, \
you can freely combine them, eg: Gaussian blur weak + Bilateral filter = 0001000 | 0100000 = 0101000 = 40(D)",
false, 4, cmdline::range(1, 127));
    opt.add<unsigned int>("postFilters", 'e',
        "Enhancement filter, only working when postProcessing is true,there are 5 options by binary:\
Median blur=0000001, Mean blur=0000010, CAS Sharpening=0000100, Gaussian blur weak=0001000, Gaussian blur=0010000, Bilateral filter=0100000, Bilateral filter faster=1000000, \
you can freely combine them, eg: Gaussian blur weak + Bilateral filter = 0001000 | 0100000 = 0101000 = 40(D), \
so you can put 40 to enable Gaussian blur weak and Bilateral filter, which also is what I recommend for image that < 1080P, \
48 for image that >= 1080P, and for performance I recommend to use 72 for video that < 1080P, 80 for video that >=1080P",
false, 40, cmdline::range(1, 127));
    opt.add("GPUMode", 'q', "Enable GPU acceleration");
    opt.add("listGPUs", 'l', "list GPUs");
    opt.add<unsigned int>("platformID", 'h', "Specify the platform ID", false, 0);
    opt.add<unsigned int>("deviceID", 'd', "Specify the device ID", false, 0);
    opt.add<std::string>("codec", 'x', "Specify the codec for encoding from mp4v(recommended in Windows), dxva(for Windows), avc1(H264, recommended in Linux), vp09(very slow), \
hevc(not support in Windowds), av01(not support in Windowds)", false, "mp4v");

    opt.parse_check(argc, argv);

    std::string input = opt.get<std::string>("input");
    std::string output = opt.get<std::string>("output");
    int passes = opt.get<int>("passes");
    int pushColorCount = opt.get<int>("pushColorCount");
    double strengthColor = opt.get<double>("strengthColor");
    double strengthGradient = opt.get<double>("strengthGradient");
    double zoomFactor = opt.get<double>("zoomFactor");
    uint8_t preFilters = (uint8_t)opt.get<unsigned int>("preFilters");
    uint8_t postFilters = (uint8_t)opt.get<unsigned int>("postFilters");
    unsigned int threads = opt.get<unsigned int>("threads");
    bool fastMode = opt.exist("fastMode");
    bool videoMode = opt.exist("videoMode");
    bool preview = opt.exist("preview");
    bool preProcessing = opt.exist("preProcessing");
    bool postProcessing = opt.exist("postProcessing");
    bool GPU = opt.exist("GPUMode");
    bool listGPUs = opt.exist("listGPUs");
    unsigned int pID = opt.get<unsigned int>("platformID");
    unsigned int dID = opt.get<unsigned int>("deviceID");
    std::string codec = opt.get<std::string>("codec");


    if (listGPUs)
    {
        std::pair<std::pair<int, std::vector<int>>, std::string> ret = Anime4KGPU::listGPUs();
        if (ret.first.first == 0)
            std::cout << "Error:" << std::endl;
        std::cout << ret.second << std::endl;
        return 0;
    }

    Anime4K* anime4k;
    try
    {
        if (GPU)
        {
            std::cout << "GPU mode" << std::endl;
            std::pair<bool, std::string> ret = Anime4KGPU::checkGPUSupport(pID, dID);
            if (!ret.first)
            {
                std::cout << ret.second << std::endl;
                return 0;
            }
            else
            {
                std::cout << ret.second << std::endl;
            }
            anime4k = new Anime4KGPU(
                passes,
                pushColorCount,
                strengthColor,
                strengthGradient,
                zoomFactor,
                fastMode,
                videoMode,
                preProcessing,
                postProcessing,
                preFilters,
                postFilters,
                threads,
                pID,
                dID
            );
        }
        else
        {
            std::cout << "CPU mode" << std::endl;
            anime4k = new Anime4K(
                passes,
                pushColorCount,
                strengthColor,
                strengthGradient,
                zoomFactor,
                fastMode,
                videoMode,
                preProcessing,
                postProcessing,
                preFilters,
                postFilters,
                threads
            );
        }

        if (!videoMode)//Image
        {
            anime4k->loadImage(input);
            anime4k->showInfo();
            anime4k->showFiltersInfo();

            std::cout << "Processing..." << std::endl;
            std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();
            anime4k->process();
            std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
            std::cout << "Total process time: " << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0 << " s" << std::endl;

            if (preview)
                anime4k->showImage();
            anime4k->saveImage(output);
        }
        else//Video
        {
            //Suffix check
            if (output.substr(output.size() - 3) == "png")
                output.replace(output.size() - 3, 3, "mp4");

            bool ffmpeg = checkFFmpeg();
            std::string outputTmpName = output;

            if (!ffmpeg)
                std::cout << "Please install ffmpeg, otherwise the output file will be silent." << std::endl;
            else
                outputTmpName = "tmp_out.mp4";

            try
            {
                anime4k->loadVideo(input);
                anime4k->setVideoSaveInfo(outputTmpName, string2Codec(codec));
            }
            catch (const char* err)
            {
                std::cout << err << std::endl;
                return 0;
            }
            anime4k->showInfo();
            anime4k->showFiltersInfo();

            std::cout << "Processing..." << std::endl;
            std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();
            anime4k->process();
            std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
            std::cout << "Total process time: " << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0 / 60.0 << " min" << std::endl;

            anime4k->saveVideo();

            if (ffmpeg && mergrAudio2Video(output, input))
            {
#ifdef _WIN32
                std::string command("del /q " + outputTmpName);
#else
                std::string command("rm " + outputTmpName);
#endif // SYSTEM
                system(command.data());
            }
        }
    }
    catch (const char* err)
    {
        std::cout << err << std::endl;
        return 0;
}
    
    delete anime4k;
    
    return 0;
}
