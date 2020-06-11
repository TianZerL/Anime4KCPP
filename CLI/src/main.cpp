#include "Anime4KCPP.h"
#include <cmdline.h>

#include <iostream>
#include <filesystem>

#ifndef COMPILER
#define COMPILER "Unknown"
#endif // !COMPILER


bool checkFFmpeg()
{
    if (!system("ffmpeg -version"))
        return true;
    return false;
}

bool mergeAudio2Video(const std::string& dstFile, const std::string& srcFile, const std::string& tmpFile)
{
    std::string command("ffmpeg -loglevel 40 -i \"" + tmpFile + "\" -i \"" + srcFile + "\" -c copy -map 0:v -map 1 -map -1:v  -y \"" + dstFile + "\"");
    std::cout << command << std::endl;

    if (!system(command.data()))
        return true;
    return false;
}

Anime4KCPP::CODEC string2Codec(const std::string& codec)
{
    if (codec == "mp4v")
        return Anime4KCPP::CODEC::MP4V;
    else if (codec == "dxva")
        return Anime4KCPP::CODEC::DXVA;
    else if (codec == "avc1")
        return Anime4KCPP::CODEC::AVC1;
    else if (codec == "vp09")
        return Anime4KCPP::CODEC::VP09;
    else if (codec == "hevc")
        return Anime4KCPP::CODEC::HEVC;
    else if (codec == "av01")
        return Anime4KCPP::CODEC::AV01;
    else if (codec == "other")
        return Anime4KCPP::CODEC::OTHER;
    else
        return Anime4KCPP::CODEC::MP4V;
}

inline void showVersionInfo()
{
    std::cerr
        << "Anime4KCPPCLI" << std::endl
        << "Anime4KCPP core version: " << ANIME4KCPP_CORE_VERSION << std::endl
        << "Build date: " << __DATE__ << " " << __TIME__ << std::endl
        << "Compiler: " << COMPILER << std::endl
        << "GitHub: https://github.com/TianZerL/Anime4KCPP" << std::endl;
}

int main(int argc, char* argv[])
{
    //Options
    cmdline::parser opt;
    opt.add<std::string>("input", 'i', "File for loading", false, "./pic/p1.png");
    opt.add<std::string>("output", 'o', "File for outputting", false, "output.png");
    opt.add<int>("passes", 'p', "Passes for processing", false, 2);
    opt.add<int>("pushColorCount", 'n', "Limit the number of color pushes", false, 2);
    opt.add<float>("strengthColor", 'c', "Strength for pushing color,range 0 to 1,higher for thinner", false, 0.3F, cmdline::range(0.0F, 1.0F));
    opt.add<float>("strengthGradient", 'g', "Strength for pushing gradient,range 0 to 1,higher for sharper", false, 1.0F, cmdline::range(0.0F, 1.0F));
    opt.add<float>("zoomFactor", 'z', "zoom factor for resizing", false, 2.0F);
    opt.add<unsigned int>("threads", 't', "Threads count for video processing", false, std::thread::hardware_concurrency(), cmdline::range(1, int(32 * std::thread::hardware_concurrency())));
    opt.add("fastMode", 'f', "Faster but maybe low quality");
    opt.add("videoMode", 'v', "Video process");
    opt.add("preview", 's', "Preview image");
    opt.add("preprocessing", 'b', "Enable preprocessing");
    opt.add("postprocessing", 'a', "Enable postprocessing");
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
    opt.add("CNNMode", 'w', "Enable ACNet");
    opt.add("HDN", 'H', "Enable HDN mode for ACNet");
    opt.add("listGPUs", 'l', "list GPUs");
    opt.add<unsigned int>("platformID", 'h', "Specify the platform ID", false, 0);
    opt.add<unsigned int>("deviceID", 'd', "Specify the device ID", false, 0);
    opt.add<std::string>("codec", 'C', "Specify the codec for encoding from mp4v(recommended in Windows), dxva(for Windows), avc1(H264, recommended in Linux), vp09(very slow), \
hevc(not support in Windows), av01(not support in Windows)", false, "mp4v");
    opt.add("version", 'V', "print version information");

    opt.parse_check(argc, argv);

    std::string input = opt.get<std::string>("input");
    std::string output = opt.get<std::string>("output");
    int passes = opt.get<int>("passes");
    int pushColorCount = opt.get<int>("pushColorCount");
    float strengthColor = opt.get<float>("strengthColor");
    float strengthGradient = opt.get<float>("strengthGradient");
    float zoomFactor = opt.get<float>("zoomFactor");
    uint8_t preFilters = (uint8_t)opt.get<unsigned int>("preFilters");
    uint8_t postFilters = (uint8_t)opt.get<unsigned int>("postFilters");
    unsigned int threads = opt.get<unsigned int>("threads");
    bool fastMode = opt.exist("fastMode");
    bool videoMode = opt.exist("videoMode");
    bool preview = opt.exist("preview");
    bool preProcessing = opt.exist("preprocessing");
    bool postProcessing = opt.exist("postprocessing");
    bool GPU = opt.exist("GPUMode");
    bool CNN = opt.exist("CNNMode");
    bool HDN = opt.exist("HDN");
    bool listGPUs = opt.exist("listGPUs");
    unsigned int pID = opt.get<unsigned int>("platformID");
    unsigned int dID = opt.get<unsigned int>("deviceID");
    std::string codec = opt.get<std::string>("codec");
    bool version = opt.exist("version");

    if (version)
    {
        showVersionInfo();
        return 0;
    }

    if (listGPUs)
    {
        std::pair<std::pair<int, std::vector<int>>, std::string> ret = Anime4KCPP::Anime4KGPU::listGPUs();
        if (ret.first.first == 0)
            std::cout << "Error:" << std::endl;
        std::cout << ret.second << std::endl;
        return 0;
    }

    std::filesystem::path inputPath(input), outputPath(output);
    if (!std::filesystem::exists(inputPath))
    {
        std::cerr << "input file or directory does not exist." << std::endl;
        return 0;
    }
    Anime4KCPP::CNNType type;
    if (HDN)
        type = Anime4KCPP::CNNType::ACNetHDN;
    else
        type = Anime4KCPP::CNNType::ACNet;
    Anime4KCPP::Anime4KCreator creator(GPU, CNN, pID, dID, type);
    Anime4KCPP::Anime4K* anime4k = nullptr;
    Anime4KCPP::Parameters parameters(
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
        HDN
    );

    try
    {
        if (CNN)
        {
            if (GPU)
            {
                std::cout << "GPUCNN mode" << std::endl;
                std::pair<bool, std::string> ret = Anime4KCPP::Anime4KGPU::checkGPUSupport(pID, dID);
                if (!ret.first)
                {
                    std::cout << ret.second << std::endl;
                    return 0;
                }
                else
                {
                    std::cout << ret.second << std::endl;
                }
                anime4k = creator.create(parameters, Anime4KCPP::ProcessorType::GPUCNN);
            }
            else
            {
                std::cout << "CPUCNN mode" << std::endl;
                anime4k = creator.create(parameters, Anime4KCPP::ProcessorType::CPUCNN);
            }
        }
        else
        {
            if (GPU)
            {
                std::cout << "GPU mode" << std::endl;
                std::pair<bool, std::string> ret = Anime4KCPP::Anime4KGPU::checkGPUSupport(pID, dID);
                if (!ret.first)
                {
                    std::cout << ret.second << std::endl;
                    return 0;
                }
                else
                {
                    std::cout << ret.second << std::endl;
                }
                anime4k = creator.create(parameters, Anime4KCPP::ProcessorType::GPU);
            }
            else
            {
                std::cout << "CPU mode" << std::endl;
                anime4k = creator.create(parameters, Anime4KCPP::ProcessorType::CPU);
            }
        }

        if (!videoMode)//Image
        {
            if (std::filesystem::is_directory(inputPath))
            {
                if (!std::filesystem::is_directory(outputPath))
                    outputPath = outputPath.parent_path().append(outputPath.stem().native());
                std::filesystem::create_directories(outputPath);
                std::filesystem::directory_iterator currDir(inputPath);
                for (auto& file : currDir)
                {
                    if (file.is_directory())
                        continue;
                    std::string currInputPath = file.path().string();
                    std::string currOnputPath = (outputPath / (file.path().filename().string() + ".png")).string();
                    anime4k->loadImage(file.path().string());
                    anime4k->showInfo();
                    anime4k->showFiltersInfo();

                    std::cout << "Processing..." << std::endl;
                    std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();
                    anime4k->process();
                    std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
                    std::cout << "Total process time: " << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0 << " s" << std::endl;

                    anime4k->saveImage(currOnputPath);
                }
            }
            else
            {
                std::string currInputPath = inputPath.string();
                std::string currOnputPath = outputPath.string();

                anime4k->loadImage(currInputPath);
                anime4k->showInfo();
                anime4k->showFiltersInfo();

                std::cout << "Processing..." << std::endl;
                std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();
                anime4k->process();
                std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
                std::cout << "Total process time: " << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0 << " s" << std::endl;

                if (preview)
                    anime4k->showImage();

                anime4k->saveImage(currOnputPath);
            }
        }
        else//Video
        {
            //Suffix check
            outputPath.replace_extension(".mp4");

            bool ffmpeg = checkFFmpeg();
            std::string outputTmpName = output;

            if (!ffmpeg)
                std::cout << "Please install ffmpeg, otherwise the output file will be silent." << std::endl;
            else
                outputTmpName = "tmp_out.mp4";

            if (std::filesystem::is_directory(inputPath))
            {
                if (!std::filesystem::is_directory(outputPath))
                    outputPath = outputPath.parent_path().append(outputPath.stem().native());
                std::filesystem::create_directories(outputPath);
                std::filesystem::directory_iterator currDir(inputPath);
                for (auto& file : currDir)
                {
                    if (file.is_directory())
                        continue;
                    std::string currInputPath = file.path().string();
                    std::string currOnputPath = (outputPath / (file.path().filename().string() + ".mp4")).string();

                    anime4k->loadVideo(currInputPath);
                    anime4k->setVideoSaveInfo(outputTmpName, string2Codec(codec));

                    anime4k->showInfo();
                    anime4k->showFiltersInfo();

                    std::cout << "Processing..." << std::endl;
                    std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();
                    anime4k->process();
                    std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
                    std::cout << "Total process time: " << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0 / 60.0 << " min" << std::endl;

                    anime4k->saveVideo();

                    if (ffmpeg && mergeAudio2Video(currOnputPath, currInputPath, outputTmpName))
                        std::filesystem::remove(outputTmpName);
                }
            }
            else
            {
                std::string currInputPath = inputPath.string();
                std::string currOnputPath = outputPath.string();

                anime4k->loadVideo(currInputPath);
                anime4k->setVideoSaveInfo(outputTmpName, string2Codec(codec));

                anime4k->showInfo();
                anime4k->showFiltersInfo();

                std::cout << "Processing..." << std::endl;
                std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();
                anime4k->process();
                std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
                std::cout << "Total process time: " << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0 / 60.0 << " min" << std::endl;

                anime4k->saveVideo();

                if (ffmpeg && mergeAudio2Video(currOnputPath, currInputPath, outputTmpName))
                    std::filesystem::remove(outputTmpName);
            }
        }
    }
    catch (const char* err)
    {
        std::cout << err << std::endl;
    }

    creator.release(anime4k);

    return 0;
}
