#include <iostream>

#ifdef USE_BOOST_FILESYSTEM
#include<boost/filesystem.hpp>
namespace filesystem = boost::filesystem;
#else
#include <filesystem>
namespace filesystem = std::filesystem;
#endif // USE_BOOST_FILESYSTEM

#include "Anime4KCPP.hpp"

#include "Config.hpp"

#ifndef COMPILER
#define COMPILER "Unknown"
#endif // !COMPILER

enum class GPGPU
{
    OpenCL, CUDA
};

static bool checkFFmpeg()
{
    std::cout << "Checking ffmpeg..." << std::endl;
    if (!system("ffmpeg -version"))
        return true;
    return false;
}

static bool mergeAudio2Video(const std::string& dstFile, const std::string& srcFile, const std::string& tmpFile)
{
    std::cout << "Merging audio..." << std::endl;
    std::string command("ffmpeg -loglevel 40 -i \"" + tmpFile + "\" -i \"" + srcFile + "\" -c copy -map 0:v -map 1 -map -1:v  -y \"" + dstFile + "\"");
    std::cout << command << std::endl;

    return !system(command.data());
}

static bool video2GIF(const std::string& srcFile, const std::string& dstFile)
{
    std::string commandGeneratePalette("ffmpeg -i \"" + srcFile + "\" -vf palettegen -y palette.png");
    std::cout << commandGeneratePalette << std::endl;

    std::string command2Gif("ffmpeg -i \"" + srcFile + "\" -i palette.png -y -lavfi paletteuse \"" + dstFile + "\"");
    std::cout << command2Gif << std::endl;

    bool flag = !system(commandGeneratePalette.data()) && !system(command2Gif.data());
    filesystem::remove("palette.png");
    return flag;
}

static Anime4KCPP::CODEC string2Codec(const std::string& codec)
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

inline static void showVersionInfo()
{
    std::cerr
        << "Anime4KCPPCLI" << std::endl
        << "Anime4KCPP core version: " << ANIME4KCPP_CORE_VERSION << std::endl
        << "Parallel library: " << PARALLEL_LIBRARY << std::endl
        << "Build date: " << __DATE__ << " " << __TIME__ << std::endl
        << "Compiler: " << COMPILER << std::endl
        << "GitHub: https://github.com/TianZerL/Anime4KCPP" << std::endl;
}

static bool genConfigTemplate(const std::string& path, Config& config)
{
    ConfigWriter configWriter;
    if (!configWriter.initFile(path))
        return false;
    configWriter.set(config).write("Anime4KCPP_CLI config file");
    return true;
}

int main(int argc, char* argv[])
{
    Config config;
    auto& opt = config.getParser();
    //Options
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
    opt.add<unsigned int>("start", 'S', "Specify the start frame number for video previewing", false, 0);
    opt.add("preprocessing", 'b', "Enable preprocessing");
    opt.add("postprocessing", 'a', "Enable postprocessing");
    opt.add<unsigned int>("preFilters", 'r',
        "Enhancement filter, only working when preprocessing is true,there are 5 options by binary:"
        "Median blur=0000001, Mean blur=0000010, CAS Sharpening=0000100, Gaussian blur weak=0001000, Gaussian blur=0010000, Bilateral filter=0100000, Bilateral filter faster=1000000, "
        "you can freely combine them, eg: Gaussian blur weak + Bilateral filter = 0001000 | 0100000 = 0101000 = 40(D)",
        false, 4, cmdline::range(1, 127));
    opt.add<unsigned int>("postFilters", 'e',
        "Enhancement filter, only working when postprocessing is true,there are 5 options by binary:"
        "Median blur=0000001, Mean blur=0000010, CAS Sharpening=0000100, Gaussian blur weak=0001000, Gaussian blur=0010000, Bilateral filter=0100000, Bilateral filter faster=1000000, "
        "you can freely combine them, eg: Gaussian blur weak + Bilateral filter = 0001000 | 0100000 = 0101000 = 40(D), "
        "so you can put 40 to enable Gaussian blur weak and Bilateral filter, which also is what I recommend for image that < 1080P, "
        "48 for image that >= 1080P, and for performance I recommend to use 72 for video that < 1080P, 80 for video that >=1080P",
        false, 40, cmdline::range(1, 127));
    opt.add("GPUMode", 'q', "Enable GPU acceleration");
    opt.add("CNNMode", 'w', "Enable ACNet");
    opt.add("HDN", 'H', "Enable HDN mode for ACNet");
    opt.add<int>("HDNLevel", 'L', "Set HDN level", false, 1, cmdline::range(1, 3));
    opt.add("listGPUs", 'l', "list GPUs");
    opt.add<unsigned int>("platformID", 'h', "Specify the platform ID", false, 0);
    opt.add<unsigned int>("deviceID", 'd', "Specify the device ID", false, 0);
    opt.add<int>("OpenCLQueueNumber", 'Q', "Specify the number of command queues for OpenCL, this may affect performance of video processing for OpenCL", false, 1);
    opt.add("OpenCLParallelIO", 'P', "Use a parallel IO command queue for OpenCL, this may affect performance for OpenCL");
    opt.add<std::string>("codec", 'C', "Specify the codec for encoding from mp4v(recommended in Windows), dxva(for Windows), avc1(H264, recommended in Linux), vp09(very slow), "
        "hevc(not support in Windows), av01(not support in Windows)", false, "mp4v",
        cmdline::oneof<std::string>("mp4v", "dxva", "avc1", "vp09", "hevc", "av01", "other"));
    opt.add("version", 'V', "print version information");
    opt.add<double>("forceFps", 'F', "Set output video fps to the specifying number, 0 to disable", false, 0.0);
    opt.add("disableProgress", 'D', "disable progress display");
    opt.add("webVideo", 'W', "process the video from URL");
    opt.add("alpha", 'A', "preserve the Alpha channel for transparent image");
    opt.add("benchmark", 'B', "do benchmarking");
    opt.add<std::string>("GPGPUModel", 'M', "Specify the GPGPU model for processing", false, "opencl");
    opt.set_program_name("Anime4KCPP_CLI");
    opt.add<std::string>("configTemplate", '\000', "Generate config template", false, "./config");

    opt.parse_check(argc, argv);

    if (!opt.rest().empty() && !config.initFile(opt.rest().front()).parseFile())
    {
        config.getLastError().printErrorInfo();
        return 0;
    }

    std::string input = opt.get<std::string>("input");
    std::string output = opt.get<std::string>("output");
    bool videoMode = opt.exist("videoMode");
    bool preview = opt.exist("preview");
    bool listGPUs = opt.exist("listGPUs");
    bool version = opt.exist("version");
    bool webVideo = opt.exist("webVideo");
    bool doBenchmark = opt.exist("benchmark");
    unsigned int frameStart = opt.get<unsigned int>("start");

    int passes = config.get<int>("passes");
    int pushColorCount = config.get<int>("pushColorCount");
    int HDNLevel = config.get<int>("HDNLevel");
    double strengthColor = config.get<double>("strengthColor");
    double strengthGradient = config.get<double>("strengthGradient");
    double zoomFactor = config.get<double>("zoomFactor");
    double forceFps = config.get<double>("forceFps");
    bool fastMode = config.get<bool>("fastMode");
    bool preprocessing = config.get<bool>("preprocessing");
    bool postprocessing = config.get<bool>("postprocessing");
    bool GPU = config.get<bool>("GPUMode");
    bool CNN = config.get<bool>("CNNMode");
    bool HDN = config.get<bool>("HDN");
    bool disableProgress = config.get<bool>("disableProgress");
    bool alpha = config.get<bool>("alpha");
    uint8_t preFilters = config.get<unsigned int>("preFilters");
    uint8_t postFilters = config.get<unsigned int>("postFilters");
    unsigned int pID = config.get<unsigned int>("platformID");
    unsigned int dID = config.get<unsigned int>("deviceID");
    int OpenCLQueueNum = config.get<int>("OpenCLQueueNumber");
    bool OpenCLParallelIO = config.get<bool>("OpenCLParallelIO");
    unsigned int threads = config.get<unsigned int>("threads");
    std::string codec = config.get<std::string>("codec");
    std::string GPGPUModelString = config.get<std::string>("GPGPUModel");

    if (opt.exist("configTemplate"))
    {
        const std::string& path = opt.get<std::string>("configTemplate");
        if (genConfigTemplate(path, config))
            std::cout << "Generated config template to: " << path << std::endl;
        else
            std::cerr << "Failed to generate config template." << path << std::endl;
        return 0;
    }

    GPGPU GPGPUModel;
    std::transform(GPGPUModelString.begin(), GPGPUModelString.end(), GPGPUModelString.begin(), ::tolower);
    if (GPGPUModelString == "opencl")
        GPGPUModel = GPGPU::OpenCL;
    else if (GPGPUModelString == "cuda")
        GPGPUModel = GPGPU::CUDA;
    else
    {
        std::cerr << "Unknown GPGPU model, it must be \"cuda\" or \"opencl\"" << std::endl;
        return 0;
    }

    // -V
    if (version)
    {
        showVersionInfo();
        return 0;
    }
    // -l
    if (listGPUs)
    {
        std::cout << "OpenCL:" << std::endl;
        Anime4KCPP::OpenCL::GPUList OpenCLGPUList = Anime4KCPP::OpenCL::listGPUs();
        if (OpenCLGPUList.platforms == 0)
            std::cerr << "Error: No OpenCL GPU found" << std::endl;
        else
            std::cout << OpenCLGPUList() << std::endl;

#ifdef ENABLE_CUDA
        std::cout << "Cuda:" << std::endl;
        Anime4KCPP::Cuda::GPUList CUDAGPUList = Anime4KCPP::Cuda::listGPUs();
        if (CUDAGPUList.devices == 0)
            std::cerr << "Error: No CUDA GPU found" << std::endl;
        else
            std::cout << CUDAGPUList() << std::endl;
#endif
        return 0;
    }
    // -b
    if (doBenchmark)
    {
        std::cout << "Benchmarking..." << std::endl;

        double CPUScore = Anime4KCPP::benchmark<Anime4KCPP::CPU::ACNet>();
        double OpenCLScore = Anime4KCPP::benchmark<Anime4KCPP::OpenCL::ACNet>(pID, dID);
#ifdef ENABLE_CUDA
        double CudaScore = Anime4KCPP::benchmark<Anime4KCPP::Cuda::ACNet>(dID);
#endif 

        std::cout
            << "CPU score: "
            << CPUScore
            << std::endl;

        std::cout
            << "OpenCL score: "
            << OpenCLScore
            << " (pID = " << pID << ", dID = " << dID << ")"
            << std::endl;

#ifdef ENABLE_CUDA
        std::cout
            << "CUDA score: "
            << CudaScore
            << " (dID = " << dID << ")"
            << std::endl;
#endif 
        return 0;
    }

    filesystem::path inputPath(input), outputPath(output);
    if ((!videoMode || !webVideo) && !filesystem::exists(inputPath))
    {
        std::cerr << "input file or directory does not exist." << std::endl;
        return 0;
    }

    Anime4KCPP::CNNType type;
    if (HDN)
    {
        switch (HDNLevel)
        {
        case 1:
            type = Anime4KCPP::CNNType::ACNetHDNL1;
            break;
        case 2:
            type = Anime4KCPP::CNNType::ACNetHDNL2;
            break;
        case 3:
            type = Anime4KCPP::CNNType::ACNetHDNL3;
            break;
        default:
            type = Anime4KCPP::CNNType::ACNetHDNL1;
            break;
        }
    }
    else
        type = Anime4KCPP::CNNType::ACNetHDNL0;

    Anime4KCPP::ACCreator creator;
    std::unique_ptr<Anime4KCPP::AC> ac;
    Anime4KCPP::Parameters parameters(
        passes,
        pushColorCount,
        strengthColor,
        strengthGradient,
        zoomFactor,
        fastMode,
        videoMode,
        preprocessing,
        postprocessing,
        preFilters,
        postFilters,
        threads,
        HDN,
        HDNLevel,
        alpha
    );

    std::cout
        << "----------------------------------------------" << std::endl
        << "Welcome to Anime4KCPP" << std::endl
        << "----------------------------------------------" << std::endl;

    try
    {
        //init
        if (GPU)
        {
            switch (GPGPUModel)
            {
            case GPGPU::OpenCL:
                if (CNN)
                    creator.pushManager<Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::ACNet>>(
                        pID, dID, 
                        type,
                        OpenCLQueueNum,
                        OpenCLParallelIO);
                else
                    creator.pushManager<Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::Anime4K09>>(
                        pID, dID,
                        OpenCLQueueNum,
                        OpenCLParallelIO);
                break;
            case GPGPU::CUDA:
#ifndef ENABLE_CUDA
                std::cerr << "CUDA is not supported" << std::endl;
                return 0;
#else
                creator.pushManager<Anime4KCPP::Cuda::Manager>();
                break;
#endif
            }
            creator.init();
        }

        //create instance
        if (GPU)
        {
            switch (GPGPUModel)
            {
            case GPGPU::OpenCL:
            {
                Anime4KCPP::OpenCL::GPUInfo ret = Anime4KCPP::OpenCL::checkGPUSupport(pID, dID);
                if (!ret)
                {
                    std::cerr << ret() << std::endl;
                    return 0;
                }
                else
                    std::cerr << ret() << std::endl;
                if (CNN)
                    ac = creator.createUP(parameters, Anime4KCPP::Processor::Type::OpenCL_ACNet);
                else
                    ac = creator.createUP(parameters, Anime4KCPP::Processor::Type::OpenCL_Anime4K09);
            }
            break;
            case GPGPU::CUDA:
            {
#ifdef ENABLE_CUDA
                Anime4KCPP::Cuda::GPUInfo ret = Anime4KCPP::Cuda::checkGPUSupport(dID);
                if (!ret)
                {
                    std::cerr << ret() << std::endl;
                    return 0;
                }
                else
                    std::cerr << ret() << std::endl;
                if (CNN)
                    ac = creator.createUP(parameters, Anime4KCPP::Processor::Type::Cuda_ACNet);
                else
                    ac = creator.createUP(parameters, Anime4KCPP::Processor::Type::Cuda_Anime4K09);
#endif
            }
            break;
            }
        }
        else
        {
            if (CNN)
                ac = creator.createUP(parameters, Anime4KCPP::Processor::Type::CPU_ACNet);
            else
                ac = creator.createUP(parameters, Anime4KCPP::Processor::Type::CPU_Anime4K09);
        }
        //processing
        if (!videoMode)//Image
        {
            if (filesystem::is_directory(inputPath))
            {
                if (!filesystem::is_directory(outputPath))
                    outputPath = outputPath.parent_path().append(outputPath.stem().native());
                filesystem::create_directories(outputPath);
                filesystem::directory_iterator currDir(inputPath);
                std::vector<std::pair<std::string, std::string>> filePaths;

                std::cout
                    << ac->getInfo() << std::endl
                    << ac->getFiltersInfo() << std::endl
                    << "Scanning..." << std::endl;

                for (auto& file : currDir)
                {
                    if (filesystem::is_directory(file.path()))
                        continue;
                    std::string currInputPath = file.path().string();
                    std::string currOnputPath = (outputPath / (file.path().filename().replace_extension(".png"))).string();

                    filePaths.emplace_back(std::make_pair(currInputPath, currOnputPath));
                }

                std::cout << filePaths.size() << " files total" << std::endl
                    << "Threads: " << threads << std::endl
                    << "Start processing..." << std::endl;

                Anime4KCPP::Utils::ThreadPool threadPool(threads);
                std::atomic_uint64_t progress = 0;
                std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();

                for (int i = 0; i < filePaths.size(); i++)
                {
                    threadPool.exec([i, &filePaths, &progress, &creator, &parameters, &ac]()
                        {
                            Anime4KCPP::AC* currac = creator.create(parameters, ac->getProcessorType());
                            currac->loadImage(filePaths[i].first);
                            currac->process();
                            currac->saveImage(filePaths[i].second);
                            progress++;
                            creator.release(currac);
                        });
                }

                for (;;)
                {
                    std::this_thread::yield();
                    std::cout << '\r' << '[' << progress << '/' << filePaths.size() << ']';
                    if (progress >= filePaths.size())
                    {
                        std::cout << '\r' << '[' << filePaths.size() << '/' << filePaths.size() << ']';
                        break;
                    }
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();

                std::cout
                    << std::endl
                    << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0 << " s" << std::endl
                    << "All finished." << std::endl;
            }
            else
            {
                std::string currInputPath = inputPath.string();
                std::string currOnputPath = outputPath.string();

                ac->loadImage(currInputPath);

                std::cout << ac->getInfo() << std::endl;
                std::cout << ac->getFiltersInfo() << std::endl;

                std::cout << "Processing..." << std::endl;
                std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();
                ac->process();
                std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
                std::cout << "Total process time: " << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0 << " s" << std::endl;

                if (preview)
                    ac->showImage();
                else
                    ac->saveImage(currOnputPath);
            }
        }
        else // Video
        {
            if (preview)
            {
                std::string currInputPath = inputPath.string();
                ac->loadVideo(currInputPath);

                std::cout << ac->getInfo() << std::endl;
                std::cout << ac->getFiltersInfo() << std::endl;

                cv::VideoCapture videoCapture(currInputPath, cv::CAP_FFMPEG);
                if (!videoCapture.isOpened())
                    throw std::runtime_error("Error: Unable to open the video file");

                size_t totalFrameCount = videoCapture.get(cv::CAP_PROP_FRAME_COUNT);
                if (frameStart >= totalFrameCount)
                    throw std::runtime_error(
                        "Error: Unable to locate frame position: " +
                        std::to_string(frameStart) + " of " +
                        std::to_string(totalFrameCount - 1));

                videoCapture.set(cv::CAP_PROP_POS_FRAMES, frameStart);
                int delay = 500.0 / (forceFps < 1.0 ? videoCapture.get(cv::CAP_PROP_FPS) : forceFps);
                char keyCode = 'q';
                std::string windowName =
                    "Previewing, press 'q','ESC' or 'Enter' to exit, "
                    "'space' to pause, 'd' to fast forward, 'a' to fast backward, "
                    "'w' to forward, 's' to backward";

                cv::Mat frame;
                cv::namedWindow(windowName, cv::WindowFlags::WINDOW_NORMAL);
                cv::resizeWindow(windowName,
                    videoCapture.get(cv::CAP_PROP_FRAME_WIDTH) * zoomFactor + 0.5,
                    videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT) * zoomFactor + 0.5);
                ac->setVideoMode(false);

                std::cout
                    << "Previewing..." << std::endl
                    << "  Start frame: " << frameStart << std::endl
                    << "  Total frame: " << totalFrameCount << std::endl;
            
                while (videoCapture.read(frame))
                {
                    ac->loadImage(frame);
                    ac->process();
                    ac->saveImage(frame);
                    cv::imshow(windowName, frame);

                    keyCode = cv::waitKey(delay) & 0xff;

                    if (cv::getWindowProperty(windowName, cv::WindowPropertyFlags::WND_PROP_AUTOSIZE) != cv::WindowFlags::WINDOW_NORMAL ||
                        keyCode == 'q' || keyCode == 0x1b || keyCode == 0x0d)
                        break;
                    else if (keyCode == 0x20)
                    {
                        keyCode = cv::waitKey(0);
                        if (keyCode == 'q' || keyCode == 0x1b || keyCode == 0x0d)
                            break;
                    }
                    else
                    {
                        switch (keyCode)
                        {
                        case 'a':
                            videoCapture.set(
                                cv::CAP_PROP_POS_FRAMES,
                                videoCapture.get(cv::CAP_PROP_POS_FRAMES) - videoCapture.get(cv::CAP_PROP_FPS) * 10.0);
                            break;
                        case 'd':
                            videoCapture.set(
                                cv::CAP_PROP_POS_FRAMES,
                                videoCapture.get(cv::CAP_PROP_POS_FRAMES) + videoCapture.get(cv::CAP_PROP_FPS) * 10.0);
                            break;
                        case 's':
                            videoCapture.set(
                                cv::CAP_PROP_POS_FRAMES,
                                videoCapture.get(cv::CAP_PROP_POS_FRAMES) - videoCapture.get(cv::CAP_PROP_FPS) * 2.0);
                            break;
                        case 'w':
                            videoCapture.set(
                                cv::CAP_PROP_POS_FRAMES,
                                videoCapture.get(cv::CAP_PROP_POS_FRAMES) + videoCapture.get(cv::CAP_PROP_FPS) * 2.0);
                            break;
                        }
                    }
                }

                videoCapture.release();
                cv::destroyWindow(windowName);
                std::cout << "Exit" << std::endl;
            }
            else
            {
                // output suffix
                std::string outputSuffix = outputPath.extension().string();
                // transform to lower
                std::transform(outputSuffix.begin(), outputSuffix.end(), outputSuffix.begin(), ::tolower);

                if (std::string(".png.jpg.jpeg.bmp")
                    .find(outputSuffix) != std::string::npos)
                    outputPath.replace_extension(".mkv");

                bool gif = outputSuffix == std::string(".gif");

                bool ffmpeg = checkFFmpeg();
                std::string outputTmpName = outputPath.string();

                if (!ffmpeg)
                    std::cerr << "Please install ffmpeg, otherwise the output file will be silent." << std::endl;
                else
                    outputTmpName = "tmp_out.mp4";

                if (filesystem::is_directory(inputPath))
                {
                    if (!filesystem::is_directory(outputPath))
                        outputPath = outputPath.parent_path().append(outputPath.stem().native());
                    filesystem::create_directories(outputPath);
                    filesystem::directory_iterator currDir(inputPath);
                    for (auto& file : currDir)
                    {
                        if (filesystem::is_directory(file.path()))
                            continue;
                        //Check GIF
                        std::string inputSuffix = file.path().extension().string();
                        std::transform(inputSuffix.begin(), inputSuffix.end(), inputSuffix.begin(), ::tolower);
                        gif = inputSuffix == std::string(".gif");

                        std::string currInputPath = file.path().string();
                        std::string currOutputPath = (outputPath / (file.path().filename().replace_extension(gif ? ".gif" : ".mkv"))).string();

                        ac->loadVideo(currInputPath);
                        ac->setVideoSaveInfo(outputTmpName, string2Codec(codec), forceFps);

                        std::cout << ac->getInfo() << std::endl;
                        std::cout << ac->getFiltersInfo() << std::endl;

                        std::cout << "Processing..." << std::endl;
                        std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();
                        if (disableProgress)
                            ac->process();
                        else
                            ac->processWithPrintProgress();
                        std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
                        std::cout << "Total process time: " << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0 / 60.0 << " min" << std::endl;

                        ac->saveVideo();

                        if (ffmpeg)
                        {
                            if (!gif)
                            {
                                if (mergeAudio2Video(currOutputPath, currInputPath, outputTmpName))
                                    filesystem::remove(outputTmpName);
                            }
                            else
                            {
                                if (video2GIF(outputTmpName, currOutputPath))
                                    filesystem::remove(outputTmpName);
                            }
                        }
                    }
                }
                else
                {
                    std::string currInputPath = inputPath.string();
                    std::string currOutputPath = outputPath.string();

                    ac->loadVideo(currInputPath);
                    ac->setVideoSaveInfo(outputTmpName, string2Codec(codec), forceFps);

                    std::cout << ac->getInfo() << std::endl;
                    std::cout << ac->getFiltersInfo() << std::endl;

                    std::cout << "Processing..." << std::endl;
                    std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();
                    if (disableProgress)
                        ac->process();
                    else
                        ac->processWithPrintProgress();
                    std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
                    std::cout << "Total process time: " << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0 / 60.0 << " min" << std::endl;

                    ac->saveVideo();

                    if (ffmpeg)
                    {
                        if (!gif)
                        {
                            if (mergeAudio2Video(currOutputPath, currInputPath, outputTmpName))
                                filesystem::remove(outputTmpName);
                        }
                        else
                        {
                            if (video2GIF(outputTmpName, currOutputPath))
                                filesystem::remove(outputTmpName);
                        }
                    }
                }
            }
        }
    }
    catch (const std::exception& err)
    {
        std::cerr
            << std::endl
            << err.what()
            << std::endl;
    }
    return 0;
}
