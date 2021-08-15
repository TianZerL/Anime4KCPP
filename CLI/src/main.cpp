#include<iostream>

#ifdef USE_BOOST_FILESYSTEM
#include<boost/filesystem.hpp>
namespace filesystem = boost::filesystem;
#else
#include<filesystem>
namespace filesystem = std::filesystem;
#endif // USE_BOOST_FILESYSTEM

#include"Anime4KCPP.hpp"
#include"Parallel.hpp"

#include"Config.hpp"

#ifdef ENABLE_LIBCURL
#include"Downloader.hpp"
#endif // ENABLE_LIBCURL

#ifndef COMPILER
#define COMPILER "Unknown"
#endif // !COMPILER

enum class GPGPU
{
    OpenCL, CUDA, NCNN
};

static bool checkFFmpeg()
{
    std::cout << "Checking ffmpeg..." << std::endl;

    return !std::system("ffmpeg -version");
}

static void processVideoWithProgress(Anime4KCPP::VideoProcessor& videoPeocessor)
{
    auto s = std::chrono::steady_clock::now();
    videoPeocessor.processWithProgress(
        [&s](double progress)
        {
            auto e = std::chrono::steady_clock::now();
            double currTime = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0;

            std::fprintf(stderr,
                "%7.2f%%     elpsed: %8.2fs    remaining: %8.2fs\r",
                progress * 100,
                currTime,
                currTime / progress - currTime);

            if (progress == 1.0)
                std::putc('\n', stderr);
        });
}

static bool mergeAudio2Video(const std::string& dstFile, const std::string& srcFile, const std::string& tmpFile)
{
    std::cout << "Merging audio..." << std::endl;

    std::string command(
        "ffmpeg -loglevel 40 -i \"" +
        tmpFile + "\" -i \"" + srcFile +
        "\" -c copy -map 0:v -map 1 -map -1:v -y \"" +
        dstFile + "\"");

    std::cout << command << std::endl;

    return !std::system(command.data());
}

static bool video2GIF(const std::string& srcFile, const std::string& dstFile)
{
    std::string commandGeneratePalette(
        "ffmpeg -i \"" + srcFile +
        "\" -vf palettegen -y palette.png");
    std::string command2Gif(
        "ffmpeg -i \"" + srcFile +
        "\" -i palette.png -y -lavfi paletteuse \"" +
        dstFile + "\"");

    std::cout << commandGeneratePalette << std::endl;
    std::cout << command2Gif << std::endl;

    bool flag = !std::system(commandGeneratePalette.data()) && !std::system(command2Gif.data());

    flag &= filesystem::remove("palette.png");

    return flag;
}

static Anime4KCPP::Codec string2Codec(const std::string& codec)
{
    if (codec == "mp4v")
        return Anime4KCPP::Codec::MP4V;
    if (codec == "dxva")
        return Anime4KCPP::Codec::DXVA;
    if (codec == "avc1")
        return Anime4KCPP::Codec::AVC1;
    if (codec == "vp09")
        return Anime4KCPP::Codec::VP09;
    if (codec == "hevc")
        return Anime4KCPP::Codec::HEVC;
    if (codec == "av01")
        return Anime4KCPP::Codec::AV01;
    if (codec == "other")
        return Anime4KCPP::Codec::OTHER;
    return Anime4KCPP::Codec::MP4V;
}

static void showVersionInfo()
{
    std::cerr
        << "Anime4KCPPCLI" << std::endl
        << "Anime4KCPP core version: " << ANIME4KCPP_CORE_VERSION << std::endl
        << "Parallel library: " << PARALLEL_LIBRARY << std::endl
        << "Build date: " << __DATE__ << " " << __TIME__ << std::endl
        << "Compiler: " << COMPILER << std::endl
        << "GitHub: https://github.com/TianZerL/Anime4KCPP" << std::endl;
}

static void showGPUList()
{
#if !defined(ENABLE_OPENCL) && !defined(ENABLE_OCUDA) && !defined(ENABLE_NCNN)
    std::cerr << "Error: No GPU acceleration mode supported\n";
#endif

#ifdef ENABLE_OPENCL
    std::cout << "OpenCL:\n";
    Anime4KCPP::OpenCL::GPUList OpenCLGPUList = Anime4KCPP::OpenCL::listGPUs();
    if (OpenCLGPUList.platforms == 0)
        std::cerr << "Error: No OpenCL GPU found\n";
    else
        std::cout << OpenCLGPUList();
#endif

#ifdef ENABLE_CUDA
    std::cout << "CUDA:\n";
    Anime4KCPP::Cuda::GPUList CUDAGPUList = Anime4KCPP::Cuda::listGPUs();
    if (CUDAGPUList.devices == 0)
        std::cerr << "Error: No CUDA GPU found\n";
    else
        std::cout << CUDAGPUList();
#endif

#ifdef ENABLE_NCNN
    std::cout << "ncnn:\n";
    Anime4KCPP::NCNN::GPUList NCNNGPUList = Anime4KCPP::NCNN::listGPUs();
    if (NCNNGPUList.devices == 0)
        std::cerr << "Error: No ncnn Vulkan GPU found\n";
    else
        std::cout << NCNNGPUList();
#endif
}

static void benchmark(const int pID, const int dID)
{
    std::cout << "Benchmark test under 8-bit integer input and serial processing...\n" << std::endl;

    double CPUScoreDVD = Anime4KCPP::benchmark<Anime4KCPP::CPU::ACNet, 720, 480>();
    double CPUScoreHD = Anime4KCPP::benchmark<Anime4KCPP::CPU::ACNet, 1280, 720>();
    double CPUScoreFHD = Anime4KCPP::benchmark<Anime4KCPP::CPU::ACNet, 1920, 1080>();

#ifdef ENABLE_OPENCL
    double OpenCLScoreDVD = Anime4KCPP::benchmark<Anime4KCPP::OpenCL::ACNet, 720, 480>(pID, dID);
    double OpenCLScoreHD = Anime4KCPP::benchmark<Anime4KCPP::OpenCL::ACNet, 1280, 720>(pID, dID);
    double OpenCLScoreFHD = Anime4KCPP::benchmark<Anime4KCPP::OpenCL::ACNet, 1920, 1080>(pID, dID);
#endif 

#ifdef ENABLE_CUDA
    double CudaScoreDVD = Anime4KCPP::benchmark<Anime4KCPP::Cuda::ACNet, 720, 480>(dID);
    double CudaScoreHD = Anime4KCPP::benchmark<Anime4KCPP::Cuda::ACNet, 1280, 720>(dID);
    double CudaScoreFHD = Anime4KCPP::benchmark<Anime4KCPP::Cuda::ACNet, 1920, 1080>(dID);
#endif 

#ifdef ENABLE_NCNN
    double NCNNCPUScoreDVD = Anime4KCPP::benchmark<Anime4KCPP::NCNN::ACNet, 720, 480>(-1, Anime4KCPP::CNNType::ACNetHDNL0, 4);
    double NCNNCPUScoreHD = Anime4KCPP::benchmark<Anime4KCPP::NCNN::ACNet, 1280, 720>(-1, Anime4KCPP::CNNType::ACNetHDNL0, 4);
    double NCNNCPUScoreFHD = Anime4KCPP::benchmark<Anime4KCPP::NCNN::ACNet, 1920, 1080>(-1, Anime4KCPP::CNNType::ACNetHDNL0, 4);

    double NCNNVKScoreDVD = Anime4KCPP::benchmark<Anime4KCPP::NCNN::ACNet, 720, 480>(dID, Anime4KCPP::CNNType::ACNetHDNL0, 4);
    double NCNNVKScoreHD = Anime4KCPP::benchmark<Anime4KCPP::NCNN::ACNet, 1280, 720>(dID, Anime4KCPP::CNNType::ACNetHDNL0, 4);
    double NCNNVKScoreFHD = Anime4KCPP::benchmark<Anime4KCPP::NCNN::ACNet, 1920, 1080>(dID, Anime4KCPP::CNNType::ACNetHDNL0, 4);
#endif 

    std::cout
        << "CPU score:\n"
        << " DVD(480P->960P): " << CPUScoreDVD << " FPS\n"
        << " HD(720P->1440P): " << CPUScoreHD << " FPS\n"
        << " FHD(1080P->2160P): " << CPUScoreFHD << " FPS\n" << std::endl;

#ifdef ENABLE_OPENCL
    std::cout
        << "OpenCL score: (pID = " << pID << ", dID = " << dID << ")\n"
        << " DVD(480P->960P): " << OpenCLScoreDVD << " FPS\n"
        << " HD(720P->1440P): " << OpenCLScoreHD << " FPS\n"
        << " FHD(1080P->2160P): " << OpenCLScoreFHD << " FPS\n" << std::endl;
#endif 

#ifdef ENABLE_CUDA
    std::cout
        << "CUDA score: (dID = " << dID << ")\n"
        << " DVD(480P->960P): " << CudaScoreDVD << " FPS\n"
        << " HD(720P->1440P): " << CudaScoreHD << " FPS\n"
        << " FHD(1080P->2160P): " << CudaScoreFHD << " FPS\n" << std::endl;
#endif 

#ifdef ENABLE_NCNN
    std::cout
        << "NCNN CPU score:\n"
        << " DVD(480P->960P):" << NCNNCPUScoreDVD << " FPS\n"
        << " HD(720P->1440P):" << NCNNCPUScoreHD << " FPS\n"
        << " FHD(1080P->2160P):" << NCNNCPUScoreFHD << " FPS\n" << std::endl;

    std::cout
        << "NCNN Vulkan score: (dID = " << dID << ")\n"
        << " DVD(480P->960P):" << NCNNVKScoreDVD << " FPS\n"
        << " HD(720P->1440P):" << NCNNVKScoreHD << " FPS\n"
        << " FHD(1080P->2160P):" << NCNNVKScoreFHD << " FPS\n" << std::endl;
#endif 
}

static bool createConfigTemplate(const std::string& path, Config& config)
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
    opt.add<std::string>("input", 'i', "File for loading", false, "p1.png");
    opt.add<std::string>("output", 'o', "File for outputting", false);
    opt.add<int>("passes", 'p', "Passes for processing", false, 2);
    opt.add<int>("pushColorCount", 'n', "Limit the number of color pushes", false, 2);
    opt.add<double>("strengthColor", 'c',
        "Strength for pushing color,range 0 to 1,higher for thinner", false, 0.3, cmdline::range(0.0, 1.0));
    opt.add<double>("strengthGradient", 'g',
        "Strength for pushing gradient,range 0 to 1,higher for sharper", false, 1.0, cmdline::range(0.0, 1.0));
    opt.add<double>("zoomFactor", 'z', "zoom factor for resizing", false, 2.0);
    opt.add<unsigned int>("threads", 't',
        "Threads count for video processing", false,
        std::thread::hardware_concurrency(), cmdline::range(1u, 32 * std::thread::hardware_concurrency()));
    opt.add<unsigned int>("ncnnThreads", 'T',
        "Threads count for ncnn module", false, 4,
        cmdline::range(1u, std::thread::hardware_concurrency()));
    opt.add("fastMode", 'f', "Faster but maybe low quality");
    opt.add("videoMode", 'v', "Video process");
    opt.add("preview", 's', "Preview image");
    opt.add<std::size_t>("start", 'S', "Specify the start frame number for video previewing", false, 0);
    opt.add("preprocessing", 'b', "Enable preprocessing");
    opt.add("postprocessing", 'a', "Enable postprocessing");
    opt.add<uint8_t>("preFilters", 'r',
        "Enhancement filter, only working when preprocessing is true,there are 5 options by binary:"
        "Median blur=0000001, Mean blur=0000010, CAS Sharpening=0000100, Gaussian blur weak=0001000, "
        "Gaussian blur=0010000, Bilateral filter=0100000, Bilateral filter faster=1000000, "
        "you can freely combine them, eg: Gaussian blur weak + Bilateral filter = 0001000 | "
        "0100000 = 0101000 = 40(D)", false, Anime4KCPP::Filter::CAS_Sharpening, cmdline::range(1, 127));
    opt.add<uint8_t>("postFilters", 'e',
        "Enhancement filter, only working when postprocessing is true,there are 5 options by binary:"
        "Median blur=0000001, Mean blur=0000010, CAS Sharpening=0000100, Gaussian blur weak=0001000, "
        "Gaussian blur=0010000, Bilateral filter=0100000, Bilateral filter faster=1000000, "
        "you can freely combine them, eg: Gaussian blur weak + Bilateral filter = 0001000 | 0100000 = "
        "0101000 = 40(D), so you can put 40 to enable Gaussian blur weak and Bilateral filter, "
        "which also is what I recommend for image that < 1080P, 48 for image that >= 1080P, "
        "and for performance I recommend to use 72 for video that < 1080P, 80 for video that >=1080P",
        false, Anime4KCPP::Filter::Gaussian_Blur_Weak | Anime4KCPP::Filter::Bilateral_Filter, cmdline::range(1, 127));
    opt.add("GPUMode", 'q', "Enable GPU acceleration");
    opt.add("CNNMode", 'w', "Enable ACNet");
    opt.add("HDN", 'H', "Enable HDN mode for ACNet");
    opt.add<int>("HDNLevel", 'L', "Set HDN level", false, 1, cmdline::range(1, 3));
    opt.add("listGPUs", 'l', "list GPUs");
    opt.add<int>("platformID", 'h', "Specify the platform ID", false, 0);
    opt.add<int>("deviceID", 'd', "Specify the device ID", false, 0);
    opt.add<int>("OpenCLQueueNumber", 'Q',
        "Specify the number of command queues for OpenCL, this may affect performance of "
        "video processing for OpenCL", false, 1);
    opt.add("OpenCLParallelIO", 'P',
        "Use a parallel IO command queue for OpenCL, this may affect performance for OpenCL");
    opt.add<std::string>("codec", 'C',
        "Specify the codec for encoding from mp4v(recommended in Windows), dxva(for Windows), "
        "avc1(H264, recommended in Linux), vp09(very slow), hevc(not support in Windows), "
        "av01(not support in Windows)", false, "mp4v",
        cmdline::oneof<std::string>("mp4v", "dxva", "avc1", "vp09", "hevc", "av01", "other"));
    opt.add("version", 'V', "print version information");
    opt.add<double>("forceFps", 'F', "Set output video fps to the specifying number, 0 to disable", false, 0.0);
    opt.add("disableProgress", 'D', "disable progress display");
    opt.add("web", 'W', "process the file from URL");
    opt.add("alpha", 'A', "preserve the Alpha channel for transparent image");
    opt.add("benchmark", 'B', "do benchmarking");
    opt.add("ncnn", 'N', "Open ncnn and ACNet");
    opt.add<std::string>("GPGPUModel", 'M', "Specify the GPGPU model for processing", false, "opencl");
    opt.add<std::string>("ncnnModelPath", 'Z', "Specify the path for NCNN model and param", false, "./ncnn-models");
    opt.set_program_name("Anime4KCPP_CLI");
    opt.add<std::string>("configTemplate", '\000', "Generate config template", false);
    opt.add<std::string>("testMode", '\000', "function test for development only", false);

    opt.parse_check(argc, argv);

    if (!opt.rest().empty() && !config.initFile(opt.rest().front()).parseFile())
    {
        config.getLastError().printErrorInfo();
        return 0;
    }

    auto input = opt.get<std::string>("input");
    auto defaultOutputName = !opt.exist("output");
    auto output = defaultOutputName ? std::string{} : opt.get<std::string>("output");
    auto videoMode = opt.exist("videoMode");
    auto preview = opt.exist("preview");
    auto listGPUs = opt.exist("listGPUs");
    auto version = opt.exist("version");
    auto web = opt.exist("web");
    auto doBenchmark = opt.exist("benchmark");
    auto ncnn = opt.exist("ncnn");
    auto frameStart = opt.get<std::size_t>("start");
    auto configTemplate = opt.exist("configTemplate");
    auto testMode = opt.get<std::string>("testMode");

    //args which can be saved to config
    auto passes = config.get<int>("passes");
    auto pushColorCount = config.get<int>("pushColorCount");
    auto HDNLevel = config.get<int>("HDNLevel");
    auto strengthColor = config.get<double>("strengthColor");
    auto strengthGradient = config.get<double>("strengthGradient");
    auto zoomFactor = config.get<double>("zoomFactor");
    auto forceFps = config.get<double>("forceFps");
    auto fastMode = config.get<bool>("fastMode");
    auto preprocessing = config.get<bool>("preprocessing");
    auto postprocessing = config.get<bool>("postprocessing");
    auto GPU = config.get<bool>("GPUMode");
    auto CNN = config.get<bool>("CNNMode");
    auto HDN = config.get<bool>("HDN");
    auto disableProgress = config.get<bool>("disableProgress");
    auto alpha = config.get<bool>("alpha");
    auto preFilters = config.get<uint8_t>("preFilters");
    auto postFilters = config.get<uint8_t>("postFilters");
    auto pID = config.get<int>("platformID");
    auto dID = config.get<int>("deviceID");
    auto OpenCLQueueNum = config.get<int>("OpenCLQueueNumber");
    auto OpenCLParallelIO = config.get<bool>("OpenCLParallelIO");
    auto threads = config.get<unsigned int>("threads");
    auto ncnnThreads = config.get<unsigned int>("ncnnThreads");
    auto codec = config.get<std::string>("codec");
    auto GPGPUModelString = config.get<std::string>("GPGPUModel");
    auto ncnnModelPath = config.get<std::string>("ncnnModelPath");

    //Generate config template
    if (configTemplate)
    {
        const std::string& path = opt.get<std::string>("configTemplate");
        if (createConfigTemplate(path, config))
            std::cout << "Generated config template to: " << path << '\n';
        else
            std::cerr << "Failed to generate config template: " << path << '\n';
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
        showGPUList();
        return 0;
    }
    // -b
    if (doBenchmark)
    {
        benchmark(pID, dID);
        return 0;
    }

    GPGPU GPGPUModel;
    std::transform(GPGPUModelString.begin(), GPGPUModelString.end(), GPGPUModelString.begin(), ::tolower);
    if (GPGPUModelString == "opencl")
    {
#ifdef ENABLE_OPENCL
        GPGPUModel = GPGPU::OpenCL;
#elif defined(ENABLE_CUDA)
        GPGPUModel = GPGPU::CUDA;
#elif defined(ENABLE_NCNN)
        GPGPUModel = GPGPU::NCNN;
#else
        if (GPU)
        {
            std::cerr << "No GPU processor available\n";
            return 0;
        }
#endif
    }
    else if (GPGPUModelString == "cuda")
        GPGPUModel = GPGPU::CUDA;
    else if (GPGPUModelString == "ncnn")
        GPGPUModel = GPGPU::NCNN;
    else
    {
        std::cerr << R"(Unknown GPGPU model, it must be "ncnn", "cuda" or "opencl")" << '\n';
        return 0;
    }

    if (ncnn)
    {
        GPGPUModel = GPGPU::NCNN;
        if (!GPU)
            dID = -1;
        GPU = true;
    }

    filesystem::path inputPath(input), outputPath(output);
    if (defaultOutputName)
    {
        std::ostringstream oss;
        if (CNN)
            oss << "_ACNet_HDNL" << (HDN ? HDNLevel : 0);
        else
            oss << "_Anime4K09";
        oss << "_x" << std::fixed << std::setprecision(2) << zoomFactor
            << inputPath.extension().string();

        outputPath = inputPath.parent_path() /
            ((filesystem::is_directory(inputPath) ? "output" : inputPath.stem().string()) + oss.str());
    }

    if (!web && !filesystem::exists(inputPath))
    {
        std::cerr << "input file or directory does not exist.\n";
        return 0;
    }

    Anime4KCPP::CNNType type{};
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

    Anime4KCPP::ACInitializer initializer;
    std::unique_ptr<Anime4KCPP::AC> ac;
    Anime4KCPP::Parameters parameters(
        passes,
        pushColorCount,
        strengthColor,
        strengthGradient,
        zoomFactor,
        fastMode,
        preprocessing,
        postprocessing,
        preFilters,
        postFilters,
        HDN,
        HDNLevel,
        alpha
    );

    std::cout
        << "----------------------------------------------\n"
        << "Welcome to Anime4KCPP\n"
        << "----------------------------------------------\n";

    try
    {
        //init
        if (GPU)
        {
            switch (GPGPUModel)
            {
            case GPGPU::OpenCL:
#ifndef ENABLE_OPENCL
                std::cerr << "OpenCL is not supported\n";
                return 0;
#else
                if (CNN)
                    initializer.pushManager<Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::ACNet>>(
                        pID, dID,
                        type,
                        OpenCLQueueNum,
                        OpenCLParallelIO);
                else
                    initializer.pushManager<Anime4KCPP::OpenCL::Manager<Anime4KCPP::OpenCL::Anime4K09>>(
                        pID, dID,
                        OpenCLQueueNum,
                        OpenCLParallelIO);
#endif
                break;
            case GPGPU::CUDA:
#ifndef ENABLE_CUDA
                std::cerr << "CUDA is not supported\n";
                return 0;
#else
                initializer.pushManager<Anime4KCPP::Cuda::Manager>(dID);
                break;
#endif
            case GPGPU::NCNN:
#ifndef ENABLE_NCNN
                std::cerr << "ncnn is not supported\n";
                return 0;
#else
                {
                    if (testMode == "ncnn_load_model")
                    {
                        filesystem::path modelPath = filesystem::weakly_canonical(ncnnModelPath);

                        if (!filesystem::exists(modelPath))
                        {
                            std::cerr << "ncnn model or param file does not exist.\n";
                            return 0;
                        }
                        initializer.pushManager<Anime4KCPP::NCNN::Manager>(
                            (modelPath / (type.toString() + std::string(".bin"))).generic_string(),
                            (modelPath / "ACNet.param").generic_string(),
                            dID, type, ncnnThreads);
                    }
                    else
                    {
                        initializer.pushManager<Anime4KCPP::NCNN::Manager>(dID, type, ncnnThreads);
                    }
                }
                break;
#endif
            }
            initializer.init();
        }

        //create instance
        if (GPU)
        {
            switch (GPGPUModel)
            {
            case GPGPU::OpenCL:
            {
#ifdef ENABLE_OPENCL
                Anime4KCPP::OpenCL::GPUInfo ret = Anime4KCPP::OpenCL::checkGPUSupport(pID, dID);
                if (!ret)
                {
                    std::cerr << ret() << '\n';
                    return 0;
                }
                else
                    std::cerr << ret() << '\n';
                if (CNN)
                    ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::OpenCL_ACNet);
                else
                    ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::OpenCL_Anime4K09);
#endif
            }
            break;
            case GPGPU::CUDA:
            {
#ifdef ENABLE_CUDA
                Anime4KCPP::Cuda::GPUInfo ret = Anime4KCPP::Cuda::checkGPUSupport(dID);
                if (!ret)
                {
                    std::cerr << ret() << '\n';
                    return 0;
                }
                else
                    std::cerr << ret() << '\n';
                if (CNN)
                    ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::Cuda_ACNet);
                else
                    ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::Cuda_Anime4K09);
#endif
            }
            break;
            case GPGPU::NCNN:
            {
#ifdef ENABLE_NCNN
                if (dID < 0)
                    std::cerr << "ncnn uses CPU\n";

                if (CNN)
                    ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::NCNN_ACNet);
                else
                {
                    std::cerr << "ncnn only for ACNet\n";
                    return 0;
                }
#endif
            }
            break;
            }
        }
        else
        {
            if (CNN)
                ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::CPU_ACNet);
            else
                ac = Anime4KCPP::ACCreator::createUP(parameters, Anime4KCPP::Processor::Type::CPU_Anime4K09);
        }
        //processing
        if (!videoMode)//Image
        {
            if (filesystem::is_directory(inputPath))
            {
                if (!filesystem::is_directory(outputPath))
                    outputPath = outputPath.parent_path() / outputPath.stem();
                filesystem::recursive_directory_iterator currDir(inputPath);
                std::vector<std::pair<std::string, std::string>> filePaths;

                std::cout
                    << ac->getInfo() << '\n'
                    << ac->getFiltersInfo() << '\n'
                    << "Scanning..." << std::endl;;

                for (auto& file : currDir)
                {
                    if (filesystem::is_directory(file.path()))
                        continue;
                    auto tmpOutputPath = outputPath / file.path().lexically_relative(inputPath);
                    filesystem::create_directories(tmpOutputPath.parent_path());
                    std::string currInputPath = file.path().string();
                    std::string currOutputPath = tmpOutputPath.string();

                    filePaths.emplace_back(std::make_pair(currInputPath, currOutputPath));
                }

                std::cout << filePaths.size() << " files total" << '\n'
                    << "Start processing...\n" << std::endl;

                std::atomic_uint64_t progress = 0;
                std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();

                Anime4KCPP::Utils::ParallelFor(0, static_cast<const int>(filePaths.size()),
                    [&](const int i) {
                        Anime4KCPP::AC* pAc = Anime4KCPP::ACCreator::create(parameters, ac->getProcessorType());
                        pAc->loadImage(filePaths[i].first);
                        pAc->process();
                        pAc->saveImage(filePaths[i].second);
                        Anime4KCPP::ACCreator::release(pAc);
                        if (!disableProgress)
                        {
                            progress++;
                            std::cout << '\r' << progress << '/' << filePaths.size();
                        }
                    });

                std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();

                std::cout
                    << "Total time: "
                    << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0
                    << " s\n"
                    << "All finished." << std::endl;
            }
            else
            {
                std::string currInputPath = inputPath.string();
                std::string currOutputPath = outputPath.string();

#ifdef ENABLE_LIBCURL
                if (web)
                {
                    std::vector<std::uint8_t> buf;

                    Downloader downloader;
                    downloader.init();

                    downloader.download(currInputPath, buf);

                    ac->loadImage(buf);
                }
                else
#endif // ENABLE_LIBCURL
                    ac->loadImage(currInputPath);

                std::cout << ac->getInfo() << '\n';
                std::cout << ac->getFiltersInfo() << '\n';

                std::cout << "Processing..." << std::endl;
                std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();
                ac->process();
                std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
                std::cout
                    << "Total process time: "
                    << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0
                    << " s"
                    << std::endl;

                if (preview)
                    ac->showImage();
                else
                    ac->saveImage(currOutputPath);
            }
        }
        else // Video
        {
            if (preview)
            {
#ifdef ENABLE_PREVIEW_GUI
                std::string currInputPath = inputPath.string();

                std::cout << ac->getInfo() << '\n';
                std::cout << ac->getFiltersInfo() << '\n';

                cv::VideoCapture videoCapture(currInputPath);
                if (!videoCapture.isOpened())
                    throw std::runtime_error("Error: Unable to open the video file");

                std::size_t totalFrameCount = videoCapture.get(cv::CAP_PROP_FRAME_COUNT);
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

                std::cout
                    << "Previewing...\n"
                    << "  Start frame: " << frameStart << '\n'
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
                cv::destroyAllWindows();
                std::cout << "Exit" << std::endl;
#else
                throw Anime4KCPP::ACException<Anime4KCPP::ExceptionType::RunTimeError>("Preview video is not currently supported.");
#endif // ENABLE_PREVIEW_GUI
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
                    std::cerr << "Please install ffmpeg, otherwise the output file will be silent.\n";
                else
                    outputTmpName = "tmp_out.mp4";

                Anime4KCPP::VideoProcessor videoProcessor(*ac, threads);
                if (filesystem::is_directory(inputPath))
                {
                    if (!filesystem::is_directory(outputPath))
                        outputPath = outputPath.parent_path() / outputPath.stem();

                    filesystem::recursive_directory_iterator currDir(inputPath);
                    for (auto& file : currDir)
                    {
                        if (filesystem::is_directory(file.path()))
                            continue;
                        //Check GIF
                        std::string inputSuffix = file.path().extension().string();
                        std::transform(inputSuffix.begin(), inputSuffix.end(), inputSuffix.begin(), ::tolower);
                        gif = inputSuffix == std::string(".gif");

                        auto tmpOutputPath = outputPath / file.path().lexically_relative(inputPath);
                        filesystem::create_directories(tmpOutputPath.parent_path());
                        std::string currInputPath = file.path().string();
                        std::string currOutputPath = tmpOutputPath.replace_extension(gif ? ".gif" : ".mkv").string();

                        videoProcessor.loadVideo(currInputPath);
                        videoProcessor.setVideoSaveInfo(outputTmpName, string2Codec(codec), forceFps);

                        std::cout << ac->getInfo() << '\n';
                        std::cout << videoProcessor.getInfo() << '\n';
                        std::cout << ac->getFiltersInfo() << '\n';

                        std::cout << "Processing..." << std::endl;
                        std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();
                        if (disableProgress)
                            videoProcessor.process();
                        else
                            processVideoWithProgress(videoProcessor);
                        std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
                        std::cout 
                            << "Total process time: " 
                            << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0 / 60.0 
                            << " min" 
                            << std::endl;

                        videoProcessor.saveVideo();

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

                    videoProcessor.loadVideo(currInputPath);
                    videoProcessor.setVideoSaveInfo(outputTmpName, string2Codec(codec), forceFps);

                    std::cout << ac->getInfo() << '\n';
                    std::cout << videoProcessor.getInfo() << '\n';
                    std::cout << ac->getFiltersInfo() << '\n';

                    std::cout << "Processing..." << std::endl;
                    std::chrono::steady_clock::time_point s = std::chrono::steady_clock::now();
                    if (disableProgress)
                        videoProcessor.process();
                    else
                        processVideoWithProgress(videoProcessor);
                    std::chrono::steady_clock::time_point e = std::chrono::steady_clock::now();
                    std::cout 
                        << "Total process time: " 
                        << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() / 1000.0 / 60.0 
                        << " min" 
                        << std::endl;

                    videoProcessor.saveVideo();

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
            << '\n'
            << err.what()
            << '\n';
    }
    return 0;
}
