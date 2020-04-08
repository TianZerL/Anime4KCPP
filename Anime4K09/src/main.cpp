#include "Anime4K.h"
#include<cmdline.h>

#include<iostream>
#include<ctime>


bool checkFFmpeg()
{
    if (!system("ffmpeg -version"))
        return true;
    return false;
}

bool mergrAudio2Video(const std::string &output, const std::string &srcFile)
{
    std::string command("ffmpeg -i \"tmp_out.mp4\" -i \"" + srcFile + "\" -c copy -map 0 -map 1:1 -y \"" + output + "\"");
    std::cout << command << std::endl;

    if (!system(command.data()))
        return true;
    return false;
}

int main(int argc,char *argv[]) 
{
    //Options
    cmdline::parser opt;
    opt.add<std::string>("input", 'i', "File for loading", false, "./pic/p1.png");
    opt.add<std::string>("output", 'o', "File for outputting", false, "output.png");
    opt.add<int>("passes", 'p', "Passes for processing", false, 2);
    opt.add<double>("strengthColor", 'c', "Strength for pushing color,range 0 to 1,higher for thinner", false, 0.3, cmdline::range(0.0, 1.0));
    opt.add<double>("strengthGradient", 'g', "Strength for pushing gradient,range 0 to 1,higher for sharper", false, 1.0, cmdline::range(0.0, 1.0));
    opt.add<double>("zoomFactor", 'z', "zoom factor for resizing", false, 2.0);
    opt.add<unsigned int>("threads", 't', "Threads count for video processing", false, std::thread::hardware_concurrency(), cmdline::range(1, int(4 * std::thread::hardware_concurrency())));
    opt.add("fastMode", 'f', "Faster but maybe low quality");
    opt.add("videoMode", 'v', "Video process");
    opt.add("preview", 's', "Preview image");
    opt.add("postProcessing", 'a', "Enable post processing");
    opt.add<unsigned int>("filters", 'e', 
        "Enhancement filter, only working when postProcessing is true,there are 5 options by binary:\
median blur=000001, mean blur=000010, gaussian blur weak=000100, gaussian blur=001000, bilateral filter=010000, bilateral filter faster=100000, \
you can freely combine them, eg: gaussian blur weak + bilateral filter = 000100 & 010000 = 010100 = 20(D), \
so you can put 20 to enable gaussian blur weak and bilateral filter, which also is what I recommend for image that < 1080P, \
24 for image that >= 1080P, and for performance I recommend to use 36 for video that < 1080P, 40 for video that >=1080P", 
        false, 20, cmdline::range(1, 63));
    
    opt.parse_check(argc, argv);

    std::string input = opt.get<std::string>("input");
    std::string output = opt.get<std::string>("output");
    int passes = opt.get<int>("passes");
    double strengthColor = opt.get<double>("strengthColor");
    double strengthGradient = opt.get<double>("strengthGradient");
    double zoomFactor = opt.get<double>("zoomFactor");
    uint8_t filters = (uint8_t)opt.get<unsigned int>("filters");
    unsigned int threads = opt.get<unsigned int>("threads");
    bool fastMode = opt.exist("fastMode");
    bool videoMode = opt.exist("videoMode");
    bool preview = opt.exist("preview");
    bool postProcessing = opt.exist("postProcessing");

    //Anime4K
    Anime4K anime4k(
        passes, 
        strengthColor, 
        strengthGradient, 
        zoomFactor, 
        fastMode,
        videoMode,
        postProcessing,
        filters,
        threads
    );
    if (!videoMode)//Image
    {
        try
        {
            anime4k.loadImage(input);
        }
        catch (const char* err)
        {
            std::cout << err << std::endl;
            return 0;
        }

        anime4k.showInfo();
        anime4k.showFiltersInfo();

        std::cout << "Processing..." << std::endl;
        time_t s = std::clock();
        anime4k.process();
        time_t e = std::clock();
        std::cout << "Total process time: " << double(e - s) / CLOCKS_PER_SEC << " s" << std::endl;

        if(preview)
            anime4k.showImage();
        anime4k.saveImage(output);
    }
    else//Video
    {
        //Suffix check
        if (output.substr(output.size() - 3) == "png")
            output.replace(output.size() - 3, 3, "mp4");

        bool ffmpeg = checkFFmpeg();
        std::string outputTmpName = output;

        if (!ffmpeg)
            std::cout<<"Please install ffmpeg, otherwise the output file will be silent."<<std::endl;
        else 
            outputTmpName = "tmp_out.mp4";

        try
        {
            anime4k.loadVideo(input);
            anime4k.setVideoSaveInfo(outputTmpName);
        }
        catch (const char* err)
        {
            std::cout << err << std::endl;
            return 0;
        }
        anime4k.showInfo();
        anime4k.showFiltersInfo();

        std::cout << "Processing..." << std::endl;
        time_t s = std::clock();
        anime4k.process();
        time_t e = std::clock();
        std::cout << "Total process time: " << double(e - s) / CLOCKS_PER_SEC / 60 << " min" << std::endl;
        
        anime4k.saveVideo();

        if (ffmpeg && mergrAudio2Video(output, input))
        {
#ifdef _WIN32
            std::string command("del /q " + outputTmpName);
#elif defined(__linux)
            std::string command("rm " + outputTmpName);
#endif // SYSTEM
            system(command.data());
        }       
    }
    
    return 0;
}
