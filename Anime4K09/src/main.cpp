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

    opt.parse_check(argc, argv);

    std::string input = opt.get<std::string>("input");
    std::string output = opt.get<std::string>("output");
    int passes = opt.get<int>("passes");
    double strengthColor = opt.get<double>("strengthColor");
    double strengthGradient = opt.get<double>("strengthGradient");
    double zoomFactor = opt.get<double>("zoomFactor");
    unsigned int threads = opt.get<unsigned int>("threads");
    bool fastMode = opt.exist("fastMode");
    bool videoMode = opt.exist("videoMode");
    bool preview = opt.exist("preview");

    //Anime4K
    Anime4K anime4k(passes, strengthColor, strengthGradient, zoomFactor, fastMode, videoMode, threads);
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
