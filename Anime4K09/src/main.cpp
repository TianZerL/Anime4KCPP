#include "Anime4K.h"
#include<cmdline.h>

#include<iostream>
#include<ctime>

int main(int argc,char *argv[]) 
{
    cmdline::parser opt;
    opt.add<std::string>("input", 'i', "File for loading", false, "./pic/p1.png");
    opt.add<std::string>("output", 'o', "File for outputting", false, "output.png");
    opt.add<int>("passes", 'p', "Passes for processing", false, 2);
    opt.add<double>("strengthColor", 'c', "Strength for pushing color,range 0 to 1,higher for thinner", false, 0.3, cmdline::range(0.0, 1.0));
    opt.add<double>("strengthGradient", 'g', "Strength for pushing gradient,range 0 to 1,higher for sharper", false, 1.0, cmdline::range(0.0, 1.0));
    opt.add("fastMode", 'f', "Faster but maybe low quality");

    opt.parse_check(argc, argv);

    std::string input = opt.get<std::string>("input");
    std::string output = opt.get<std::string>("output");
    int passes = opt.get<int>("passes");
    double strengthColor = opt.get<double>("strengthColor");
    double strengthGradient = opt.get<double>("strengthGradient");
    bool fastMode = opt.exist("fastMode");

    Anime4K anime4k(passes, strengthColor, strengthGradient, fastMode);

    try
    {
        anime4k.loadImage(input);
    }
    catch (const char* err)
    {
        std::cout << err << std::endl;
    }

    anime4k.showInfo();

    time_t s = std::clock();
    anime4k.process();
    time_t e = std::clock();
    std::cout <<"Total process time: "<< double(e - s) / CLOCKS_PER_SEC << "s" << std::endl;

    anime4k.showImg();
    anime4k.saveImage(output);

    return 0;
}