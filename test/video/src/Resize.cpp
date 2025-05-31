#include <cstdio>

#include "AC/Core.hpp"
#include "AC/Video.hpp"
#include "AC/Util/Stopwatch.hpp"

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::printf("usage: input output [encoder] [format]\n");
        return 0;
    }

    ac::video::Pipeline pipeline{};
    if (!pipeline.openDecoder(argv[1]))
    {
        std::printf("Failed to open decoder");
        return 0;
    }
    if (!pipeline.openEncoder(argv[2], 2.0, { argc >= 3 ? argv[3] : nullptr, argc >= 4 ? argv[4] : nullptr, 0 }))
    {
        std::printf("Failed to open encoder");
        return 0;
    }

    auto info = pipeline.getInfo();

    struct {
        double frames;
    } data{};
    data.frames = info.fps * info.duration;

    ac::util::Stopwatch stopwatch{};
    ac::video::filter(pipeline, [](ac::video::Frame& src, ac::video::Frame& dst, void* userdata) -> bool {
        auto ctx = static_cast<decltype(data)*>(userdata);
        for (int i = 0; i < src.planes; i++)
        {
            ac::core::Image srcp{src.plane[i].width, src.plane[i].height, src.plane[i].channel, src.elementType, src.plane[i].data, src.plane[i].stride};
            ac::core::Image dstp{dst.plane[i].width, dst.plane[i].height, dst.plane[i].channel, dst.elementType, dst.plane[i].data, dst.plane[i].stride};
            ac::core::resize(srcp, dstp, 0.0, 0.0);
        }
        if (src.number % 32 == 0)
        {
            std::printf("%lf%%\r", 100 * src.number / ctx->frames); // printf is thread safe
            std::fflush(stdout);
        }
        return true;
    }, &data, ac::video::FILTER_AUTO);
    stopwatch.stop();

    std::printf("it takes %lfs\n", stopwatch.elapsed());

    pipeline.close();

    return 0;
}
