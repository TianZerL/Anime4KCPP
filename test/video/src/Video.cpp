#include <cstdio>

#include "AC/Core.hpp"
#include "AC/Video.hpp"
#include "AC/Util/Stopwatch.hpp"

int main(int argc, const char* argv[])
{
    if (argc < 3)
    {
        std::printf("usage: input output [encoder]\n");
        return 0;
    }

    ac::video::Pipeline pipeline{};
    if (!pipeline.openDecoder(argv[1]))
    {
        std::printf("Failed to open decoder");
        return 0;
    }
    if (!pipeline.openEncoder(argv[2], 2.0, { argc >= 3 ? argv[3] : nullptr, 0 }))
    {
        std::printf("Failed to open encoder");
        return 0;
    }

    auto info = pipeline.getInfo();
    double total = info.fps * info.length;

    struct {
        double total{};
    } data;
    data.total = total;

    ac::util::Stopwatch stopwatch{};
    ac::video::filter(pipeline, [](ac::video::Frame& src, ac::video::Frame& dst, void* userdata) {
        auto ctx = static_cast<decltype(data)*>(userdata);
        for (int i = 0; i < 3; i++)
        {
            ac::core::Image srcp{src.planar[i].width, src.planar[i].height, 1, src.elementType, src.planar[i].data, src.planar[i].stride};
            ac::core::Image dstp{dst.planar[i].width, dst.planar[i].height, 1, dst.elementType, dst.planar[i].data, dst.planar[i].stride};
            ac::core::resize(srcp, dstp, 0.0, 0.0);
        }
        if (src.number % 32 == 0) std::printf("%lf%\r", 100 * src.number / ctx->total); // printf is thread safe
    }, &data, ac::video::FILTER_AUTO);
    stopwatch.stop();

    std::printf("it takes %lfs\n", stopwatch.elapsed());

    pipeline.close();

    return 0;
}
