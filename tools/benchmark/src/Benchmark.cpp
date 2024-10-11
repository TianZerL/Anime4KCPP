#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "AC/Core.hpp"
#include "AC/Util/Stopwatch.hpp"
#include "AC/Util/ThreadPool.hpp"

static void benchmark(const std::shared_ptr<ac::core::Processor>& processor, const ac::core::Image& image, int batch)
{
    ac::util::Stopwatch stopwatch{};
    for (int i = 0; i < batch; i++) processor->process(image, 2.0);
    stopwatch.stop();
    std::printf("%s: serial average FPS %lf\n", processor->name(), batch / stopwatch.elapsed());
}

static void benchmarkParallel(const std::shared_ptr<ac::core::Processor>& processor, const ac::core::Image& image, int batch, std::size_t threads)
{
    ac::util::Stopwatch stopwatch{};
    {
        ac::util::ThreadPool pool{ threads };
        for (int i = 0; i < batch; i++) pool.exec([&]() { processor->process(image, 2.0); });
    }
    stopwatch.stop();
    std::printf("%s: parallel average FPS %lf\n", processor->name(), batch / stopwatch.elapsed());
}

int main(int argc, char* argv[])
{
    std::printf("usage: [processor] [device] [width] [height] [batch] [threads]\n");

    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::uniform_int_distribution<unsigned short> d{ 0, 255 };

    auto processor = [&]() {
        ac::core::model::ACNet model{ ac::core::model::ACNet::Variant::HDN0 };
        int device = argc > 2 ? std::atoi(argv[2]) : 0;
        if (argc > 1)
        {
#       ifdef AC_CORE_WITH_OPENCL
            if (!std::strcmp(argv[1], "opencl")) return ac::core::Processor::create<ac::core::Processor::OpenCL>(device, model);
#       endif
#       ifdef AC_CORE_WITH_CUDA
            if (!std::strcmp(argv[1], "cuda")) return ac::core::Processor::create<ac::core::Processor::CUDA>(device, model);
#       endif
        }
        return ac::core::Processor::create<ac::core::Processor::CPU>(device, model);
    }();

    int w = argc > 3 ? std::atoi(argv[3]) : 720;
    int h = argc > 4 ? std::atoi(argv[4]) : 480;
    int batch = argc > 5 ? std::atoi(argv[5]) : 60;
    int threads = argc > 6 ? std::atoi(argv[6]) : 0;

    std::printf("random images: %d x %d x %d\n", w, h, batch);

    ac::core::Image image{};

    image.create(w, h, 1, ac::core::Image::UInt8);
    for (int i = 0; i < image.height(); i++)
        for (int j = 0; j < image.width(); j++)
            *image.pixel(j, i) = static_cast<std::uint8_t>(d(gen));

    benchmark(processor, image, batch);
    benchmarkParallel(processor, image, batch, threads > 0 ? threads : (ac::util::ThreadPool::hardwareThreads() + 1));

    return 0;
}
