#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include "AC/Core.hpp"
#include "AC/Util/Stopwatch.hpp"
#include "AC/Util/ThreadPool.hpp"

struct ImagePool
{
    std::vector<ac::core::Image> images{};
    std::vector<int> indices{};

    ImagePool(const int w, const int h, const int batch, const int size)
    {
        std::random_device rd{};
        std::mt19937 gen{ rd() };
        std::uniform_int_distribution<int> indice{ 0, size - 1 };

        images.reserve(size);
        indices.reserve(batch);

        for (int idx = 0; idx < size; idx++)
        {
            ac::core::Image image{ w, h, 1, ac::core::Image::UInt8 };
            auto length = image.size() & -4;
            for (int i = 0; i < length; i += 4)
            {
                auto pixels = gen();
                std::memcpy(image.data() + i, &pixels, 4);
            }

            images.emplace_back(image);
        }

        for (int idx = 0; idx < batch; idx++) indices.emplace_back(indice(gen));
    }

    const ac::core::Image& select(const int idx) const
    {
        return images[indices[idx]];
    }
};

static void benchmark(const std::shared_ptr<ac::core::Processor>& processor, const ImagePool& images, int batch, std::size_t threads)
{
    for (int i = 0; i < std::max(static_cast<int>(batch * 0.05), 1); i++)
    {
        processor->process(images.select(i), 2.0);
        if (!processor->ok())
        {
            std::printf("%s\n", processor->error());
            return;
        }
    }

    ac::util::Stopwatch stopwatch{};
    if (threads > 1)
    {
        ac::util::ThreadPool pool{ threads };
        for (int i = 0; i < batch; i++) pool.exec([&, i]() { processor->process(images.select(i), 2.0); });
    }
    else for (int i = 0; i < batch; i++) processor->process(images.select(i), 2.0);
    stopwatch.stop();
    std::printf("FPS: %lf\n", batch / stopwatch.elapsed());
}

int main(int argc, char* argv[])
{
    std::printf("usage: [model] [processor] [device] [width] [height] [batch] [threads]\n");
    std::printf("\n");
    std::printf("%s\n", ac::core::Processor::listInfo());

    auto model = argc > 1 ? argv[1] : "acnet";
    auto processorType = argc > 2 ? argv[2] : "cpu";
    auto processor = ac::core::Processor::create(processorType, argc > 3 ? std::atoi(argv[3]) : 0, model);
    if (!processor->ok())
    {
        std::printf("%s\n", processor->error());
        return 0;
    }
    int w = argc > 4 ? std::max(std::atoi(argv[4]), 3) : 720;
    int h = argc > 5 ? std::max(std::atoi(argv[5]), 3) : 480;
    int batch = argc > 6 ? std::max(std::atoi(argv[6]), 1) : processor->type() == ac::core::Processor::CPU ? 60 : 600;
    int threads = argc > 7 ? std::max(std::atoi(argv[7]), 1) : ac::util::ThreadPool::hardwareThreads();

    std::printf("model: %s, processor: %s, device: %s, input: %d x %d x %d, threads: %d\n", model, processor->typeName(), processor->name(), w, h, batch, threads);

    // max image pool size: 128m
    ImagePool images{ w, h, batch, std::min(128 * 1024 * 1024 / (w * h), batch) };

    benchmark(processor, images, batch, threads);

    return 0;
}
