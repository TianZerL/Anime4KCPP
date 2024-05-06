#include <cstdio>
#include <cstdint>
#include <memory>
#include <random>

#include "AC/Core.hpp"
#include "AC/Util/Stopwatch.hpp"

template<int batch>
static void benchmark(const std::shared_ptr<ac::core::Processor>& processor, const ac::core::Image (&images)[batch])
{
    double total = 0.0;
    ac::util::Stopwatch stopwatch{};

    for(auto&& src : images)
    {
        stopwatch.reset();
        auto dst = processor->process(src, 2.0);
        stopwatch.stop();
        total += stopwatch.elapsed();
    }

    std::printf("%s: average FPS %lfs\n", processor->name(), batch / total);
}

int main(int argc, const char* argv[])
{
    std::random_device rd{};
    std::mt19937 gen{ rd() };
    std::uniform_int_distribution<unsigned short> d{ 0, 255 };

    auto processor = ac::core::Processor::create<ac::core::Processor::CPU>(0, ac::core::model::ACNet{ ac::core::model::ACNet::Variant::HDN0 });

    ac::core::Image images[60]{};

    for(auto&& image : images)
    {
        image.create(720, 480, 1, ac::core::Image::UInt8);
        for (int i = 0; i < image.height(); i++)
            for (int j = 0; j < image.width(); j++)
                *image.pixel(j, i) = static_cast<std::uint8_t>(d(gen));
    }

    benchmark(processor, images);

    return 0;
}
