#ifndef ANIME4KCPP_CORE_VIDEO_IO_CONCURRENT_HPP
#define ANIME4KCPP_CORE_VIDEO_IO_CONCURRENT_HPP

#if defined(ENABLE_VIDEO) && !defined(DISABLE_PARALLEL) && (defined(USE_PPL) || defined(USE_TBB))

#define ENABLE_VIDEOIO_CONCURRENT

#if defined(USE_PPL)
#include<ppl.h>
namespace Parallel = Concurrency;
#elif defined(USE_TBB)
#include<tbb/tbb.h>
namespace Parallel = tbb;
#endif

#include"VideoIO.hpp"

namespace Anime4KCPP::Video
{
    class VideoIOConcurrent;
}

class Anime4KCPP::Video::VideoIOConcurrent :public VideoIO
{
public:
    void process() override;
private:
    std::atomic<std::size_t> finished;
};

#endif // ENABLE_VIDEO

#endif // !ANIME4KCPP_CORE_VIDEO_IO_CONCURRENT_HPP
