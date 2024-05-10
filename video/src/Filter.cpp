#include <atomic>
#include <cstddef>
#include <functional>
#include <queue>

#include "AC/Util/Channel.hpp"
#include "AC/Util/ThreadPool.hpp"
#include "AC/Video/Filter.hpp"

namespace ac::video::detail
{
    inline static void filterSerial(Pipeline& pipeline, void (* const callback)(Frame& /*src*/, Frame& /*dst*/, void* /*userdata*/), void* const userdata)
    {
        Frame src{};
        Frame dst{};

        while (pipeline >> src)
        {
            pipeline.request(dst, src);

            callback(src, dst, userdata);

            pipeline.release(src);
            pipeline << dst;
        }
        pipeline.remux();
    }

    inline static void filterParallel(Pipeline& pipeline, void (* const callback)(Frame& /*src*/, Frame& /*dst*/, void* /*userdata*/), void* const userdata)
    {
        std::atomic_size_t threads = util::ThreadPool::hardwareThreads();
        util::Channel<Frame> decodeChan{ threads };
        util::AscendingChannel<Frame> encodeChan{ threads };
        util::ThreadPool pool{ threads + 1 };

        pool.exec([&](){
            int idx = 1;
            std::priority_queue<Frame, std::vector<Frame>, std::greater<Frame>> buffer{};
            auto process = [&](){
                Frame dst{};
                encodeChan >> dst;
                if (!dst.ref) return;
                if (dst.number != idx) buffer.emplace(dst);
                else
                {
                    pipeline << dst;
                    idx++;
                    while (!buffer.empty())
                    {
                        dst = buffer.top();
                        if (dst.number != idx) break;
                        else
                        {
                            buffer.pop();
                            pipeline << dst;
                            idx++;
                        }
                    }
                }
            };
            while(!encodeChan.isClose()) process();
            while(!encodeChan.empty()) process();
            pipeline.remux();
        });

        for (std::size_t i = 0; i < threads; i++)
        {
            pool.exec([&](){
                auto process = [&](){
                    Frame src{};
                    Frame dst{};
                    decodeChan >> src;
                    if (!src.ref) return;
                    pipeline.request(dst, src);
                    callback(src, dst, userdata);
                    pipeline.release(src);
                    encodeChan << dst;
                };
                while (!decodeChan.isClose()) process();
                while (!decodeChan.empty()) process();
                // last one close the door
                if(--threads == 0) encodeChan.close();
            });
        }

        Frame src{};
        while (pipeline >> src) decodeChan << src;
        decodeChan.close();
    }
}

void ac::video::filter(Pipeline& pipeline, void (* const callback)(Frame& /*src*/, Frame& /*dst*/, void* /*userdata*/), void* const userdata, const int flag)
{
    if (!callback) return;

    if (flag == FILTER_PARALLEL || (flag == FILTER_AUTO && util::ThreadPool::hardwareThreads() > 1))
        detail::filterParallel(pipeline, callback, userdata);
    else
        detail::filterSerial(pipeline, callback, userdata);
}
