#include <atomic>
#include <cstddef>
#include <functional>
#include <queue>

#include "AC/Util/Channel.hpp"
#include "AC/Util/ThreadPool.hpp"
#include "AC/Video/Filter.hpp"

namespace ac::video::detail
{
    static inline void filterSerial(Pipeline& pipeline, bool (* const callback)(Frame& /*src*/, Frame& /*dst*/, void* /*userdata*/), void* const userdata)
    {
        Frame src{};
        Frame dst{};

        while (pipeline >> src)
        {
            bool ret = true;
            ret = pipeline.request(dst, src); if (!ret) break;

            ret = callback(src, dst, userdata); if (!ret) break;

            pipeline.release(src);
            ret = pipeline << dst; if (!ret) break;
            pipeline.release(dst);
        }
        // make sure that we have released all frames
        pipeline.release(src);
        pipeline.release(dst);
    }

    static inline void filterParallel(Pipeline& pipeline, bool (* const callback)(Frame& /*src*/, Frame& /*dst*/, void* /*userdata*/), void* const userdata)
    {
        auto threads = util::ThreadPool::hardwareThreads();
        util::Channel<Frame> decodeChan{ threads };
        util::AscendingChannel<Frame> encodeChan{ threads };
        util::ThreadPool pool{ threads + 1 };
        std::atomic_bool success = true;
        std::atomic_size_t remainingThreads = threads;

        pool.exec([&](){
            decltype(Frame::number) idx = 1;
            std::priority_queue<Frame, std::vector<Frame>, std::greater<Frame>> buffer{};
            auto process = [&](){
                Frame dst{};
                if (!(encodeChan >> dst)) return;
                if (dst.number != idx) buffer.emplace(dst);
                else
                {
                    if (success.load(std::memory_order_relaxed))
                    {
                        bool ret = pipeline << dst;
                        if (!ret)
                        {
                            bool expected = true;
                            success.compare_exchange_strong(expected, false, std::memory_order_relaxed, std::memory_order_relaxed);
                        }
                    }
                    pipeline.release(dst);
                    idx++;
                    while (!buffer.empty())
                    {
                        dst = buffer.top();
                        if (dst.number != idx) break;
                        else
                        {
                            buffer.pop();
                            if (success.load(std::memory_order_relaxed))
                            {
                                bool ret = pipeline << dst;
                                if (!ret)
                                {
                                    bool expected = true;
                                    success.compare_exchange_strong(expected, false, std::memory_order_relaxed, std::memory_order_relaxed);
                                }
                            }
                            pipeline.release(dst);
                            idx++;
                        }
                    }
                }
            };
            while(!encodeChan.isClosed()) process();
            while(!encodeChan.empty()) process();
        });

        for (std::size_t i = 0; i < threads; i++)
        {
            pool.exec([&](){
                auto process = [&](){
                    Frame src{};
                    Frame dst{};
                    if (!(decodeChan >> src)) return;
                    bool ret = pipeline.request(dst, src);
                    if (ret)
                    {
                        ret = callback(src, dst, userdata);
                        pipeline.release(src);
                        if (ret) encodeChan << dst;
                        else pipeline.release(dst);
                    }
                    else pipeline.release(src);
                    if (!success.load(std::memory_order_relaxed)) return;
                    if (!ret)
                    {
                        bool expected = true;
                        success.compare_exchange_strong(expected, false, std::memory_order_relaxed, std::memory_order_relaxed);
                    }
                };
                while (!decodeChan.isClosed()) process();
                while (!decodeChan.empty()) process();
                // last one close the door
                if(remainingThreads.fetch_sub(1, std::memory_order_relaxed) == 1) encodeChan.close();
            });
        }

        Frame src{};
        while (success.load(std::memory_order_relaxed) && pipeline >> src) decodeChan << src;
        decodeChan.close();
    }
}

void ac::video::filter(Pipeline& pipeline, bool (* const callback)(Frame& /*src*/, Frame& /*dst*/, void* /*userdata*/), void* const userdata, const int flag)
{
    if (!callback) return;

    if (flag == FILTER_PARALLEL || (flag == FILTER_AUTO && util::ThreadPool::hardwareThreads() > 1))
        detail::filterParallel(pipeline, callback, userdata);
    else
        detail::filterSerial(pipeline, callback, userdata);
}
