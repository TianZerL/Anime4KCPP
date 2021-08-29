#ifdef ENABLE_VIDEO

#include"VideoIOThreads.hpp"
#include"ThreadPool.hpp"

void Anime4KCPP::Video::VideoIOThreads::process()
{
    std::promise<void> barrier;
    std::future<void> barrierFuture = barrier.get_future();

    Utils::ThreadPool pool(threads + 1);

    stop = false;

    finished = 0;

    pool.exec([this, &barrier]()
        {
            double totalFrame = reader.get(cv::CAP_PROP_FRAME_COUNT);

            for (std::size_t frameCount = 0; finished == 0 || frameCount < finished; frameCount++)
            {
                std::unique_lock<std::mutex> lock(mtxWrite);
                std::unordered_map<std::size_t, cv::Mat>::iterator it;

                while (!stop && ((it = frameMap.find(frameCount)) == frameMap.end()))
                    cndWrite.wait(lock);

                if (stop)
                    return barrier.set_value();

                writer.write(it->second);
                frameMap.erase(it);
                setProgress(static_cast<double>(frameCount) / totalFrame);
            }

            barrier.set_value();
        });


    for (std::size_t frameCount = 0;; frameCount++)
    {
        cv::Mat frame;
        if (!reader.read(frame))
        {
            finished = frameCount;
            break;
        }
        {
            std::unique_lock<std::mutex> lock(mtxRead);

            while (!stop && rawFrames.size() >= threads)
                cndRead.wait(lock);

            if (stop)
                break;

            rawFrames.emplace(frame, frameCount);
        }
        pool.exec(processor);
    }

    barrierFuture.wait();
}

#endif
