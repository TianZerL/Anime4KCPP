#include"VideoIOConcurrent.hpp"

#ifdef ENABLE_VIDEOIO_CONCURRENT

void Anime4KCPP::Video::VideoIOConcurrent::process()
{
    Parallel::task_group tasks;

    stop = false;

    finished = 0;

    tasks.run([&]()
        {
            double totalFrame = reader.get(cv::CAP_PROP_FRAME_COUNT);

            for (std::size_t frameCount = 0; finished == 0 || frameCount < finished; frameCount++)
            {
                std::unique_lock<std::mutex> lock(mtxWrite);
                std::unordered_map<std::size_t, cv::Mat>::iterator it;

                while (!stop && ((it = frameMap.find(frameCount)) == frameMap.end()))
                    cndWrite.wait(lock);

                if (stop)
                    return;

                writer.write(it->second);
                frameMap.erase(it);
                setProgress(static_cast<double>(frameCount) / totalFrame);
            }
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
        tasks.run(processor);
    }

    tasks.wait();
}

#endif
