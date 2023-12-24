#ifdef ENABLE_VIDEO

#include <exception>
#include <atomic>
#include <iostream>
#include <vector>
#include <mutex>
// #ifndef DISABLE_PARALLEL
// 	#include "oneapi/tbb/global_control.h"
// 	#include "oneapi/tbb/parallel_pipeline.h"
// 	#include "oneapi/tbb/tbb_allocator.h"
// 	#include "oneapi/tbb/tick_count.h"
// #endif

#include "ACCreator.hpp"
#include "VideoProcessor.hpp"
#include "VideoIOAsync.hpp"
#include "VideoIOThreads.hpp"
#include "VideoIOSerial.hpp"

Anime4KCPP::VideoProcessor::VideoProcessor(const Parameters& parameters, const Processor::Type type, const unsigned int threads)
    :fps(0.0), totalFrameCount(0.0), height(0), width(0), threads(threads), param(parameters), type(type)
{
#ifdef DISABLE_PARALLEL
    videoIO = std::make_unique<Video::VideoIOSerial>();
#else
    if (threads > 1)
        videoIO = std::make_unique<Video::VideoIOThreads>();
    else if (threads == 1)
        videoIO = std::make_unique<Video::VideoIOSerial>();
    else
        videoIO = std::make_unique<Video::VideoIOAsync>();
#endif
}

Anime4KCPP::VideoProcessor::VideoProcessor(const AC& config, const unsigned int threads)
    :VideoProcessor(config.getParameters(), config.getProcessorType(), threads) {}

void Anime4KCPP::VideoProcessor::loadVideo(const std::string& srcFile, const bool hw)
{
    if (!videoIO->openReader(srcFile, hw))
        throw ACException<ExceptionType::IO>("Failed to load file: file doesn't not exist or decoder isn't installed.");

    fps = videoIO->get(cv::CAP_PROP_FPS);
    totalFrameCount = videoIO->get(cv::CAP_PROP_FRAME_COUNT);
    height = static_cast<int>(std::round(param.zoomFactor * videoIO->get(cv::CAP_PROP_FRAME_HEIGHT)));
    width = static_cast<int>(std::round(param.zoomFactor * videoIO->get(cv::CAP_PROP_FRAME_WIDTH)));
}

void Anime4KCPP::VideoProcessor::setVideoSaveInfo(const std::string& dstFile, const Codec codec, const double fps, const bool hw)
{
    if (!videoIO->openWriter(dstFile, codec, cv::Size(width, height), fps, hw))
        throw ACException<ExceptionType::IO>("Failed to initialize video writer.");
}

void Anime4KCPP::VideoProcessor::saveVideo()
{
    videoIO->release();
}

void Anime4KCPP::VideoProcessor::process()
{
    std::once_flag eptrFlag;
    std::exception_ptr eptr;

	std::queue<AC*> creators;
	for (int i = 0; i < threads; i++)
	{
		creators.push(ACCreator::create(param, type));
	}

	std::mutex m;
	videoIO->init(
	    [&]()
	    {
	        Video::Frame frame;
	        videoIO->read(frame);
	
	        try
	        { // Reduce memory usage
				m.lock();
	            auto ac = creators.front(); creators.pop();
				m.unlock();
	            ac->loadImage(frame.first);
	            ac->process();
	            ac->saveImage(frame.first);

				m.lock();
				creators.push(ac);
				m.unlock();
	        }
	        catch (...)
	        {
	            std::call_once(eptrFlag,
	                [&]()
	                {
	                    videoIO->stopProcess();
	                    eptr = std::current_exception();
	                });
	            return;
	        }
	
	        videoIO->write(frame);
	    }
	, threads).process();

	// std::vector<Video::Frame> frames(threads);
	// int startedFrame = 0, finishedFrame = 0;
	// auto inputFilter = oneapi::tbb::make_filter<void, AC*>(
	// 	oneapi::tbb::filter_mode::serial_in_order,
	// 	[&](oneapi::tbb::flow_control& fc) -> AC* {
	// 		if (startedFrame < totalFrameCount)
	// 		{
	// 			assert(!creators.empty());
	// 			AC* ac = creators.front(); creators.pop();
	// 			auto& frame = frames.at(startedFrame % threads);
	// 			videoIO->read(frame);
	// 			std::cerr << "Frame " << startedFrame << " started" << std::endl;
	// 			ac->loadImage(frame.first);
	// 			std::cerr << "Frame " << startedFrame << " started" << std::endl;
	// 			startedFrame++;
	// 			// frames.at(startedFrame % threads) = std::move(frame);
	// 			return ac;
	// 		}
	// 		else
	// 		{
	// 			fc.stop();
	// 			return nullptr;
	// 		}
	// 	}
	// );
	// auto processFilter = oneapi::tbb::make_filter<AC*, AC*>(
	// 	oneapi::tbb::filter_mode::parallel,
	// 	[&](AC* ac) -> AC* {
	// 		std::cerr << "Processing..." << std::endl;
	// 		ac->process();
	// 		return ac;
	// 	}
	// );
	// auto outputFilter = oneapi::tbb::make_filter<AC*, void>(
	// 	oneapi::tbb::filter_mode::serial_in_order,
	// 	[&](AC* ac) {
	// 		auto& frame = frames.at(finishedFrame % threads);
	// 		ac->saveImage(frame.first);
	// 		videoIO->write(frame);
	// 		std::cerr << "Frame " << finishedFrame << " finished" << std::endl;
	// 		finishedFrame++;
	// 	}
	// );

	// oneapi::tbb::parallel_pipeline(threads, inputFilter & processFilter & outputFilter);

    if (eptr)
        std::rethrow_exception(eptr);
}

void Anime4KCPP::VideoProcessor::processWithProgress(const std::function<void(double)>&& callBack)
{
    std::future<void> p = std::async(std::launch::async, &VideoProcessor::process, this);
    std::chrono::milliseconds timeout(1000);
    for (;;)
    {
        std::future_status status = p.wait_for(timeout);
        if (status == std::future_status::ready)
        {
            callBack(1.0);
            p.get();
            break;
        }
        double progress = videoIO->getProgress();
        callBack(progress);
    }
}

void Anime4KCPP::VideoProcessor::stopVideoProcess() noexcept
{
    if (videoIO->isPaused())
        videoIO->continueProcess();

    videoIO->stopProcess();
}

void Anime4KCPP::VideoProcessor::pauseVideoProcess()
{
    videoIO->pauseProcess();
}

void Anime4KCPP::VideoProcessor::continueVideoProcess()
{
    videoIO->continueProcess();
}

std::string Anime4KCPP::VideoProcessor::getInfo() const
{
    std::ostringstream oss;
    oss << "----------------------------------------------" << '\n'
        << "Video information" << '\n'
        << "----------------------------------------------" << '\n'
        << "FPS: " << fps << '\n'
        << "Threads: " << threads << '\n'
        << "Total frames: " << totalFrameCount << '\n'
        << "----------------------------------------------" << '\n';
    return oss.str();
}

#endif // ENABLE_VIDEO
