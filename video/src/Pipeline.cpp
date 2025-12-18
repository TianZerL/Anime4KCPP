#include <memory>
#include <sstream>
#include <utility>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

#include "AC/Util/Defer.hpp"
#include "AC/Video/Pipeline.hpp"

namespace ac::video
{
    struct FrameData
    {
        AVFrame* frame = nullptr;
        std::vector<AVPacket*> packets{};
    };

    namespace detail
    {
        class PipelineImpl
        {
        public:
            PipelineImpl() noexcept;
            ~PipelineImpl() noexcept;

            bool openDecoder(const char* filename, const DecoderHints& hints) noexcept;
            bool openEncoder(const char* filename, double factor, const EncoderHints& hints) noexcept;
            bool decode(Frame& dst) noexcept;
            bool encode(const Frame& src) noexcept;
            bool request(Frame& dst, const Frame& src) const noexcept;
            void release(Frame& frame) noexcept;
            void close() noexcept;
            Info getInfo() const noexcept;

        private:
            bool fetch() noexcept;
            bool remux(const std::vector<AVPacket*>& packets) const noexcept;

        private:
            static void fill(Frame& dst, AVFrame* src, std::vector<AVPacket*>& packets) noexcept;
            static Info::BitDepth getBitDepth(AVPixelFormat format) noexcept;

        private:
            bool writeHeaderFlag = false;
            SwsContext* swsCtx = nullptr;
            AVFormatContext* dfmtCtx = nullptr;
            AVFormatContext* efmtCtx = nullptr;
            AVPacket* dpacket = nullptr;
            AVPacket* epacket = nullptr;
            AVCodecContext* decoderCtx = nullptr;
            AVCodecContext* encoderCtx = nullptr;
            AVStream* dvideoStream = nullptr;
            AVStream* evideoStream = nullptr;
            AVRational timeBase{}; // should be 1/fps
            std::vector<int> streamIdxMap{};
            std::vector<AVPacket*> packetBuffer{};
        };

        PipelineImpl::PipelineImpl() noexcept = default;
        PipelineImpl::~PipelineImpl() noexcept
        {
            close();
        }

        inline bool PipelineImpl::openDecoder(const char* const filename, const DecoderHints& hints) noexcept
        {
            int ret = 0;
            dpacket = av_packet_alloc(); if (!dpacket) return false;
            ret = avformat_open_input(&dfmtCtx, filename, nullptr, nullptr); if (ret < 0) return false;

            ret = avformat_find_stream_info(dfmtCtx, nullptr); if (ret < 0) return false;
            for (unsigned int i = 0; i < dfmtCtx->nb_streams; i++)
                if (dfmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
                {
                    dvideoStream = dfmtCtx->streams[i];
                    break;
                }
            if (!dvideoStream) return false;

            auto codec = (hints.decoder && *hints.decoder) ? avcodec_find_decoder_by_name(hints.decoder) : avcodec_find_decoder(dvideoStream->codecpar->codec_id); if (!codec) return false;
            decoderCtx = avcodec_alloc_context3(codec); if (!decoderCtx) return false;
            ret = avcodec_parameters_to_context(decoderCtx, dvideoStream->codecpar); if (ret < 0) return false;
            decoderCtx->pkt_timebase = dvideoStream->time_base;
            ret = avcodec_open2(decoderCtx, codec, nullptr); if (ret < 0) return false;
            auto framerate = av_guess_frame_rate(dfmtCtx, dvideoStream, nullptr);
            timeBase = av_inv_q(framerate.num ? framerate : av_make_q(24000, 1001));
            return true;
        }
        inline bool PipelineImpl::openEncoder(const char* const filename, const double factor, const EncoderHints& hints) noexcept
        {
            int ret = 0;
            epacket = av_packet_alloc(); if (!epacket) return false;
            ret = avformat_alloc_output_context2(&efmtCtx, av_guess_format(nullptr, filename, nullptr), "matroska", filename); if (ret < 0) return false;

            AVPixelFormat targetPixFmt = AV_PIX_FMT_NONE;
            if (hints.format && *hints.format) targetPixFmt = av_get_pix_fmt(hints.format);
            auto codec = (hints.encoder && *hints.encoder) ? avcodec_find_encoder_by_name(hints.encoder) : avcodec_find_encoder(AV_CODEC_ID_H264); if (!codec) return false;
            encoderCtx = avcodec_alloc_context3(codec); if (!encoderCtx) return false;

            encoderCtx->pix_fmt = targetPixFmt != AV_PIX_FMT_NONE ? targetPixFmt : decoderCtx->pix_fmt;
            switch (encoderCtx->pix_fmt)
            {
            case AV_PIX_FMT_GRAY8:
            case AV_PIX_FMT_GRAY10:
            case AV_PIX_FMT_GRAY12:
            case AV_PIX_FMT_GRAY16:
            // planar
            case AV_PIX_FMT_YUV420P:
            case AV_PIX_FMT_YUV422P:
            case AV_PIX_FMT_YUV444P:
            case AV_PIX_FMT_YUV420P10:
            case AV_PIX_FMT_YUV422P10:
            case AV_PIX_FMT_YUV444P10:
            case AV_PIX_FMT_YUV420P12:
            case AV_PIX_FMT_YUV422P12:
            case AV_PIX_FMT_YUV444P12:
            case AV_PIX_FMT_YUV420P16:
            case AV_PIX_FMT_YUV422P16:
            case AV_PIX_FMT_YUV444P16:
            // packed
            case AV_PIX_FMT_NV12:
            case AV_PIX_FMT_NV21:
            case AV_PIX_FMT_NV16:
            case AV_PIX_FMT_NV24:
            case AV_PIX_FMT_NV42:
            case AV_PIX_FMT_P010:
            case AV_PIX_FMT_P210:
            case AV_PIX_FMT_NV20:
            case AV_PIX_FMT_P410:
            case AV_PIX_FMT_P012:
            case AV_PIX_FMT_P212:
            case AV_PIX_FMT_P412:
            case AV_PIX_FMT_P016:
            case AV_PIX_FMT_P216:
            case AV_PIX_FMT_P416: break;
            default: return false;
            }
            // just let encoder to choose a profile
#       if LIBAVCODEC_VERSION_MAJOR < 60 // ffmpeg 6, libavcodec 60
            encoderCtx->profile = FF_PROFILE_UNKNOWN;
#       else
            encoderCtx->profile = AV_PROFILE_UNKNOWN;
#       endif
            encoderCtx->bit_rate = hints.bitrate > 0 ? hints.bitrate : static_cast<decltype(encoderCtx->bit_rate)>(decoderCtx->bit_rate * factor * factor);
            encoderCtx->rc_max_rate = encoderCtx->bit_rate * 2; // is this too big?
            encoderCtx->rc_buffer_size = encoderCtx->rc_max_rate * 5; // 10s
            encoderCtx->gop_size = static_cast<decltype(encoderCtx->gop_size)>(10 * av_q2d(decoderCtx->framerate) + 0.5); // 10s gop size, maybe we should use dynamic gop size.
            encoderCtx->time_base = timeBase;
            encoderCtx->width = static_cast<decltype(encoderCtx->width)>(decoderCtx->width * factor);
            encoderCtx->height = static_cast<decltype(encoderCtx->height)>(decoderCtx->height * factor);
            encoderCtx->sample_aspect_ratio = decoderCtx->sample_aspect_ratio;
            encoderCtx->color_primaries = decoderCtx->color_primaries;
            encoderCtx->color_trc = decoderCtx->color_trc;
            encoderCtx->colorspace = decoderCtx->colorspace;
            encoderCtx->color_range = decoderCtx->color_range;
            if (efmtCtx->oformat->flags & AVFMT_GLOBALHEADER) encoderCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
            ret = avcodec_open2(encoderCtx, codec, nullptr); if (ret < 0) return false;
            // copy all streams
            streamIdxMap.resize(dfmtCtx->nb_streams);
            int streamIdx = 0;
            for (unsigned int i = 0; i < dfmtCtx->nb_streams; i++)
            {
                bool isOtherStream = (dfmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_ATTACHMENT) || (dfmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_DATA);
                if ((isOtherStream && (strcmp(efmtCtx->oformat->name, "matroska") != 0)) || // check if mkv
                    (!isOtherStream && (avformat_query_codec(efmtCtx->oformat, dfmtCtx->streams[i]->codecpar->codec_id, FF_COMPLIANCE_NORMAL) < 1))) // check if the given container can store a codec
                {
                    streamIdxMap[i] = -1;
                    continue;
                }
                streamIdxMap[i] = streamIdx++;
                auto stream = avformat_new_stream(efmtCtx, nullptr); if (!stream) return false;
                if (dfmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) evideoStream = stream;
                else avcodec_parameters_copy(stream->codecpar, dfmtCtx->streams[i]->codecpar); // copy stream info
                stream->codecpar->codec_tag = 0; // avoid unnecessary additional codec tag checks for MKV
                stream->time_base = dfmtCtx->streams[i]->time_base;
                stream->duration = dfmtCtx->streams[i]->duration;
                stream->disposition = dfmtCtx->streams[i]->disposition; // a series of flags that tells a player or media player how to handle a stream
                stream->sample_aspect_ratio = dfmtCtx->streams[i]->sample_aspect_ratio; // for mkv to keep DAR
                stream->avg_frame_rate = dfmtCtx->streams[i]->avg_frame_rate;
                av_dict_copy(&stream->metadata, dfmtCtx->streams[i]->metadata, 0); // keep metadata
            }
            // copy chapters
            if (dfmtCtx->nb_chapters > 0)
            {
                efmtCtx->chapters = static_cast<AVChapter**>(av_malloc_array(dfmtCtx->nb_chapters, sizeof(AVChapter*)));
                if (efmtCtx->chapters)
                {
                    unsigned int idx{};
                    for (idx = 0; idx < dfmtCtx->nb_chapters; idx++)
                    {
                        efmtCtx->chapters[idx] = static_cast<AVChapter*>(av_mallocz(sizeof(AVChapter)));
                        if (!efmtCtx->chapters[idx]) break;
                        *efmtCtx->chapters[idx] = *dfmtCtx->chapters[idx];
                        efmtCtx->chapters[idx]->metadata = nullptr;
                        av_dict_copy(&efmtCtx->chapters[idx]->metadata, dfmtCtx->chapters[idx]->metadata, 0);
                    }
                    efmtCtx->nb_chapters = idx;
                }
            }

            ret = avcodec_parameters_from_context(evideoStream->codecpar, encoderCtx); if (ret < 0) return false;
            ret = avio_open2(&efmtCtx->pb, filename, AVIO_FLAG_WRITE, &efmtCtx->interrupt_callback, nullptr); if (ret < 0) return false;
            ret = avformat_write_header(efmtCtx, nullptr); if (ret < 0) return false;
            writeHeaderFlag = true;

            return true;
        }
        inline bool PipelineImpl::decode(Frame& dst) noexcept
        {
            int ret = 0;
            auto frame = av_frame_alloc(); if (!frame) return false;
            bool finishFrame = false;
            util::Defer deferFreeFrame { [&]() { if (!finishFrame) av_frame_free(&frame); } };
            for (;;)
            {
                ret = avcodec_receive_frame(decoderCtx, frame);
                if (ret == 0) break;
                else if (ret == AVERROR(EAGAIN) && fetch()) continue;
                else return false;
            }

            if (!swsCtx && (frame->format != encoderCtx->pix_fmt))// I think it can be assumed that the frame format will not change during decoding.
            {
                swsCtx = sws_getContext(frame->width, frame->height, static_cast<AVPixelFormat>(frame->format), frame->width, frame->height, encoderCtx->pix_fmt, SWS_FAST_BILINEAR | SWS_PRINT_INFO, nullptr, nullptr, nullptr);
                if (!swsCtx) return false;
            }
            if (swsCtx)
            {
                auto dstFrame = av_frame_alloc(); if (!dstFrame) return false;
                bool finishDstFrame = false;
                util::Defer deferFreeDstFrame{ [&]() { if (!finishDstFrame) av_frame_free(&dstFrame); } };
                ret = av_frame_copy_props(dstFrame, frame); if (ret < 0) return false;
                dstFrame->width = frame->width;
                dstFrame->height = frame->height;
                dstFrame->format = encoderCtx->pix_fmt;
    #   if LIBAVUTIL_VERSION_MAJOR < 57 // ffmpeg 5, libavutil 57
                ret = av_frame_get_buffer(dstFrame, 0); if (ret < 0) return false;
                if (sws_scale(swsCtx, frame->data, frame->linesize, 0, frame->height, dstFrame->data, dstFrame->linesize) == dstFrame->height)
    #   else
                if (sws_scale_frame(swsCtx, dstFrame, frame) >= 0)
    #   endif
                {
                    av_frame_free(&frame);
                    frame = dstFrame;
                    finishDstFrame = true;
                }
                else return false;
            }

            finishFrame = true;
            fill(dst, frame, packetBuffer);

    #   if LIBAVCODEC_VERSION_MAJOR < 60 // ffmpeg 6, libavcodec 60
            dst.number = decoderCtx->frame_number;
    #   else
            dst.number = decoderCtx->frame_num;
    #   endif
            return true;
        }
        inline bool PipelineImpl::encode(const Frame& src) noexcept
        {
            int ret = 0;
            if (!src.dptr) return false;
            if (!remux(src.dptr->packets)) return false;
            ret = avcodec_send_frame(encoderCtx, src.dptr->frame); if (ret < 0) return false;
            for (;;)
            {
                ret = avcodec_receive_packet(encoderCtx, epacket);
                if (ret == AVERROR(EAGAIN)) break;
                else if (ret < 0) return false;
                av_packet_rescale_ts(epacket, encoderCtx->time_base, evideoStream->time_base);
                epacket->stream_index = evideoStream->index;
                ret = av_interleaved_write_frame(efmtCtx, epacket); if (ret < 0) return false;
            }
            return true;
        }
        inline bool PipelineImpl::request(Frame& dst, const Frame& src) const noexcept
        {
            if (!src.dptr) return false;
            auto srcFrame = src.dptr->frame;
            auto dstFrame = av_frame_alloc(); if (!dstFrame) return false;

            dstFrame->width = encoderCtx->width;
            dstFrame->height = encoderCtx->height;
            dstFrame->format = encoderCtx->pix_fmt;

            dstFrame->pts = srcFrame->pts;

            dstFrame->color_primaries = srcFrame->color_primaries;
            dstFrame->color_trc = srcFrame->color_trc;
            dstFrame->colorspace = srcFrame->colorspace;
            dstFrame->color_range = srcFrame->color_range;

            dstFrame->sample_aspect_ratio = srcFrame->sample_aspect_ratio;

            if (av_frame_get_buffer(dstFrame, 0) < 0)
            {
                av_frame_free(&dstFrame);
                return false;
            }
            // copy some side datas (basically for HDR)
            for (int i = 0; i < srcFrame->nb_side_data; i++)
            {
                const AVFrameSideData* srcSideData = srcFrame->side_data[i];
                switch (srcSideData->type)
                {
                case AV_FRAME_DATA_A53_CC:
                case AV_FRAME_DATA_MASTERING_DISPLAY_METADATA:
                case AV_FRAME_DATA_CONTENT_LIGHT_LEVEL: break;
                default: continue;
                }
                AVBufferRef* bufferRef = av_buffer_ref(srcSideData->buf);
                if (!bufferRef) continue;
                AVFrameSideData* dstSideData = av_frame_new_side_data_from_buf(dstFrame, srcSideData->type, bufferRef);
                if (!dstSideData) av_buffer_unref(&bufferRef);
                else av_dict_copy(&dstSideData->metadata, srcSideData->metadata, 0);
            }

            fill(dst, dstFrame, src.dptr->packets);
            dst.number = src.number;
            return true;
        }
        inline void PipelineImpl::release(Frame& frame) noexcept
        {
            if (frame.dptr)
            {
                av_frame_free(&frame.dptr->frame);
                for (auto packet : frame.dptr->packets) av_packet_free(&packet);
                frame.dptr.reset();
            }
        }
        inline void PipelineImpl::close() noexcept
        {
            if (!packetBuffer.empty())
            {
                remux(packetBuffer);
                for (auto packet : packetBuffer) av_packet_free(&packet);
                packetBuffer.clear();
            }
            if (writeHeaderFlag)
            {
                av_write_trailer(efmtCtx);
                writeHeaderFlag = false;
            }
            if (swsCtx)
            {
                sws_freeContext(swsCtx);
                swsCtx = nullptr;
            }
            if (encoderCtx) avcodec_free_context(&encoderCtx);
            if (decoderCtx) avcodec_free_context(&decoderCtx);
            if (efmtCtx)
            {
                avio_closep(&efmtCtx->pb);
                avformat_free_context(efmtCtx);
                efmtCtx = nullptr;
            }
            if (dfmtCtx) avformat_close_input(&dfmtCtx);
            if (epacket) av_packet_free(&epacket);
            if (dpacket) av_packet_free(&dpacket);

            streamIdxMap.clear();
            timeBase = {};
            evideoStream = nullptr;
            dvideoStream = nullptr;
        }
        inline Info PipelineImpl::getInfo() const noexcept
        {
            Info info{};
            info.width = decoderCtx->width;
            info.height = decoderCtx->height;
            info.bitDepth = getBitDepth(encoderCtx->pix_fmt);
            info.fps = av_q2d(av_inv_q(timeBase));

            if (dvideoStream->duration != AV_NOPTS_VALUE) info.duration = dvideoStream->duration * av_q2d(dvideoStream->time_base);
            else if (auto duration = av_dict_get(dvideoStream->metadata, "DURATION", nullptr, 0); duration != nullptr)
            {
                int hours = 0, minutes = 0;
                double secones = 0.0;

                std::istringstream iss{ duration->value };
                iss >> hours;
                iss.ignore();
                iss >> minutes;
                iss.ignore();
                iss >> secones;

                info.duration = hours * 3600 + minutes * 60 + secones;
            }
            else info.duration = dfmtCtx->duration * av_q2d(AV_TIME_BASE_Q);

            return info;
        }

        inline bool PipelineImpl::fetch() noexcept
        {
            while (av_read_frame(dfmtCtx, dpacket) >= 0)
            {
                if (dpacket->stream_index == dvideoStream->index)
                {
                    av_packet_rescale_ts(dpacket, dvideoStream->time_base, timeBase);
                    int ret = avcodec_send_packet(decoderCtx, dpacket);
                    av_packet_unref(dpacket);
                    return ret == 0;
                }
                else packetBuffer.emplace_back(av_packet_clone(dpacket));
                av_packet_unref(dpacket);
            }
            return avcodec_send_packet(decoderCtx, nullptr) == 0;
        }
        inline bool PipelineImpl::remux(const std::vector<AVPacket*>& packets) const noexcept
        {
            for (auto packet : packets)
            {
                if (streamIdxMap[packet->stream_index] < 0) continue;

                av_packet_rescale_ts(packet, dfmtCtx->streams[packet->stream_index]->time_base, efmtCtx->streams[streamIdxMap[packet->stream_index]]->time_base);
                packet->stream_index = streamIdxMap[packet->stream_index];
                if (av_interleaved_write_frame(efmtCtx, packet) < 0) return false;
            }
            return true;
        }

        inline void PipelineImpl::fill(Frame& dst, AVFrame* const src, std::vector<AVPacket*>& packets) noexcept
        {
            int wscale = 2, hscale = 2, elementSize = sizeof(std::uint8_t);
            int planes = 3;
            switch (src->format)
            {
            case AV_PIX_FMT_GRAY8: planes = 1; break;
            case AV_PIX_FMT_GRAY10:
            case AV_PIX_FMT_GRAY12:
            case AV_PIX_FMT_GRAY16: elementSize = sizeof(std::uint16_t); planes = 1; break;
            // planar
            case AV_PIX_FMT_YUV444P: wscale = 1; [[fallthrough]];
            case AV_PIX_FMT_YUV422P: hscale = 1; break;
            case AV_PIX_FMT_YUV444P10:
            case AV_PIX_FMT_YUV444P12:
            case AV_PIX_FMT_YUV444P16: wscale = 1; [[fallthrough]];
            case AV_PIX_FMT_YUV422P10:
            case AV_PIX_FMT_YUV422P12:
            case AV_PIX_FMT_YUV422P16: hscale = 1; [[fallthrough]];
            case AV_PIX_FMT_YUV420P10:
            case AV_PIX_FMT_YUV420P12:
            case AV_PIX_FMT_YUV420P16: elementSize = sizeof(std::uint16_t); break;
            // packed
            case AV_PIX_FMT_NV42:
            case AV_PIX_FMT_NV24: wscale = 1; [[fallthrough]];
            case AV_PIX_FMT_NV16: hscale = 1; [[fallthrough]];
            case AV_PIX_FMT_NV21:
            case AV_PIX_FMT_NV12: planes = 2; break;
            case AV_PIX_FMT_P410:
            case AV_PIX_FMT_P412:
            case AV_PIX_FMT_P416: wscale = 1; [[fallthrough]];
            case AV_PIX_FMT_NV20:
            case AV_PIX_FMT_P210:
            case AV_PIX_FMT_P212:
            case AV_PIX_FMT_P216: hscale = 1; [[fallthrough]];
            case AV_PIX_FMT_P010:
            case AV_PIX_FMT_P012:
            case AV_PIX_FMT_P016: elementSize = sizeof(std::uint16_t); planes = 2; break;
            default: break;
            }
            dst.planes = planes;
            for (int i = 0; i < dst.planes; i++)
            {
                dst.plane[i].width = src->width / (i > 0 ? wscale : 1);
                dst.plane[i].height = src->height / (i > 0 ? hscale : 1);
                dst.plane[i].channel = 1;
                dst.plane[i].stride = src->linesize[i];
                dst.plane[i].data = src->data[i];
            }
            if (planes == 2) dst.plane[1].channel = 2;
            dst.elementType = (0 << 8) | elementSize; // same as ac::core::Image
            dst.dptr = std::make_shared<FrameData>();
            dst.dptr->frame = src;
            dst.dptr->packets = std::move(packets);
            packets.clear();
        }
        inline Info::BitDepth PipelineImpl::getBitDepth(const AVPixelFormat format) noexcept
        {
            Info::BitDepth bitDepth{ false, 8 };
            switch (format)
            {
            case AV_PIX_FMT_GRAY10:
            case AV_PIX_FMT_YUV420P10:
            case AV_PIX_FMT_YUV422P10:
            case AV_PIX_FMT_YUV444P10:
            case AV_PIX_FMT_NV20: bitDepth.lsb = true; [[fallthrough]];
            case AV_PIX_FMT_P410:
            case AV_PIX_FMT_P210:
            case AV_PIX_FMT_P010: bitDepth.bits = 10; break;
            case AV_PIX_FMT_GRAY12:
            case AV_PIX_FMT_YUV420P12:
            case AV_PIX_FMT_YUV422P12:
            case AV_PIX_FMT_YUV444P12: bitDepth.lsb = true; [[fallthrough]];
            case AV_PIX_FMT_P412:
            case AV_PIX_FMT_P212:
            case AV_PIX_FMT_P012: bitDepth.bits = 12; break;
            case AV_PIX_FMT_GRAY16:
            case AV_PIX_FMT_YUV420P16:
            case AV_PIX_FMT_YUV422P16:
            case AV_PIX_FMT_YUV444P16:
            case AV_PIX_FMT_P416:
            case AV_PIX_FMT_P216:
            case AV_PIX_FMT_P016: bitDepth.bits = 16; break;
            default: break;
            }
            return bitDepth;
        }
    }
}

struct ac::video::Pipeline::PipelineData
{
    detail::PipelineImpl impl{};
};

ac::video::Pipeline::Pipeline() noexcept : dptr(std::make_unique<PipelineData>()) {}
ac::video::Pipeline::~Pipeline() noexcept = default;

bool ac::video::Pipeline::openDecoder(const char* const fileame, DecoderHints hints) noexcept
{
    return dptr->impl.openDecoder(fileame, hints);
}
bool ac::video::Pipeline::openEncoder(const char* const filename, const double factor, EncoderHints hints) noexcept
{
    return dptr->impl.openEncoder(filename, factor, hints);
}
void ac::video::Pipeline::close() noexcept
{
    dptr->impl.close();
}
bool ac::video::Pipeline::operator>>(Frame& frame) noexcept
{
    return dptr->impl.decode(frame);
}
bool ac::video::Pipeline::operator<<(const Frame& frame) noexcept
{
    return dptr->impl.encode(frame);
}
bool ac::video::Pipeline::request(Frame& dst, const Frame& src) const noexcept
{
    return dptr->impl.request(dst, src);
}
void ac::video::Pipeline::release(Frame& frame) noexcept
{
    dptr->impl.release(frame);
}
ac::video::Info ac::video::Pipeline::getInfo() const noexcept
{
    return dptr->impl.getInfo();
}
