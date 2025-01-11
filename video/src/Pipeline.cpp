#include <queue>
#include <utility>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

#include "AC/Video/Pipeline.hpp"

namespace ac::video::detail
{
    struct FrameRefData
    {
        AVFrame* frame = nullptr;
        std::queue<AVPacket*> packets{};
    };

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
        bool remux(std::queue<AVPacket*>& packets) const noexcept;
        bool fetch(std::queue<AVPacket*>& packets) noexcept;
        void fill(Frame& dst, AVFrame* src, std::queue<AVPacket*>& packets) const noexcept;
        Info::BitDepth getBitDepth(AVPixelFormat format) const noexcept;
    private:
        bool dfmtCtxOpenFlag = false;
        bool writeHeaderFlag = false;
        AVPixelFormat targetPixFmt = AV_PIX_FMT_NONE;
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
        dfmtCtx = avformat_alloc_context(); if (!dfmtCtx) return false;
        ret = avformat_open_input(&dfmtCtx, filename, nullptr, nullptr); if (ret < 0) return false;
        dfmtCtxOpenFlag = true;

        ret = avformat_find_stream_info(dfmtCtx, nullptr); if (ret < 0) return false;
        for (unsigned int i = 0; i < dfmtCtx->nb_streams; i++)
            if (dfmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
            {
                dvideoStream = dfmtCtx->streams[i];
                break;
            }
        if (!dvideoStream) return false;

        switch (dvideoStream->codecpar->format)
        {
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_YUV422P:
        case AV_PIX_FMT_YUV444P:
        case AV_PIX_FMT_YUV420P10:
        case AV_PIX_FMT_YUV422P10:
        case AV_PIX_FMT_YUV444P10:
        case AV_PIX_FMT_YUV420P16:
        case AV_PIX_FMT_YUV422P16:
        case AV_PIX_FMT_YUV444P16: break;
        default: return false;
        }
        auto codec = (hints.decoder && *hints.decoder) ? avcodec_find_decoder_by_name(hints.decoder) : avcodec_find_decoder(dvideoStream->codecpar->codec_id); if (!codec) return false;
        decoderCtx = avcodec_alloc_context3(codec); if (!decoderCtx) return false;
        ret = avcodec_parameters_to_context(decoderCtx, dvideoStream->codecpar); if (ret < 0) return false;
        decoderCtx->pkt_timebase = dvideoStream->time_base;
        if (hints.format && *hints.format) decoderCtx->pix_fmt = targetPixFmt = av_get_pix_fmt(hints.format);
        ret = avcodec_open2(decoderCtx, codec, nullptr); if (ret < 0) return false;
        auto framerate = av_guess_frame_rate(dfmtCtx, dvideoStream, nullptr);
        timeBase = av_inv_q(framerate.num ? framerate : av_make_q(24000, 1001));
        return true;
    }
    inline bool PipelineImpl::openEncoder(const char* const filename, const double factor, const EncoderHints& hints) noexcept
    {
        int ret = 0;
        epacket = av_packet_alloc(); if (!epacket) return false;
        ret = avformat_alloc_output_context2(&efmtCtx, nullptr, nullptr, filename); if (ret < 0) return false;

        auto codec = (hints.encoder && *hints.encoder) ? avcodec_find_encoder_by_name(hints.encoder) : avcodec_find_encoder(AV_CODEC_ID_H264); if (!codec) return false;
        encoderCtx = avcodec_alloc_context3(codec); if (!encoderCtx) return false;

        encoderCtx->pix_fmt = targetPixFmt != AV_PIX_FMT_NONE ? targetPixFmt : decoderCtx->pix_fmt;
        auto bitDepth = getBitDepth(encoderCtx->pix_fmt);
        switch (codec->id)
        {
#       if LIBAVCODEC_VERSION_MAJOR < 60 // ffmpeg 6, libavcodec 60
        case AV_CODEC_ID_H264: encoderCtx->profile = bitDepth.bits > 8 ? FF_PROFILE_H264_HIGH_10 : FF_PROFILE_H264_HIGH; break;
        case AV_CODEC_ID_HEVC: encoderCtx->profile = bitDepth.bits > 8 ? FF_PROFILE_HEVC_MAIN_10 : FF_PROFILE_HEVC_MAIN; break;
#       else
        case AV_CODEC_ID_H264: encoderCtx->profile = bitDepth.bits > 8 ? AV_PROFILE_H264_HIGH_10 : AV_PROFILE_H264_HIGH; break;
        case AV_CODEC_ID_HEVC: encoderCtx->profile = bitDepth.bits > 8 ? AV_PROFILE_HEVC_MAIN_10 : AV_PROFILE_HEVC_MAIN; break;
#       endif
        default: break;
        }
        encoderCtx->codec_type = AVMEDIA_TYPE_VIDEO;
        encoderCtx->bit_rate = hints.bitrate > 0 ? hints.bitrate : static_cast<decltype(encoderCtx->bit_rate)>(decoderCtx->bit_rate * factor * factor);
        encoderCtx->framerate = decoderCtx->framerate;
        encoderCtx->gop_size = 12;
        encoderCtx->time_base = timeBase;
        encoderCtx->width = static_cast<decltype(encoderCtx->width)>(decoderCtx->width * factor);
        encoderCtx->height = static_cast<decltype(encoderCtx->height)>(decoderCtx->height * factor);
        encoderCtx->sample_aspect_ratio = decoderCtx->sample_aspect_ratio;
        encoderCtx->color_primaries = decoderCtx->color_primaries;
        encoderCtx->color_trc = decoderCtx->color_trc;
        encoderCtx->colorspace = decoderCtx->colorspace;
        if (efmtCtx->oformat->flags & AVFMT_GLOBALHEADER) encoderCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        ret = avcodec_open2(encoderCtx, codec, nullptr); if (ret < 0) return false;
        // copy all streams
        streamIdxMap.resize(dfmtCtx->nb_streams);
        int streamIdx = 0;
        for (unsigned int i = 0; i < dfmtCtx->nb_streams; i++)
        {
            if (dfmtCtx->streams[i]->codecpar->codec_type != AVMEDIA_TYPE_VIDEO &&
                dfmtCtx->streams[i]->codecpar->codec_type != AVMEDIA_TYPE_AUDIO &&
                dfmtCtx->streams[i]->codecpar->codec_type != AVMEDIA_TYPE_SUBTITLE)
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
            stream->disposition = dfmtCtx->streams[i]->disposition; // a series of flags that tells a player or media player how to handle a stream.
            stream->sample_aspect_ratio = dfmtCtx->streams[i]->sample_aspect_ratio; // for mkv to keep DAR
            stream->avg_frame_rate = dfmtCtx->streams[i]->avg_frame_rate;
        }
        if (encoderCtx->pix_fmt != decoderCtx->pix_fmt) swsCtx = sws_getContext(decoderCtx->width, decoderCtx->height, decoderCtx->pix_fmt, decoderCtx->width, decoderCtx->height, encoderCtx->pix_fmt, SWS_FAST_BILINEAR | SWS_PRINT_INFO, nullptr, nullptr, nullptr);
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
        std::queue<AVPacket*> packets{};
        for (;;)
        {
            ret = avcodec_receive_frame(decoderCtx, frame);
            if (ret == 0) break;
            else if (ret == AVERROR(EAGAIN) && fetch(packets)) continue;
            else
            {
                av_frame_free(&frame);
                return false;
            }
        }
        if (swsCtx)
        {
            auto* dstFrame = av_frame_alloc(); if (!dstFrame) return false;
            if ((av_frame_copy_props(dstFrame, frame) >= 0) && (sws_scale_frame(swsCtx, dstFrame, frame) >= 0))
            {
                av_frame_free(&frame);
                frame = dstFrame;
            }
            else
            {
                av_frame_free(&dstFrame);
                av_frame_free(&frame);
                return false;
            }
        }
        fill(dst, frame, packets);
#       if LIBAVCODEC_VERSION_MAJOR < 60 // ffmpeg 6, libavcodec 60
        dst.number = decoderCtx->frame_number;
#       else
        dst.number = decoderCtx->frame_num;
#       endif
        return true;
    }
    inline bool PipelineImpl::encode(const Frame& src) noexcept
    {
        int ret = 0;
        if (!src.ref)  return false;

        auto frameRefData = static_cast<FrameRefData*>(src.ref);
        if (!remux(frameRefData->packets)) return false;
        ret = avcodec_send_frame(encoderCtx, frameRefData->frame); if (ret < 0) return false;
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
        if (!src.ref)  return false;

        auto srcFrameRefData = static_cast<FrameRefData*>(src.ref);
        auto srcFrame = srcFrameRefData->frame;
        auto dstFrame = av_frame_alloc(); if (!dstFrame) return false;
        dstFrame->width = encoderCtx->width;
        dstFrame->height = encoderCtx->height;
        dstFrame->format = srcFrame->format;
        dstFrame->pts = srcFrame->pts;
#       if LIBAVUTIL_VERSION_MAJOR > 57 // ffmpeg 5, libavutil 57
        dstFrame->duration = srcFrame->duration;
#       endif
        if (av_frame_get_buffer(dstFrame, 0) < 0)
        {
            av_frame_free(&dstFrame);
            return false;
        }

        fill(dst, dstFrame, srcFrameRefData->packets);
        dst.number = src.number;
        return true;
    }
    inline void PipelineImpl::release(Frame& frame) noexcept
    {
        if (frame.ref)
        {
            auto frameRefData = static_cast<FrameRefData*>(frame.ref);
            av_frame_free(&frameRefData->frame);
            while (!frameRefData->packets.empty())
            {
                av_packet_free(&frameRefData->packets.front());
                frameRefData->packets.pop();
            }
            delete frameRefData;
            frame.ref = nullptr;
        }
    }
    inline void PipelineImpl::close() noexcept
    {
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
        if (dfmtCtx)
        {
            if (dfmtCtxOpenFlag)
            {
                avformat_close_input(&dfmtCtx);
                dfmtCtxOpenFlag = false;
            }
            else
            {
                avformat_free_context(dfmtCtx);
                dfmtCtx = nullptr;
            }
        }
        if (epacket) av_packet_free(&epacket);
        if (dpacket) av_packet_free(&dpacket);

        streamIdxMap.clear();
        timeBase = {};
        evideoStream = nullptr;
        dvideoStream = nullptr;
        targetPixFmt = AV_PIX_FMT_NONE;
    }
    inline Info PipelineImpl::getInfo() const noexcept
    {
        Info info{};
        info.width = decoderCtx->width;
        info.height = decoderCtx->height;
        info.bitDepth = getBitDepth(encoderCtx->pix_fmt);
        info.duration = dvideoStream->duration * av_q2d(dvideoStream->time_base);
        info.fps = av_q2d(av_inv_q(timeBase));
        return info;
    }

    inline bool PipelineImpl::remux(std::queue<AVPacket*>& packets) const noexcept
    {
        int ret = 0;
        while (!packets.empty())
        {
            AVPacket* packet = packets.front();
            packets.pop();
            if (streamIdxMap[packet->stream_index] >= 0)
            {
                av_packet_rescale_ts(packet, dfmtCtx->streams[packet->stream_index]->time_base, efmtCtx->streams[streamIdxMap[packet->stream_index]]->time_base);
                packet->stream_index = streamIdxMap[packet->stream_index];
                ret = av_interleaved_write_frame(efmtCtx, packet);
            }
            av_packet_unref(packet);
            av_packet_free(&packet);
            if (ret < 0) return false;
        }
        return true;
    }
    inline bool PipelineImpl::fetch(std::queue<AVPacket*>& packets) noexcept
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
            else packets.emplace(av_packet_clone(dpacket));
            av_packet_unref(dpacket);
        }
        return avcodec_send_packet(decoderCtx, nullptr) == 0;
    }
    inline void PipelineImpl::fill(Frame& dst, AVFrame* const src, std::queue<AVPacket*>& packets) const noexcept
    {
        int wscale = 2, hscale = 2, elementSize = sizeof(std::uint8_t);
        bool packed = false;
        switch (src->format)
        {// planar
        case AV_PIX_FMT_YUV444P: wscale = 1; [[fallthrough]];
        case AV_PIX_FMT_YUV422P: hscale = 1; break;
        case AV_PIX_FMT_YUV444P10:
        case AV_PIX_FMT_YUV444P16: wscale = 1; [[fallthrough]];
        case AV_PIX_FMT_YUV422P10:
        case AV_PIX_FMT_YUV422P16: hscale = 1; [[fallthrough]];
        case AV_PIX_FMT_YUV420P10:
        case AV_PIX_FMT_YUV420P16: elementSize = sizeof(std::uint16_t); break;
         // packed
        case AV_PIX_FMT_P010:
        case AV_PIX_FMT_P016: elementSize = sizeof(std::uint16_t); [[fallthrough]];
        case AV_PIX_FMT_NV12: packed = true; break;
        default: break;
        }
        dst.plane[0].width = src->width;
        dst.plane[0].height = src->height;
        dst.plane[1].width = src->width / wscale;
        dst.plane[1].height = src->height / hscale;
        dst.plane[2].width = src->width / wscale;
        dst.plane[2].height = src->height / hscale;
        dst.planes = packed ? 2 : 3;
        for (int i = 0; i < dst.planes; i++)
        {
            dst.plane[i].stride = src->linesize[i];
            dst.plane[i].data = src->data[i];
            dst.plane[i].channel = 1;
        }
        if (packed) dst.plane[1].channel = 2;
        dst.elementType = (0 << 8) | elementSize; // same as ac::core::Image
        dst.ref = new FrameRefData{ src, std::move(packets) };
    }
    inline Info::BitDepth PipelineImpl::getBitDepth(const AVPixelFormat format) const noexcept
    {
        Info::BitDepth bitDepth{false, 8};
        switch (format)
        {
        case AV_PIX_FMT_YUV420P10:
        case AV_PIX_FMT_YUV422P10:
        case AV_PIX_FMT_YUV444P10: bitDepth.lsb = true; [[fallthrough]];
        case AV_PIX_FMT_P010: bitDepth.bits = 10; break;
        case AV_PIX_FMT_YUV420P16:
        case AV_PIX_FMT_YUV422P16:
        case AV_PIX_FMT_YUV444P16:
        case AV_PIX_FMT_P016: bitDepth.bits = 16; break;
        }
        return bitDepth;
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
