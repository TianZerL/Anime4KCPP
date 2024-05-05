extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/pixdesc.h>
}

#include "AC/Video/Pipeline.hpp"

namespace ac::video::detail
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
        bool request(Frame& dst, const Frame& src) noexcept;
        void release(AVFrame* frame) noexcept;
        bool remux() noexcept;
        void close() noexcept;
        Info getInfo() noexcept;
    private:
        bool fetch() noexcept;
        void fill(Frame& dst, AVFrame* src) noexcept;
    private:
        AVFormatContext* dfmtCtx = nullptr;
        AVFormatContext* efmtCtx = nullptr;
        AVPacket* dpacket = nullptr;
        AVPacket* epacket = nullptr;
        AVCodecContext* decodecCtx = nullptr;
        AVCodecContext* encodecCtx = nullptr;
        AVStream* dvideoStream = nullptr;
        AVStream* evideoStream = nullptr;
        AVRational timeBase{}; // should be 1/fps
        int videoIdx = 0;
        bool dfmtCtxOpenFlag = false;
        bool writeHaderFlag = false;
    };

    PipelineImpl::PipelineImpl() noexcept = default;
    PipelineImpl::~PipelineImpl() noexcept
    {
        close();
    }
    inline bool PipelineImpl::openDecoder(const char* const filename, const DecoderHints& hints) noexcept
    {
        int ret = 0;
        epacket = av_packet_alloc(); if (!epacket) return false;
        dfmtCtx = avformat_alloc_context(); if (!dfmtCtx) return false;
        ret = avformat_open_input(&dfmtCtx, filename, nullptr, nullptr); if (ret < 0) return false;
        dfmtCtxOpenFlag = true;

        ret = avformat_find_stream_info(dfmtCtx, nullptr); if (ret < 0) return false;
        for(unsigned int i = 0; i < dfmtCtx->nb_streams; i++)
            if(dfmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
            {
                dvideoStream = dfmtCtx->streams[i];
                videoIdx = i;
                break;
            }
        if (!dvideoStream) return false;

        switch (dvideoStream->codecpar->format)
        {
        case AV_PIX_FMT_YUV420P:
        case AV_PIX_FMT_YUV422P:
        case AV_PIX_FMT_YUV444P:
        case AV_PIX_FMT_YUV420P16:
        case AV_PIX_FMT_YUV422P16:
        case AV_PIX_FMT_YUV444P16: break;
        default: return false;
        }
        auto codec = (hints.decoder && *hints.decoder) ? avcodec_find_decoder_by_name(hints.decoder) : avcodec_find_decoder(dvideoStream->codecpar->codec_id); if (!codec) return false;
        decodecCtx = avcodec_alloc_context3(codec); if (!decodecCtx) return false;
        ret = avcodec_parameters_to_context(decodecCtx, dvideoStream->codecpar); if (ret < 0) return false;
        decodecCtx->pkt_timebase = dvideoStream->time_base;
        if (hints.format && *hints.format) decodecCtx->pix_fmt = av_get_pix_fmt(hints.format);
        ret = avcodec_open2(decodecCtx, codec, nullptr); if (ret < 0) return false;
        timeBase = av_inv_q(decodecCtx->framerate.num ? decodecCtx->framerate : (dvideoStream->avg_frame_rate.num ? dvideoStream->avg_frame_rate : av_make_q(24000, 1001)));
        return true;
    }
    inline bool PipelineImpl::openEncoder(const char* const filename, const double factor, const EncoderHints& hints) noexcept
    {
        int ret = 0;
        dpacket = av_packet_alloc(); if (!dpacket) return false;
        ret = avformat_alloc_output_context2(&efmtCtx, nullptr, nullptr, filename); if (ret < 0) return false;

        auto codec = (hints.encoder && *hints.encoder) ? avcodec_find_encoder_by_name(hints.encoder) : avcodec_find_encoder(AV_CODEC_ID_H264); if (!codec) return false;
        encodecCtx = avcodec_alloc_context3(codec); if (!encodecCtx) return false;

        switch (codec->id)
        {
#       if LIBAVCODEC_VERSION_MAJOR < 60 // ffmpeg 6, libavcodec 60
            case AV_CODEC_ID_H264: encodecCtx->profile = FF_PROFILE_H264_HIGH; break;
            case AV_CODEC_ID_HEVC: encodecCtx->profile = FF_PROFILE_HEVC_MAIN_10; break;
#       else
            case AV_CODEC_ID_H264: encodecCtx->profile = AV_PROFILE_H264_HIGH; break;
            case AV_CODEC_ID_HEVC: encodecCtx->profile = AV_PROFILE_HEVC_MAIN_10; break;
#       endif
        default: break;
        }
        encodecCtx->pix_fmt = decodecCtx->pix_fmt;
        encodecCtx->codec_type = AVMEDIA_TYPE_VIDEO;
        encodecCtx->bit_rate = hints.bitrate > 0 ? hints.bitrate : static_cast<decltype(encodecCtx->bit_rate)>(decodecCtx->bit_rate * factor * factor);
        encodecCtx->framerate = decodecCtx->framerate;
        encodecCtx->gop_size = 12;
        encodecCtx->time_base = timeBase;
        encodecCtx->width = static_cast<decltype(encodecCtx->width)>(decodecCtx->width * factor);
        encodecCtx->height = static_cast<decltype(encodecCtx->height)>(decodecCtx->height * factor);
        encodecCtx->sample_aspect_ratio = decodecCtx->sample_aspect_ratio;
        encodecCtx->color_primaries = decodecCtx->color_primaries;
        encodecCtx->color_trc = decodecCtx->color_trc;
        encodecCtx->colorspace = decodecCtx->colorspace;

        ret = avcodec_open2(encodecCtx, codec, nullptr); if (ret < 0) return false;
        // copy all streams
        for(unsigned int i = 0; i < dfmtCtx->nb_streams; i++)
        {
            auto stream = avformat_new_stream(efmtCtx, nullptr); if (!stream) return false;
            if(dfmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) evideoStream = stream;
            else
            {   // copy stream info except pointers
                *stream->codecpar = *dfmtCtx->streams[i]->codecpar;
#               if LIBAVCODEC_VERSION_MAJOR > 60 // ffmpeg 7, libavcodec 61
                    stream->codecpar->coded_side_data = nullptr;
                    stream->codecpar->nb_coded_side_data = 0;
#               endif
                stream->codecpar->extradata = nullptr;
                stream->codecpar->extradata_size = 0;
            }
        }
        evideoStream->time_base = dvideoStream->time_base;
        evideoStream->avg_frame_rate = dvideoStream->avg_frame_rate;
        ret = avcodec_parameters_from_context(evideoStream->codecpar, encodecCtx); if (ret < 0) return false;
        ret = avio_open2(&efmtCtx->pb, filename, AVIO_FLAG_WRITE, &efmtCtx->interrupt_callback, nullptr); if (ret < 0) return false;
        ret = avformat_write_header(efmtCtx, nullptr); if (ret < 0) return false;
        writeHaderFlag = true;

        return true;
    }
    inline bool PipelineImpl::decode(Frame& dst) noexcept
    {
        int ret = 0;
        auto frame = av_frame_alloc(); if (!frame) return false;
        for(;;)
        {
            ret = avcodec_receive_frame(decodecCtx, frame);
            if(ret == 0) break;
            else if (ret == AVERROR(EAGAIN) && fetch()) continue;
            else return false;
        }
        fill(dst, frame);
#       if LIBAVCODEC_VERSION_MAJOR < 60 // ffmpeg 6, libavcodec 60
            dst.number = decodecCtx->frame_number;
#       else
            dst.number = decodecCtx->frame_num;
#       endif
        return true;
    }
    inline bool PipelineImpl::encode(const Frame& src) noexcept
    {
        int ret = 0;
        auto frame = static_cast<AVFrame*>(src.ref);
        ret = avcodec_send_frame(encodecCtx, frame); if (ret < 0) return false;
        for (;;)
        {
            ret = avcodec_receive_packet(encodecCtx, epacket);
            if (ret == AVERROR(EAGAIN)) break;
            else if (ret < 0) return false;
            av_packet_rescale_ts(epacket, encodecCtx->time_base, evideoStream->time_base);
            epacket->stream_index = evideoStream->index;
            ret = av_interleaved_write_frame(efmtCtx, epacket); if (ret < 0) return false;
        }
        release(frame);
        return true;
    }
    inline bool PipelineImpl::request(Frame& dst, const Frame& src) noexcept
    {
        int ret = 0;
        auto srcFrame = static_cast<AVFrame*>(src.ref);
        auto dstFrame = av_frame_alloc(); if (!dstFrame) return false;
        dstFrame->width = encodecCtx->width;
        dstFrame->height = encodecCtx->height;
        dstFrame->format = srcFrame->format;
        dstFrame->pts = srcFrame->pts;
#       if LIBAVUTIL_VERSION_MAJOR > 57 // ffmpeg 5, libavutil 57
            dstFrame->duration = srcFrame->duration;
#       endif
        ret = av_frame_get_buffer(dstFrame, 0); if (ret < 0) return false;

        fill(dst, dstFrame);
        dst.number = src.number;
        return true;
    }
    inline void PipelineImpl::release(AVFrame* frame) noexcept
    {
        av_frame_unref(frame);
        av_frame_free(&frame);
    }
    inline bool PipelineImpl::remux() noexcept
    {
        if(avformat_seek_file(dfmtCtx, videoIdx, 0, 0, dvideoStream->duration, 0) < 0) return false;
        while (av_read_frame(dfmtCtx, dpacket) >= 0)
            if(dpacket->stream_index != videoIdx)
                if(av_interleaved_write_frame(efmtCtx, dpacket) < 0) return false;
        return true;
    }
    inline void PipelineImpl::close() noexcept
    {
        if (writeHaderFlag)
        {
            av_write_trailer(efmtCtx);
            writeHaderFlag = false;
        }
        if (encodecCtx) avcodec_free_context(&encodecCtx);
        if (decodecCtx) avcodec_free_context(&decodecCtx);
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
    }
    inline bool PipelineImpl::fetch() noexcept
    {
        while (av_read_frame(dfmtCtx, dpacket) >= 0)
        {
            if(dpacket->stream_index == videoIdx)
            {
                av_packet_rescale_ts(dpacket, dvideoStream->time_base, timeBase);
                int ret = avcodec_send_packet(decodecCtx, dpacket);
                av_packet_unref(dpacket);
                return ret == 0;
            }
        }
        return false;
    }
    inline void PipelineImpl::fill(Frame& dst, AVFrame* const src) noexcept
    {
        int wscale = 2, hscale = 2, elementSize = sizeof(std::uint8_t);
        bool packed = false;
        switch (decodecCtx->pix_fmt)
        {// planar
        case AV_PIX_FMT_YUV444P: wscale = 1; [[fallthrough]];
        case AV_PIX_FMT_YUV422P: hscale = 1; break;
        case AV_PIX_FMT_YUV444P16: wscale = 1; [[fallthrough]];
        case AV_PIX_FMT_YUV422P16: hscale = 1; [[fallthrough]];
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
        dst.ref = src;
    }
    inline Info PipelineImpl::getInfo() noexcept
    {
        Info info{};
        info.length = dfmtCtx->duration / AV_TIME_BASE;
        info.width = decodecCtx->width;
        info.height = decodecCtx->height;
        info.fps = av_q2d(av_inv_q(timeBase));
        return info;
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
bool ac::video::Pipeline::request(Frame& dst, const Frame& src) noexcept
{
    return dptr->impl.request(dst, src);
}
void ac::video::Pipeline::release(Frame& frame) noexcept
{
    dptr->impl.release(static_cast<AVFrame*>(frame.ref));
}
bool ac::video::Pipeline::remux() noexcept
{
    return dptr->impl.remux();
}
ac::video::Info ac::video::Pipeline::getInfo() noexcept
{
    return dptr->impl.getInfo();
}
