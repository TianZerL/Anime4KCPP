extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
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
        AVStream* evideoStream = nullptr;
        AVStream* dvideoStream = nullptr;
        int videoIdx = -1;
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
        for(int i = 0; i < dfmtCtx->nb_streams; i++)
            if(dfmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
            {
                dvideoStream = dfmtCtx->streams[i];
                videoIdx = i;
                break;
            }
        if (videoIdx == -1) return false;

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
        auto codec = hints.decoder ? avcodec_find_decoder_by_name(hints.decoder) : avcodec_find_decoder(dvideoStream->codecpar->codec_id); if (!codec) return false;
        decodecCtx = avcodec_alloc_context3(codec); if (!decodecCtx) return false;
        ret = avcodec_parameters_to_context(decodecCtx, dvideoStream->codecpar); if (ret < 0) return false;
        ret = avcodec_open2(decodecCtx, codec, nullptr); if (ret < 0) return false;
        return true;
    }
    inline bool PipelineImpl::openEncoder(const char* const filename, const double factor, const EncoderHints& hints) noexcept
    {
        int ret = 0;
        dpacket = av_packet_alloc(); if (!dpacket) return false;
        ret = avformat_alloc_output_context2(&efmtCtx, nullptr, nullptr, filename); if (ret < 0) return false;

        auto codec = hints.encoder ? avcodec_find_encoder_by_name(hints.encoder) : avcodec_find_encoder(AV_CODEC_ID_H264); if (!codec) return false;
        encodecCtx = avcodec_alloc_context3(codec); if (!encodecCtx) return false;

        switch (codec->id)
        {
        case AV_CODEC_ID_H264: encodecCtx->profile = AV_PROFILE_H264_HIGH; break;
        case AV_CODEC_ID_HEVC: encodecCtx->profile = AV_PROFILE_HEVC_MAIN_10; break;
        }
        encodecCtx->pix_fmt = decodecCtx->pix_fmt;
        encodecCtx->codec_type = AVMEDIA_TYPE_VIDEO;
        encodecCtx->bit_rate = hints.bitrate > 0 ? hints.bitrate : static_cast<decltype(encodecCtx->bit_rate)>(decodecCtx->bit_rate * factor * factor);
        encodecCtx->framerate = decodecCtx->framerate;
        encodecCtx->gop_size = 10;
        encodecCtx->max_b_frames = 5;
        encodecCtx->time_base = dvideoStream->time_base;
        encodecCtx->width = static_cast<decltype(encodecCtx->width)>(decodecCtx->width * factor);
        encodecCtx->height = static_cast<decltype(encodecCtx->height)>(decodecCtx->height * factor);
        encodecCtx->sample_aspect_ratio = decodecCtx->sample_aspect_ratio;
        encodecCtx->color_range = decodecCtx->color_range;
        encodecCtx->color_primaries = decodecCtx->color_primaries;
        encodecCtx->color_trc = decodecCtx->color_trc;
        encodecCtx->colorspace = decodecCtx->colorspace;
        encodecCtx->chroma_sample_location = decodecCtx->chroma_sample_location;

        ret = avcodec_open2(encodecCtx, codec, nullptr); if (ret < 0) return false;
        // copy all streams
        for(int i = 0; i < dfmtCtx->nb_streams; i++)
        {
            auto stream = avformat_new_stream(efmtCtx, nullptr); if (!stream) return false;
            if(dfmtCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) evideoStream = stream;
            else
            {   // copy stream info except pointers
                *stream->codecpar = *dfmtCtx->streams[i]->codecpar;
                stream->codecpar->coded_side_data = nullptr;
                stream->codecpar->nb_coded_side_data = 0;
                stream->codecpar->extradata = nullptr;
                stream->codecpar->extradata_size = 0;
            }
        }
        evideoStream->time_base = dvideoStream->time_base;
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
            else switch (ret)
            {
            case AVERROR(EAGAIN): if(fetch()) break;
            default: release(frame); return false;
            }
        }
        fill(dst, frame);
        dst.number = decodecCtx->frame_num;
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
            epacket->stream_index = evideoStream->index;
            av_packet_rescale_ts(epacket, encodecCtx->time_base, evideoStream->time_base);
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
        dstFrame->format = encodecCtx->pix_fmt;
        dstFrame->pts = srcFrame->pts;
        ret = av_frame_get_buffer(dstFrame, 0); if (ret < 0) return false;

        fill(dst, dstFrame);
        dst.number = encodecCtx->frame_num;
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
        switch (decodecCtx->pix_fmt)
        {
        case AV_PIX_FMT_YUV444P: wscale = 1;
        case AV_PIX_FMT_YUV422P: hscale = 1; break;
        case AV_PIX_FMT_YUV444P16: wscale = 1;
        case AV_PIX_FMT_YUV422P16: hscale = 1;
        case AV_PIX_FMT_YUV420P16: elementSize = sizeof(std::uint16_t);
        }
        dst.planar[0].width = src->width;
        dst.planar[0].height = src->height;
        dst.planar[1].width = src->width / wscale;
        dst.planar[1].height = src->height / hscale;
        dst.planar[2].width = src->width / wscale;
        dst.planar[2].height = src->height / hscale;
        for (int i = 0; i < 3; i++)
        {
            dst.planar[i].stride = src->linesize[i];
            dst.planar[i].data = src->data[i];
        }
        dst.elementType = (0 << 8) | elementSize;
        dst.ref = src;
    }
    inline Info PipelineImpl::getInfo() noexcept
    {
        Info info{};
        info.length = dfmtCtx->duration / AV_TIME_BASE;
        info.width = decodecCtx->width;
        info.height = decodecCtx->height;
        info.fps = av_q2d(decodecCtx->framerate);
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
