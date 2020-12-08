package github.tianzerl.anime4kcpp.utility;

import android.media.MediaCodec;
import android.media.MediaExtractor;
import android.media.MediaFormat;
import android.media.MediaMuxer;
import android.util.Log;

import androidx.annotation.NonNull;

import java.io.IOException;
import java.nio.ByteBuffer;

public class VideoAudioProcessor {
    private final String src;
    private final String tmp;
    private final String dst;

    public VideoAudioProcessor(@NonNull String srcPath, @NonNull String tmpPath, @NonNull String dstPath) {
        src = srcPath;
        tmp = tmpPath;
        dst = dstPath;
    }

    public void merge() throws IOException {
        MediaExtractor srcVideoExtractor = new MediaExtractor();
        MediaExtractor tmpVideoExtractor = new MediaExtractor();
        MediaMuxer dstVideoMuxer = new MediaMuxer(dst,MediaMuxer.OutputFormat.MUXER_OUTPUT_MPEG_4);
        srcVideoExtractor.setDataSource(src);
        tmpVideoExtractor.setDataSource(tmp);
        int srcTrackCount = srcVideoExtractor.getTrackCount();
        int srcAudioTrackIndex = -1;
        int dstAudioTrackIndex = -1;
        int dstVideoTrackIndex;
        int dstAudioMaxInputSize = -1;
        int tmpVideoMaxInputSize;
        //Find src audio track;
        for (int i = 0; i < srcTrackCount; i++)
        {
            MediaFormat mimeFormat = srcVideoExtractor.getTrackFormat(i);
            String mimeType = mimeFormat.getString(MediaFormat.KEY_MIME);
            assert mimeType != null;
            Log.d("MediaType",mimeType);
            if(mimeType.startsWith(MediaFormat.MIMETYPE_AUDIO_AAC))
            {
                srcAudioTrackIndex = i;
                dstAudioTrackIndex = dstVideoMuxer.addTrack(mimeFormat);
                dstAudioMaxInputSize = mimeFormat.getInteger(MediaFormat.KEY_MAX_INPUT_SIZE);
            }
        }

        if(srcAudioTrackIndex == -1)
        {
            throw new IOException("Only supports to merge aac audio track");
        }

        //tmp video track;
        {
            MediaFormat mimeFormat = tmpVideoExtractor.getTrackFormat(0);
            dstVideoTrackIndex = dstVideoMuxer.addTrack(mimeFormat);
            tmpVideoMaxInputSize = mimeFormat.getInteger(MediaFormat.KEY_MAX_INPUT_SIZE);
        }

        dstVideoMuxer.start();

        srcVideoExtractor.selectTrack(srcAudioTrackIndex);
        MediaCodec.BufferInfo srcAudioBufferInfo = new MediaCodec.BufferInfo();
        ByteBuffer srcAudioBuffer = ByteBuffer.allocate(dstAudioMaxInputSize);
        while (true)
        {
            int readSampleSize = srcVideoExtractor.readSampleData(srcAudioBuffer,0);
            if (readSampleSize < 0)
            {
                srcVideoExtractor.unselectTrack(srcAudioTrackIndex);
                break;
            }

            srcAudioBufferInfo.size = readSampleSize;
            srcAudioBufferInfo.offset = 0;
            srcAudioBufferInfo.flags = srcVideoExtractor.getSampleFlags();
            srcAudioBufferInfo.presentationTimeUs = srcVideoExtractor.getSampleTime();

            dstVideoMuxer.writeSampleData(dstAudioTrackIndex,srcAudioBuffer,srcAudioBufferInfo);

            srcVideoExtractor.advance();
        }

        tmpVideoExtractor.selectTrack(0);
        MediaCodec.BufferInfo tmpVideoBufferInfo = new MediaCodec.BufferInfo();
        ByteBuffer tmpVideoBuffer = ByteBuffer.allocate(tmpVideoMaxInputSize);
        while (true)
        {
            int readSampleSize = tmpVideoExtractor.readSampleData(tmpVideoBuffer,0);
            if (readSampleSize < 0)
            {
                tmpVideoExtractor.unselectTrack(0);
                break;
            }

            tmpVideoBufferInfo.size = readSampleSize;
            tmpVideoBufferInfo.offset = 0;
            tmpVideoBufferInfo.flags = tmpVideoExtractor.getSampleFlags();
            tmpVideoBufferInfo.presentationTimeUs = tmpVideoExtractor.getSampleTime();

            dstVideoMuxer.writeSampleData(dstVideoTrackIndex, tmpVideoBuffer,tmpVideoBufferInfo);
            tmpVideoExtractor.advance();
        }

        dstVideoMuxer.stop();
        dstVideoMuxer.release();
    }
}
