package github.tianzerl.anime4kcpp.wrapper;

import github.tianzerl.anime4kcpp.CallbackProxy;

public class VideoProcessor {
    static {
        System.loadLibrary("Anime4KCPPCore");
    }

    private final long videoProcessor;

    private CallbackProxy callbackProxy;

    public VideoProcessor(Anime4K ac) {
                videoProcessor = createVideoProcessor(ac.anime4k);
    }

    public void loadVideo(final String src) throws Exception {
        loadVideoVideoProcessor(videoProcessor, src);
    }

    public void setVideoSaveInfo(final String dst) throws Exception {
        setVideoSaveInfoVideoProcessor(videoProcessor, dst);
    }

    public void processWithProgress() {
        processWithProgressVideoProcessor(videoProcessor);
    }

    public void progressCallback(double v, double t) {
        callbackProxy.callback(v, t);
    }

    public void saveVideo() {
        saveVideoVideoProcessor(videoProcessor);
    }

    public void setCallbackProxy(CallbackProxy callbackProxy) {
        this.callbackProxy = callbackProxy;
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        releaseVideoProcessor(videoProcessor);
    }

    private native void releaseVideoProcessor(long ptr);

    protected native void saveVideoVideoProcessor(long ptr);

    protected native void processWithProgressVideoProcessor(long ptr);

    protected native void loadVideoVideoProcessor(long ptr, final String src) throws Exception;

    protected native void setVideoSaveInfoVideoProcessor(long ptr, final String dst) throws Exception;

    private native long createVideoProcessor(long acPtr);

}
