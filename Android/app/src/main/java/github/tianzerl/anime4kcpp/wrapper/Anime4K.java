package github.tianzerl.anime4kcpp.wrapper;

import github.tianzerl.anime4kcpp.CallbackProxy;

public abstract class Anime4K {
    static {
        System.loadLibrary("Anime4KCPPCore");
    }

    long anime4k;
    private CallbackProxy callbackProxy;

    public void setArguments(Parameters parameters) {
        setArgumentsAnime4K(anime4k,
                parameters.passes,
                parameters.pushColorCount,
                parameters.strengthColor,
                parameters.strengthGradient,
                parameters.zoomFactor,
                parameters.fastMode,
                parameters.videoMode,
                parameters.preprocessing,
                parameters.postprocessing,
                parameters.preFilters,
                parameters.postFilters
        );
    }

    public void setVideoMode(boolean flag) {
        setVideoModeAnime4K(anime4k, flag);
    }

    public void loadImage(final String src) throws Exception {
        loadImageAnime4K(anime4k, src);
    }

    public void loadVideo(final String src) throws Exception {
        loadVideoAnime4K(anime4k, src);
    }

    public void setVideoSaveInfo(final String dst) throws Exception {
        setVideoSaveInfoAnime4K(anime4k, dst);
    }

    public void process() {
        processAnime4K(anime4k);
    }

    public void processWithProgress() {
        processWithProgressAnime4K(anime4k);
    }

    public void progressCallback(double v, double t) {
        callbackProxy.callback(v, t);
    }

    public void setCallbackProxy(CallbackProxy callbackProxy) {
        this.callbackProxy = callbackProxy;
    }

    public void saveImage(final String dst) {
        saveImageAnime4K(anime4k, dst);
    }

    public void saveVideo() {
        saveVideoAnime4K(anime4k);
    }

    public static String getCoreVersion() {
        return getCoreVersionAnime4K();
    }

    public static double[] benchmark() {
        return benchmarkAnime4K();
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        releaseAnime4K(anime4k);
    }

    private native void releaseAnime4K(long ptr);

    protected native void setArgumentsAnime4K(long ptr, int passes, int pushColorCount,
                                              double strengthColor, double strengthGradient,
                                              double zoomFactor, boolean fastMode, boolean videoMode,
                                              boolean preprocessing, boolean postprocessing,
                                              byte preFilters, byte postFilters);

    protected native void setVideoModeAnime4K(long ptr, boolean flag);

    protected native void loadImageAnime4K(long ptr, final String src) throws Exception;

    protected native void loadVideoAnime4K(long ptr, final String src) throws Exception;

    protected native void setVideoSaveInfoAnime4K(long ptr, final String dst) throws Exception;

    protected native void processAnime4K(long ptr);

    protected native void processWithProgressAnime4K(long ptr);

    protected native void saveImageAnime4K(long ptr, final String dst);

    protected native void saveVideoAnime4K(long ptr);

    protected native static String getCoreVersionAnime4K();

    protected native static double[] benchmarkAnime4K();
}
