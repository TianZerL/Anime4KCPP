package github.tianzerl.anime4kcpp;

public class Anime4KCPP {

    static {
        System.loadLibrary("Anime4KCPPCore");
    }

    public static String getCoreVersion() {
        return getCoreVersionAnime4KCPP();
    }

    public Anime4KCPP() {
        anime4k = createAnime4KCPP();
    }

    public Anime4KCPP(
            int passes,
            int pushColorCount,
            double strengthColor,
            double strengthGradient,
            double zoomFactor,
            boolean fastMode,
            boolean videoMode,
            boolean preprocessing,
            boolean postprocessing,
            byte preFilters,
            byte postFilters) {
        anime4k = createAnime4KCPPByArgs(
        passes,
        pushColorCount,
        strengthColor,
        strengthGradient,
        zoomFactor,
        fastMode,
        videoMode,
        preprocessing,
        postprocessing,
        preFilters,
        postFilters
        );
    }

    public void setArguments(int passes,
                             int pushColorCount,
                             double strengthColor,
                             double strengthGradient,
                             double zoomFactor,
                             boolean fastMode,
                             boolean videoMode,
                             boolean preprocessing,
                             boolean postprocessing,
                             byte preFilters,
                             byte postFilters) {
        setArgumentsAnime4KCPP( anime4k,
                passes,
                pushColorCount,
                strengthColor,
                strengthGradient,
                zoomFactor,
                fastMode,
                videoMode,
                preprocessing,
                postprocessing,
                preFilters,
                postFilters);
    }

    public void setVideoMode(boolean flag) {
        setVideoModeAnime4KCPP(anime4k, flag);
    }

    public void loadImage(final String src) {
        loadImageAnime4KCPP(anime4k, src);
    }

    public void loadVideo(final String src) {
        loadVideoAnime4KCPP(anime4k, src);
    }

    public void setVideoSaveInfo(final String dst) {
        setVideoSaveInfoAnime4KCPP(anime4k, dst);
    }

    public void process() {
        processAnime4KCPP(anime4k);
    }

    public void saveImage(final String dst) {
        saveImageAnime4KCPP(anime4k, dst);
    }

    public void saveVideo() {
        saveVideoAnime4KCPP(anime4k);
    }

    long anime4k;

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        releaseAnime4KCPP(anime4k);
    }

    private native long createAnime4KCPP();
    private native long createAnime4KCPPByArgs(int passes,
                                               int pushColorCount,
                                               double strengthColor,
                                               double strengthGradient,
                                               double zoomFactor,
                                               boolean fastMode,
                                               boolean videoMode,
                                               boolean preprocessing,
                                               boolean postProcessing,
                                               byte preFilters,
                                               byte postFilters);
    private native void releaseAnime4KCPP(long ptr);

    protected native void setArgumentsAnime4KCPP(long ptr,
                                                 int passes,
                                                 int pushColorCount,
                                                 double strengthColor,
                                                 double strengthGradient,
                                                 double zoomFactor,
                                                 boolean fastMode,
                                                 boolean videoMode,
                                                 boolean preprocessing,
                                                 boolean postProcessing,
                                                 byte preFilters,
                                                 byte postFilters);

    protected native void setVideoModeAnime4KCPP(long ptr, boolean flag);
    protected native void loadImageAnime4KCPP(long ptr, final String src);
    protected native void loadVideoAnime4KCPP(long ptr, final String src);
    protected native void setVideoSaveInfoAnime4KCPP(long ptr, final String dst);
    protected native void processAnime4KCPP(long ptr);
    protected native void saveImageAnime4KCPP(long ptr, final String dst);
    protected native void saveVideoAnime4KCPP(long ptr);
    protected native static String getCoreVersionAnime4KCPP();
}
