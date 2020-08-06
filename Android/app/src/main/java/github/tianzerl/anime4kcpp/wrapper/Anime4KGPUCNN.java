package github.tianzerl.anime4kcpp.wrapper;

public class Anime4KGPUCNN extends Anime4K {
    public Anime4KGPUCNN() {
        anime4k = createAnime4KGPUCNN();
    }

    public Anime4KGPUCNN(Parameters parameters) {
        anime4k = createAnime4KGPUCNNByArgs(
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
                parameters.postFilters,
                parameters.HDN,
                parameters.alpha
        );
    }

    public static void initGPU() {
        initGPUAnime4KGPUCNN();
    }

    public static void releaseGPU() {
        releaseGPUAnime4KGPUCNN();
    }

    public static boolean isInitializedGPU() {
        return isInitializedGPUAnime4KGPUCNN();
    }

    private native long createAnime4KGPUCNN();
    private native long createAnime4KGPUCNNByArgs(int passes,
                                               int pushColorCount,
                                               double strengthColor,
                                               double strengthGradient,
                                               double zoomFactor,
                                               boolean fastMode,
                                               boolean videoMode,
                                               boolean preprocessing,
                                               boolean postProcessing,
                                               byte preFilters,
                                               byte postFilters,
                                               boolean HDN,
                                               boolean alpha);
    private native static void initGPUAnime4KGPUCNN();
    private native static void releaseGPUAnime4KGPUCNN();
    private native static boolean isInitializedGPUAnime4KGPUCNN();
}
