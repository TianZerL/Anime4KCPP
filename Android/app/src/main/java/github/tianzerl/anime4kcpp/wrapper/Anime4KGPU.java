package github.tianzerl.anime4kcpp.wrapper;

public class Anime4KGPU extends Anime4K {

    public Anime4KGPU(Parameters parameters) {
        anime4k = createAnime4KGPUByArgs(
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
                parameters.alpha
        );
    }

    public static boolean checkGPUSupport() {
        return checkGPUSupportAnime4KGPU();
    }

    public static void initGPU() {
        initGPUAnime4KGPU();
    }

    public static void releaseGPU() {
        releaseGPUAnime4KGPU();
    }

    public static boolean isInitializedGPU() {
        return isInitializedGPUAnime4KGPU();
    }

    private native long createAnime4KGPUByArgs(int passes,
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
                                               boolean alpha);

    private native static boolean checkGPUSupportAnime4KGPU();

    private native static void initGPUAnime4KGPU();

    private native static void releaseGPUAnime4KGPU();

    private native static boolean isInitializedGPUAnime4KGPU();
}
