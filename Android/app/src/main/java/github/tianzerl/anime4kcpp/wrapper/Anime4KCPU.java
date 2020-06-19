package github.tianzerl.anime4kcpp.wrapper;

public class Anime4KCPU extends Anime4K {

    public Anime4KCPU() {
        anime4k = createAnime4KCPU();
    }

    public Anime4KCPU(Parameters parameters) {
        anime4k = createAnime4KCPUByArgs(
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

    private native long createAnime4KCPU();
    private native long createAnime4KCPUByArgs(int passes,
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
}
