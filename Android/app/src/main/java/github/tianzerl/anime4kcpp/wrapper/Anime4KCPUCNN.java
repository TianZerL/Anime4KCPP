package github.tianzerl.anime4kcpp.wrapper;

public class Anime4KCPUCNN extends Anime4K {
    public Anime4KCPUCNN() {
        anime4k = createAnime4KCPUCNN();
    }

    public Anime4KCPUCNN(Parameters parameters) {
        anime4k = createAnime4KCPUCNNByArgs(
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
                parameters.HDN
        );
    }

    private native long createAnime4KCPUCNN();
    private native long createAnime4KCPUCNNByArgs(int passes,
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
                                               boolean HDN);
}
