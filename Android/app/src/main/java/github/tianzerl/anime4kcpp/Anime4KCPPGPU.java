package github.tianzerl.anime4kcpp;

public class Anime4KCPPGPU extends Anime4KCPP {

    public static boolean checkGPUSupport() {
        return checkGPUSupportAnime4KCPPGPU();
    }

    public Anime4KCPPGPU() throws Exception {
        anime4k = createAnime4KCPPGPU();
    }

    public Anime4KCPPGPU(
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
            byte postFilters) throws Exception {
        anime4k = createAnime4KCPPGPUByArgs(
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

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        releaseAnime4KCPPGPU(anime4k);
    }

    private native long createAnime4KCPPGPU() throws Exception;
    private native long createAnime4KCPPGPUByArgs(int passes,
                                               int pushColorCount,
                                               double strengthColor,
                                               double strengthGradient,
                                               double zoomFactor,
                                               boolean fastMode,
                                               boolean videoMode,
                                               boolean preprocessing,
                                               boolean postProcessing,
                                               byte preFilters,
                                               byte postFilters) throws Exception;
    private native void releaseAnime4KCPPGPU(long ptr);
    protected native static boolean checkGPUSupportAnime4KCPPGPU();
}
