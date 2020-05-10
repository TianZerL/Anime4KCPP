package github.tianzerl.anime4kcpp;

public class Anime4KCPPGPU extends Anime4KCPP {

    public static boolean checkGPUSupport() {
        try {
            return checkGPUSupportAnime4KCPPGPU();
        } catch (Exception exp) {
            throw exp;
        }
    }

    public Anime4KCPPGPU() {
        try {
            anime4k = createAnime4KCPPGPU();
        } catch (Exception exp) {
            throw exp;
        }
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
            byte postFilters) {
        try {
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
        } catch (Exception exp) {
            throw exp;
        }
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        releaseAnime4KCPPGPU(anime4k);
    }

    private native long createAnime4KCPPGPU();
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
                                               byte postFilters);
    private native void releaseAnime4KCPPGPU(long ptr);
    protected native static boolean checkGPUSupportAnime4KCPPGPU();
}
