package github.tianzerl.anime4kcpp.wrapper;

public final class Parameters {
    public int passes;
    public int pushColorCount;
    public double strengthColor;
    public double strengthGradient;
    public double zoomFactor;
    public boolean fastMode;
    public boolean videoMode;
    public boolean preprocessing;
    public boolean postprocessing;
    public byte preFilters;
    public byte postFilters;
    public boolean HDN;
    public int HDNLevel;
    public boolean alpha;

    public Parameters(
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
            byte postFilters,
            boolean HDN,
            int HDNLevel,
            boolean alpha
    ) {
        this.passes = passes;
        this.pushColorCount = pushColorCount;
        this.strengthColor = strengthColor;
        this.strengthGradient = strengthGradient;
        this.zoomFactor = zoomFactor;
        this.fastMode = fastMode;
        this.videoMode = videoMode;
        this.preprocessing = preprocessing;
        this.postprocessing = postprocessing;
        this.preFilters = preFilters;
        this.postFilters = postFilters;
        this.HDN = HDN;
        this.HDNLevel = HDNLevel;
        this.alpha = alpha;
    }

    public void reset() {
        passes = 2;
        pushColorCount = 2;
        strengthColor = 0.3;
        strengthGradient = 1.0;
        zoomFactor = 2.0;
        fastMode = false;
        videoMode = false;
        preprocessing = false;
        postprocessing = false;
        preFilters = 4;
        postFilters = 48;
        HDN = false;
        alpha = false;
    }

}
