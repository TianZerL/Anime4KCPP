package github.tianzerl.anime4kcpp.wrapper;

public final class Parameters {
    public int passes = 2;
    public int pushColorCount = 2;
    public double strengthColor = 0.3 ;
    public double strengthGradient = 1.0 ;
    public double zoomFactor = 2.0 ;
    public boolean fastMode = false ;
    public boolean videoMode = false ;
    public boolean preprocessing = false ;
    public boolean postprocessing = false ;
    public byte preFilters = 4 ;
    public byte postFilters = 48 ;
    public boolean HDN = false;
    public boolean alpha = false;

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
            boolean alpha
    ) {
        this.passes = passes;
        this.pushColorCount = pushColorCount;
        this.strengthColor = strengthColor;
        this.strengthGradient = strengthGradient;
        this.zoomFactor = zoomFactor;
        this.fastMode = fastMode;
        this.videoMode = videoMode ;
        this.preprocessing = preprocessing ;
        this.postprocessing = postprocessing ;
        this.preFilters = preFilters ;
        this.postFilters = postFilters ;
        this.HDN = HDN;
        this.alpha = alpha;
    }

    public void reset() {
        passes = 2;
        pushColorCount = 2;
        strengthColor = 0.3 ;
        strengthGradient = 1.0 ;
        zoomFactor = 2.0 ;
        fastMode = false ;
        videoMode = false ;
        preprocessing = false ;
        postprocessing = false ;
        preFilters = 4 ;
        postFilters = 48 ;
        HDN = false;
        alpha = false;
    }

}
