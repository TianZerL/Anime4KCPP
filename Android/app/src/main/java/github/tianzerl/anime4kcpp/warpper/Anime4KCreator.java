package github.tianzerl.anime4kcpp.warpper;

public class Anime4KCreator {
    public Anime4KCreator(boolean initGPU) {
        if (initGPU && !Anime4KGPU.isInitializedGPU())
            Anime4KGPU.initGPU();
    }

    public Anime4K create(Parameters parameters, ProcessorType type) {
        switch (type)
        {
            case CPU:
                return new Anime4KCPU(parameters);
            case GPU:
                return new Anime4KGPU(parameters);
            case CPUCNN:
                return new Anime4KCPUCNN(parameters);
            case GPUCNN:
                return new Anime4KGPUCNN(parameters);
            default:
                return null;
        }
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        if (Anime4KGPU.isInitializedGPU())
            Anime4KGPU.releaseGPU();
    }
}
