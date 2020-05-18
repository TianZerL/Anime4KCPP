package github.tianzerl.anime4kcpp.warpper;

public class Anime4KCreator {
    public Anime4KCreator(boolean initGPU) {
        if (initGPU && !Anime4KGPU.isInitializedGPU())
            Anime4KGPU.initGPU();
    }

    public Anime4K create(Parameters parameters, ProcessorType type) {
        if (type == ProcessorType.CPU)
            return new Anime4KCPU(parameters);
        else
            return new Anime4KGPU(parameters);
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        if (Anime4KGPU.isInitializedGPU())
            Anime4KGPU.releaseGPU();
    }
}
