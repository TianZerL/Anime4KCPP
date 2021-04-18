package github.tianzerl.anime4kcpp.wrapper;

import github.tianzerl.anime4kcpp.CallbackProxy;

public abstract class Anime4K {
    static {
        System.loadLibrary("Anime4KCPPCore");
    }

    long anime4k;

    public void loadImage(final String src) throws Exception {
        loadImageAnime4K(anime4k, src);
    }

    public void process() {
        processAnime4K(anime4k);
    }

    public void saveImage(final String dst) {
        saveImageAnime4K(anime4k, dst);
    }

    public static String getCoreVersion() {
        return getCoreVersionAnime4K();
    }

    public static double[] benchmark() {
        return benchmarkAnime4K();
    }

    @Override
    protected void finalize() throws Throwable {
        super.finalize();
        releaseAnime4K(anime4k);
    }

    private native void releaseAnime4K(long ptr);

    protected native void loadImageAnime4K(long ptr, final String src) throws Exception;

    protected native void processAnime4K(long ptr);

    protected native void saveImageAnime4K(long ptr, final String dst);

    protected native static String getCoreVersionAnime4K();

    protected native static double[] benchmarkAnime4K();
}
