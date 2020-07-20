package github.tianzerl.anime4kcpp;

public class CallbackProxy {
    private MainActivity.Anime4KProcessor object;
    public CallbackProxy(MainActivity.Anime4KProcessor object) {
        this.object = object;
    }

    public void callback(double v, double t) {
        object.updateProgressForVideoProcessing(v, t);
    }
}
