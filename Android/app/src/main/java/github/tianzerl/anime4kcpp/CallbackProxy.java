package github.tianzerl.anime4kcpp;

public class CallbackProxy {
    private final MainActivity.ACProcessor object;

    public CallbackProxy(MainActivity.ACProcessor object) {
        this.object = object;
    }

    public void callback(double v, double t) {
        object.updateProgressForVideoProcessing(v, t);
    }
}
