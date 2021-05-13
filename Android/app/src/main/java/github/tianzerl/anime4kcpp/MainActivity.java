package github.tianzerl.anime4kcpp;

import android.content.Intent;
import android.os.Bundle;
import android.os.Handler;
import android.os.Message;
import android.util.Pair;
import android.view.Menu;
import android.view.MenuItem;
import android.view.View;
import android.view.WindowManager;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ListView;
import android.widget.ProgressBar;
import android.widget.RadioGroup;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.SwitchCompat;

import com.hjq.permissions.OnPermissionCallback;
import com.hjq.permissions.Permission;
import com.hjq.permissions.XXPermissions;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.lang.ref.WeakReference;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Properties;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import github.tianzerl.anime4kcpp.utility.VideoAudioProcessor;
import github.tianzerl.anime4kcpp.wrapper.Anime4K;
import github.tianzerl.anime4kcpp.wrapper.Anime4KCreator;
import github.tianzerl.anime4kcpp.wrapper.Anime4KGPU;
import github.tianzerl.anime4kcpp.wrapper.Anime4KGPUCNN;
import github.tianzerl.anime4kcpp.wrapper.Parameters;
import github.tianzerl.anime4kcpp.wrapper.ProcessorType;
import github.tianzerl.anime4kcpp.wrapper.VideoProcessor;
import me.rosuh.filepicker.bean.FileItemBeanImpl;
import me.rosuh.filepicker.config.AbstractFileFilter;
import me.rosuh.filepicker.config.FilePickerManager;

public class MainActivity extends AppCompatActivity {

    private enum Error {
        Anime4KCPPError, FailedToCreateFolders, FailedToDeleteTmpFile, FailedToLoadConfig, FailedToWriteConfig
    }

    private enum GPUState {
        Initialized, UnInitialized, Unsupported
    }

    private enum GPUCNNState {
        Initialized, UnInitialized, Unsupported
    }

    private enum FileType {
        Image, Video, Unknown
    }

    AbstractFileFilter fileFilter = new AbstractFileFilter() {
        @Override
        public @NonNull
        ArrayList<FileItemBeanImpl> doFilter(@NonNull ArrayList<FileItemBeanImpl> arrayList) {
            ArrayList<FileItemBeanImpl> newArrayList = new ArrayList<>();
            for (FileItemBeanImpl file : arrayList) {
                if (file.isDir() || getFileType(file.getFilePath()) != FileType.Unknown)
                    newArrayList.add(file);
            }
            return newArrayList;
        }
    };

    private ArrayAdapter<String> adapterForProcessingList;
    private ProgressBar mainProgressBar;
    private TextView textViewTime;
    private ListView processingList;
    private TextView textViewState;
    private Config config;

    private GPUState GPU = GPUState.UnInitialized;
    private GPUCNNState GPUCNN = GPUCNNState.UnInitialized;
    private final Anime4KCreator anime4KCreator = new Anime4KCreator(false, false);

    private final Handler anime4KCPPHandler = new Handler(msg -> {
        errorHandler(Error.Anime4KCPPError, new Exception((String) msg.obj));
        return false;
    });

    private final Handler otherErrorHandler = new Handler(msg -> {
        errorHandler((Error) msg.obj, null);
        return false;
    });

    private final ExecutorService ACProcessorThread = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r);
        t.setDaemon(true);
        return t;
    });

    private final ExecutorService ACBenchmarkThread = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r);
        t.setDaemon(true);
        return t;
    });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mainProgressBar = findViewById(R.id.progressBar);
        textViewTime = findViewById(R.id.textViewTime);
        textViewState = findViewById(R.id.textViewState);

        adapterForProcessingList = new ArrayAdapter<>(
                this,
                android.R.layout.simple_list_item_multiple_choice,
                new ArrayList<>()
        );

        processingList = findViewById(R.id.processingList);
        processingList.setAdapter(adapterForProcessingList);

        RadioGroup settingsGroup = findViewById(R.id.radioGroupSettings);
        settingsGroup.check(R.id.radioButtonBalance);

        setSwitches();

        config = new Config();
        config.read();
        //Keep screen on
        this.getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        XXPermissions.with(this)
                .permission(Permission.MANAGE_EXTERNAL_STORAGE)
                .request(new OnPermissionCallback() {

                @Override
                public void onGranted(List<String> permissions, boolean all) {
                    if (all) {
                        Toast.makeText(MainActivity.this,
                                "Access to storage allowed\ndo not set the output path to Android/data",
                                Toast.LENGTH_LONG).show();
                    }
                }

                @Override
                public void onDenied(List<String> permissions, boolean never) {
                    if (never) {
                        Toast.makeText(MainActivity.this,"Access to storage not allowed, please set it manually", Toast.LENGTH_LONG).show();
                        XXPermissions.startPermissionActivity(MainActivity.this, permissions);
                    } else {
                        Toast.makeText(MainActivity.this,"Access to storage not allowed", Toast.LENGTH_LONG).show();
                    }
                }
            });
    }

    @Override
    protected void onPause() {
        super.onPause();
        config.write();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        int id = item.getItemId();
        if (id == R.id.menuItemAbout) {
            new AlertDialog.Builder(MainActivity.this)
                    .setTitle("ABOUT")
                    .setMessage("Anime4KCPP for Android\n\n"
                            + "Anime4KCPP core version: " + Anime4K.getCoreVersion()
                            + "\n\nGitHub: https://github.com/TianZerL/Anime4KCPP")
                    .setPositiveButton("OK", null)
                    .show();
        } else if (id == R.id.menuItemBenchmark) {
            Runnable runnable = new ACBenchmark(this);
            ACBenchmarkThread.execute(runnable);
        } else if (id == R.id.menuItemQuit) {
            finish();
        }
        return true;
    }

    public void onClickButtonPick(View v) {
        FilePickerManager.INSTANCE
                .from(this)
                .filter(fileFilter)
                .forResult(1);
    }

    public void onClickButtonClear(View v) {
        adapterForProcessingList.clear();
        adapterForProcessingList.notifyDataSetInvalidated();
    }

    public void onClickButtonDelete(View v) {
        List<String> readyToRemove = new ArrayList<>();
        for (int i = 0; i < processingList.getCount(); i++) {
            if (processingList.isItemChecked(i))
                readyToRemove.add(adapterForProcessingList.getItem(i));
        }
        for (String item : readyToRemove) {
            adapterForProcessingList.remove(item);
        }
        adapterForProcessingList.notifyDataSetInvalidated();
    }

    public void onClickButtonStart(View v) {
        if (adapterForProcessingList.isEmpty()) {
            Toast.makeText(MainActivity.this, "Nothing to do", Toast.LENGTH_SHORT).show();
            return;
        }
        EditText editTextOutputPath = findViewById(R.id.editTextOutputPath);
        EditText editTextOutputPrefix = findViewById(R.id.editTextOutputPrefix);
        Runnable runnable = new ACProcessor(
                this, editTextOutputPrefix.getText().toString(),
                "/storage/emulated/0/" + editTextOutputPath.getText().toString());
        ACProcessorThread.execute(runnable);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != AppCompatActivity.RESULT_OK)
            return;
        if (requestCode == 1) {
            List<String> files = FilePickerManager.INSTANCE.obtainData();
            for (String file : files) {
                if (file.isEmpty())
                    return;
                adapterForProcessingList.add(file);
            }
        }
    }

    private FileType getFileType(@NonNull String src) {
        String imageSuffix = ((EditText) findViewById(R.id.editTextSuffixImage)).getText().toString();
        String VideoSuffix = ((EditText) findViewById(R.id.editTextSuffixVideo)).getText().toString();
        if (imageSuffix.contains(src.substring(src.lastIndexOf('.') + 1)))
            return FileType.Image;
        if (VideoSuffix.contains(src.substring(src.lastIndexOf('.') + 1)))
            return FileType.Video;
        return FileType.Unknown;
    }

    private void setSwitches() {
        final SwitchCompat GPUMode = findViewById(R.id.switchGPUMode);
        GPUMode.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked && GPU == GPUState.UnInitialized) {
                try {
                    if (!Anime4KGPU.isInitializedGPU()) {
                        if (Anime4KGPU.checkGPUSupport()) {
                            Toast.makeText(MainActivity.this, "GPU Mode enabled", Toast.LENGTH_SHORT).show();
                            Anime4KGPU.initGPU();
                            GPU = GPUState.Initialized;
                        } else {
                            Toast.makeText(MainActivity.this, "Unsupported GPU Mode", Toast.LENGTH_SHORT).show();
                            GPU = GPUState.Unsupported;
                            buttonView.setChecked(false);
                        }
                    }
                } catch (Exception exp) {
                    errorHandler(Error.Anime4KCPPError, exp);
                    GPU = GPUState.Unsupported;
                    buttonView.setChecked(false);
                }
            } else if (isChecked && GPU == GPUState.Unsupported) {
                Toast.makeText(MainActivity.this, "Unsupported GPU Mode", Toast.LENGTH_SHORT).show();
                buttonView.setChecked(false);
            }
        });

        final SwitchCompat ACNetGPUMode = findViewById(R.id.switchACNetGPUMode);
        ACNetGPUMode.setOnCheckedChangeListener((buttonView, isChecked) -> {
            if (isChecked && GPUCNN == GPUCNNState.UnInitialized) {
                try {
                    if (!Anime4KGPUCNN.isInitializedGPU()) {
                        if (Anime4KGPU.checkGPUSupport()) {
                            Toast.makeText(MainActivity.this, "ACNet GPU Mode enabled", Toast.LENGTH_SHORT).show();
                            Anime4KGPUCNN.initGPU();
                            GPUCNN = GPUCNNState.Initialized;
                        } else {
                            Toast.makeText(MainActivity.this, "Unsupported ACNet GPU Mode", Toast.LENGTH_SHORT).show();
                            GPUCNN = GPUCNNState.Unsupported;
                            buttonView.setChecked(false);
                        }
                    }
                } catch (Exception exp) {
                    errorHandler(Error.Anime4KCPPError, exp);
                    GPUCNN = GPUCNNState.Unsupported;
                    buttonView.setChecked(false);
                }
            } else if (isChecked && GPUCNN == GPUCNNState.Unsupported) {
                Toast.makeText(MainActivity.this, "Unsupported ACNet GPU Mode", Toast.LENGTH_SHORT).show();
                buttonView.setChecked(false);
            }
        });

        final SwitchCompat preprocessingCAS = findViewById(R.id.switchPreprocessingCAS);
        preprocessingCAS.setChecked(true);
        final SwitchCompat postprocessingGaussianBlurWeak = findViewById(R.id.switchPostprocessingGaussianBlurWeak);
        postprocessingGaussianBlurWeak.setChecked(true);
        final SwitchCompat postprocessingBilateralFilter = findViewById(R.id.switchPostprocessingBilateralFilter);
        postprocessingBilateralFilter.setChecked(true);
    }

    private boolean getSwitchSate(int id) {
        SwitchCompat sw = findViewById(id);
        return sw.isChecked();
    }

    private void errorHandler(@NonNull Error errorType, @Nullable Exception exp) {
        switch (errorType) {
            case Anime4KCPPError:
                assert exp != null;
                new AlertDialog.Builder(MainActivity.this)
                        .setTitle("ERROR")
                        .setMessage(exp.getMessage())
                        .setPositiveButton("OK", null)
                        .show();
                break;
            case FailedToCreateFolders:
                new AlertDialog.Builder(MainActivity.this)
                        .setTitle("ERROR")
                        .setMessage("Failed to create folders")
                        .setPositiveButton("OK", null)
                        .show();
                break;
            case FailedToDeleteTmpFile:
                new AlertDialog.Builder(MainActivity.this)
                        .setTitle("ERROR")
                        .setMessage("Failed to delete Temporary output video file, delete it manually.")
                        .setPositiveButton("OK", null)
                        .show();
                break;
            case FailedToLoadConfig:
                new AlertDialog.Builder(MainActivity.this)
                        .setTitle("ERROR")
                        .setMessage("Failed to load or create config file.")
                        .setPositiveButton("OK", null)
                        .show();
                break;
            case FailedToWriteConfig:
                new AlertDialog.Builder(MainActivity.this)
                        .setTitle("ERROR")
                        .setMessage("Failed to write config file.")
                        .setPositiveButton("OK", null)
                        .show();
                break;
        }
    }

    private Anime4K initAnime4KCPP() {
        int passes = 2;
        int pushColorCount = 2;
        double strengthColor = 0.3;
        double strengthGradient = 1;
        double zoomFactor = 2;
        boolean fastMode = getSwitchSate(R.id.switchFastMode);
        boolean preprocessing = false;
        boolean postprocessing = false;
        byte preFilters = 0;
        byte postFilters = 0;
        boolean HDN = getSwitchSate(R.id.switchHDN);
        int HDNLevel = Integer.parseInt(((EditText) findViewById(R.id.editTextHDNLevel)).getText().toString());
        boolean alpha = getSwitchSate(R.id.switchAlphaChannel);

        RadioGroup settingsGroup = findViewById(R.id.radioGroupSettings);
        int id = settingsGroup.getCheckedRadioButtonId();
        if (id == R.id.radioButtonFast) {
            passes = 1;
            preprocessing = false;
            postprocessing = false;
        } else if (id == R.id.radioButtonQuality) {
            preprocessing = true;
            postprocessing = true;
            preFilters = 4;
            postFilters = 48;
        } else if (id == R.id.radioButtonCustom) {
            passes = Integer.parseInt(((EditText) findViewById(R.id.editTextPasses)).getText().toString());
            pushColorCount = Integer.parseInt(((EditText) findViewById(R.id.editTextPushColorCount)).getText().toString());
            strengthColor = Double.parseDouble(((EditText) findViewById(R.id.editTextStrengthColor)).getText().toString());
            strengthGradient = Double.parseDouble(((EditText) findViewById(R.id.editTextStrengthGradien)).getText().toString());
            zoomFactor = Double.parseDouble(((EditText) findViewById(R.id.editTextZoomFactor)).getText().toString());
            preprocessing = ((Switch) findViewById(R.id.switchPreprocessing)).isChecked();
            postprocessing = ((Switch) findViewById(R.id.switchPostprocessing)).isChecked();

            if (((Switch) findViewById(R.id.switchPreprocessingMedianBlur)).isChecked())
                preFilters |= 1;
            if (((Switch) findViewById(R.id.switchPreprocessingMeanBlur)).isChecked())
                preFilters |= 1 << 1;
            if (((Switch) findViewById(R.id.switchPreprocessingCAS)).isChecked())
                preFilters |= 1 << 2;
            if (((Switch) findViewById(R.id.switchPreprocessingGaussianBlurWeak)).isChecked())
                preFilters |= 1 << 3;
            if (((Switch) findViewById(R.id.switchPreprocessingGaussianBlur)).isChecked())
                preFilters |= 1 << 4;
            if (((Switch) findViewById(R.id.switchPreprocessingBilateralFilter)).isChecked())
                preFilters |= 1 << 5;
            if (((Switch) findViewById(R.id.switchPreprocessingBilateralFilterFaster)).isChecked())
                preFilters |= 1 << 6;

            if (((Switch) findViewById(R.id.switchPostprocessingMedianBlur)).isChecked())
                postFilters |= 1;
            if (((Switch) findViewById(R.id.switchPostprocessingMeanBlur)).isChecked())
                postFilters |= 1 << 1;
            if (((Switch) findViewById(R.id.switchPostprocessingCAS)).isChecked())
                postFilters |= 1 << 2;
            if (((Switch) findViewById(R.id.switchPostprocessingGaussianBlurWeak)).isChecked())
                postFilters |= 1 << 3;
            if (((Switch) findViewById(R.id.switchPostprocessingGaussianBlur)).isChecked())
                postFilters |= 1 << 4;
            if (((Switch) findViewById(R.id.switchPostprocessingBilateralFilter)).isChecked())
                postFilters |= 1 << 5;
            if (((Switch) findViewById(R.id.switchPostprocessingBilateralFilterFaster)).isChecked())
                postFilters |= 1 << 6;
        }

        Parameters parameters = new Parameters(
                passes,
                pushColorCount,
                strengthColor,
                strengthGradient,
                zoomFactor,
                fastMode,
                false,
                preprocessing,
                postprocessing,
                preFilters,
                postFilters,
                HDN,
                HDNLevel,
                alpha
        );

        if (getSwitchSate(R.id.switchACNet)) {
            if (getSwitchSate(R.id.switchACNetGPUMode)) {
                return anime4KCreator.create(parameters, ProcessorType.GPUCNN);
            } else {
                return anime4KCreator.create(parameters, ProcessorType.CPUCNN);
            }
        } else {
            if (getSwitchSate(R.id.switchGPUMode)) {
                return anime4KCreator.create(parameters, ProcessorType.GPU);
            } else {
                return anime4KCreator.create(parameters, ProcessorType.CPU);
            }
        }
    }

    class Config {
        Properties properties;
        File path = new File(getFilesDir(), "config.properties");

        protected Config() {
            properties = new Properties();
            if (path.exists()) {
                try {
                    FileInputStream fi = new FileInputStream(path);
                    properties.load(fi);
                    fi.close();
                } catch (Exception ignored) {
                    errorHandler(Error.FailedToLoadConfig, null);
                }
            } else {
                try {
                    if (!path.createNewFile())
                        throw new Exception();
                    FileInputStream fi = new FileInputStream(path);
                    properties.load(fi);
                    fi.close();
                } catch (Exception ignored) {
                    errorHandler(Error.FailedToLoadConfig, null);
                }
            }
        }

        protected void write() {
            properties.setProperty("ouputPath", ((EditText) findViewById(R.id.editTextOutputPath)).getText().toString());
            properties.setProperty("ouputPrefix", ((EditText) findViewById(R.id.editTextOutputPrefix)).getText().toString());
            properties.setProperty("imageSuffix", ((EditText) findViewById(R.id.editTextSuffixImage)).getText().toString());
            properties.setProperty("videoSuffix", ((EditText) findViewById(R.id.editTextSuffixVideo)).getText().toString());
            try {
                FileOutputStream fo = new FileOutputStream(path);
                properties.store(fo, null);
                fo.flush();
                fo.close();
            } catch (Exception ignored) {
                errorHandler(Error.FailedToWriteConfig, null);
            }
        }

        protected void read() {
            ((EditText) findViewById(R.id.editTextOutputPath))
                    .setText(properties.getProperty("ouputPath", "Anime4KCPP/output"));
            ((EditText) findViewById(R.id.editTextOutputPrefix))
                    .setText(properties.getProperty("ouputPrefix", "anime4kcpp_output_"));
            ((EditText) findViewById(R.id.editTextSuffixImage))
                    .setText(properties.getProperty("imageSuffix", "png:jpg:jpeg:bmp"));
            ((EditText) findViewById(R.id.editTextSuffixVideo))
                    .setText(properties.getProperty("videoSuffix", "mp4:mkv:avi:m4v:flv:3gp:wmv:mov"));
        }
    }

    protected static class ACProcessor implements Runnable {

        private final WeakReference<MainActivity> activityReference;
        private final String prefix;
        private final String dst;

        ACProcessor(MainActivity context, String prefix, String dst) {
            super();
            this.activityReference = new WeakReference<>(context);
            this.prefix = prefix;
            this.dst = dst;
        }

        private void publishProgress(int p, int r, int idx) {
            MainActivity activity = activityReference.get();
            if (activity == null || activity.isFinishing())
                return;

            (activityReference.get()).runOnUiThread(() -> {
                activity.mainProgressBar.setProgress(p);
                activity.textViewTime.setText(String.format(Locale.ENGLISH, "remaining time:  %d s", r));
                if (idx != -1) {
                    activity.processingList.setItemChecked(idx, true);
                }
            });
        }

        protected void updateProgressForVideoProcessing(double v, double t) {
            publishProgress((int) (v * 100), (int) (t / v - t), -1);
        }

        private void onPreExecute() {
            MainActivity activity = activityReference.get();
            if (activity == null || activity.isFinishing())
                return;

            Button buttonStart = activity.findViewById(R.id.buttonStart);

            buttonStart.setEnabled(false);
            Toast.makeText(activity, "Processing...", Toast.LENGTH_SHORT).show();
            activity.textViewState.setText(R.string.processing);
        }

        private Double doInBackground() {
            MainActivity activity = activityReference.get();
            if (activity == null || activity.isFinishing())
                return 0.0;

            long totalTime = 0;
            int taskCount = activity.adapterForProcessingList.getCount();

            File dstPath = new File(dst);
            if (!dstPath.exists() && !dstPath.mkdirs()) {
                Message message = new Message();
                message.obj = Error.FailedToCreateFolders;
                activity.otherErrorHandler.sendMessage(message);
            }

            List<Pair<String, Integer>> images = new ArrayList<>();
            List<Pair<String, Integer>> videos = new ArrayList<>();

            //add images and video to processing list
            for (int i = 0; i < taskCount; i++) {
                String filePath = activity.adapterForProcessingList.getItem(i);
                assert filePath != null;
                FileType fileType = activity.getFileType(filePath);
                if (fileType == FileType.Image)
                    images.add(Pair.create(filePath, i));
                else if (fileType == FileType.Video)
                    videos.add(Pair.create(filePath, i));
            }

            Anime4K ac = activity.initAnime4KCPP();

            int imageCount = 0, videoCount = 0;

            try {
                //processing images
                for (Pair<String, Integer> image : images) {
                    File srcFile = new File(image.first);
                    ac.loadImage(srcFile.getPath());

                    long start = System.currentTimeMillis();
                    ac.process();
                    long end = System.currentTimeMillis();

                    ac.saveImage(dst + "/" + prefix + srcFile.getName());

                    totalTime += end - start;
                    publishProgress((++imageCount) * 100 / taskCount, 0, image.second);
                }

                //processing videos
                if (!videos.isEmpty()) {
                    CallbackProxy callbackProxy = new CallbackProxy(this);
                    VideoProcessor videoProcessor = new VideoProcessor(ac);
                    videoProcessor.setCallbackProxy(callbackProxy);
                    for (Pair<String, Integer> video : videos) {
                        File srcFile = new File(video.first);
                        String tmpOutputPath = dst + "/" + "tmpOutput" + videoCount + ".mp4";
                        String OutputPath = dst + "/" + prefix + srcFile.getName() + ".mp4";

                        videoProcessor.loadVideo(srcFile.getPath());
                        videoProcessor.setVideoSaveInfo(tmpOutputPath);

                        long start = System.currentTimeMillis();
                        videoProcessor.processWithProgress();
                        long end = System.currentTimeMillis();

                        videoProcessor.saveVideo();

                        new VideoAudioProcessor(srcFile.getPath(), tmpOutputPath, OutputPath).merge();

                        if (!(new File(tmpOutputPath).delete())) {
                            Message message = new Message();
                            message.obj = Error.FailedToDeleteTmpFile;
                            activity.otherErrorHandler.sendMessage(message);
                        }
                        totalTime += end - start;
                        publishProgress((++videoCount + imageCount) * 100 / taskCount, 0, video.second);
                    }
                }

            } catch (Exception exp) {
                Message message = new Message();
                message.obj = exp.getMessage();
                activity.anime4KCPPHandler.sendMessage(message);
                return 0.0;
            }

            return (double) (totalTime) / 1000.0;
        }

        private void onPostExecute(Double t) {
            MainActivity activity = activityReference.get();
            if (activity == null || activity.isFinishing())
                return;

            Button buttonStart = activity.findViewById(R.id.buttonStart);

            buttonStart.setEnabled(true);

            if (t == 0.0) {
                activity.mainProgressBar.setProgress(0);
                return;
            }

            new AlertDialog.Builder(activity)
                    .setTitle("NOTICE")
                    .setMessage("Finished in " + t.toString() + "s")
                    .setPositiveButton("OK", null)
                    .show();
            activity.textViewState.setText(R.string.waitting);
            activity.mainProgressBar.setProgress(0);
        }

        @Override
        public void run() {
            (activityReference.get()).runOnUiThread(this::onPreExecute);

            double t = doInBackground();

            (activityReference.get()).runOnUiThread(() -> onPostExecute(t));
        }
    }

    protected static class ACBenchmark implements Runnable {

        private final WeakReference<MainActivity> activityReference;

        ACBenchmark(MainActivity context) {
            super();
            this.activityReference = new WeakReference<>(context);
        }

        private void onPreExecute() {
            MainActivity activity = activityReference.get();
            if (activity == null || activity.isFinishing())
                return;

            Toast.makeText(activity, "Benchmarking in the background", Toast.LENGTH_SHORT).show();
        }

        private double[] doInBackground() {
            return Anime4K.benchmark();
        }

        private void onPostExecute(double[] s) {
            MainActivity activity = activityReference.get();
            if (activity == null || activity.isFinishing())
                return;

            String outputMsg = String.format(Locale.ENGLISH, "CPU score: %.2f FPS\nGPU score: %.2f FPS", s[0], s[1]);

            new AlertDialog.Builder(activity)
                    .setTitle("Benchmark done (480P->960P)")
                    .setMessage(outputMsg)
                    .setPositiveButton("OK", null)
                    .show();
        }

        @Override
        public void run() {
            (activityReference.get()).runOnUiThread(this::onPreExecute);

            double[] s = doInBackground();

            (activityReference.get()).runOnUiThread(() -> onPostExecute(s));
        }
    }
}
