package github.tianzerl.anime4kcpp;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.os.AsyncTask;
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
import android.widget.CompoundButton;
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

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import github.tianzerl.anime4kcpp.warpper.Anime4K;
import github.tianzerl.anime4kcpp.warpper.Anime4KCreator;
import github.tianzerl.anime4kcpp.warpper.Anime4KGPU;
import github.tianzerl.anime4kcpp.warpper.Anime4KGPUCNN;
import github.tianzerl.anime4kcpp.warpper.Parameters;
import github.tianzerl.anime4kcpp.warpper.ProcessorType;
import me.rosuh.filepicker.bean.FileItemBeanImpl;
import me.rosuh.filepicker.config.AbstractFileFilter;
import me.rosuh.filepicker.config.FilePickerManager;

public class MainActivity extends AppCompatActivity {

    private enum Error {
        Anime4KCPPError, FailedToCreateFolders, FailedToDeleteTmpFile
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
        public @NonNull ArrayList<FileItemBeanImpl> doFilter(@NonNull ArrayList<FileItemBeanImpl> arrayList) {
            ArrayList<FileItemBeanImpl> newArrayList = new ArrayList<>();
            for (FileItemBeanImpl file: arrayList)
            {
                if(file.isDir() || getFileType(file.getFilePath()) != FileType.Unknown)
                    newArrayList.add(file);
            }
            return newArrayList;
        }
    };

    private ArrayAdapter<String> adapterForProcessingList;
    private ProgressBar mainProgressBar;
    private ListView processingList;
    private TextView textViewState;

    private GPUState GPU = GPUState.UnInitialized;
    private GPUCNNState GPUCNN = GPUCNNState.UnInitialized;
    private Anime4KCreator anime4KCreator = new Anime4KCreator(false);

    private Handler anime4KCPPHandler = new Handler(new Handler.Callback() {
        @Override
        public boolean handleMessage(@NonNull Message msg) {
            errorHandler(Error.Anime4KCPPError, new Exception((String)msg.obj));
            return false;
        }
    });

    private Handler otherErrorHandler = new Handler(new Handler.Callback() {
        @Override
        public boolean handleMessage(@NonNull Message msg) {
            errorHandler((Error) msg.obj,null);
            return false;
        }
    });

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mainProgressBar = findViewById(R.id.progressBar);
        textViewState = findViewById(R.id.textViewState);

        adapterForProcessingList = new ArrayAdapter<>(
                this,
                android.R.layout.simple_list_item_multiple_choice,
                new ArrayList<String>()
        );

        processingList = findViewById(R.id.processingList);
        processingList.setAdapter(adapterForProcessingList);

        RadioGroup settingsGroup = findViewById(R.id.radioGroupSettings);
        settingsGroup.check(R.id.radioButtonBalance);

        setSwitches();

        //Keep screen on
        this.getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        switch (item.getItemId()) {
            case R.id.menuItemAbout:
                new AlertDialog.Builder(MainActivity.this)
                        .setTitle("ABOUT")
                        .setMessage("Anime4KCPP for Android\n\n"
                                +"Anime4KCPP core version: "+ Anime4K.getCoreVersion()
                                +"\n\nGitHub: https://github.com/TianZerL/Anime4KCPP")
                        .setPositiveButton("OK",null)
                        .show();
                break;
            case R.id.menuItemQuit:
                finish();
                break;
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
        for (int i = 0; i< processingList.getCount(); i++) {
            if(processingList.isItemChecked(i))
                readyToRemove.add(adapterForProcessingList.getItem(i));
        }
        for (String item:readyToRemove) {
            adapterForProcessingList.remove(item);
        }
        adapterForProcessingList.notifyDataSetInvalidated();
    }

    public void onClickButtonStart(View v) {
        if (adapterForProcessingList.isEmpty())
        {
            Toast.makeText(MainActivity.this,"Nothing to do",Toast.LENGTH_SHORT).show();
            return;
        }
        EditText editTextOutputPath = findViewById(R.id.editTextOutputPath);
        EditText editTextOutputPrefix = findViewById(R.id.editTextOutputPrefix);
        new Anime4KProcessor()
                .execute(editTextOutputPrefix.getText().toString(),
                        "/storage/emulated/0/"+editTextOutputPath.getText().toString());
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(resultCode!=AppCompatActivity.RESULT_OK)
            return;
        if (requestCode == 1) {
            List<String> files = FilePickerManager.INSTANCE.obtainData();
            for (String file: files) {
                if(file.isEmpty())
                    return;
                adapterForProcessingList.add(file);
            }
        }
    }

    private FileType getFileType(@NonNull String src) {
        String imageSuffix = ((EditText)findViewById(R.id.editTextSuffixImage)).getText().toString();
        String VideoSuffix = ((EditText)findViewById(R.id.editTextSuffixVideo)).getText().toString();
        if(imageSuffix.contains(src.substring(src.lastIndexOf('.') + 1)))
            return FileType.Image;
        if(VideoSuffix.contains(src.substring(src.lastIndexOf('.') + 1)))
            return FileType.Video;
        return FileType.Unknown;
    }

    private void setSwitches() {
        final Switch GPUMode = findViewById(R.id.switchGPUMode);
        GPUMode.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked && GPU == GPUState.UnInitialized)
                {
                    try {
                        if (!Anime4KGPU.isInitializedGPU() && Anime4KGPU.checkGPUSupport())
                        {
                            Toast.makeText(MainActivity.this,"GPU Mode enabled",Toast.LENGTH_SHORT).show();
                            Anime4KGPU.initGPU();
                            GPU = GPUState.Initialized;
                        }
                        else
                        {
                            Toast.makeText(MainActivity.this,"Unsupported GPU Mode",Toast.LENGTH_SHORT).show();
                            GPU = GPUState.Unsupported;
                            buttonView.setChecked(false);
                        }
                    } catch (Exception exp) {
                        errorHandler(Error.Anime4KCPPError, exp);
                        GPU = GPUState.Unsupported;
                        buttonView.setChecked(false);
                    }
                }
                else if (isChecked && GPU == GPUState.Unsupported)
                {
                    Toast.makeText(MainActivity.this,"Unsupported GPU Mode",Toast.LENGTH_SHORT).show();
                    buttonView.setChecked(false);
                }
            }
        });

        final Switch ACNetGPUMode = findViewById(R.id.switchACNetGPUMode);
        ACNetGPUMode.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked && GPUCNN == GPUCNNState.UnInitialized)
                {
                    try {
                        if (!Anime4KGPUCNN.isInitializedGPU() && Anime4KGPU.checkGPUSupport())
                        {
                            Toast.makeText(MainActivity.this,"ACNet GPU Mode enabled",Toast.LENGTH_SHORT).show();
                            Anime4KGPUCNN.initGPU();
                            GPUCNN = GPUCNNState.Initialized;
                        }
                        else
                        {
                            Toast.makeText(MainActivity.this,"Unsupported ACNet GPU Mode",Toast.LENGTH_SHORT).show();
                            GPUCNN = GPUCNNState.Unsupported;
                            buttonView.setChecked(false);
                        }
                    } catch (Exception exp) {
                        errorHandler(Error.Anime4KCPPError, exp);
                        GPUCNN = GPUCNNState.Unsupported;
                        buttonView.setChecked(false);
                    }
                }
                else if (isChecked && GPUCNN == GPUCNNState.Unsupported)
                {
                    Toast.makeText(MainActivity.this,"Unsupported ACNet GPU Mode",Toast.LENGTH_SHORT).show();
                    buttonView.setChecked(false);
                }
            }
        });

        final Switch preprocessingCAS = findViewById(R.id.switchPreprocessingCAS);
        preprocessingCAS.setChecked(true);
        final Switch postprocessingGaussianBlurWeak = findViewById(R.id.switchPostprocessingGaussianBlurWeak);
        postprocessingGaussianBlurWeak.setChecked(true);
        final Switch postprocessingBilateralFilter = findViewById(R.id.switchPostprocessingBilateralFilter);
        postprocessingBilateralFilter.setChecked(true);
    }

    private boolean getSwitchSate(int id) {
        Switch sw = findViewById(id);
        return sw.isChecked();
    }

    private void errorHandler(@NonNull Error errorType,@Nullable Exception exp) {
        switch (errorType) {
            case Anime4KCPPError:
                assert exp != null;
                new AlertDialog.Builder(MainActivity.this)
                        .setTitle("ERROR")
                        .setMessage(exp.getMessage())
                        .setPositiveButton("OK",null)
                        .show();
                break;
            case FailedToCreateFolders:
                new AlertDialog.Builder(MainActivity.this)
                        .setTitle("ERROR")
                        .setMessage("Failed to create folders")
                        .setPositiveButton("OK",null)
                        .show();
                break;
            case FailedToDeleteTmpFile:
                new AlertDialog.Builder(MainActivity.this)
                        .setTitle("ERROR")
                        .setMessage("Failed to delete Temporary output video file, delete it manually.")
                        .setPositiveButton("OK",null)
                        .show();
                break;
        }
    }

    private Anime4K initAnime4KCPP() {
        int passes = 2;
        int pushColorCount=2;
        double strengthColor = 0.3;
        double strengthGradient = 1;
        double zoomFactor=2;
        boolean fastMode = getSwitchSate(R.id.switchFastMode);
        boolean preprocessing = false;
        boolean postprocessing = false;
        byte preFilters=0;
        byte postFilters=0;

        RadioGroup settingsGroup = findViewById(R.id.radioGroupSettings);
        switch (settingsGroup.getCheckedRadioButtonId()) {
            case R.id.radioButtonFast:
                passes = 1;
                preprocessing=false;
                postprocessing=false;
                break;
            case R.id.radioButtonBalance:
                passes = 2;
                preprocessing=false;
                postprocessing=false;
                break;
            case R.id.radioButtonQuality:
                passes = 2;
                preprocessing=true;
                postprocessing=true;
                preFilters=4;
                postFilters=48;
                break;
            case  R.id.radioButtonCustom:
                passes = Integer.parseInt(((EditText)findViewById(R.id.editTextPasses)).getText().toString());
                pushColorCount = Integer.parseInt(((EditText)findViewById(R.id.editTextPushColorCount)).getText().toString());
                strengthColor = Double.parseDouble(((EditText)findViewById(R.id.editTextStrengthColor)).getText().toString());
                strengthGradient = Double.parseDouble(((EditText)findViewById(R.id.editTextStrengthGradien)).getText().toString());
                zoomFactor = Double.parseDouble(((EditText)findViewById(R.id.editTextZoomFactor)).getText().toString());
                preprocessing = ((Switch)findViewById(R.id.switchPreprocessing)).isChecked();
                postprocessing = ((Switch)findViewById(R.id.switchPostprocessing)).isChecked();

                if(((Switch)findViewById(R.id.switchPreprocessingMedianBlur)).isChecked())
                    preFilters|=1;
                if(((Switch)findViewById(R.id.switchPreprocessingMeanBlur)).isChecked())
                    preFilters|=1<<1;
                if(((Switch)findViewById(R.id.switchPreprocessingCAS)).isChecked())
                    preFilters|=1<<2;
                if(((Switch)findViewById(R.id.switchPreprocessingGaussianBlurWeak)).isChecked())
                    preFilters|=1<<3;
                if(((Switch)findViewById(R.id.switchPreprocessingGaussianBlur)).isChecked())
                    preFilters|=1<<4;
                if(((Switch)findViewById(R.id.switchPreprocessingBilateralFilter)).isChecked())
                    preFilters|=1<<5;
                if(((Switch)findViewById(R.id.switchPreprocessingBilateralFilterFaster)).isChecked())
                    preFilters|=1<<6;

                if(((Switch)findViewById(R.id.switchPostprocessingMedianBlur)).isChecked())
                    postFilters|=1;
                if(((Switch)findViewById(R.id.switchPostprocessingMeanBlur)).isChecked())
                    postFilters|=1<<1;
                if(((Switch)findViewById(R.id.switchPostprocessingCAS)).isChecked())
                    postFilters|=1<<2;
                if(((Switch)findViewById(R.id.switchPostprocessingGaussianBlurWeak)).isChecked())
                    postFilters|=1<<3;
                if(((Switch)findViewById(R.id.switchPostprocessingGaussianBlur)).isChecked())
                    postFilters|=1<<4;
                if(((Switch)findViewById(R.id.switchPostprocessingBilateralFilter)).isChecked())
                    postFilters|=1<<5;
                if(((Switch)findViewById(R.id.switchPostprocessingBilateralFilterFaster)).isChecked())
                    postFilters|=1<<6;
                break;
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
            postFilters
        );

        if (getSwitchSate(R.id.switchACNet))
        {
            if (getSwitchSate(R.id.switchACNetGPUMode))
            {
                return anime4KCreator.create(parameters, ProcessorType.GPUCNN);
            }
            else
            {
                return anime4KCreator.create(parameters, ProcessorType.CPUCNN);
            }
        }
        else
        {
            if (getSwitchSate(R.id.switchGPUMode))
            {
                return anime4KCreator.create(parameters, ProcessorType.GPU);
            }
            else
            {
                return anime4KCreator.create(parameters, ProcessorType.CPU);
            }
        }
    }


    @SuppressLint("StaticFieldLeak")
    class Anime4KProcessor extends AsyncTask<String, Integer, Double> {

        //private final boolean GPU;
        Button buttonStart = findViewById(R.id.buttonStart);

        @Override
        protected void onPreExecute() {
            buttonStart.setEnabled(false);
            Toast.makeText(MainActivity.this,"Processing...",Toast.LENGTH_SHORT).show();
            textViewState.setText(R.string.processing);
        }

        @Override
        protected Double doInBackground(@NonNull String... strings) {
            String prefix = strings[0], dst = strings[1];
            long totalTime = 0;
            int taskCount = adapterForProcessingList.getCount();

            File dstPath = new File(dst);
            if(!dstPath.exists() && !dstPath.mkdirs())
             {
                 Message message = new Message();
                 message.obj = Error.FailedToCreateFolders;
                 otherErrorHandler.sendMessage(message);
             }

            List<Pair<String,Integer>> images = new ArrayList<>();
            List<Pair<String,Integer>> videos = new ArrayList<>();

            for(int i = 0; i < taskCount; i++)
            {
                String filePath = adapterForProcessingList.getItem(i);
                assert filePath != null;
                FileType fileType = getFileType(filePath);
                if(fileType == FileType.Image)
                    images.add(Pair.create(filePath,i));
                else if(fileType == FileType.Video)
                    videos.add(Pair.create(filePath,i));
            }

            Anime4K anime4K = initAnime4KCPP();

            int imageCount = 0, videoCount = 0;

            try {
                anime4K.setVideoMode(false);
                for (Pair<String,Integer> image: images) {
                    File srcFile = new File(image.first);
                    anime4K.loadImage(srcFile.getPath());

                    long start = System.currentTimeMillis();
                    anime4K.process();
                    long end = System.currentTimeMillis();

                    anime4K.saveImage(dst+"/"+prefix+srcFile.getName());

                    totalTime += end - start;
                    publishProgress((++imageCount) * 100 / taskCount, image.second);
                }

                anime4K.setVideoMode(true);
                for (Pair<String,Integer> video: videos) {
                    File srcFile = new File(video.first);
                    String tmpOutputPath = dst+"/" + "tmpOutput" + videoCount +".mp4";
                    String OutputPath = dst+"/" + prefix+srcFile.getName() + ".mp4";

                    anime4K.loadVideo(srcFile.getPath());
                    anime4K.setVideoSaveInfo(tmpOutputPath);

                    long start = System.currentTimeMillis();
                    anime4K.process();
                    long end = System.currentTimeMillis();

                    anime4K.saveVideo();

                    new VideoAudioProcessor(srcFile.getPath(), tmpOutputPath, OutputPath).merge();

                    if (!(new File(tmpOutputPath).delete()))
                    {
                        Message message = new Message();
                        message.obj = Error.FailedToDeleteTmpFile;
                        otherErrorHandler.sendMessage(message);
                    }
                    totalTime += end - start;
                    publishProgress((++videoCount + imageCount) * 100 / taskCount, video.second);
                }
            } catch (Exception exp) {
                Message message = new Message();
                message.obj = exp.getMessage();
                anime4KCPPHandler.sendMessage(message);
                return 0.0;
            }

            return (double)(totalTime)/1000.0;
        }

        @Override
        protected void onProgressUpdate(@NonNull Integer... values) {
            mainProgressBar.setProgress(values[0]);
            processingList.setItemChecked(values[1], true);
        }

        @Override
        protected void onPostExecute(Double aDouble) {
            buttonStart.setEnabled(true);

            if (aDouble==0.0)
            {
                mainProgressBar.setProgress(0);
                return;
            }

            new AlertDialog.Builder(MainActivity.this)
                    .setTitle("NOTICE")
                    .setMessage("Finished in "+aDouble.toString()+"s")
                    .setPositiveButton("OK",null)
                    .show();
            textViewState.setText(R.string.waitting);
            mainProgressBar.setProgress(0);
        }
    }

}
