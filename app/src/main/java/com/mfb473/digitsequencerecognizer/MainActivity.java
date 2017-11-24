package com.mfb473.digitsequencerecognizer;

import android.Manifest;
import android.app.Activity;
import android.graphics.Bitmap;
import android.os.AsyncTask;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

import pub.devrel.easypermissions.AfterPermissionGranted;
import pub.devrel.easypermissions.AppSettingsDialog;
import pub.devrel.easypermissions.EasyPermissions;

public class MainActivity extends Activity implements
        EasyPermissions.PermissionCallbacks,
        CvCameraViewListener2 {
    private static final String TAG = "MainActivity";
    private static final int RC_CAMERA_AND_STORAGE = 123;

    private Mat mRgba;
    private Mat mGray;
    private Mat mOut;

    private CameraBridgeViewBase mOpenCvCameraView;
    private Button mButton;
    private TextView mTextView;

    private Classifier classifier;

    private static final String MODEL_FILE = "opt_mnist_convnet.pb";
    private static final String LABEL_FILE = "labels.txt";
    private static final int INPUT_SIZE = 28;
    private static final String INPUT_NAME = "conv2d_1_input";
    private static final String OUTPUT_NAME = "activation_6/Softmax";

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    System.loadLibrary("native-lib");
                    enablePreview();
                } break;
                default: super.onManagerConnected(status); break;
            }
        }
    };

    public void onDetectClick(){
        processImage(mGray.getNativeObjAddr(), mOut.getNativeObjAddr(), mRgba.getNativeObjAddr());
        Bitmap bitmap;
        Mat tmp = new Mat(mOut.rows(), mOut.cols(), CvType.CV_8UC4);
        Imgproc.cvtColor(mOut, tmp, Imgproc.COLOR_GRAY2RGBA);
        bitmap = Bitmap.createBitmap(tmp.cols(), tmp.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(tmp, bitmap);
        float[] pixels = new float[(int) mOut.total()];
        mOut.convertTo(mOut, CvType.CV_32F);
        mOut.get(0,0, pixels);
        for(int i=0; i<pixels.length; i++){
            pixels[i] /= 255;
        }
        int num_digits = pixels.length / 784;
        float[][] digits = new float[num_digits][784];
        for(int i=0; i<num_digits; i++){
            for(int j=0; j<784; j++) {
                int y = j / 28;
                int x = j % 28;
                digits[i][j] = pixels[y * num_digits * 28 + i * 28 + x];
            }
        }
        List<List<Classifier.Recognition>> results = new ArrayList<>();
        String string_digits = "Result: ";
        for(int i=0; i<num_digits; i++){
            results.add(classifier.recognizeImage(digits[i]));
            string_digits += results.get(i).get(0).getTitle();
        }
        mTextView.setText(string_digits);
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);

        mOpenCvCameraView = findViewById(R.id.main_surface);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        mButton = findViewById(R.id.button);
        mTextView = findViewById(R.id.textView);

        new LoadModel().execute();

    }

    private class LoadModel extends AsyncTask<Void, Void, Void>{

        @Override
        protected Void doInBackground(Void... voids) {
            classifier = TensorFlowImageClassifier.create(
                    getAssets(),
                    MODEL_FILE,
                    LABEL_FILE,
                    INPUT_SIZE,
                    INPUT_NAME,
                    OUTPUT_NAME);
            return null;
        }

        @Override
        protected void onPostExecute(Void aVoid) {
            mButton.setVisibility(View.VISIBLE);
            mButton.setOnClickListener(new View.OnClickListener() {
                @Override
                public void onClick(View v) {
                    onDetectClick();
                }
            });
        }
    }

    @AfterPermissionGranted(RC_CAMERA_AND_STORAGE)
    public void enablePreview(){
        Log.i(TAG, "OpenCV loaded successfully");
        String[] perms = {Manifest.permission.CAMERA, Manifest.permission.READ_EXTERNAL_STORAGE};
        if (EasyPermissions.hasPermissions(this, perms)) {
            mOpenCvCameraView.enableView();
        } else {
            EasyPermissions.requestPermissions(this, getString(R.string.camera_and_storage_rationale),
                    RC_CAMERA_AND_STORAGE, perms);
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onCameraViewStarted(int width, int height) {
        mRgba = new Mat(height, width, CvType.CV_8UC4);
        mGray = new Mat(height, width, CvType.CV_8UC1);
        mOut = new Mat();
    }

    public void onCameraViewStopped() {
        mRgba.release();
        mGray.release();
        mOut.release();
    }

    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        mRgba = inputFrame.rgba();
        mGray = inputFrame.gray();
        addGuideLines(mRgba.getNativeObjAddr());
        return mRgba;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        EasyPermissions.onRequestPermissionsResult(requestCode, permissions, grantResults, this);
    }

    @Override
    public void onPermissionsGranted(int requestCode, List<String> perms) {}

    @Override
    public void onPermissionsDenied(int requestCode, List<String> perms) {
        if (EasyPermissions.somePermissionPermanentlyDenied(this, perms)) {
            new AppSettingsDialog.Builder(this).build().show();
        }
    }

    public native void addGuideLines(long matAddrRgba);
    public native void processImage(long matAddrGr, long matAddrOut, long matAddrRgba);
}
