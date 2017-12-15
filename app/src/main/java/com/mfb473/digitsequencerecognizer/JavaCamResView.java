package com.mfb473.digitsequencerecognizer;

import android.content.Context;
import android.hardware.Camera;
import android.util.AttributeSet;
import android.widget.Toast;

import org.opencv.android.JavaCameraView;

import java.util.List;

public class JavaCamResView extends JavaCameraView {

    public JavaCamResView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public void setFocusMode (){

        Camera.Parameters params = mCamera.getParameters();

        List<String> FocusModes = params.getSupportedFocusModes();

        if (FocusModes.contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE))
            params.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE);

        mCamera.setParameters(params);
    }
}
