#include <jni.h>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <vector>

using namespace std;
using namespace cv;

extern "C" {

JNIEXPORT void JNICALL Java_com_mfb473_digitsequencerecognizer_MainActivity_addGuideLines(JNIEnv *env, jobject instance,
                                                                                          jlong addrRgba) {
    Mat& mRgba = *(Mat*)addrRgba;
    int h = mRgba.rows;
    int w = mRgba.cols;
    int y1 = h / 2 - h / 6;
    int y2 = h / 2 + h / 6;
    line(mRgba, Point(0, y1), Point(w, y1), Scalar(255, 0, 0));
    line(mRgba, Point(0, y2), Point(w, y2), Scalar(255, 0, 0));
}

JNIEXPORT void JNICALL Java_com_mfb473_digitsequencerecognizer_MainActivity_processImage(JNIEnv *env, jobject instance,
                                                                                          jlong addrGray, jlong addrOut, jlong addrRgba) {
    Mat& mGray = *(Mat*)addrGray;
    Mat* mOut = (Mat*)addrOut;
    Mat& mRgba = *(Mat*)addrRgba;
    int h = mGray.rows;
    int w = mGray.cols;
    int y1 = h / 2 - h / 6;
    int roi_edge = h / 3;

    Mat mInter(roi_edge, w, CV_8UC1, Scalar(0));
    adaptiveThreshold(mGray(Rect(0, y1, w, roi_edge)), mInter, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 61, 15);

    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours(mInter.clone(), contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    int num_contours = contours.size();
    vector<vector<Point> > contours_poly(num_contours);
    vector<Rect> boundRect(num_contours);
    for (int i = 0; i < contours.size(); i++) {
        approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
        boundRect[i] = boundingRect(Mat(contours_poly[i]));
    }

    for (int i = 0; i < num_contours; i++) {
        for (int j = i + 1; j < num_contours; j++) {
            if (boundRect[i].x > boundRect[j].x)
                iter_swap(boundRect.begin() + i, boundRect.begin() + j);
        }
    }

    int sum = 0;
    vector<int> areas(num_contours);
    for (int i = 0; i < num_contours; i++) {
        areas[i] = boundRect[i].height * boundRect[i].width;
        sum += areas[i];
    }
    float mean = sum / num_contours;

    vector<int> digit_idx(num_contours);
    int num_digits = 0;
    for (int i = 0; i < num_contours; i++) {
        if (areas[i] >= mean / 8) {
            digit_idx[num_digits] = i;
            num_digits += 1;
        }
    }

    mOut -> create(28, num_digits * 28, CV_8UC1);
    Mat& mAddr = *mOut;
    mAddr = Mat::zeros(28, num_digits * 28, CV_8UC1);

    for (int i = 0; i < num_digits; i++) {
        int h = boundRect[digit_idx[i]].height;
        int w = boundRect[digit_idx[i]].width;
        int max_edge = max(h, w);
        Mat digit = Mat::zeros(max_edge, max_edge, CV_8UC1);
        mInter(boundRect[digit_idx[i]]).copyTo(digit(Rect((max_edge - w) / 2, (max_edge - h) / 2, w, h)));
        Moments m = moments(digit);
        Mat deskewed_digit = Mat::zeros(digit.rows, digit.cols, digit.type());
        if (abs(m.mu02) >= 0.01) {
            double skew = m.mu11 / m.mu02;
            Mat warpMat = Mat_<double>(2, 3);
            warpMat.at<double>(0, 0) = 1;
            warpMat.at<double>(0, 1) = skew;
            warpMat.at<double>(0, 2) = -10 * skew;
            warpMat.at<double>(1, 0) = 0;
            warpMat.at<double>(1, 1) = 1;
            warpMat.at<double>(1, 2) = 0;
            warpAffine(digit, deskewed_digit, warpMat, deskewed_digit.size(), WARP_INVERSE_MAP | INTER_LINEAR);
        }
        else {
            digit.copyTo(deskewed_digit);
        }
        resize(digit, mAddr(Rect(28 * i + 4, 4, 20, 20)), Size(20, 20));
    }
}
}