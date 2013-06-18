//
//  repetitionExtract.cpp
//  repetitionExtract
//
//  Created by Rosani Lin on 13/4/15.
//  Copyright (c) 2013年 Rosani Lin. All rights reserved.
//

#include "detectSliceNum.h"


detectSliceNum::detectSliceNum () {}

detectSliceNum::~detectSliceNum () {}


int detectSliceNum::extractSliceNum (const Mat& pattern, const Point& centroid) {

    
    Mat shrink_img = pattern.clone();
    int slices = 0;
    
    resizeMat(shrink_img, 0.5);
    
    
    Mat rotate_diff = getRotateDiff(pattern, centroid);
    
    threshold(rotate_diff, rotate_diff, 0, 255, THRESH_OTSU);
    
//    Mat temp = Mat(rotate_diff.rows, 100, CV_64FC1);
//    for (int i = 0; i < temp.rows; i ++) {
//        for (int j = 0; j < temp.cols; j ++) {
//            
//            temp.at<double>(i, j) = rotate_diff.at<uchar>(i, 0);
//            
//        }
//    }
//    
//    showMat(temp);
    
    
    for (int i = 1; i < rotate_diff.rows; i ++) {
        
        if ( rotate_diff.at<uchar>(i, 0) > rotate_diff.at<uchar>(i - 1, 0)) {
            slices ++;
        }
    }
    
    
    return slices;

    
    

//
//    

//    
//    Mat slice_clip = tatting.pattern.clone();
//    
//    Mat temp = Mat::zeros(tatting.pattern.size(), CV_64FC1);
//    
//    
//    for (int i = 2/*slices/2 - 1*/; i <= 2/* slices/2*/; i ++) {
//        
//        
//
//        temp.setTo(0);
//        
//        double m1 = tan(((i - 1)*(360/slices) + sym_angle)/180*PI), m2 = tan((i*(360/slices) + sym_angle)/180*PI);
//        
//        for (int i = 0; i < temp.rows; i ++) {
//            for (int j = 0; j < temp.cols; j ++) {
//                
//                if ( (j - tatting.centroid.x)*m1 + (i - tatting.centroid.y) >= 0 && (j - tatting.centroid.x)*m2 + (i - tatting.centroid.y) < 0 ) {
//                    
//                    temp.at<double>(i, j) = 1;
//                }
//                
//                else {
//                    
//                    slice_clip.at<uchar>(i, j) = 0;
//                }
//                
//                
//            }
//        }
//        
////        showMat(temp);
//    }
//    
////    bitwise_and(slice_clip, temp, slice_clip);
//    
//    showMat(slice_clip);
//    
//    Mat slice_temp = slice_clip.clone();
//    
//    
//    
//    
//    Mat output = slice_clip.clone();
//    
//    for (int i = 0; i < slices; i ++) {
//        
//        Mat rot_mat = getRotationMatrix2D(tatting.centroid, (360.0/slices)*(i + 1), 1.0);
//        
//        warpAffine(slice_clip, slice_temp, rot_mat, slice_clip.size());
//        
//        bitwise_or(output, slice_temp, output);
//        
//        showMat(output);
//    }
    

    

    
    

    

    

}





Mat detectSliceNum::getRotateDiff (const Mat& pattern, const Point& centroid) {
    
    Mat ang_diff = Mat(360, 1, CV_64FC1);
    double diff_max = 0.0, diff_min = 0.0;
    
    for (double ang = 0; ang < 360.0; ang += 1) {
        
        Mat rotate_mat = getRotationMatrix2D(centroid, ang, 1.0);
        
        Mat rotated_img = Mat::zeros(pattern.size(), CV_8UC1);
        
        warpAffine(pattern, rotated_img, rotate_mat, rotated_img.size());
        
        
        
        Mat diff = Mat::zeros(rotated_img.size(), CV_8UC1);
        absdiff(pattern, rotated_img, diff);
        
        ang_diff.at<double>(ang, 0) = sum(diff)[0];
        
    }
    
    
    minMaxIdx(ang_diff, &diff_min, &diff_max);
    
    ang_diff = (ang_diff - diff_min)*(255/(diff_max - diff_min));
    
    ang_diff.convertTo(ang_diff, CV_8UC1);
    
    
    return ang_diff;
    
}



