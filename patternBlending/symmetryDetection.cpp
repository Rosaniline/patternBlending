//
//  symmetryDetection.cpp
//  repetitionExtract
//
//  Created by Rosani Lin on 13/4/16.
//  Copyright (c) 2013年 Rosani Lin. All rights reserved.
//

#include "symmetryDetection.h"

symmetryDetection::symmetryDetection() {
    
}


symmetryDetection::~symmetryDetection() {
    
}


double symmetryDetection::detectSymmetryAngle (const Mat& pattern, const Point& centroid) {
    
    
    vector<int> angles;
    
    for (int radius = CONCENTRIC_MIN_RADIUS; radius < getMinContainingRadius(pattern, centroid); radius += CONCENTRIC_SAMPLE_STEP) {
        
        vector<Point> circle_pts = getCirclePoints(centroid, radius);

        
        Mat circle_pt_mat = Mat((int)circle_pts.size(), 1, CV_64FC1);
        Mat convolved_mat = Mat((int)circle_pts.size(), 1, CV_64FC1);
        
        
        for (int i = 0; i < circle_pts.size(); i ++) {
            
            circle_pt_mat.at<double>(i, 0) = pattern.at<double>(circle_pts[i]);
            
        }
        
        
        for (int i = 0; i < circle_pts.size(); i ++) {
            
            convolved_mat.at<double>(i, 0) = convolveMat(circle_pt_mat);
            shiftMatElement(circle_pt_mat);
            
        }
        
        double max = 0.0;
        int idx = 0;
        
        for (int i = 0; i < convolved_mat.rows; i ++) {
            
            if ( max < convolved_mat.at<double>(i, 0) ) {
                
                max = convolved_mat.at<double>(i, 0);
                idx = i;
            }
        }
        
        int angle = ceil(135 - idx*(360.0 / convolved_mat.rows));
        angle = angle > 0 ? angle : 360 + angle;
        
        angles.push_back(angle);
        
        
    }
    
    
    return mode(angles);

    
}


int symmetryDetection::getMinContainingRadius (const Mat& pattern, const Point& centroid) {
    
    int min_radius = 0;
    
    for (int r = INNER_MOST_RADIUS; r < pattern.rows/2; r ++) {
        
        vector<Point>circle_p = getCirclePoints(centroid, r);
        
        for (int i = 0; i < circle_p.size(); i ++) {
            
            if ( pattern.at<double>(circle_p[i]) ) {
                
                min_radius = r;
                break;
            }
             
        }
    }
    
    
    return min_radius;    
}




double symmetryDetection::convolveMat(const cv::Mat &mat) {
    
    double value = 0.0;
    
    for (int i = 0; i < mat.rows; i ++) {
        
        value += mat.at<double>(i, 0)*mat.at<double>(mat.rows - 1 - i, 0);
    }
    
    return value;
}


void symmetryDetection::shiftMatElement(cv::Mat &mat) {
    
    double temp = mat.at<double>(0, 0);
    
    for (int i = 1; i < mat.rows; i ++) {
        
        mat.at<double>(i - 1, 0) = mat.at<double>(i, 0);
    }
    
    mat.at<double>(mat.rows - 1) = temp;
    
}


int symmetryDetection::mode(const vector<int>& array) {
    
    vector<int> angles, counts;
    
    for (int idx = 0; idx < array.size(); idx ++) {
        
        bool found = false;
        for (int i = 0; i < angles.size(); i ++) {
            
            if ( angles[i] == array[idx] ) {
                counts[i] ++;
                found = true;
                break;
            }
        }
        
        if ( !found ) {
            angles.push_back(array[idx]);
            counts.push_back(1);
        }
        
    }
    
    int max_index = -1;
    int max_temp = 0;
    
    for (int i = 0; i < counts.size(); i ++) {
        
        if ( max_temp < counts[i] ) {
            max_temp = counts[i];
            max_index = i;
        }
    }
    

    
    return angles[max_index];

}









