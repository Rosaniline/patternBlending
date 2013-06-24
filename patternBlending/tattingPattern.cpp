//
//  tattingPattern.cpp
//  repetitionExtract
//
//  Created by Rosani Lin on 13/5/7.
//  Copyright (c) 2013年 Rosani Lin. All rights reserved.
//

#include "tattingPattern.h"

tattingPattern::tattingPattern(const Mat& img) {
    
    preprocess(img);
    slice = Mat::zeros(img.size(), CV_64FC1);
    slice_mask = Mat::zeros(img.size(), CV_64FC1);
    
    
    
    centroid = Point(296, 301);
//    centroid = Point(148, 150);
//    centroid = setRotMinCentroid();
//    cout<<centroid<<endl;
    
    max_radius = getMaxContainingRadius();
    
    slice_num = 10;
//    detectSliceNum sliceNum_detector = detectSliceNum();
//    slice_num = sliceNum_detector.extractSliceNum(pattern, centroid);
//    
//    cout<<slice_num<<endl;
    
    Mat rot_mat = getRotationMatrix2D(centroid, -2, 1.0);
    warpAffine(pattern, pattern, rot_mat, pattern.size(), INTER_CUBIC);

//    sym_angle = 92;
    sym_angle = 90;
    
//    symmetryDetection sym_detector = symmetryDetection();
//    sym_angle = sym_detector.detectSymmetryAngle(pattern, centroid);
//
//    sym_angle = symAngleFinder();
//    cout<<sym_angle<<endl;
    
//    sym_lines = symLineExtract();
    
//    minSectorExtraction();

    
    
    
    extractSlices();
    
    extractSliceBlend();
    
    sliceSelection();
    
    
    
    
    

   
    
    
    
}


tattingPattern::~tattingPattern() {
    
    pattern.release();
    centroid.~Point_();
    
    vector<Mat>().swap(slices);
    vector<Mat>().swap(slices_mask);
    
}


void tattingPattern::preprocess(const cv::Mat &img) {
    
    pattern = img.clone();
    
    if ( pattern.type() != CV_8UC1 && pattern.type() != CV_64FC1 ) {
        cvtColor(pattern, pattern, CV_BGR2GRAY);
        
        threshold(pattern, pattern, 0, 255, THRESH_OTSU);
        
        pattern.convertTo(pattern, CV_64FC1);
        pattern /= 255.0;
    }
    
}


Point tattingPattern::setRotMinCentroid() {
    
    vector<Mat> pattern_pyr = constructPyramid(pattern, ROT_MIN_PYRAMID_LEVEL);
    
    Point centroid_candi(0);
    
    
    for (int level = 0; level <= ROT_MIN_PYRAMID_LEVEL; level ++) {
        
        if ( level == 0 ) {
            
            Moments mo = moments(pattern_pyr[level], true);
            centroid_candi = Point((int)(mo.m10/mo.m00), (int)(mo.m01/mo.m00));
        }
        
        else {
            
            centroid_candi = centroid*2;
        }
    
        
        
        double min_error = INFINITY;
        
        #pragma omp parallel for
        for (int i = -ROT_MIN_SEARCH_WIDTH; i <= ROT_MIN_SEARCH_WIDTH; i ++) {
            for (int j = -ROT_MIN_SEARCH_WIDTH; j <= ROT_MIN_SEARCH_WIDTH; j ++) {
                
                vector<double> diff;
                double local_error = 0.0;
                
                for (double ang = 0; ang < 360.0; ang += 10) {
                    
                    Mat rotate_mat = getRotationMatrix2D(centroid_candi + Point(j, i), ang, 1.0);
                    
                    Mat rotated_img = Mat::zeros(pattern_pyr[level].size(), CV_8UC1);
                    
                    warpAffine(pattern_pyr[level], rotated_img, rotate_mat, pattern_pyr[level].size());
                
                    Mat rot_diff = Mat::zeros(pattern_pyr[level].size(), CV_8UC1);
                    absdiff(pattern_pyr[level], rotated_img, rot_diff);
                    
                    diff.push_back(sum(rot_diff)[0]);
                    
                }
                
                sort(diff.begin(), diff.end());
                
                local_error = accumulate(diff.begin(), diff.begin() + 10, 0);
                
                vector<double>().swap(diff);
                
                
                #pragma omp critical
                {
                    if ( min_error > local_error ) {
                        
                        centroid = centroid_candi + Point(j, i);
                        min_error = local_error;
                    }
                }
            }
        }
        
    }
    
    vector<Mat>().swap(pattern_pyr);
    
    return centroid;
}


void tattingPattern::extractSlices() {
    
    
    double ang_one = sym_angle, ang_two = ang_one + 360.0/slice_num;
    
    for (int s = 0; s < slice_num/2; s ++) {
        
        Mat mask_one = Mat::zeros(pattern.size(), CV_64FC1), mask_two = Mat::zeros(pattern.size(), CV_64FC1);

        for (int i = 0; i < mask_one.rows; i ++) {
            for (int j = 0; j < mask_one.cols; j ++) {
                
                int map_i = i - centroid.y, map_j = j - centroid.x, reverse_sign = 0;
                
                double radian_one = ang_one*PI/180.0, radian_two = ang_two*PI/180.0;
                double r = sqrt(pow(map_i, 2.0) + pow(map_j, 2.0));
                
                // to compensate the sign reversion of tan at 90 degree
                reverse_sign = (int)ang_two%180 > 90 && (int)ang_one%180 < 90 ? -1 : 1;
                
                
                Mat mask_local = map_j*tan(radian_one - PI/2) + map_i < 0 ? mask_one : mask_two;
                
                if ( reverse_sign*(map_j*tan(radian_one) + map_i)*(map_j*tan(radian_two) + map_i ) < 0 && r < max_radius ) {
                    
                    mask_local.at<double>(i, j) = 1.0;
                    
                }
                
            }
        }
        
        Mat slice_one = Mat::zeros(pattern.size(), CV_64FC1), slice_two = Mat::zeros(pattern.size(), CV_64FC1);
        
        bitwise_and(pattern, mask_one, slice_one);
        bitwise_and(pattern, mask_two, slice_two);
        
        slices_mask.push_back(mask_one);
        slices_mask.push_back(mask_two);
        
        slices.push_back(slice_one);
        slices.push_back(slice_two);
                
        
        ang_one += 360.0/slice_num;
        ang_two += 360.0/slice_num;
    }
    
    
}


void tattingPattern::extractSliceBlend() {
    
    
    double ang_one = sym_angle - BLEND_OFFSET, ang_two = ang_one + 360.0/slice_num + 2*BLEND_OFFSET;
    
    for (int s = 0; s < slice_num/2; s ++) {
        
        Mat mask_one = Mat::zeros(pattern.size(), CV_64FC1), mask_two = Mat::zeros(pattern.size(), CV_64FC1);
        
        for (int i = 0; i < mask_one.rows; i ++) {
            for (int j = 0; j < mask_one.cols; j ++) {
                
                int map_i = i - centroid.y, map_j = j - centroid.x, reverse_sign = 0;
                
                double radian_one = ang_one*PI/180.0, radian_two = ang_two*PI/180.0;
                double r = sqrt(pow(map_i, 2.0) + pow(map_j, 2.0));
                
                // to compensate the sign reversion of tan at 90 degree
                reverse_sign = (int)ang_two%180 > 90 && (int)ang_one%180 < 90 ? -1 : 1;
                
                
                Mat mask_local = map_j*tan(radian_one - PI/2) + map_i < 0 ? mask_one : mask_two;
                
                if ( reverse_sign*(map_j*tan(radian_one) + map_i)*(map_j*tan(radian_two) + map_i ) < 0 && r < max_radius ) {
                    
                    mask_local.at<double>(i, j) = 1.0;
                    
                }
                
            }
        }
        
        Mat slice_one = Mat::zeros(pattern.size(), CV_64FC1), slice_two = Mat::zeros(pattern.size(), CV_64FC1);
        
        bitwise_and(pattern, mask_one, slice_one);
        bitwise_and(pattern, mask_two, slice_two);
        
        blend_mask.push_back(mask_one);
        blend_mask.push_back(mask_two);
        
        slice_blend.push_back(slice_one);
        slice_blend.push_back(slice_two);
        
        
        ang_one += 360.0/slice_num;
        ang_two += 360.0/slice_num;
    }
    
    
}


void tattingPattern::sliceSelection() {
    
    
}


int tattingPattern::getMaxContainingRadius () {
    
    int min_radius = 0;
    
    for (int r = INNER_MOST_RADIUS; r < MIN(pattern.rows, pattern.cols)/2; r ++) {
        
        vector<Point>circle_p = getCirclePoints(centroid, r);
        
        for (int i = 0; i < circle_p.size(); i ++) {
            
            if ( pattern.at<double>(circle_p[i]) ) {
                
                min_radius = r;
                break;
            }
            
        }
        
        vector<Point>().swap(circle_p);
    }
    
    
    return min_radius;
}


vector<int> tattingPattern::symLineExtract() {
    
    vector<int> local_sym_lines;
    
    Mat boundary = Mat::zeros(slice_num, pattern.cols - centroid.x, CV_64FC1);
    
    for (int lines = 0; lines < slice_num; lines ++) {
        
        int min_angle = 0;
        double min_error = INFINITY;
        
        for (int near_angle = -5 + lines*(360/slice_num) + sym_angle; near_angle <= 5 + lines*(360/slice_num) + sym_angle; near_angle ++) {
            
            
            Mat rotated = Mat::zeros(pattern.size(), pattern.type());
            
            Mat rot_mat = getRotationMatrix2D(centroid, -(near_angle), 1.0);
            warpAffine(pattern, rotated, rot_mat, rotated.size());
            
            double sym_error = 0.0;
            for (int i = 1; i <= 3; i ++) {
                
                for (int j = centroid.x; j < rotated.cols; j ++) {
                    
                    sym_error += abs(rotated.at<double>(centroid.y + i, j) - rotated.at<double>(centroid.y - i, j))/i;
                    
                }
            }
            
            for (int j = centroid.x; j < rotated.cols; j ++) {
                boundary.at<double>(lines, j - centroid.x) = rotated.at<double>(centroid.y, j);
            }
            
            if ( sym_error < min_error ) {
                
                min_error = sym_error;
                min_angle = near_angle;
            }
        }
        
        local_sym_lines.push_back(min_angle);
        
    }
    
    
//    Mat plot = pattern.clone();
//    for (int i = 0; i < local_sym_lines.size(); i ++) {
//        
//        if ( local_sym_lines[i] > 180 ) {
//            cout<<local_sym_lines[i] - 180 <<endl;
//        }
//        
//        else {
//            cout<<local_sym_lines[i]<<endl;
//        }
//        infLine(plot, centroid, tan(local_sym_lines[i]*PI/180));
//        
//    }
    
    double min_error = INFINITY;
    int min_index = 0;
    
    for (int i = 0; i < boundary.rows; i ++) {
        
        double local_error = 0.0;
        for (int j = 0; j < boundary.cols; j ++) {
            local_error += abs(boundary.at<double>(i, j) - boundary.at<double>((i + 1)%boundary.rows, j));
        }
        
        if ( local_error < min_error ) {
            
            min_error = local_error;
            min_index = i;
        }
    }

    
    for (int i = 0; i < slice_num; i ++) {
        
        min_index = i;
        
        int ang_one = local_sym_lines[min_index]%360, ang_two = local_sym_lines[(min_index + 1)%slice_num]%360;
        
        if ( ang_two < ang_one ) {
            ang_two += 360;
        }
        
        cout<<ang_one<<", "<<ang_two<<endl;
        
        Mat mask = Mat::zeros(pattern.size(), CV_64FC1);
        
        for (int i = 0; i < mask.rows; i ++) {
            for (int j = 0; j < mask.cols; j ++) {
                
                int map_i = centroid.y - i, map_j = j - centroid.x;
                
                double r = sqrt(pow(map_i, 2.0) + pow(map_j, 2.0));
                
                int map_ang = (int)(atan2(map_i, map_j)*180.0/PI);
                
                map_ang = map_ang < 0 ? map_ang + 360 : map_ang;
                
                if ( ang_two > 360 && map_ang < 180 ) {
                    map_ang += 360;
                }
                
                if ( map_ang <= ang_two && map_ang >= ang_one && r < max_radius ) {
                    
                    mask.at<double>(i, j) = 1.0;
                }
                
            }
        }
        
        showMat(mask);
    }
    

    
//    Mat slice_one = Mat::zeros(pattern.size(), CV_64FC1), slice_two = Mat::zeros(pattern.size(), CV_64FC1);
    
//    bitwise_and(pattern, mask, slice_one);
//    bitwise_and(pattern, mask_two, slice_two);
    
//    showMat(mask_one);
    

    
    
//    cout<<min_index<<", "<<min_error;
//    
//    showMat(boundary, "bo", 1);
//    showMat(plot);
    
    
    
    return local_sym_lines;
    
}


int tattingPattern::symAngleFinder() {
    
    int min_degree = 0;
    double min_error = INFINITY;
    
    for (int rot_degree = 0; rot_degree < 180; rot_degree ++) {
        
        Mat rotated = Mat::zeros(pattern.size(), pattern.type());
        Mat rot_mat = getRotationMatrix2D(centroid, -rot_degree, 1.0);
        warpAffine(pattern, rotated, rot_mat, rotated.size(), CV_INTER_CUBIC);
        
        
        double local_error = 0.0;
        for (int i = 1; i <= 3; i ++) {
            
            for (int j = 0; j < rotated.cols; j ++) {
                
                local_error += abs(rotated.at<double>(centroid.y + i, j) - rotated.at<double>(centroid.y - i, j))/i;
            }
            
        }
        
        if ( local_error < min_error ) {
            
            min_error = local_error;
            min_degree = rot_degree;
        }
        
    }
    
    return min_degree;
    
}


void tattingPattern::minSectorExtraction() {
    
    Mat boudaries = Mat::zeros(360, pattern.cols - centroid.x, pattern.type());
    
    for (int degree = 0; degree < 360; degree ++) {
        
        Mat rot_mat = getRotationMatrix2D(centroid, -degree, 1.0), rotated;
        warpAffine(pattern, rotated, rot_mat, pattern.size(), CV_INTER_CUBIC);
        
        for (int j = 0; j < boudaries.cols; j ++) {
            
            boudaries.at<double>(degree, j) = rotated.at<double>(centroid.y, centroid.x + j);
        }
    }
    
    int opt_angle = 0;
    double min_error = INFINITY;
    
    for (int i = 0; i < boudaries.rows; i ++) {
        
        double local_error = 0.0;
        int count = 0;
        
        for (int j = 0; j < boudaries.cols; j ++) {
            
            local_error += boudaries.at<double>(i, j) != boudaries.at<double>((i + 36)%boudaries.rows, j) ? 1 : 0;
            
            if ( boudaries.at<double>(i, j) != 0) {
                count ++;
            }
        }
        
        local_error /= count;
        
        if ( local_error < min_error ) {
            
            min_error = local_error;
            opt_angle = i;
            
        }
    }
    
//    int ang_one = opt_angle, ang_two = (opt_angle + 360/slice_num);
//
//    cout<<ang_one<<", "<<ang_two<<endl;
//    
//    Mat mask = Mat::zeros(pattern.size(), CV_64FC1);
//    
//    for (int i = 0; i < mask.rows; i ++) {
//        for (int j = 0; j < mask.cols; j ++) {
//            
//            int map_i = centroid.y - i, map_j = j - centroid.x;
//            
//            double r = sqrt(pow(map_i, 2.0) + pow(map_j, 2.0));
//            
//            int map_ang = (int)(atan2(map_i, map_j)*180.0/PI);
//            
//            map_ang = map_ang < 0 ? map_ang + 360 : map_ang;
//            
//            if ( ang_two > 360 && map_ang < 180 ) {
//                map_ang += 360;
//            }
//            
//            if ( map_ang <= ang_two && map_ang >= ang_one && r < max_radius ) {
//                
//                mask.at<double>(i, j) = 1.0;
//            }
//            
//        }
//    }
    

    for (int i = 0; i < slice_mask.rows; i ++) {
        for (int j = 0; j < slice_mask.cols; j ++) {

            int map_i = centroid.y - i, map_j = j - centroid.x;

            double r = sqrt(pow(map_i, 2.0) + pow(map_j, 2.0));

            int map_ang = (int)(atan2(map_i, map_j)*180.0/PI);

            map_ang = map_ang < 0 ? map_ang + 360 : map_ang;

            if ( map_ang <= (180 + 180/slice_num) && map_ang >= (180 - 180/slice_num) && r < max_radius ) {

                slice_mask.at<double>(i, j) = 1.0;
            }
            
        }
    }

    
    Mat rot_mat = getRotationMatrix2D(centroid, 180 - (opt_angle + 180/slice_num), 1.0);
    warpAffine(pattern, slice, rot_mat, pattern.size(), INTER_CUBIC);
    
    multiply(slice, slice_mask, slice);
    
    showMat(slice);
    

    
}




