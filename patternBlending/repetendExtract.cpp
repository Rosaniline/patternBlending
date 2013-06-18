//
//  repetendExtract.cpp
//  repetitionExtract
//
//  Created by Rosani Lin on 13/5/6.
//  Copyright (c) 2013年 Rosani Lin. All rights reserved.
//

#include "repetendExtract.h"

repetendExtract::repetendExtract() {};

repetendExtract::~repetendExtract() {};


bool aa(const element A, const element B) {
    
    return A.min_radius < B.min_radius;
}


void repetendExtract::extract(tattingPattern tatting) {
    
//    SIFTFeatureSaliency(pattern);
    
    vector<vector<Point> > elements = extractElementBoundary(tatting);
    
    vector<element> re_element;
    
    for (int i = 0; i < elements.size(); i ++) {
        
        if ( contourArea(elements[i]) > 20 ) {
            element local_element = element(elements[i], tatting);
            
            re_element.push_back(local_element);
        }
        

    }
    
    
    Point centroid = tatting.centroid;//Point(296, 301);//getCentroid(src);

    sort(re_element.begin(), re_element.end(), aa);
    
    re_element.erase(re_element.begin());
    
    Mat plot = Mat::zeros(tatting.pattern.size(), CV_8UC1);
    
    int class_id_counter = 0;
    
    for (int i = 0; i < re_element.size(); i ++) {
        
        plot.setTo(0);
        
        vector<int> similar_idx;
        vector<int> temp_distance;
        
        for (int j = 0; j < re_element.size(); j ++) {
            
            if ( i != j && ((re_element[j].min_radius <= re_element[i].min_radius && re_element[j].max_radius > re_element[i].min_radius) || (re_element[j].min_radius >= re_element[i].min_radius && re_element[j].min_radius < re_element[i].max_radius ) ) ){
                
                temp_distance.push_back(abs(re_element[i].area - re_element[j].area));
                
                similar_idx.push_back(j);

//                vector< vector<Point> > temp_contour;
//                temp_contour.push_back(re_element[j].contour);
//                drawContours(plot, temp_contour, 0, 100);
//                vector< vector<Point> >().swap(temp_contour);
            }
            
        }
        
        double dis_threshold = *min_element(temp_distance.begin(), temp_distance.end())*1.5;
        
        for (int i = 0; i < similar_idx.size(); ) {
            
            if ( temp_distance[i] > dis_threshold ) {
                
                temp_distance.erase(temp_distance.begin() + i);
                similar_idx.erase(similar_idx.begin() + i);
                
            }
            
            else {
                i ++;
            }
            
        }
        
        
        vector< vector<Point> > temp_contour;
        temp_contour.push_back(re_element[i].contour);
        drawContours(plot, temp_contour, 0, 255);
//
//        if ( re_element[i].class_id == -1 ) {
//            
//            re_element[i].class_id = class_id_counter ++;
//            
//            cout<<"not found, assign id = "<<class_id_counter - 1;
//        }
        
        
        bool exist_found = false;
        int exist_id = 0;
        
        for (int k = 0; k < similar_idx.size(); k ++) {
            
            if ( re_element[similar_idx[k]].class_id != -1 ) {
                
                exist_found = true;
                exist_id = re_element[similar_idx[k]].class_id;
                
                break;
            }
            
//            vector< vector<Point> > temp_contour;
//            temp_contour.push_back(re_element[similar_idx[k]].contour);
//            drawContours(plot, temp_contour, 0, 100);
//            vector< vector<Point> >().swap(temp_contour);

        }
        
        if ( exist_found ) {
            for (int k = 0; k < similar_idx.size(); k ++) {
                
                re_element[similar_idx[k]].class_id = exist_id;
                
            }
            
            re_element[i].class_id = exist_id;
        }
        
        else {
            
            for (int k = 0; k < similar_idx.size(); k ++) {
                
                re_element[similar_idx[k]].class_id = class_id_counter;
                
            }
            
            re_element[i].class_id = class_id_counter;
            
            class_id_counter ++;
            
        }
        
        
//        showMat(plot, 0);
//        vector< vector<Point> >().swap(temp_contour);

        

        


    }
    
    
    plot.setTo(0);
    for (int i = 0; i < class_id_counter; i ++) {
        
        
        
        for (int j = 0; j < re_element.size(); j ++) {
            
            if ( re_element[j].class_id == i ) {
                
                vector< vector<Point> > temp_contour;
                temp_contour.push_back(re_element[j].contour);
                drawContours(plot, temp_contour, 0, 255);
                vector< vector<Point> >().swap(temp_contour);
                
            }
            
        }
        

        
    }
    
//    showMat(plot);
    
    imwrite("/Users/xup6qup3/Desktop/t.jpg", plot);

//    multimap<double, vector<Point> > concentrc_pt;
//    
//    for (int i = 0; i < elements.size(); i ++) {
//        
//        pair<double, vector<Point> > temp_pair;
//        
//        temp_pair.first = getElementRadius(elements[i], centroid);
//        temp_pair.second = elements[i];
//        
//        concentrc_pt.insert(temp_pair);
//        
//    }
//    
//
//    
//    
//    Mat temp = Mat::zeros(pattern.getPattern().size(), CV_8UC1);
//    
//    for (multimap<double, vector<Point> >::iterator it = concentrc_pt.begin(); it != concentrc_pt.end(); it ++) {
//        
//        if ( contourArea(it->second) > 20 ) {
//        
//            
//            for (int i = 0; i < it->second.size(); i ++) {
//                temp.at<uchar>(it->second[i]) = 255;
//            }
//            
//            showMat(temp, "t", 1);
//        }
//        
//    }
//    
//    showMat(temp);
    
    
//    for (int i = 0; i < elements.size(); i ++) {
//        
//        int min_idx = 0;
//        double min_error = INFINITY;
//        
//        for (int j = 0; j < elements.size(); j ++) {
//            
//            if ( contourArea(elements[i]) > 20 && contourArea(elements[j]) > 20 && i != j ) {
//
//                double local_error = matchShapes(elements[i], elements[j], CV_CONTOURS_MATCH_I1, 0);
//                
//    //            cout<<local_error<<", ";
//                
//                if ( local_error < min_error ) {
//                    
//                    min_error = local_error;
//                    min_idx = j;
//                }
//            }
//            
//        }
//        
//        Mat temp = Mat::zeros(pattern.getPattern().size(), CV_8UC1);
//        
//        for (int j = 0; j < elements[i].size(); j ++) {
//            temp.at<uchar>(elements[i][j]) = 255;
//        }
//        
//        showMat(temp, "t1", 1);
//        
//        temp.setTo(0);
//        
//        cout<<min_idx<<endl; 
//        
//        for (int j = 0; j < elements[min_idx].size(); j ++) {
//            temp.at<uchar>(elements[min_idx][j]) = 255;
//        }
//        
//        showMat(temp, "t2", 0);
//    }
    
}



vector<vector<Point> > repetendExtract::extractElementBoundary(tattingPattern tatting) {
    
    vector<vector<Point> > contours;
    
    Mat src_temp = tatting.pattern.clone();
    
    threshold(src_temp, src_temp, 0, 255, THRESH_OTSU);
    
    findContours(src_temp, contours, CV_RETR_CCOMP, CV_CHAIN_APPROX_NONE);
    
    src_temp.release();
    
    return contours;
    
}





double repetendExtract::getElementRadius (const vector<Point> &element, const Point &centroid) {
    
    double avg_radius = 0.0;
    
    for (int i = 0; i < element.size(); i ++) {
        
        avg_radius += sqrt( pow(element[i].x - centroid.x, 2.0) + pow(element[i].y - centroid.y, 2.0) );
        
    }
    
    return avg_radius/element.size();
}


Point repetendExtract::getElementCentroid(const vector<Point> &element) {
    
    Point centroid(0);
    
    for (int i = 0; i < element.size(); i ++) {
        centroid += element[i];
    }
    
    centroid.x /= element.size();
    centroid.y /= element.size();
    
    return centroid;
}


void repetendExtract::plotContour(cv::Mat &img, const vector<Point> &contour) {
    
    for (int i = 0; i < contour.size(); i ++) {
        
        img.at<uchar>(contour[i]) = 255;
        
    }
    
}


void repetendExtract::SIFTFeatureSaliency(tattingPattern &tatting) {
    
    
    /* threshold      = 0.04;
     edge_threshold = 10.0;
     magnification  = 3.0;    */
    
    // SIFT feature detector and feature extractor
    cv::SiftFeatureDetector detector( 0.05, 5.0 );
    cv::SiftDescriptorExtractor extractor( 3.0 );
    
    /* In case of SURF, you apply the below two lines
     cv::SurfFeatureDetector detector();
     cv::SurfDescriptorExtractor extractor();
     */
    
    // Feature detection
    std::vector<KeyPoint> keypoints;
    detector.detect( tatting.pattern, keypoints );
    
    // Feature display
    Mat temp = Mat::zeros(tatting.pattern.size(), CV_64FC1);
    
    RNG rng = RNG();
    
    vector<double> response;
    for (int i = 0; i < keypoints.size(); i ++) {
        response.push_back(keypoints[i].response);
    }
    
    double max_response = *max_element(response.begin(), response.end()), min_response = *min(response.begin(), response.end());
    
    
    cout<<max_response<<", "<<min_response<<endl;
    
    double threshold = (max_response - min_response)/10;
    
    vector<double> similar_count;
    

    
    for (int i = 0; i < response.size(); i ++) {
        
        double local_count = 0;
        
        Mat tt = tatting.pattern.clone();
        
        tt -= 150;
        
        
        
        
        
        
        
        for (int j = 0; j < response.size(); j ++) {
            
            if ( i != j && abs(response[i] - response[j]) < threshold ) {
                
                circle(tt, keypoints[j].pt, 5, 200, 1);
                local_count ++;
            }
            
        }
        
        circle(tt, keypoints[i].pt, 7, 255, 2);
        
//        showMat(tt);
        
        similar_count.push_back(local_count);
    }
    
    
    normalizeVector(similar_count);
    
    for (int i = 0; i < similar_count.size(); i ++) {
        
//        cout<<similar_count[i]<<", ";
//        temp.at<double>(keypoints[i].pt) = similar_count[i];
        
        circle(temp, keypoints[i].pt, 3, similar_count[i], 3);
        
    }
    
//    showMat(temp);
    
    
    
    
//    double threshold = (*max_element(keypoint_res.begin(), keypoint_res.end()) - *min_element(keypoint_res.begin(), keypoint_res.end())) / 10;
    
//    vector<int> cohere_count;
//    
//    for (int i = 0; i < keypoints.size(); i ++) {
//        
//        int cohere = 0;
//        
//        for (int j = 0; j < keypoints.size(); j ++) {
//            
//            if ( threshold > abs(keypoints[i].response - keypoints[j].response) ) {
//                cohere ++;
//            }
//        
//        }
//        
//        cohere_count.push_back(cohere);
//        
//    }
//    
//    
//    for (int i = 0; i < cohere_count.size(); i ++) {
//        
//        int color = ((double)cohere_count[i] / *max_element(cohere_count.begin(), cohere_count.end()))*255;
//        
//        circle(temp, keypoints[i].pt, 3, CV_RGB(color, color, color), 3);
//        
//        
//    }
    
//    Mat gau_kernel = getGaussianKernel(23, 7.0);
//    
//    gau_kernel = gau_kernel*gau_kernel.t();
//    s
//    cout<<gau_kernel.size();
//    
//    
//    for (int i = 0; i < keypoints1.size(); i ++) {
//        
//        for (int m = -gau_kernel.rows/2; m <= gau_kernel.rows/2; m ++) {
//            for (int n = -gau_kernel.cols/2; n <= gau_kernel.cols/2; n ++) {
//                
//                temp.at<double>(keypoints1[i].pt.y + m, keypoints1[i].pt.x + n) += gau_kernel.at<double>(11 + m, 11 + n);
//                
//            }
//        }
//
//    }
//    
//    
//    double max, min;
//    
//    minMaxIdx(temp, &min, &max);
//    
//    temp = (temp - min) / (max - min);
//
//    
//    Mat tt = pattern.getPattern().clone();
//    tt.convertTo(tt, CV_64FC1);
    
//    temp = temp + tt;
    
    
    
//    showMat(temp);
    
    
}
















