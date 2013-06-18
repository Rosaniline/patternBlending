//
//  anchorDetection.cpp
//  repetitionExtract
//
//  Created by Rosani Lin on 13/5/19.
//  Copyright (c) 2013年 Rosani Lin. All rights reserved.
//

#include "anchorDetection.h"


anchorDetection::anchorDetection() {
    
}

anchorDetection::~anchorDetection() {
    
}


vector<anchorPoint> anchorDetection::detectAnchor(const Mat& pattern, const Point& centroid) {
    
    
    Mat thinned = thinningGuoHall(pattern);
    
//    Mat temp = thinned.clone();
//    vector< vector<Point> > pts;
//    findContours(temp, pts, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
//    
//    temp.setTo(0);
//    for (int i = 0; i < pts.size(); i ++) {
//        drawContours(temp, pts, i, 255);
//            showMat(temp);
//    }
    


    
    vector<anchorPoint> anchor_pts;
    
    for (int i = 0; i < thinned.rows*thinned.cols; i ++) {
        
        if ( thinned.at<uchar>(i/thinned.rows, i%thinned.rows) == 255 ) {
            
            Mat trace = Mat::zeros(thinned.size(), CV_8UC1);
            
            findRawAnchorPoints(thinned, trace, Point(i%thinned.rows, i/thinned.rows), anchor_pts);
            
            trace.release();
            
            break;
        }
        
    }
    
    
    consecutiveAnchorPtsGrouping(anchor_pts, pattern.size());
    
    establishAnchorPoints(thinned, anchor_pts);
    
    
//    thinned -= 100;
//    
//    Mat temp = thinned.clone().setTo(0);
//    for (int i = 0; i < anchor_pts.size(); i ++) {
//        
//        circle(thinned, anchor_pts[i].loc, 2, 255);
//        
//        temp.at<uchar>(anchor_pts[i].loc) = 255;
//        
//    }
//    
////    showMat(thinned, "g", 1);
////    showMat(temp, "thin", 0);
//    
//    
//    thinned.release();
    
    
    return anchor_pts;
}



void anchorDetection::findRawAnchorPoints (const cv::Mat &thinned, cv::Mat &trace, Point loc, vector<anchorPoint> &anchor_pts) {

    trace.at<uchar>(loc) = 255;
    
    int count = 0;
    for (int i = -1; i <= 1; i ++) {
        for (int j = -1; j <= 1; j ++) {
            
            Point nex_loc = loc + Point(j, i);
            
            if ( thinned.at<uchar>(nex_loc) == 255) {
                
                count ++;
                
                if ( trace.at<uchar>(nex_loc) != 255 ) {
                    findRawAnchorPoints(thinned, trace, nex_loc, anchor_pts);
                }
                
            }
       
            
        }
    }
    
    if ( count >= 4 ) {
        anchor_pts.push_back(anchorPoint(loc));
    }

}



void anchorDetection::consecutiveAnchorPtsGrouping (vector<anchorPoint>& anchor_pts, Size size) {
    
    vector<anchorPoint> grouped_pts;
    
    Mat plot = Mat::zeros(size, CV_8UC1);
    
    for (int i = 0; i < anchor_pts.size(); i ++) {
        
        plot.at<uchar>(anchor_pts[i].loc) = 255;
        
    }
    
    
    for (int i = 0; i < plot.rows; i ++) {
        for (int j = 0; j < plot.cols; j ++) {
            
            if ( plot.at<uchar>(i, j) == 255 ) {
                
                vector<Point> temp_group;
                
                gatherGroup(plot, temp_group, Point(j, i));
                
                Point group_center(0);
                for (int k = 0; k < temp_group.size(); k ++) {
                    
                    group_center += temp_group[k];
                    
                }
                
                group_center.x /= (int)temp_group.size();
                group_center.y /= (int)temp_group.size();
                
                int min_idx = -1;
                double min_error = INFINITY;
                for (int k = 0; k < temp_group.size(); k ++) {
                    
                    double local_error = pointDist(group_center, temp_group[k]);
                    
                    if (local_error < min_error) {
                        
                        min_error = local_error;
                        min_idx = k;
                    }
                    
                }
                
                grouped_pts.push_back(anchorPoint(temp_group[min_idx]));
                
                vector<Point>().swap(temp_group);

            }
            
        }
    }
    
    anchor_pts.swap(grouped_pts);
    
    vector<anchorPoint>().swap(grouped_pts);
    plot.release();

}


void anchorDetection::gatherGroup(cv::Mat &plot, vector<Point> &local_group, Point loc) {
    
    local_group.push_back(loc);
    plot.at<uchar>(loc) = 0;
    
    for (int i = -1; i <= 1; i ++) {
        for (int j = -1; j <= 1; j ++) {
            
            if ( plot.at<uchar>(loc + Point(j, i)) == 255 ) {
                
                gatherGroup(plot, local_group, loc + Point(j, i));
            }
            
        }
    }
    
}


void anchorDetection::establishAnchorPoints(const cv::Mat &thinned, vector<anchorPoint> &anchor_pts) {
    
    
    
    RosMat<int> anchor_id = RosMat<int>(thinned.rows, thinned.cols, -1);
    
    for (int i = 0; i < anchor_pts.size(); i ++) {
        
        anchor_id(anchor_pts[i].loc) = i;
        
    }
    
    Mat temp_thinned = thinned.clone();
    
    for (int i = 0; i < anchor_pts.size(); i ++) {
    
        establish_RecursiveBody(temp_thinned, anchor_pts, i, anchor_pts[i].loc, anchor_id);
    }
    
    
    

//    for (int i = 0; i < anchor_pts.size(); i ++) {
//        
//        temp_thinned = thinned.clone().setTo(0);
//        temp_thinned -= 100;
//        
//        for (int j = 0; j < anchor_pts.size(); j ++) {
//            
//            circle(temp_thinned, anchor_pts[j].loc, 1, 255, -1);
//            
////            temp_thinned.at<uchar>(anchor_pts[j].loc) = 255;
//        }
//        
//        circle(temp_thinned, anchor_pts[i].loc, 3, 255);
//        
//        for (int j = 0; j < anchor_pts[i].neighbor_id.size() ; j ++) {
//            circle(temp_thinned, anchor_pts[anchor_pts[i].neighbor_id[j]].loc, 2, 200);
//        }
//        
//        resizeMat(temp_thinned, 3.0);
//        showMat(temp_thinned);
//        
//    }
//    
//    
//    showMat(temp_thinned);
    
    
    temp_thinned.release();
    
}

void anchorDetection::establish_RecursiveBody (Mat& thinned, vector<anchorPoint>& anchor_pts, int pre_id, Point loc, const RosMat<int>& anchor_id ) {
    
//    showMat(thinned);
    
    thinned.at<uchar>(loc) = 0;
    
    bool anchor_nearby = false;
    
    for (int i = -1; i <= 1; i ++) {
        for (int j = -1; j <= 1; j ++) {
            
            Point nex_loc = loc + Point(j, i);
            
            if ( thinned.at<uchar>(nex_loc) == 255 ) {
                
                if ( anchor_id(nex_loc) != -1 ) {
                    
                    anchor_pts[pre_id].neighbor_id.push_back(anchor_id(nex_loc));
                    anchor_pts[anchor_id(nex_loc)].neighbor_id.push_back(pre_id);
                    
                    anchor_nearby = true;
                    
                }
                                
            }
        }
    }
    
    
    if ( !anchor_nearby ) {
        
        for (int i = -1; i <= 1; i ++) {
            for (int j = -1; j <= 1; j ++) {
                
                Point nex_loc = loc + Point(j, i);
                
                if ( thinned.at<uchar>(nex_loc) == 255 ) {
                    
                    establish_RecursiveBody(thinned, anchor_pts, pre_id, loc + Point(j, i), anchor_id);

                    
                }
            }
        }
    }
    
    
    
}



void anchorDetection::thinningGuoHallIteration(cv::Mat& im, int iter) {
    
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);
    
    for (int i = 1; i < im.rows; i++) {
        for (int j = 1; j < im.cols; j++) {
            
            uchar p2 = im.at<uchar>(i-1, j);
            uchar p3 = im.at<uchar>(i-1, j+1);
            uchar p4 = im.at<uchar>(i, j+1);
            uchar p5 = im.at<uchar>(i+1, j+1);
            uchar p6 = im.at<uchar>(i+1, j);
            uchar p7 = im.at<uchar>(i+1, j-1);
            uchar p8 = im.at<uchar>(i, j-1);
            uchar p9 = im.at<uchar>(i-1, j-1);
            
            int C  = (!p2 & (p3 | p4)) + (!p4 & (p5 | p6)) +
            (!p6 & (p7 | p8)) + (!p8 & (p9 | p2));
            int N1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8);
            int N2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9);
            int N  = N1 < N2 ? N1 : N2;
            int m  = iter == 0 ? ((p6 | p7 | !p9) & p8) : ((p2 | p3 | !p5) & p4);
            
            if (C == 1 && (N >= 2 && N <= 3) & (m == 0))
                marker.at<uchar>(i,j) = 1;
        }
    }
    
    im &= ~marker;
}


Mat anchorDetection::thinningGuoHall(const cv::Mat& mat) {
    
    
    Mat im = mat.clone();
    
    im /= 255;
    
    cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
    cv::Mat diff;
    
    do {
        thinningGuoHallIteration(im, 0);
        thinningGuoHallIteration(im, 1);
        cv::absdiff(im, prev, diff);
        im.copyTo(prev);
    }
    while (cv::countNonZero(diff) > 0);
    
    im *= 255;
    
    return im;
}





























