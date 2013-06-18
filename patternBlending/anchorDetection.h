//
//  anchorDetection.h
//  repetitionExtract
//
//  Created by Rosani Lin on 13/5/19.
//  Copyright (c) 2013年 Rosani Lin. All rights reserved.
//

#ifndef __repetitionExtract__anchorDetection__
#define __repetitionExtract__anchorDetection__

#include <iostream>
#include "Rosaniline.h"
#include "tattingPattern.h"


class anchorPoint {
    
public:
    
    anchorPoint(const Point& pt) {
        
        loc = pt;
    };
    
    ~anchorPoint() {
    
        loc.~Point();
        vector<int>().swap(neighbor_id);
    };
    
    void addNeighbor(int nei_id) {
        
        neighbor_id.push_back(nei_id);
    }
      
    Point loc;
    vector<int> neighbor_id;

};




class anchorDetection {
    
public:
    
    anchorDetection();
    ~anchorDetection();
    
    vector<anchorPoint> detectAnchor (const Mat& pattern, const Point& centroid);
    
private:
    
    // Return points which possesses more than three neighbors
    void findRawAnchorPoints (const Mat& thinned, Mat& trace, Point loc, vector<anchorPoint>& anchor_pts);

    // Group consecutive anchor point candidates
    void consecutiveAnchorPtsGrouping (vector<anchorPoint>& anchor_pts, cv::Size size);
    
    // Group connected anchor point candidates, recursive body
    void gatherGroup (Mat& plot, vector<Point>& local_group, Point loc);
    
    
    void establishAnchorPoints (const Mat& thinned, vector<anchorPoint>& anchor_pts);
    
    void establish_RecursiveBody (Mat& thinned, vector<anchorPoint>& anchor_pts, int pre_id, Point loc, const RosMat<int>& anchor_id );
    
    
    
    /*** thinning process @ http://opencv-code.com/quick-tips/implementation-of-guo-hall-thinning-algorithm/ ***/
    
    void thinningGuoHallIteration(cv::Mat& im, int iter);
    
    Mat thinningGuoHall(const cv::Mat& mat);
    
    
};




#endif /* defined(__repetitionExtract__anchorDetection__) */
