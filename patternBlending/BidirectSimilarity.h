//
//  BidirectSimilarity.h
//  BidirectSimilarity
//
//  Created by Rosani Lin on 12/9/20.
//  Copyright (c) 2012年 Rosani Lin. All rights reserved.
//

#ifndef BidirectSimilarity_BidirectSimilarity_h
#define BidirectSimilarity_BidirectSimilarity_h

#include "Rosaniline.h"
#include "PatchMatch.h"

#include <sstream>

class BidirectSimilarity {
    
public:
    
    BidirectSimilarity();
    
    Mat retargeting (const Mat& input_img, double resize_ratio);
    
    Mat reshuffling (const Mat& src_img, const Mat& shuffled_img, const RosMat<Point>& mask);
    
    Mat inpainting (const Mat& src_img, const Mat& mask);
    
    Mat reconstruct (const Mat& src_one, const Mat& src_two, const Mat& dst_img, const Mat& weight, const Mat& mask, const Mat& slice_mask, const Point& centroid);
    
private:


    
    double gradualResize (const cv::Mat &src, cv::Mat &dst, const Mat& mask, RosMat<Point>& reconst_src, RosMat<Point>& reconst_dst, bool randomInit);
    
    double correction (const cv::Mat &src, cv::Mat &dst, const Mat& mask, RosMat<Point>& reconst_src, RosMat<Point>& reconst_dst, bool randomInit);
    
    double gradualResize (const cv::Mat &src, cv::Mat &dst, RosMat<Point>& reconst_src, RosMat<Point>& reconst_dst, bool randomInit);
    
    
    double cohereTerm (const Mat& src, const Mat& dst, RosMat<Point>& reconst_dst, Mat& cohere_mat, Mat& cohere_count);
    
    double completeTerm (const Mat& src, const Mat& dst, RosMat<Point>& reconst_src, Mat& complete_mat, Mat& complete_count);
    
    
    
    
    

    
    vector< RosMat<Point> > constructRosMatPointPyramid (const RosMat<Point>& src, int pyramid_level);
    
    

    const static int    PATCHWIDTH         = 7;
    const static int    PYRAMID_LEVEL      = 2;
    const static int    MIN_ITERATION      = 20;
    const static int    INTER_DECREASE     = 10;
    
};



#endif
