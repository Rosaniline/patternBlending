//
//  sliceSynthesis.h
//  repetitionExtract
//
//  Created by Rosani Lin on 13/5/28.
//  Copyright (c) 2013年 Rosani Lin. All rights reserved.
//

#ifndef __repetitionExtract__sliceSynthesis__
#define __repetitionExtract__sliceSynthesis__

#include <iostream>
#include <stack>
#include <sys/stat.h>

#include "Rosaniline.h"
#include "tattingPattern.h"
#include "BidirectSimilarity.h"

//#define PI 3.14159265359


class sliceSynthesis {
    
public:
    
    sliceSynthesis();
    ~sliceSynthesis();
    
    
    Mat synthesis (const tattingPattern& tatting);
    
    
private:
    
    Mat circularGaussianRG (double ratio, double sigma, const Point& centroid, Size canvas_size, int max_radius);
    
    Mat sliceGaussianRG (int slice_num, double sigma, const Point& centroid, Size canvas_size, int max_radius, double sym_angle);
    
    Mat alphaBlending (const Mat& src, const Point& centroid, int max_radius, const Mat& dst, const Mat& weight);
    
    Mat hybrid (const tattingPattern& src, const Mat& similar, double layer_ratio, double layer_sigma, double slice_sigma);
    
    Mat sliceHybrid (const tattingPattern& src, const Mat& similar, const Mat& weight, double layer_ratio, double layer_sigma, double slice_sigma);
    
    Mat hybridweight (tattingPattern src, double resize_ratio, double layer_ratio, double layer_sigma, double slice_sigma);
    
    void randomResizing (Mat& src, const Point& centroid, double ratio);
    
    Mat reconstSlice (const tattingPattern& tatting, const Mat& slice);
    
    Mat createSelfSimilar (const Mat& src, const Point& centroid, double scale, double angle);
    
    void connectedCompRefine (Mat &src);
    

};




#endif /* defined(__repetitionExtract__sliceSynthesis__) */
