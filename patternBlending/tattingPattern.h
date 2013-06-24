//
//  tattingPattern.h
//  repetitionExtract
//
//  Created by Rosani Lin on 13/5/7.
//  Copyright (c) 2013年 Rosani Lin. All rights reserved.
//

#ifndef __repetitionExtract__tattingPattern__
#define __repetitionExtract__tattingPattern__

#include <iostream>
#include <sstream>

#include "Rosaniline.h"

#include "detectSliceNum.h"
#include "symmetryDetection.h"



class tattingPattern {
    
public:
    
    tattingPattern (const Mat& img);
    ~tattingPattern ();
    
    Mat pattern, slice, slice_mask;
    Point centroid;
    
    int slice_num, max_radius, sym_angle;
    
    vector<Mat> slices, slices_mask, slice_blend, blend_mask;
    
    vector<int> sym_lines;

    

private:
    
    void preprocess (const Mat& img);
    
    Point setRotMinCentroid ();
    
    void extractSlices ();
    
    void extractSliceBlend ();
    
    void sliceSelection ();
    
    // returns the max radius that contains whole object in the input image
    int getMaxContainingRadius ();
    
    vector<int> symLineExtract ();
    
    int symAngleFinder ();
    
    void minSectorExtraction ();
    
    
    
    



    
    static const int ROT_MIN_PYRAMID_LEVEL  = 3;
    static const int ROT_MIN_SEARCH_WIDTH   = 1;
    static const int INNER_MOST_RADIUS      = 20;
    static const int BLEND_OFFSET           = 10;
    
    
};


#endif /* defined(__repetitionExtract__tattingPattern__) */
