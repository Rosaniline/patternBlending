//
//  PatchMatch.h
//  PatchMatch
//
//  Created by Rosani Lin on 13/4/27.
//  Copyright (c) 2013年 Rosani Lin. All rights reserved.
//

#ifndef __PatchMatch__PatchMatch__
#define __PatchMatch__PatchMatch__

#include <iostream>
#include "Rosaniline.h"
#include "offsetElement.h"


class PatchMatch {
    
public:
    
    PatchMatch (int patch_height, int patch_width);
    ~PatchMatch ();
    
    // Run PatchMatch according to image files
    RosMat<Point> matchFile (const string& img_src, const string& img_dst);
    
    // Run PatchMatch according to images
    RosMat<Point> matchImage (const Mat& img_src, const Mat& img_dst);
    
    // Run PatchMatch according to images and pre-specified nearest neighbor fieldn
    void matchImageWithPrior (const Mat& img_src, const Mat& img_dst, RosMat<Point>& prior, bool randomInit);
    
    
    void matchPriorWithMask (const Mat& img_src, const Mat& img_dst, const Mat& mask, RosMat<Point>& prior, bool randomInit);


    // Display the reconstructed image according to the nearest neighbor field
    void reconstDisplay (const Mat& src, RosMat<offsetElement>& NearNei);
    
    // Display the locality distance of nearest neighbor
    void distanceDisplay (RosMat<offsetElement>& array);
    
    // Display the error distance of the reconstructed image
    void reconstErrorDisplay (const Mat& dst, const Mat& mapped);
    
    
private:
    
    // Core of PatchMatch
    void match (const Mat& src, const Mat& dst, RosMat<offsetElement>& NearNei);
    
    // Multilevel version of PatchMatch
    RosMat<offsetElement> multilevelMatch (const Mat& src, const Mat& dst);
    
    
    
    // Propagation
    void propagation (const Mat& src, const Mat& dst, RosMat<offsetElement>& NearNei);
    
    // Random Search
    void randomSearch (const Mat& src, const Mat& dst, offsetElement& NearNei_pt, const Point loc);
    
    
    
    // Convergence condition test
    bool convergenceTest (RosMat<offsetElement>& NearNei_new, RosMat<offsetElement>& NearNei_old, int itr);
    
    // Upsample procedure applied in multilevel mechanism
    void upsample (const Mat& src, const Mat& dst, RosMat<offsetElement>& NearNei);
    
    // Display the offset image
    Mat coordinateMapping (const Mat& src, const RosMat<offsetElement>& NearNei);
    
    // Update the offset with minimum error
    void update (const Mat& src, const Mat& dst, offsetElement& local_off, const Point& nex_loc, const Point& loc);
    
    double errorComputation (const Mat& dst, const Mat& mapped);
    
    RosMat<Point> elementToPoint (const RosMat<offsetElement>& element);
    
    RosMat<offsetElement> pointToElement (const Mat& src, const Mat& dst, const RosMat<Point> &pt);
    
    
    double binaryPatchNorm (const Mat& src, const Point& src_loc, const Mat& dst, const Point& dst_loc, int size);
    
    


    
    bool test;
    
    RNG genRandom;
    
    Mat mask;
    
    int PATCHHEIGHT, PATCHWIDTH;
    static const int        PYRAMID_LEVEL   =   15;
    static const double     HALTING_PARA    =   0.01;
    static const int        MAX_ITERATION   =   7;
    static const int        MIN_ITERATION   =   1;

};

#endif /* defined(__PatchMatch__PatchMatch__) */
