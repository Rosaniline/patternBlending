//
//  repetitionExtract.h
//  repetitionExtract
//
//  Created by Rosani Lin on 13/4/15.
//  Copyright (c) 2013年 Rosani Lin. All rights reserved.
//

#ifndef __detectSliceNum__
#define __detectSliceNum__

#include <iostream>
#include "Rosaniline.h"




class detectSliceNum {
    
    
public:
    
    detectSliceNum();
    ~detectSliceNum();
    
    // extracts the symmetric element of the image stored in the input path
    int extractSliceNum (const Mat& pattern, const Point& centroid);
    
private:
    
    
    
    // Returns the difference after rotation for 360 degrees
    Mat getRotateDiff (const Mat& pattern, const Point& centroid);
    
    int getMinContainingRadius (const cv::Mat &img);
    
    
    
    static const int DIFF_WIDTH = 100;
    
};

#endif /* defined(__repetitionExtract__repetitionExtract__) */
