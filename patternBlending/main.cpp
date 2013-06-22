//
//  main.cpp
//  patternBlending
//
//  Created by Rosani Lin on 13/6/17.
//  Copyright (c) 2013年 Rosani Lin. All rights reserved.
//

#include <iostream>
#include "tattingPattern.h"
#include "sliceSynthesis.h"

int main(int argc, const char * argv[])
{

    Mat img = imread("/Users/xup6qup3/Dropbox/code/Dev.temp/patternBlending/tl6.jpg");
    
//    for (int i = 0; i < img.rows; i ++) {
//        for (int j = 0; j < img.cols; j ++) {
//
////            img.at<Vec3b>(i, j) = Vec3b(255, 255, 255)  - img.at<Vec3b>(i, j);
//            
//            if ( sqrt((i - 301)*(i - 301) + (j - 296)*(j - 296) ) > 297 ) {
//                
//                img.at<Vec3b>(i, j) = Vec3b(255, 255, 255);
//            }
//        }
//    }
//    
//    for (int i = 0; i < 5; i ++) {
//        
//        
//        infLine(img, Point(296, 301), tan((92.0 + i*36)/180.0*PI));
//
//    }
    
    


    tattingPattern pattern = tattingPattern(img);

    
    
    
//    Mat temp = 1 - pattern.slices_mask[4];
//    
//    multiply(pattern.pattern, (1 - pattern.slices_mask[4])*0.7, temp);
//    
//    temp += pattern.slices[4];
//    
//    
//    
//    for (int i = 0; i < img.rows; i ++) {
//        for (int j = 0; j < img.cols; j ++) {
//
////            img.at<Vec3b>(i, j) = Vec3b(255, 255, 255)  - img.at<Vec3b>(i, j);
//
//            if ( sqrt((i - 301)*(i - 301) + (j - 296)*(j - 296) ) > 297 ) {
//
//                temp.at<double>(i, j) = 1.0;
//            }
//        }
//    }
//
//    save_CV64FC1("/Users/xup6qup3/Desktop/slice_src.jpg", temp);
    

    sliceSynthesis synthesizer = sliceSynthesis();
    synthesizer.synthesis(pattern);
    
    
    return 0;
}

