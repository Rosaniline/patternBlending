//
//  repetendExtract.h
//  repetitionExtract
//
//  Created by Rosani Lin on 13/5/6.
//  Copyright (c) 2013年 Rosani Lin. All rights reserved.
//

#ifndef __repetitionExtract__repetendExtract__
#define __repetitionExtract__repetendExtract__

#include <iostream>
#include "Rosaniline.h"

#include "tattingPattern.h"
#include "element.h"

#include <opencv2/legacy/legacy.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>



class repetendExtract {

public:
    
    repetendExtract();
    ~repetendExtract();
    
    void extract (tattingPattern pattern);
    
private:
    
    vector<vector<Point> > extractElementBoundary (tattingPattern pattern);
    
    Point getElementCentroid (const vector<Point>& element);
    
    double getElementRadius (const vector<Point>& element, const Point& centroid);
    
    void plotContour (Mat& img, const vector<Point>& contour);
    
    void SIFTFeatureSaliency (tattingPattern& pattern);

};




#endif /* defined(__repetitionExtract__repetendExtract__) */
