//
//  element.h
//  repetitionExtract
//
//  Created by Rosani Lin on 13/5/9.
//  Copyright (c) 2013年 Rosani Lin. All rights reserved.
//

#ifndef __repetitionExtract__element__
#define __repetitionExtract__element__

#include <iostream>
#include "Rosaniline.h"

#include "tattingPattern.h"

using namespace std;

class element {
    
public:
    
    element (vector<Point> input_contour, tattingPattern& pattern);
    ~element ();
    
    
    
    Point centroid;
    
    int max_radius, min_radius, area, class_id;
    
    vector<Point> contour;
    vector<Point> pts;
    
private:
    
    void computeCentroid ();
    void computeMaxMinRadius (Point pattern_centroid);
    void collectPoints (tattingPattern& pattern);

    
    
};



#endif /* defined(__repetitionExtract__element__) */
