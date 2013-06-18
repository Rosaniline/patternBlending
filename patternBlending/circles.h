//
//  circles.h
//  repetitionExtract
//
//  Created by Rosani Lin on 13/5/6.
//  Copyright (c) 2013年 Rosani Lin. All rights reserved.
//

#ifndef __repetitionExtract__circles__
#define __repetitionExtract__circles__

#include <iostream>
#include "Rosaniline.h"


class circles {
    
public:
    
    circles();
    ~circles();
    
    int getMinContainingRadius (const Mat& img);
    
    Point getCentroid (const Mat& img);
    
    vector<Point> getCirclePoint (const Point& centroid, int radius);
    
    
};

#endif /* defined(__repetitionExtract__circles__) */
