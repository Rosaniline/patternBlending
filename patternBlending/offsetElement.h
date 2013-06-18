//
//  offsetElement.h
//  PatchMatch
//
//  Created by Rosani Lin on 13/5/12.
//  Copyright (c) 2013年 Rosani Lin. All rights reserved.
//

#ifndef __PatchMatch__offsetElement__
#define __PatchMatch__offsetElement__

#include <iostream>
#include <opencv2/opencv.hpp>
#include "Rosaniline.h"


using namespace cv;

class offsetElement {
    
public:
    
    Point offset;
    double error;
    
    offsetElement () {};
    offsetElement (Point pt, double er) {
        offset = pt;
        error = er;
    }
    
    offsetElement (const offsetElement& rhs) {
        offset = rhs.offset;
        error = rhs.error;
    }
    
    ~offsetElement () {
        offset.~Point_();
    }
    
    bool operator == (const offsetElement& rhs) const {
        
        return ( offset == rhs.offset && error == rhs.error );
        
    }
    
    friend ostream& operator << (ostream& os, const offsetElement& rhs) {
        
            os << rhs.offset << ", " << rhs.error;
        
        return os;
    }
    
};
        


#endif /* defined(__PatchMatch__offsetElement__) */














