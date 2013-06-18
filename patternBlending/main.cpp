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
    
    tattingPattern pattern = tattingPattern(img);
    
    sliceSynthesis synthesizer = sliceSynthesis();
    synthesizer.synthesis(pattern);
    
    return 0;
}

