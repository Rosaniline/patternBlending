//
//  element.cpp
//  repetitionExtract
//
//  Created by Rosani Lin on 13/5/9.
//  Copyright (c) 2013年 Rosani Lin. All rights reserved.
//

#include "element.h"


element::element (vector<Point> input_contour, tattingPattern& pattern) {
    
    contour = input_contour;
    
    area = contourArea(contour);
    
    computeCentroid();
    
    computeMaxMinRadius(pattern.centroid);
    
    class_id = -1;
    
//    collectPoints(pattern);
    
    
}

element::~element() {}


void element::computeCentroid() {
    
    for (int i = 0; i < contour.size(); i ++) {
        
        centroid += contour[i];
        
    }
    
    centroid.x /= contour.size();
    centroid.y /= contour.size();
    
}


void element::computeMaxMinRadius(Point pattern_centroid) {
    
    max_radius = 0, min_radius = INT_MAX;
    
    for (int i = 0; i < contour.size(); i ++) {
        
        int temp_radius = sqrt(pow(pattern_centroid.x - contour[i].x, 2.0) + pow(pattern_centroid.y - contour[i].y, 2.0));
        
        if ( temp_radius < min_radius ) {
            min_radius = temp_radius;
        }
        
        if ( temp_radius > max_radius ) {
            max_radius = temp_radius;
        }
        
    }
    
    if ( min_radius > sqrt(pow(centroid.x - pattern_centroid.x, 2.0) + pow(centroid.y - pattern_centroid.y, 2.0)) ) {
        min_radius = 0;
    }
    
    
}


void element::collectPoints(tattingPattern &tatting) {
    
    Mat temp = Mat::zeros(tatting.pattern.size(), CV_8UC1);
    
    vector<vector<Point> > temp_contour;
    
    temp_contour.push_back(contour);
    
    drawContours(temp, temp_contour, 0, 255, -1);
    
    showMat(temp);
    
    
    
    vector<vector<Point> >().swap(temp_contour);
}













