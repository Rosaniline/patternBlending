 //
//  PatchMatch.cpp
//  PatchMatch
//
//  Created by Rosani Lin on 13/4/27.
//  Copyright (c) 2013年 Rosani Lin. All rights reserved.
//

#include "PatchMatch.h"


PatchMatch::PatchMatch (int patch_height, int patch_width) {
    
    PATCHHEIGHT = patch_height, PATCHWIDTH = patch_width;
    
    genRandom = RNG(time(NULL));
    
    
}

PatchMatch::~PatchMatch () {
    
    mask.release();
}



void PatchMatch::matchImageWithPrior(const cv::Mat &img_src, const cv::Mat &img_dst, RosMat<Point> &prior, bool randomInit) {
    
    Mat src = img_src.clone(), dst = img_dst.clone();
    
    this->mask = Mat::ones(dst.size(), CV_64FC1);
    
//    showMat(src); showMat(dst); showMat(mask);
    
    RosMat<offsetElement> NearNei = RosMat<offsetElement>(prior.size());
    
    if ( randomInit == false ) {
        for (int i = 0; i < NearNei.rows; i ++) {
            for (int j = 0; j < NearNei.cols; j ++) {
                
                NearNei(i, j).offset = prior(i, j);
                
                NearNei(i, j).error = binaryPatchNorm(src, NearNei(i, j).offset + Point(j, i), dst, Point(j, i), PATCHWIDTH);
                
            }
        }
    }
    
    else {
        
        // random initialization of Nearest Neighbor Field
        for (int i = 0; i < NearNei.rows; i ++) {
            for (int j = 0; j < NearNei.cols; j ++) {
                
                NearNei(i, j).offset = Point(genRandom.uniform(0, src.cols) - j, genRandom.uniform(0, src.rows) - i);
                
                NearNei(i, j).error = binaryPatchNorm(src, NearNei(i, j).offset + Point(j, i), dst, Point(j, i), PATCHWIDTH);
                
            }
        }
        
    }
    

    match(src, dst, NearNei);
    
    
    for (int i = 0; i < prior.rows; i ++) {
        for (int j = 0; j < prior.cols; j ++) {
    
            prior.at(i, j) = NearNei.at(i, j).offset;
            
        }
    }
    
}


void PatchMatch::matchPriorWithMask(const cv::Mat &img_src, const cv::Mat &img_dst, const cv::Mat &mask, RosMat<Point> &prior, bool randomInit) {
    
    Mat src = img_src.clone(), dst = img_dst.clone();
    
    this->mask = mask.clone();
    
//    std::cout<<src.size()<<", "<<dst.size()<<", "<<mask.size()<<", "<<prior.size()<<std::endl;
    
    RosMat<offsetElement> NearNei = RosMat<offsetElement>(prior.size());
    
    if ( randomInit == false ) {
        for (int i = 0; i < NearNei.rows; i ++) {
            for (int j = 0; j < NearNei.cols; j ++) {
                
                if ( this->mask.at<double>(i, j) == 1 ) {
                    
                    NearNei(i, j).offset = prior(i, j);
                    
                    NearNei(i, j).error = binaryPatchNorm(src, NearNei(i, j).offset + Point(j, i), dst, Point(j, i), PATCHWIDTH);
                }
                
                else {
                    
                    NearNei(i, j).offset = Point(-j, -i);
                    NearNei(i, j).error = INFINITY;
                }
                

                
            }
        }
    }
    
    else {
        
        // random initialization of Nearest Neighbor Field
        for (int i = 0; i < NearNei.rows; i ++) {
            for (int j = 0; j < NearNei.cols; j ++) {
                
                if ( this->mask.at<double>(i, j) == 1 ) {
                    
                    do {
                        NearNei(i, j).offset = Point(genRandom.uniform(0, src.cols) - j, genRandom.uniform(0, src.rows) - i);
                    } while ( !this->mask.at<double>(NearNei(i, j).offset + Point(j, i)) );
                    
                    NearNei(i, j).error = binaryPatchNorm(src, NearNei(i, j).offset + Point(j, i), dst, Point(j, i), PATCHWIDTH);
                }
                
                else {
                    
                    NearNei(i, j).offset = Point(-j, -i);
                    
                    NearNei(i, j).error = INFINITY;
                }

                
            }
        }
        
    }
    
    
    
    
    match(src, dst, NearNei);
    
    
    for (int i = 0; i < prior.rows; i ++) {
        for (int j = 0; j < prior.cols; j ++) {
            
            prior.at(i, j) = NearNei.at(i, j).offset;
            
        }
    }

    
    
}


RosMat<Point> PatchMatch::matchImage (const cv::Mat &img_src, const cv::Mat &img_dst) {
    
    
    mask = Mat::ones(img_dst.size(), CV_64FC1);
    
    Mat src = img_src.clone(), dst = img_dst.clone();
    
    RosMat<offsetElement> NearNei = RosMat<offsetElement>(dst.size());
    
    
    // random initialization of Nearest Neighbor Field
    for (int i = 0; i < NearNei.rows; i ++) {
        for (int j = 0; j < NearNei.cols; j ++) {
            
            NearNei(i, j).offset = Point(genRandom.uniform(0, src.cols) - j, genRandom.uniform(0, src.rows) - i);
            
            NearNei(i, j).error = binaryPatchNorm(src, Point(j, i) + NearNei(i, j).offset, dst, Point(j, i), PATCHWIDTH);
            
        }
    }
    
    
    
    match(src, dst, NearNei);
    

    
    return elementToPoint(NearNei);
}


void PatchMatch::match(const cv::Mat &src, const cv::Mat &dst, RosMat<offsetElement> &NearNei) {
    
    double err = INFINITY, pre_err = INFINITY;
    
    for (int i = 0; i < MAX_ITERATION ; i ++) {
        
        pre_err = err;
        
        propagation(src, dst, NearNei);
        
        err = errorComputation(dst, coordinateMapping(src, NearNei));
        
        if ( (abs(err - pre_err) < HALTING_PARA*err) && i >= MIN_ITERATION ) {
            break;
        }
        
    }
    
    
    
    
//    showMat(coordinateMapping(src, NearNei), "map", 1);
//    showMat(dst, "d", 1);
//    showMat(src, "s", 0);
//    reconstErrorDisplay(dst, coordinateMapping(src, NearNei));
//    distanceDisplay(NearNei);
    
    
}



void PatchMatch::propagation(const cv::Mat &src, const cv::Mat &dst, RosMat<offsetElement> &NearNei) {
    
    for (int i = 0; i < NearNei.rows; i ++) {
        for (int j = 0; j < NearNei.cols; j ++) {
            
            if ( mask.at<double>(i, j) == 1 ) {
            
                if ( i >= 1 ) {
                    
                    update(src, dst, NearNei(i, j), NearNei(i - 1, j).offset + Point(j, i), Point(j, i));

                }
                
                if ( j >= 1 ) {
                    
                    update(src, dst, NearNei(i, j), NearNei(i, j - 1).offset + Point(j, i), Point(j, i));
                    
                }
                
                randomSearch(src, dst, NearNei(i, j), Point(j, i));
                    
            }
            
        }
    }
    
    
    for (int i = NearNei.rows - 1; i >= 0; i --) {
        for (int j = NearNei.cols - 1; j >= 0; j --) {
            
            if ( mask.at<double>(i, j) == 1) {
            
                if ( i < NearNei.rows - 1 ) {
                    
                    update(src, dst, NearNei(i, j), NearNei(i + 1, j).offset + Point(j, i), Point(j, i));

                }
                
                if ( j < NearNei.cols - 1 ) {
                    
                    update(src, dst, NearNei(i, j), NearNei(i, j + 1).offset + Point(j, i), Point(j, i));
                    
                }
                
                randomSearch(src, dst, NearNei(i, j), Point(j, i));
            
            }
        }
    }
    
    
    
}


void PatchMatch::randomSearch (const Mat& src, const Mat& dst, offsetElement& NearNei_pt, const Point loc) {
    
    double alpha = 1.0, w = MAX(dst.rows, src.rows);
    
    while ( alpha * w >= 1 ) {
        
        Point random_loc = NearNei_pt.offset + loc;
        int left_bound = 0, right_bound = 0, up_bound = 0, low_bound = 0, search_w = (int)(w*alpha);
        
        left_bound = MAX(0, random_loc.x - search_w);
        right_bound = MIN(src.cols - 1, random_loc.x + search_w);
        
        up_bound = MAX(0, random_loc.y - search_w);
        low_bound = MIN(src.rows - 1, random_loc.y + search_w);
        
        do {
            random_loc = Point(genRandom.uniform(left_bound, right_bound), genRandom.uniform(up_bound, low_bound));
        } while ( !mask.at<double>(random_loc) );
        
        

        
        update(src, dst, NearNei_pt, random_loc, loc);
        
        alpha /= 2.0;
    }
    
}



Mat PatchMatch::coordinateMapping (const Mat& src, const RosMat<offsetElement>& NearNei) {
    
    Mat mapped = Mat::zeros(NearNei.rows, NearNei.cols, src.type());
    
    double max_error = 0.0;
    for (int i = 0; i < NearNei.rows; i ++) {
        for (int j = 0; j < NearNei.cols; j ++) {
            
            if ( max_error < NearNei(i, j).error ) {
                
                max_error = NearNei(i, j).error;
            }
        }
    }
    
    for (int i = 0; i < mapped.rows; i ++) {
        for (int j = 0; j < mapped.cols; j ++) {
            
            if ( !boundaryTest(src, NearNei(i, j).offset + Point(j, i)) ) {
                
                cout<<" OUT OF BOUNDARY  size: "<<src.rows<<", "<<src.cols<<" : "<<NearNei(i, j).offset + Point(j, i)<<endl;
            }
            
            
            
            
            mapped.at<double>(i, j) = src.at<double>(NearNei.at(i, j).offset + Point(j, i));
            
        }
    }

    
    return mapped;
    
}


void PatchMatch::update (const Mat& src, const Mat& dst, offsetElement& local_off, const Point& nex_loc, const Point& loc) {
    
    
    
    if ( boundaryTest(src, nex_loc) ) {
        if ( mask.at<double>(nex_loc) ) {
        
            double nex_err = binaryPatchNorm(src, nex_loc, dst, loc, PATCHWIDTH);
            
            if ( local_off.error > nex_err ) {
                
                local_off.offset = nex_loc - loc;
                local_off.error = nex_err;
            }
        }
    }
    
    
}


void PatchMatch::reconstDisplay (const Mat& src, RosMat<offsetElement>& NearNei) {
    
    showMat(coordinateMapping(src, NearNei), "Reconstructed dst", 0);
    
}



void PatchMatch::distanceDisplay (RosMat<offsetElement> &array) {
    
    Mat temp = Mat::zeros(array.rows, array.cols, CV_64FC1);
    
    for (int i = 0; i < array.rows; i ++) {
        for (int j = 0; j < array.cols; j ++) {
            
            temp.at<double>(i, j) = sqrt(array.at(i, j).offset.x*array.at(i, j).offset.x + array.at(i, j).offset.y*array.at(i, j).offset.y);
            
        }
    }
    
    double temp_max = maxMatItem(temp);
    temp /= temp_max;
    
    showMat(temp, "distance", 0);

    temp.release();
    
}



void PatchMatch::reconstErrorDisplay (const Mat& dst, const Mat& mapped) {
    
    Mat temp = Mat::zeros(dst.size(), dst.type());
    
    for (int i = 0; i < temp.rows; i ++) {
        for (int j = 0; j < temp.cols; j ++) {
            
            temp.at<double>(i, j) = abs(dst.at<double>(i, j) - mapped.at<double>(i, j));
        }
    }
    
    showMat(temp, "error", 0);
    
    temp.release();
}


double PatchMatch::errorComputation (const Mat& dst, const Mat& mapped) {
    
    double error = 0.0;
    
    for (int i = 0; i < dst.rows; i ++) {
        for (int j = 0; j < dst.cols; j ++) {
            
            error += binaryPatchNorm(dst, Point(j, i), mapped, Point(j, i), PATCHWIDTH);
            
        }
    }
    
    return error;
    
    
}



RosMat<Point> PatchMatch::elementToPoint (const RosMat<offsetElement> &element) {
    
    RosMat<Point> pt = RosMat<Point>(element.size());
    
    for (int i = 0; i < pt.rows; i ++) {
        for (int j = 0; j < pt.cols; j ++) {
            
            pt.at(i, j) = element.at(i, j).offset;
            
        }
    }
    
    return pt;
}


RosMat<offsetElement> PatchMatch::pointToElement (const Mat& src, const Mat& dst, const RosMat<Point> &pt) {
    
    RosMat<offsetElement> element = RosMat<offsetElement>(pt.size());
    
    for (int i = 0; i < element.rows; i ++) {
        for (int j = 0; j < element.cols; j ++) {
            
            element.at(i, j).offset = pt.at(i, j);
            
            element.at(i, j).error = binaryPatchNorm(src, Point(j, i) + pt(i, j), dst, Point(j, i), PATCHWIDTH);
            
        }
    }
    
    return element;
}


inline double PatchMatch::binaryPatchNorm(const cv::Mat &src, const Point &src_loc, const cv::Mat &dst, const Point &dst_loc, int size) {
    
    
    double error = 0.0, valid_count = 0.000001;
    
    Point src_mapLoc = Point(0), dst_mapLoc = Point(0);
    
    for (int m = -size/2; m <= size/2; m ++) {
        for (int n = -size/2; n <= size/2; n ++) {
            
            src_mapLoc = src_loc + Point(n, m), dst_mapLoc = dst_loc + Point(n, m);
            
            if ( boundaryTest(src, src_mapLoc) && boundaryTest(dst, dst_mapLoc) ) {
                
//                if ( mask.at<double>(src_mapLoc) && mask.at<double>(dst_mapLoc) ) {
                
                    error += pow(src.at<double>(src_mapLoc) - dst.at<double>(dst_mapLoc), 2.0);
                    valid_count ++;
//                }
            }
            
        }
    }
    
    
    return sqrt(error)/valid_count;
    
    
}





