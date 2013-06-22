//
//  BidirectSimilarity.cpp
//  BidirectSimilarity
//
//  Created by Rosani Lin on 12/9/20.
//  Copyright (c) 2012年 Rosani Lin. All rights reserved.
//

#include "BidirectSimilarity.h"

BidirectSimilarity::BidirectSimilarity() {}


Mat BidirectSimilarity::reconstruct (const Mat& src_one, const Mat& src_two, const Mat& dst_img, const Mat& weight, const Mat& mask, const Mat& slice_mask, const Point& centroid) {
    
//    showMat(src_one, "s", 1);
//    showMat(src_two, "s2", 1);
//    showMat(dst_img, "d", 1);
//    showMat(weight, "w", 1);
//    showMat(mask, "m", 0);
    
    Mat blend = dst_img.clone();
    
    vector<Mat> src_one_pyr = constructPyramid(src_one, PYRAMID_LEVEL);
    vector<Mat> src_two_pyr = constructPyramid(src_two, PYRAMID_LEVEL);
    vector<Mat> mask_pyr = constructPyramid(mask, PYRAMID_LEVEL);
    vector<Mat> slice_mask_pyr = constructPyramid(slice_mask, PYRAMID_LEVEL);
    vector<Mat> weight_pyr = constructPyramid(weight, PYRAMID_LEVEL);
    
    
    
    for (int scale = 0; scale <= PYRAMID_LEVEL; scale ++) {
        
        cout<<"scale = "<<scale<<":\n";
        
        if ( mask_pyr[scale].rows < PATCHWIDTH + 5 || mask_pyr[scale].cols < PATCHWIDTH + 5 ) {
            
            cout<<"image size in pyramid is too small.\n";
            continue;
        }
        
        resize(blend, blend, mask_pyr[scale].size());
        
        RosMat<Point> reconst_src_one = RosMat<Point>(src_one_pyr[scale].size());
        RosMat<Point> reconst_src_two = RosMat<Point>(src_two_pyr[scale].size());
        RosMat<Point> reconst_blend_one = RosMat<Point>(blend.size());
        RosMat<Point> reconst_blend_two = RosMat<Point>(blend.size());
        
        cout<<"iteration = ";
        
        Mat temp_mask = (mask_pyr[scale] + slice_mask_pyr[scale])*0.5;
        Mat div_mask = mask_pyr[scale] - slice_mask_pyr[scale];
        Mat temp_c, temp_cc;
        
        Point temp_centroid = Point(centroid.x*pow(0.5, PYRAMID_LEVEL - scale), centroid.y*pow(0.5, PYRAMID_LEVEL - scale));
        
        Mat rot_mat_c = getRotationMatrix2D(temp_centroid, 36 + 10, 1.0);
        Mat rot_mat_cc = getRotationMatrix2D(temp_centroid, -36 - 10, 1.0);

        
        

        
        for (int s = 0; s < MIN_ITERATION + (PYRAMID_LEVEL - scale)*INTER_DECREASE; s ++) {
            
//            Mat div = Mat::zeros(blend.size(), blend.type());
//            for (int i = 0; i < div.rows; i ++) {
//                for (int j = 0; j < div.cols; j ++) {
//                    
//                    if ( div_mask.at<double>(i, j) ) {
//                        div.at<double>(i, j) = 
//                    }
//                }
//            }
            
            multiply(blend, temp_mask, blend);
            


            
            warpAffine(blend, temp_c, rot_mat_c, temp_c.size(), INTER_CUBIC);
            multiply(temp_c, mask_pyr[scale], temp_c);
            
//            showMat(temp_c, "c", 1);
            
            warpAffine(blend, temp_cc, rot_mat_cc, temp_cc.size(), INTER_CUBIC);
            multiply(temp_cc, mask_pyr[scale], temp_cc);
            
//            showMat(temp_cc, "cc", 1);
            
            blend = blend + temp_c + temp_cc;
            
//            showMat(blend, "f", 0);
            
            
            Mat blend_one = blend.clone(), blend_two = blend.clone();
            
            cout<<s<<", ";
            
//            showMat(blend_one, "1", 1);
//            showMat(blend_two, "2", 1);
            
            #pragma omp parallel
            {
                gradualResize(src_one_pyr[scale], blend_one, mask_pyr[scale], reconst_src_one, reconst_blend_one, !s);

                gradualResize(src_two_pyr[scale], blend_two, mask_pyr[scale], reconst_src_two, reconst_blend_two, !s);

            }
            
//            showMat(blend_one, "11", 1);
//            showMat(blend_two, "22", 0);
            
            multiply(blend_one, 1.0 - weight_pyr[scale], blend_one);
            multiply(blend_two, weight_pyr[scale], blend_two);

            blend = blend_one + blend_two;
            
            
            stringstream path;
            
            path << "/Users/xup6qup3/Dropbox/code/Dev.temp/patternBlending/patternBlending/exp/1/"<<scale<<"_"<<s<<".bmp";
            save_CV64FC1(path.str(), blend);
            
//            path << "/Users/xup6qup3/Dropbox/code/Dev.temp/patternBlending/patternBlending/exp/1/"<<scale<<"_"<<s<<"_1"<<".bmp";
//            save_CV64FC1(path.str(), blend_one);
//            
//            path.str("");
//            path << "/Users/xup6qup3/Dropbox/code/Dev.temp/patternBlending/patternBlending/exp/1/"<<scale<<"_"<<s<<"_2"<<".bmp";
//            save_CV64FC1(path.str(), blend_two);
            
        }
        

        cout<<endl;
    }
    
 
    vector<Mat>().swap(src_one_pyr);
    vector<Mat>().swap(src_two_pyr);
    vector<Mat>().swap(mask_pyr);
    vector<Mat>().swap(slice_mask_pyr);
    vector<Mat>().swap(weight_pyr);
    
    
    return blend;
}


double BidirectSimilarity::gradualResize (const cv::Mat &src, cv::Mat &dst, const Mat &mask, RosMat<Point> &reconst_src, RosMat<Point> &reconst_dst, bool randomInit) {

    
    double NumPatch_src = src.rows*src.cols, NumPatch_dst = dst.rows*dst.cols;
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            PatchMatch patch_match = PatchMatch(PATCHWIDTH, PATCHWIDTH);
//            patch_match.matchImageWithPrior(src, dst, reconst_dst, randomInit);
            patch_match.matchPriorWithMask(src, dst, mask, reconst_dst, randomInit);
            patch_match.~PatchMatch();
        }
        
        #pragma omp section
        {
            PatchMatch patch_match = PatchMatch(PATCHWIDTH, PATCHWIDTH);
            patch_match.matchImageWithPrior(dst, src, reconst_src, randomInit);
//            patch_match.matchPriorWithMask(src, dst, mask, reconst_src, randomInit);
            patch_match.~PatchMatch();
        }
    }
    
        
    for (int i = 0; i < mask.rows; i ++) {
        for (int j = 0; j < mask.cols; j ++) {
            
            if ( mask.at<double>(i, j) != 1 ) {
                
                reconst_dst.at(i, j) = Point(-j, -i);
            }
            
        }
    }
    
    Mat cohere_mat = Mat::zeros(dst.size(), dst.type());
    Mat cohere_count = Mat(dst.size(), CV_64FC1).setTo(0.000001);
    
    Mat complete_mat = Mat::zeros(dst.size(), dst.type());
    Mat complete_count = Mat(dst.size(), CV_64FC1).setTo(0.000001);
    
    double total_error = 0.0, cohere_error = 0.0, complete_error = 0.0;
    
    
    
//    #pragma omp parallel sections
//    {
//        #pragma omp section
//        {
            cohere_error = cohereTerm(src, dst, reconst_dst, cohere_mat, cohere_count)/NumPatch_dst;
//        }
//            
//        #pragma omp section
//        {
            complete_error = completeTerm(src, dst, reconst_src, complete_mat, complete_count)/NumPatch_src;
//        }
//    }
    
    total_error = cohere_error + complete_error;

    
    for (int i = 0; i < cohere_mat.rows; i ++) {
        for (int j = 0; j < cohere_mat.cols; j ++) {
        
            dst.at<double>(i, j) = ( cohere_mat.at<double>(i, j)/NumPatch_dst + complete_mat.at<double>(i, j)/NumPatch_src )/( cohere_count.at<double>(i, j)/NumPatch_dst + complete_count.at<double>(i, j)/NumPatch_src) * mask.at<double>(i, j);
            
        }
    }
    
    
    cohere_mat.release();
    cohere_count.release();
    complete_mat.release();
    complete_count.release();
    
    return total_error;
    
}


double BidirectSimilarity::correction (const cv::Mat &src, cv::Mat &dst, const Mat &mask, RosMat<Point> &reconst_src, RosMat<Point> &reconst_dst, bool randomInit) {
    
    

    PatchMatch patch_match = PatchMatch(PATCHWIDTH, PATCHWIDTH);
    //            patch_match.matchImageWithPrior(src, dst, reconst_dst, randomInit);
    patch_match.matchPriorWithMask(src, dst, mask, reconst_dst, randomInit);
    patch_match.~PatchMatch();

    
    for (int i = 0; i < mask.rows; i ++) {
        for (int j = 0; j < mask.cols; j ++) {
            
            if ( mask.at<double>(i, j) != 1 ) {
                
                reconst_dst.at(i, j) = Point(-j, -i);
            }
            
        }
    }
    
    double total_error = 0.0;
    
    for (int i = 0; i < dst.rows; i ++) {
        for (int j = 0; j < dst.cols; j ++) {
            
            double candi = src.at<double>(reconst_dst(i, j) + Point(j, i));
            
            total_error += abs(candi - dst.at<double>(i, j));
            
            dst.at<double>(i, j) = src.at<double>(reconst_dst(i, j) + Point(j, i));
            
        }
    }

    
    
    
    return total_error;
    
}


double BidirectSimilarity::gradualResize(const cv::Mat &src, cv::Mat &dst, RosMat<Point> &reconst_src, RosMat<Point> &reconst_dst, bool randomInit) {
    
    
    double NumPatch_src = src.rows*src.cols, NumPatch_dst = dst.rows*dst.cols;
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            PatchMatch patch_match = PatchMatch(PATCHWIDTH, PATCHWIDTH);
            patch_match.matchImageWithPrior(src, dst, reconst_dst, randomInit);
            patch_match.~PatchMatch();
        }
        
        #pragma omp section
        {
            PatchMatch patch_match = PatchMatch(PATCHWIDTH, PATCHWIDTH);
            patch_match.matchImageWithPrior(dst, src, reconst_src, randomInit);
            patch_match.~PatchMatch();
        }
    }
    
        
    Mat cohere_mat = Mat::zeros(dst.size(), dst.type());
    Mat cohere_count = Mat(dst.size(), CV_64FC1).setTo(0.000001);
    
    Mat complete_mat = Mat::zeros(dst.size(), dst.type());
    Mat complete_count = Mat(dst.size(), CV_64FC1).setTo(0.000001);
    
    double total_error = 0.0, cohere_error = 0.0, complete_error = 0.0;
    
    
    
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            cohere_error = cohereTerm(src, dst, reconst_dst, cohere_mat, cohere_count)/NumPatch_dst;
        }
        
        #pragma omp section
        {
            complete_error = completeTerm(src, dst, reconst_src, complete_mat, complete_count)/NumPatch_src;
        }
    }
    
    total_error = cohere_error + complete_error;
//    cout<<total_error<<", ";
    
    
    for (int i = 0; i < cohere_mat.rows; i ++) {
        for (int j = 0; j < cohere_mat.cols; j ++) {
            
//            dst.at<Vec3d>(i, j) = ( cohere_mat.at<Vec3d>(i, j)/NumPatch_dst + complete_mat.at<Vec3d>(i, j)/NumPatch_src )/( cohere_count.at<double>(i, j)/NumPatch_dst + complete_count.at<double>(i, j)/NumPatch_src);
            
            dst.at<double>(i, j) = cohere_mat.at<double>(i, j)/cohere_count.at<double>(i, j);
            
        }
    }
    
    
    cohere_mat.release();
    cohere_count.release();
    complete_mat.release();
    complete_count.release();
    
    return total_error;
    
}


double BidirectSimilarity::cohereTerm(const cv::Mat &src, const cv::Mat &dst, RosMat<Point> &reconst_dst, cv::Mat &cohere_mat, cv::Mat &cohere_count) {
    
    double cohere_error = 0.0;
    
    
    for (int i = 0; i < dst.rows; i ++) {
        for (int j = 0; j < dst.cols; j ++) {
            
            for (int m = -PATCHWIDTH/2; m <= PATCHWIDTH/2; m ++) {
                for (int n = -PATCHWIDTH/2; n <= PATCHWIDTH/2; n ++) {
                    
                    Point Q = Point(j + n, i + m);
                    
                    if ( boundaryTest(dst, Q) ) {
                        
                        Point P = Point(j, i) + reconst_dst(Q);
                        
                        if ( boundaryTest(src, P) ) {
                            
                            cohere_mat.at<double>(i, j) += src.at<double>(P);
                            cohere_count.at<double>(i, j) += 1;
                            
                            cohere_error += pow(dst.at<double>(i, j) - src.at<double>(P), 2.0);
//                            Vec3dDiff(dst.at<Vec3d>(i, j), src.at<Vec3d>(P));
                            
                        }
                        
                    }
                }
            }
            
            
        }
    }

    
    return cohere_error;
    
}


double BidirectSimilarity::completeTerm(const cv::Mat &src, const cv::Mat &dst, RosMat<Point> &reconst_src, cv::Mat &complete_mat, cv::Mat &complete_count) {
    
    double complete_error = 0.0;
    
    
    for (int i = 0; i < reconst_src.rows; i ++) {
        for (int j = 0; j < reconst_src.cols; j ++) {
            
            for (int m = -PATCHWIDTH/2; m <= PATCHWIDTH/2; m ++) {
                for (int n = -PATCHWIDTH/2; n <= PATCHWIDTH/2; n ++) {
                        
                    Point Q = Point(j, i) + reconst_src.at(i, j) + Point(n, m);
                    Point P = Point(j, i) + Point(n, m);
                    
                    if ( boundaryTest(dst, Q) && boundaryTest(src, P)) {
                        
                        complete_mat.at<double>(Q) += src.at<double>(P);
                        complete_count.at<double>(Q) += 1.0;
                        
                        complete_error += pow(dst.at<double>(Q) - src.at<double>(P), 2.0);
//                        Vec3dDiff(dst.at<Vec3d>(Q), src.at<Vec3d>(P));
                    }
                        
                }
            }
            
        }
    }


    return complete_error;
    
}


vector< RosMat<Point> > BidirectSimilarity::constructRosMatPointPyramid(const RosMat<Point> &src, int pyramid_level) {
    
    vector< RosMat<Point> > src_pyr;

    
    for (int level = PYRAMID_LEVEL; level >= 0; level --) {
        
        double ratio = pow(2.0, level);
        
        RosMat<Point> resized = RosMat<Point>(src.rows/ratio, src.cols/ratio);
        
        for (int i = 0; i < resized.rows; i ++) {
            for (int j = 0; j < resized.cols; j ++) {
                
                resized.at(i, j).x = src.at(i*ratio, j*ratio).x/ratio;
                resized.at(i, j).y = src.at(i*ratio, j*ratio).y/ratio;
                
            }
        }
        
        src_pyr.push_back(resized);
        
    }
    
//    reverse(src_pyr.begin(), src_pyr.end());
    
//    for (int i = 0; i < src_pyr.size(); i ++) {
//        cout<<src_pyr[i]<<endl;
//    }
    
    return src_pyr;
}























