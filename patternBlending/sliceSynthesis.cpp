//
//  sliceSynthesis.cpp
//  repetitionExtract
//
//  Created by Rosani Lin on 13/5/28.
//  Copyright (c) 2013年 Rosani Lin. All rights reserved.
//

#include "sliceSynthesis.h"


sliceSynthesis::sliceSynthesis() {}

sliceSynthesis::~sliceSynthesis() {}


Mat sliceSynthesis::synthesis(const tattingPattern &tatting) {
    
    Mat synthesized = Mat::zeros(tatting.pattern.size(), tatting.pattern.type());
    
//    RNG rng = RNG(time(NULL));
    
    
    double resize_ratio = 0.75, rotate_angle = 18, layer_ratio = 0.5, layer_sigma = 0.0, slice_sigma = 65;
    
    stringstream file;
    file <<"rsz"<<resize_ratio<<"_rot"<<rotate_angle<<"_lyr"<<layer_ratio<<"_lsg"<<layer_sigma;
    string path = file.str();
    
    mkdir(path.c_str(),  S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    
        
//    #pragma omp parallel for
    for (int k = 0; k < 20; k ++) {
        
        Mat similar = createSelfSimilar(tatting.pattern, tatting.centroid, resize_ratio, rotate_angle);
        
        Mat weight = hybridweight(tatting, resize_ratio, layer_ratio, layer_sigma, slice_sigma);
        
        Mat syn = sliceHybrid(tatting, similar, weight, layer_ratio, layer_sigma, slice_sigma);
        
        file.str("");
        file << path <<"/"<< path <<"_sgg"<<slice_sigma<<".bmp";
        
        save_CV64FC1(file.str(), syn);

        
        file.str("");
        file << path <<"/"<< path <<"_sgg"<<slice_sigma<<"_weight.bmp";
        
        save_CV64FC1(file.str(), weight);
        
        Mat merge = Mat::zeros(syn.rows, syn.cols*2, syn.type());
        
        for (int i = 0; i < merge.rows; i ++) {
            for (int j = 0; j < merge.cols; j ++) {
                
                if ( j < syn.cols ) {
                    merge.at<double>(i, j) = syn.at<double>(i, j);
                }
                
                else {
                    merge.at<double>(i, j) = weight.at<double>(i, j - weight.cols);
                }
            }
        }
        
        file.str("");
        file << path <<"/"<< path <<"_sgg"<<slice_sigma<<"_blend.bmp";
        
        save_CV64FC1(file.str(), merge);
        
        slice_sigma += 5;
    }
    
    
    
    resize_ratio = 0.75, rotate_angle = 18, layer_ratio = 0.5, layer_sigma = 5.0, slice_sigma = 0;
    
    file.str("");
    file <<"rsz"<<resize_ratio<<"_rot"<<rotate_angle<<"_lyr"<<layer_ratio<<"_ssg"<<slice_sigma;
    path = file.str();
    
    mkdir(path.c_str(),  S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);

    // variable : layer_sigma
    #pragma omp parallel for
    for (int k = 0; k < 20; k ++) {
        
        Mat similar = createSelfSimilar(tatting.pattern, tatting.centroid, resize_ratio, rotate_angle);
        
        Mat weight = hybridweight(tatting, resize_ratio, layer_ratio, layer_sigma, slice_sigma);
        
        Mat syn = sliceHybrid(tatting, similar, weight, layer_ratio, layer_sigma, slice_sigma);
        
        file.str("");
        file << path <<"/"<< path <<"_lsg"<<layer_sigma<<".bmp";
        
        save_CV64FC1(file.str(), syn);
        
        file.str("");
        file << path <<"/"<< path <<"_lsg"<<layer_sigma<<"_weight.bmp";
        
        save_CV64FC1(file.str(), weight);
        
        
        Mat merge = Mat::zeros(syn.rows, syn.cols*2, syn.type());

        for (int i = 0; i < merge.rows; i ++) {
            for (int j = 0; j < merge.cols; j ++) {

                if ( j < syn.cols ) {
                    merge.at<double>(i, j) = syn.at<double>(i, j);
                }

                else {
                    merge.at<double>(i, j) = weight.at<double>(i, j - weight.cols);
                }
            }
        }

        file.str("");
        file << path <<"/"<< path <<"_lsg"<<layer_sigma<<"_blend.bmp";
        
        save_CV64FC1(file.str(), merge);
        
        layer_sigma += 5;
    }
    
    
    resize_ratio = 0.75, rotate_angle = 18, layer_ratio = 0.1, layer_sigma = 40, slice_sigma = 0;
    
    file.str("");
    file <<"rsz"<<resize_ratio<<"_rot"<<rotate_angle<<"_lsg"<<layer_sigma<<"_ssg"<<slice_sigma;
    path = file.str();
    
    mkdir(path.c_str(),  S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    
    // variable : layer_ratio
    #pragma omp parallel for
    for (int k = 0; k < 8; k ++) {
        
        Mat similar = createSelfSimilar(tatting.pattern, tatting.centroid, resize_ratio, rotate_angle);
        
        Mat weight = hybridweight(tatting, resize_ratio, layer_ratio, layer_sigma, slice_sigma);
        
        Mat syn = sliceHybrid(tatting, similar, weight, layer_ratio, layer_sigma, slice_sigma);
        
        file.str("");
        file << path <<"/"<< path <<"_lyr"<<layer_ratio<<".bmp";
        
        save_CV64FC1(file.str(), syn);
        
        file.str("");
        file << path <<"/"<< path <<"_lyr"<<layer_ratio<<"_weight.bmp";
        
        save_CV64FC1(file.str(), weight);
        
        
        Mat merge = Mat::zeros(syn.rows, syn.cols*2, syn.type());
        
        for (int i = 0; i < merge.rows; i ++) {
            for (int j = 0; j < merge.cols; j ++) {
                
                if ( j < syn.cols ) {
                    merge.at<double>(i, j) = syn.at<double>(i, j);
                }
                
                else {
                    merge.at<double>(i, j) = weight.at<double>(i, j - weight.cols);
                }
            }
        }
        
        file.str("");
        file << path <<"/"<< path <<"_lyr"<<layer_ratio<<"blend.bmp";
        
        save_CV64FC1(file.str(), merge);
        
        layer_ratio += 0.1;
    }


    
        
//        stringstream file;
//
        
        
//        resize_ratio = 1.0;//rng.uniform(0.3, 1.0);
//        rotate_angle = 18;//rng.uniform(0.0, 360.0/tatting.slice_num);
//        layer_ratio = 1.0;//rng.uniform(0.3, 1.0);
//        layer_sigma = 50;//rng.uniform(20, 60);
//        slice_sigma = 50;//rng.uniform(20, 70);
        
//        resize_ratio = rng.uniform(0.5, 1.5);
//        rotate_angle = rng.uniform(0, 2)*180/tatting.slice_num;
//        layer_ratio = rng.uniform(0.3, MIN(resize_ratio, 0.9));
//        layer_sigma = rng.uniform(10, 30);
//        slice_sigma = rng.uniform(20, 60);

//        Mat similar = createSelfSimilar(tatting.pattern, tatting.centroid, resize_ratio, rotate_angle);
//        
//        Mat syn = sliceHybrid(tatting, similar, tatting.max_radius*resize_ratio, layer_ratio, layer_sigma, slice_sigma);
//        
//        stringstream file;
//        file << "/Users/xup6qup3/Dropbox/code/Dev.temp/patternBlending/patternBlending/exp/rsz_"<<resize_ratio<<"_rot_"<<rotate_angle<<"_lyr_"<<layer_ratio<<"_lsg_"<<layer_sigma<<"_ssg_"<<slice_sigma<<".bmp";
//        
//        save_CV64FC1(file.str(), syn);
    
    
    
    
    
    
    return synthesized;
}


Mat sliceSynthesis::circularGaussianRG (double ratio, double sigma, const Point& centroid, Size canvas_size, int max_radius) {
    
    Mat canvas = Mat::zeros(canvas_size, CV_64FC1);
    
    
    int mu = (int)(ratio*MIN(canvas_size.height, canvas_size.width)/2);
    
    double denominator_out = 1.0/(sigma*sqrt(2*PI)), denominator_in = (2*pow(sigma, 2.0));
    
    
    for (int i = 0; i < canvas.rows; i ++) {
        for (int j = 0; j < canvas.cols; j ++) {
            
            int r = (int)sqrt(pow(i - centroid.y, 2.0) + pow(j - centroid.x, 2.0));
            
            if ( r <= max_radius ) {
                
                canvas.at<double>(i, j) = denominator_out*exp(-pow(r - mu, 2.0)/denominator_in);
            }
            
        }
    }
    
    
    
    double max = 0.0;
    
    minMaxIdx(canvas, NULL, &max);
    
    canvas = canvas/max;
    
    
    
    
//    showMat(canvas, "canvas", 0);
    
    return canvas;
}


Mat sliceSynthesis::sliceGaussianRG (int slice_num, double sigma, const Point &centroid, Size canvas_size, int max_radius, double sym_angle) {
    
    Mat canvas = Mat::zeros(canvas_size, CV_64FC1);

    // Create a Gaussian distribution sample according to the max containing radius
    Mat gaussian_sample = Mat::zeros((int)(max_radius*(PI/slice_num)), 1, CV_64FC1);
    
    
    double denominator_in = 2.0*pow(sigma, 2.0);
    for (int i = 0; i < gaussian_sample.rows; i ++) {
        for (int j = 0; j < gaussian_sample.cols; j ++) {
            
            gaussian_sample.at<double>(i, j) = exp(-pow(i, 2.0)/denominator_in);
            
        }
    }
    
    double sita_seed = (sym_angle/180.0)*PI;
    vector<double> divide_sita;
    for (int i = 0; i < slice_num; i ++) {
        divide_sita.push_back(sita_seed);
        
        sita_seed += 2.0*PI/slice_num;
        
        if ( sita_seed > 2.0*PI ) {
            sita_seed -= 2.0*PI;
        }
        
    }
    
    
    for (int i = 0; i < canvas.rows; i ++) {
        for (int j = 0; j < canvas.cols; j ++) {
            
            double r = sqrt( pow(i - centroid.y, 2.0) + pow(j - centroid.x, 2.0));
            
            double sita = atan2(j - centroid.x, i - centroid.y) + PI;
            
            double closest_dist = INFINITY;
            for (int k = 0; k < divide_sita.size(); k ++) {
                
                if( closest_dist > abs(sita - divide_sita[k])) {
                    closest_dist = abs(sita - divide_sita[k]);
                }
                
                if( closest_dist > abs(sita - (divide_sita[k] + 2.0*PI)) ) {
                    closest_dist = abs(sita - (divide_sita[k] + 2.0*PI));
                }
                
                if( closest_dist > abs(sita + 2.0*PI - divide_sita[k])) {
                    closest_dist = abs(sita + 2.0*PI - divide_sita[k]);
                }
            }
            
            if ( r < max_radius ) {
                
                canvas.at<double>(i, j) = gaussian_sample.at<double>((int)(closest_dist/(PI/slice_num)*gaussian_sample.rows), 0);
                
            }
            
            
        }
    }


    vector<double>().swap(divide_sita);
    gaussian_sample.release();
    
    return canvas;
}


Mat sliceSynthesis::alphaBlending (const Mat& src, const Point& centroid, int max_radius, const Mat& dst, const Mat& weight) {
    
    Mat blended = Mat::zeros(src.size(), src.type());
    Mat temp = Mat::zeros(src.size(), src.type());
    
    multiply(src, 1.0 - weight, blended);
    multiply(dst, weight, temp);
    
    blended += temp;
    
    temp.release();
    
    return blended;
}


Mat sliceSynthesis::createSelfSimilar(const cv::Mat &src, const Point &centroid, double scale, double angle) {
    
    Mat similar = Mat::zeros(src.size(), src.type());
    
    Mat rot_mat = getRotationMatrix2D(centroid, angle, 1.0);
    warpAffine(src, similar, rot_mat, similar.size(), INTER_CUBIC);
    
    Mat resize_temp = similar.clone();
    resizeMat(resize_temp, scale);
    
    
    Point centroid_dist = centroid - centroid*scale;

    
    similar.setTo(0);
    
    for (int i = 0; i < resize_temp.rows; i ++) {
        for (int j = 0; j < resize_temp.cols; j ++) {
            
            Point map = centroid_dist + Point(j, i);
            if (boundaryTest(similar, map)) {
                similar.at<double>(i + centroid_dist.y, j + centroid_dist.x) = resize_temp.at<double>(i, j);
            }
            
            
            
        }
    }
    
    return similar;
}


Mat sliceSynthesis::hybrid(const tattingPattern &src, const cv::Mat &similar, double layer_ratio, double layer_sigma, double slice_sigma) {
    
    
    Mat weight = sliceGaussianRG(src.slice_num, slice_sigma, src.centroid, src.pattern.size(), src.max_radius, src.sym_angle);
    
//    Mat weight_p = circularGaussianRG(layer_ratio, layer_sigma, src.centroid, src.pattern.size(), src.max_radius);
//
//    for (int i = 0; i < weight.rows; i ++) {
//        for (int j = 0; j < weight.cols; j ++) {
//            
//            weight.at<double>(i, j) = MIN(weight.at<double>(i, j) + weight_p.at<double>(i, j), 1.0);
//        }
//    }
    
    stringstream file;
    file << "/Users/xup6qup3/Dropbox/code/Dev.temp/patternBlending/patternBlending/exp/s.png";
    save_CV64FC1(file.str(), src.pattern);
    
    file.str("");
    file << "/Users/xup6qup3/Dropbox/code/Dev.temp/patternBlending/patternBlending/exp/d.png";
    save_CV64FC1(file.str(), similar);
    
    file.str("");
    file << "/Users/xup6qup3/Dropbox/code/Dev.temp/patternBlending/patternBlending/exp/rsz_"<<"_lyr_"<<layer_ratio<<"_lsg_"<<layer_sigma<<"_ssg_"<<slice_sigma<<"_wei.bmp";
    save_CV64FC1(file.str(), weight);
    
    Mat temp = Mat::zeros(weight.size(), weight.type());
    Mat temp2 = Mat::zeros(weight.size(), weight.type());
    
    for (int i = 0; i < weight.rows; i ++) {
        for (int j = 0; j < weight.cols; j ++) {
            
            if ( sqrt(pow(i - src.centroid.y, 2.0) + pow(j - src.centroid.x, 2.0)) < src.max_radius ) {
                
                if ( weight.at<double>(i, j) >= 0.9 ) {
                    
                    temp.at<double>(i, j) = 1.0;
                }
                
                if ( weight.at<double>(i, j) > 0.1 && weight.at<double>(i, j) < 0.9 ) {
                    
                    temp2.at<double>(i, j) = 1.0;
                }
            }
            

            
        }
    }
    
//    showMat(temp2, "2", 1);
//    showMat(temp);
    
    file.str("");
    file << "/Users/xup6qup3/Dropbox/code/Dev.temp/patternBlending/patternBlending/exp/srcmask.png";
    save_CV64FC1(file.str(), temp);
    
    file.str("");
    file << "/Users/xup6qup3/Dropbox/code/Dev.temp/patternBlending/patternBlending/exp/synmask.png";
    save_CV64FC1(file.str(), temp2);
    

    
    
    
    Mat blended = alphaBlending(src.pattern, src.centroid, src.max_radius, similar, weight);
    
//    showMat(blended);
  
    
    Mat mask = Mat::zeros(src.pattern.size(), src.pattern.type());
    for (int i = 0; i < mask.rows; i ++) {
        for (int j = 0; j < mask.cols; j ++) {
            
            if ( sqrt(pow(i - src.centroid.y, 2.0) + pow(j - src.centroid.x, 2.0)) < src.max_radius ) {
                mask.at<double>(i, j) = 1.0;
            }
        }
    }
    
//    showMat(mask);
    
    

//    BidirectSimilarity bi_Sim = BidirectSimilarity();
//    bi_Sim.reconstruct(src.pattern, similar, blended, weight, mask).copyTo(blended);
//    
//    
//    connectedCompRefine(blended);
    
    return blended;
    
}


Mat sliceSynthesis::sliceHybrid (const tattingPattern& src, const Mat& similar, const Mat& weight, double layer_ratio, double layer_sigma, double slice_sigma) {

    
//    showMat(weight, "w", 1);
    

    Mat blended = alphaBlending(src.pattern, src.centroid, src.max_radius, similar, weight);
    Mat blended_aug = blended.clone();
    
//    showMat(blended);
    
    multiply(blended, src.slices_mask[4], blended);
    multiply(blended_aug, src.blend_mask[4], blended_aug);
    
        
    Rect roi_range = minContainingRect(src.slices_mask[4]);
    Rect aug_range = minContainingRect(src.blend_mask[4]);
    
    Mat blend_roi = blended.operator()(roi_range);
    
    Mat aug_roi = blended_aug.operator()(aug_range);
    
        
    BidirectSimilarity bi_Sim = BidirectSimilarity();
    
    bi_Sim.reconstruct(src.pattern.operator()(aug_range), similar.operator()(aug_range), aug_roi, weight.operator()(aug_range), src.blend_mask[4].operator()(aug_range), src.slices_mask[4].operator()(aug_range), Point(src.centroid.x - aug_range.x, src.centroid.y - aug_range.y)).copyTo(aug_roi);
    
//    showMat(src.slices_mask[4], "s", 1);
    
    int dilation_size = 1;
    Mat element = getStructuringElement( MORPH_RECT, Size( 2*dilation_size + 1, 2*dilation_size+1 ), Point( dilation_size, dilation_size ) );	//RECT  ELLIPSE  CROSS
    dilate(src.slices_mask[4], src.slices_mask[4], element);

    
//    showMat(src.slices_mask[4]);
    multiply(blended_aug, src.slices_mask[4], blended_aug);
    
    return reconstSlice(src, blended_aug);
    
}


Mat sliceSynthesis::reconstSlice(const tattingPattern &tatting, const cv::Mat &slice) {
    
    Mat reconst = slice.clone();
    
    
    for (int s_num = 1; s_num < tatting.slice_num; s_num ++) {
        

        
        Mat rot_mat = getRotationMatrix2D(tatting.centroid, (360.0/tatting.slice_num)*s_num, 1.0);
        Mat temp;
        warpAffine(slice, temp, rot_mat, slice.size(), INTER_CUBIC);
        
        for (int i = 0; i < slice.rows; i ++) {
            for (int j = 0; j < slice.cols; j ++) {
                
                reconst.at<double>(i, j) = MAX(reconst.at<double>(i, j) , temp.at<double>(i, j));
            }
        }
        
    }
    
    connectedCompRefine(reconst);
    
    return reconst;
}


void sliceSynthesis::connectedCompRefine(cv::Mat &src) {

    vector< vector<Point> > comp;
    
    Mat src_temp = src.clone();

    for (int i = 0; i < src.rows; i ++) {
        for (int j = 0; j < src.cols; j ++) {
            
            if ( src.at<double>(i, j) ) {
                
                stack<Point> local_comp;
                vector<Point> temp_comp;
                
                local_comp.push(Point(j, i));
                
                do {
                    
                    Point local_p = local_comp.top();
                    local_comp.pop();
                    
                    src.at<double>(local_p) = 0;
                    temp_comp.push_back(local_p);
                    
                    for (int m = -1; m <= 1; m ++) {
                        for (int n = -1; n <= 1; n ++) {
                            
                            Point nex = local_p + Point(n, m);
                            
                            if ( boundaryTest(src, nex) ) {
                                if ( src.at<double>(nex) >= 0.5 ) {
                                    local_comp.push(local_p + Point(n, m));
                                }
                            }
                            
                        }
                    }
                    
                    
                } while (!local_comp.empty());
                
                
                comp.push_back(temp_comp);
                
            }
            
        }
    }
    
    int max_idx = 0, max_size = 0;
    for (int i = 0; i < comp.size(); i ++) {
        
        if ( max_size < comp[i].size() ) {
            
            max_size = (int)comp[i].size();
            max_idx = i;
        }
    }
    
    
    for (int i = 0; i < comp[max_idx].size(); i ++) {
        src.at<double>(comp[max_idx][i]) = src_temp.at<double>(comp[max_idx][i]);
    }

    
    vector< vector<Point> >().swap(comp);
    
}


Mat sliceSynthesis::hybridweight (tattingPattern src, double resize_ratio, double layer_ratio, double layer_sigma, double slice_sigma) {
    
    
    Mat weight_circular = Mat::zeros(src.pattern.size(), src.pattern.type()), weight_slice = Mat::zeros(src.pattern.size(), src.pattern.type());
    
    if ( layer_sigma != 0 ) {
         weight_circular = circularGaussianRG(layer_ratio, layer_sigma, src.centroid, src.pattern.size(), src.max_radius);
    }
    
    if ( slice_sigma != 0 ) {
        weight_slice = sliceGaussianRG(src.slice_num, slice_sigma, src.centroid, src.pattern.size(), src.max_radius, src.sym_angle);
    }
    

    
    for (int i = 0; i < weight_circular.rows; i ++) {
        for (int j = 0; j < weight_circular.cols; j ++) {
            
            if ( sqrt(pow(i - src.centroid.y, 2.0) + pow(j - src.centroid.x, 2.0)) < src.max_radius*resize_ratio ) {
                weight_circular.at<double>(i, j) = MIN(weight_circular.at<double>(i, j) + weight_slice.at<double>(i, j), 1.0);
            }
            
        }
    }
    
    
    return weight_circular;
}






