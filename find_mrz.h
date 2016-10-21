#ifndef FIND_MRZ_H
#define FIND_MRZ_H 1

#include <opencv2/opencv.hpp>

bool find_mrz(const cv::Mat &original, cv::Mat &mrz);

#endif /* FIND_MRZ_H */
