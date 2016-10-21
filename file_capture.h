#ifndef FILE_CAPTURE_H
#define FILE_CAPTURE_H 1

#include <opencv2/opencv.hpp>

using namespace cv;

/**
 * OpenCV provides a class like this, but I can't get it to work.
 * It provides images with non-NULL data, but all zeroes.
 * So this subclass over-rides the interesting functionality
 * in the directoryful-of-files use case with fixed versions.
 */
class FileCapture : public VideoCapture {
private:
    const char *patt;
    unsigned int count;
public:
    FileCapture(const char *input_pattern)
        : VideoCapture(), patt(input_pattern), count(0) {};
    virtual ~FileCapture() {}
    CV_WRAP virtual bool read(CV_OUT Mat& image) {
        char *filename;
        // NOTE: Use of potentially untrusted format string passed in by caller
        asprintf(&filename, patt, count);
        image = imread(filename);
        bool ret = !!image.data;
        free(filename);
        count++;
        return ret;
    };
    // We are always open, but there may not be any images when we look.
    CV_WRAP virtual bool isOpened() const { return true; };
};

#endif /* FILE_CAPTURE_H */
