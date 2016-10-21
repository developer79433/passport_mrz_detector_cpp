#ifndef SLIDING_WINDOW_CAPTURE_H
#define SLIDING_WINDOW_CAPTURE_H 1

#include <opencv2/opencv.hpp>

using namespace cv;

class SlidingWindowCapture : public VideoCapture {
private:
    Mat &img;
    Rect current;
    Point off;
    unsigned int i;
    int lim;
public:
    SlidingWindowCapture(Mat &image, const Rect &start_window, const Point &inter_window_offset, int num_windows = -1)
        : img(image), current(start_window), off(inter_window_offset), i(0), lim(num_windows) {};
    SlidingWindowCapture(Mat &image, const Size &window_size, const Point &inter_window_offset, int num_windows = -1, const Point &start_offset = Point(0, 0))
        : img(image), current(Rect(start_offset, window_size)), off(inter_window_offset), i(0), lim(num_windows) {};
    virtual ~SlidingWindowCapture(void) {};
    CV_WRAP virtual bool read(CV_OUT Mat& image) {
        if (lim >= 0 && i >= static_cast<unsigned int>(lim)) {
            return false; // Reached frame count limit
        }
        if (
            current.x < 0 || current.x + current.width > img.size().width ||
            current.y < 0 || current.y + current.height > img.size().height
        ) {
            return false; // Run out of image
        }
        image = img(current);
        current += off;
        return true;
    };
    // We are always open, but we may have run out of images.
    CV_WRAP virtual bool isOpened() const { return true; };
};

#endif /* SLIDING_WINDOW_CAPTURE_H */
