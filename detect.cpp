#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <cerrno>

#include <opencv2/opencv.hpp>

#include "debug.h"
#include "ocr.h"

using namespace std;
using namespace cv;
using namespace ocr;

#define MRZ_CHARS "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"

std::string getcwd(void) {
    string result(1024, '\0');
    while (getcwd(&result[0], result.size()) == 0) {
        if( errno != ERANGE ) {
          throw runtime_error(strerror(errno));
        }
        result.resize(result.size() * 2);
    }
    result.resize(result.find('\0'));
    return result;
}

#if 0
#define DISPLAY_INTERMEDIATE_IMAGES
#endif

static void process(Mat &original)
{
    // initialize a rectangular and square structuring kernel
    Mat rectKernel = getStructuringElement(MORPH_RECT, Size(13, 5));
    Mat sqKernel = getStructuringElement(MORPH_RECT, Size(21, 21));

    // resize the image and convert it to grayscale
    Mat image;
    resize(original, image, Size(original.size().width * 600 / original.size().height, 600));
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
 
    // smooth the image using a 3x3 Gaussian, then apply the blackhat
    // morphological operator to find dark regions on a light background
    GaussianBlur(gray, gray, Size(3, 3), 0);
    Mat blackhat;
    morphologyEx(gray, blackhat, MORPH_BLACKHAT, rectKernel);

#ifdef DISPLAY_INTERMEDIATE_IMAGES
    display_image("Blackhat", blackhat);
#endif /* DISPLAY_INTERMEDIATE_IMAGES */

    // compute the Scharr gradient of the blackhat image and scale the
    // result into the range [0, 255]
    Mat gradX;
    Sobel(blackhat, gradX, CV_32F, 1, 0, -1);
    gradX = abs(gradX);
    double minVal, maxVal;
    minMaxIdx(gradX, &minVal, &maxVal);
    Mat gradXfloat = (255 * ((gradX - minVal) / (maxVal - minVal)));
    gradXfloat.convertTo(gradX, CV_8UC1);

#ifdef DISPLAY_INTERMEDIATE_IMAGES
    display_image("Gx", gradX);
#endif /* DISPLAY_INTERMEDIATE_IMAGES */
 
    // apply a closing operation using the rectangular kernel to close
    // gaps in between letters -- then apply Otsu's thresholding method
    morphologyEx(gradX, gradX, MORPH_CLOSE, rectKernel);
    Mat thresh;
    threshold(gradX, thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);

#ifdef DISPLAY_INTERMEDIATE_IMAGES
    display_image("Horizontal closing", thresh);
#endif /* DISPLAY_INTERMEDIATE_IMAGES */

    // perform another closing operation, this time using the square
    // kernel to close gaps between lines of the MRZ, then perform a
    // series of erosions to break apart connected components
    morphologyEx(thresh, thresh, MORPH_CLOSE, sqKernel);
    Mat nullKernel;
    erode(thresh, thresh, nullKernel, Point(-1, -1), 4);

#ifdef DISPLAY_INTERMEDIATE_IMAGES
    display_image("Vertical closing", thresh);
#endif /* DISPLAY_INTERMEDIATE_IMAGES */

    // during thresholding, it's possible that border pixels were
    // included in the thresholding, so let's set 5% of the left and
    // right borders to zero
    double p = image.size().height * 0.05;
    /*
    thresh[:, 0:p] = 0;
    thresh[:, image.size().height - p:] = 0;
    */
    thresh = thresh(Rect(p, p, image.size().width - 2 * p, image.size().height - 2 * p));

#ifdef DISPLAY_INTERMEDIATE_IMAGES
    display_image("Border removal", thresh);
#endif /* DISPLAY_INTERMEDIATE_IMAGES */

    // find contours in the thresholded image and sort them by their
    // size
    vector<vector<Point> > contours;
    findContours(thresh, contours, RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE);
    // Sort the contours in decreasing area
    sort(contours.begin(), contours.end(), [](const vector<Point>& c1, const vector<Point>& c2){
        return contourArea(c1, false) > contourArea(c2, false);
    });
 
    // Find the first contour with the right aspect ratio and a large width relative to the width of the image
    Rect roiRect(0, 0, 0, 0);
    vector<vector<Point> >::iterator border_iter = find_if(contours.begin(), contours.end(), [&roiRect, gray](vector<Point> &contour) {
        // compute the bounding box of the contour and use the contour to
        // compute the aspect ratio and coverage ratio of the bounding box
        // width to the width of the image
        roiRect = boundingRect(contour);
        dump_rect("Bounding rect", roiRect);
        // pprint([x, y, w, h])
        double aspect = (double) roiRect.size().width / (double) roiRect.size().height;
        double coverageWidth = (double) roiRect.size().width / (double) gray.size().height;
        cerr << "aspect=" << aspect << "; coverageWidth=" << coverageWidth << endl;
        // check to see if the aspect ratio and coverage width are within
        // acceptable criteria
        if (aspect > 5 and coverageWidth > 0.5) {
            return true;
        }
        return false;
    });

    if (border_iter == contours.end()) {
    } else {
        // Correct ROI for border removal offset
        roiRect.x += p;
        roiRect.y += p;
        // pad the bounding box since we applied erosions and now need
        // to re-grow it
        int pX = (roiRect.x + roiRect.size().width) * 0.03;
        int pY = (roiRect.y + roiRect.size().height) * 0.03;
        int x = roiRect.x - pX, y = roiRect.y - pY;
        int w = roiRect.size().width + pX * 2, h = roiRect.size().height + pY * 2;
        // Ensure ROI is within image
        if (x < 0) {
            w += x;
            x = 0;
        }
        if (y < 0) {
            h += y;
            y = 0;
        }
        if (x + w > image.size().width) {
            w = image.size().width - x;
        }
        if (y + h > image.size().height) {
            h = image.size().height - y;
        }
        roiRect.x = x;
        roiRect.y = y;
        roiRect.width = w;
        roiRect.height = h;
 
        // extract the ROI from the image and draw a bounding box
        // surrounding the MRZ
        Mat results = image.clone();
        rectangle(results, roiRect, Scalar(0, 255, 0), 2);
        // show the output images
#ifdef DISPLAY_INTERMEDIATE_IMAGES
        display_image("MRZ detection results", results);
#endif /* DISPLAY_INTERMEDIATE_IMAGES */
        Mat roiImage(image, roiRect);
#ifdef DISPLAY_INTERMEDIATE_IMAGES
        display_image("ROI", roiImage);
#endif /* DISPLAY_INTERMEDIATE_IMAGES */
        Mat grey, thresh;
        cvtColor(roiImage, grey, COLOR_BGR2GRAY);
        threshold(grey, thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);
        display_image("Threshold", thresh);
		vector<uchar> buf;
		imencode(".bmp", thresh, buf);
		string data_dir = getcwd();
		data_dir.append("/tessdata");
		cerr << "data dir: " << data_dir << endl;
		Recogniser tess("eng", &data_dir[0], MRZ_CHARS);
		tess.set_image_bmp(&buf[0]);
		tess.ocr();
    }

}

static int process_cmdline_args(int argc, char *argv[])
{
    char **arg;
    int ret = EXIT_SUCCESS;

    for (arg = &argv[1]; arg < &argv[argc]; arg++) {
        Mat input = imread(*arg);
        if (input.data) {
            process(input);
        } else {
            cerr << "Failed to load image from " << *arg << endl;
            ret = EXIT_FAILURE;
            break;
        }
    }

    return ret;
}

int main(int argc, char *argv[])
{
    return process_cmdline_args(argc, argv);
}
