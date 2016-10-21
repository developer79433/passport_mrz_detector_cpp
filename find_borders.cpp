#include "find_borders.h"

using namespace cv;

Rect find_borders(const Mat &image)
{
    Mat work = image.clone();
    vector<vector<Point> > contours;
    findContours(work, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    // Sort the contours in decreasing area
    sort(contours.begin(), contours.end(), [](const vector<Point>& c1, const vector<Point>& c2) {
        return contourArea(c1, false) > contourArea(c2, false);
    });
    if (contours.size() > 0) {
        return boundingRect(contours[0]);
    }
    return Rect();
}
