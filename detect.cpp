#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <cerrno>

#include <opencv2/opencv.hpp>

#include "debug.h"
#include "ocr.h"
#include "mrz.h"

using namespace std;
using namespace cv;
using namespace ocr;

#define MRZ_CHARS "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"
#define CHAR_SIZE_TOLERANCE 0.1
#define MRZ_LINE_SPACING 1.0

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

static Rect find_borders(const Mat &image)
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

static void calc_char_cell(const Mat &image, Size &char_min, Size &char_max, enum MRZ::mrz_type type = MRZ::mrz_type::UNKNOWN)
{
    unsigned int min_lines, max_lines;
    unsigned int min_chars_per_line, max_chars_per_line;
    switch (type) {
    case MRZ::mrz_type::TYPE_1:
		min_chars_per_line = MRZType1::getCharsPerLine();
		max_chars_per_line = min_chars_per_line;
		min_lines = MRZType1::getLineCount();
		max_lines = min_lines;
		break;
    case MRZ::mrz_type::TYPE_3:
		min_chars_per_line = MRZType3::getCharsPerLine();
		max_chars_per_line = min_chars_per_line;
		min_lines = MRZType3::getLineCount();
		max_lines = min_lines;
		break;
    case MRZ::mrz_type::UNKNOWN:
    default:
		min_chars_per_line = min(MRZType1::getCharsPerLine(), MRZType3::getCharsPerLine());
		max_chars_per_line = max(MRZType1::getCharsPerLine(), MRZType3::getCharsPerLine());
		min_lines = min(MRZType1::getLineCount(), MRZType3::getLineCount());
		max_lines = max(MRZType1::getLineCount(), MRZType3::getLineCount());
		break;
    };
    // Account for inter-line spacing
    min_lines *= MRZ_LINE_SPACING + 1;
    min_lines -= 1;
    max_lines *= MRZ_LINE_SPACING + 1;
    max_lines -= 1;
    char_min = Size((double) image.size().width / (double) max_chars_per_line, (double) image.size().height / (double) max_lines);
    char_max = Size((double) image.size().width / (double) min_chars_per_line, (double) image.size().height / (double) min_lines);
    // Add a tolerance
    char_min.width /= (1 + CHAR_SIZE_TOLERANCE);
    char_min.height /= (1 + CHAR_SIZE_TOLERANCE);
    char_max.width *= (1 + CHAR_SIZE_TOLERANCE);
    char_max.height *= (1 + CHAR_SIZE_TOLERANCE);
#if 0
    cerr << "Char min rect: " << char_min << endl;
    cerr << "Char max rect: " << char_max << endl;
#endif
    // Additional tuning for minimum width.
    // Although OCR B is monospaced, some character glyphs are much narrower than others.
    char_min.width *= 0.25;
    // Additional tuning for minimum height.
    // Line spacing varies widely.
    char_min.height *= 0.75;
}

static bool is_character(const Rect boundingRect, const Size &minSize, const Size &maxSize)
{
	return
			boundingRect.width  >= minSize.width  &&
			boundingRect.height >= minSize.height &&
			boundingRect.width  <= maxSize.width  &&
			boundingRect.height <= maxSize.height
	;
}

static void find_character_contours(const Mat &image, vector<vector<Point> > &characterContours, enum MRZ::mrz_type type = MRZ::mrz_type::UNKNOWN)
{
    vector<vector<Point> > contours;
    Mat work = image.clone();
    findContours(work, contours, RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE);
    Size char_min, char_max;
    calc_char_cell(image, char_min, char_max, type);
    copy_if(contours.begin(), contours.end(), inserter(characterContours, characterContours.begin()), [char_min, char_max](vector<Point> &contour) {
        Rect br = boundingRect(contour);
        if (is_character(br, char_min, char_max)) {
            // dump_rect("Character", br);
            return true;
        }
    	// Not the right size
        // dump_rect("Rejected char", br);
    	return false;
    });
}

static void find_character_bboxes(const Mat &image, vector<Rect> &char_bboxes, enum MRZ::mrz_type type = MRZ::mrz_type::UNKNOWN)
{
    vector<vector<Point> > contours;
    Mat work = image.clone();
    findContours(work, contours, RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE);
    Size char_min, char_max;
    calc_char_cell(image, char_min, char_max, type);
    for_each(contours.begin(), contours.end(), [&char_bboxes, char_min, char_max](vector<Point> &contour) {
        Rect br = boundingRect(contour);
        if (is_character(br, char_min, char_max)) {
            // dump_rect("Character", br);
            char_bboxes.push_back(br);
        } else {
        	// Not the right size
            // dump_rect("Rejected char", br);
        }
    });
}

static void count_lines(const Mat &image, const vector<Rect> &char_bboxes, unsigned int num_lines, vector<unsigned int> &line_counts, unsigned int &num_indeterminate)
{
	num_indeterminate = 0;
	for (unsigned int line_num = 0; line_num < num_lines; line_num++) {
		line_counts[line_num] = 0;
	}
	for_each(char_bboxes.begin(), char_bboxes.end(), [image, num_lines, &line_counts, &num_indeterminate](const Rect &bbox) {
		unsigned int line_num;
		bool found = false;
		for (line_num = 0; line_num < num_lines; line_num++) {
			int top = image.size().height * line_num / num_lines;
			int middle = image.size().height * (line_num + 0.5) / num_lines;
			int bottom = image.size().height * (line_num + 1) / num_lines;
			if (
				bbox.y >= top && bbox.y + bbox.height <= bottom &&
				abs((bbox.y + bbox.y + bbox.height) / 2 - middle) < abs(image.size().height / (3 * num_lines))
			) {
				found = true;
				break;
			}
		}
		if (found) {
			line_counts[line_num]++;
		} else {
			num_indeterminate++;
		}
	});
}

static bool looks_like_type(const Mat &image, unsigned int num_lines, unsigned int chars_per_line, const vector<Rect> &char_bboxes)
{
	if (char_bboxes.size() < num_lines * chars_per_line / 2) {
		return false; // Less than 50% characters recognised
	}
	vector<unsigned int> line_counts(num_lines);
	unsigned int num_indeterminate;
	count_lines(image, char_bboxes, line_counts.size(), line_counts, num_indeterminate);
	if (num_indeterminate > char_bboxes.size() / 10) {
		return false; // More than 20% of characters not aligned
	}
	for (unsigned int line_num = 0; line_num < num_lines; line_num++) {
		if (line_counts[line_num] > chars_per_line) {
			return false; // Line too long
		}
	}
	return true;
}

static bool looks_like_type_1(const Mat &image, const vector<Rect> &char_bboxes)
{
	return looks_like_type(image, MRZ::getLineCount(MRZ::mrz_type::TYPE_1), MRZ::getCharsPerLine(MRZ::mrz_type::TYPE_1), char_bboxes);
}

static bool looks_like_type_3(const Mat &image, const vector<Rect> &char_bboxes)
{
	return looks_like_type(image, MRZ::getLineCount(MRZ::mrz_type::TYPE_3), MRZ::getCharsPerLine(MRZ::mrz_type::TYPE_3), char_bboxes);
}

static void find_chars(Mat &roi_thresh)
{
	Rect borders = find_borders(roi_thresh);
	// dump_rect("ROI border", borders);
	roi_thresh = roi_thresh(borders);
	roi_thresh = 255 - roi_thresh;
	// display_image("Inverted cropped ROI", roi_thresh);
	enum MRZ::mrz_type type = MRZ::mrz_type::UNKNOWN;
    vector<Rect> bboxes;
    find_character_bboxes(roi_thresh, bboxes, type);
    // drawContours(roi_thresh, characterContours, -1, Scalar(127, 127, 127));
    // drawContourBoundingRects(roi_thresh, characterContours, -1, Scalar(127, 127, 127));
    for_each(bboxes.begin(), bboxes.end(), [&roi_thresh](const Rect &bbox) {
    	rectangle(roi_thresh, bbox, Scalar(127, 127, 127));
    });
    display_image("Character contours", roi_thresh);
    if (looks_like_type_1(roi_thresh, bboxes)) {
    	cerr << "Looks like type 1" << endl;
    } else if (looks_like_type_3(roi_thresh, bboxes)) {
    	cerr << "Looks like type 3" << endl;
    }
}

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
        // dump_rect("Bounding rect", roiRect);
        // pprint([x, y, w, h])
        double aspect = (double) roiRect.size().width / (double) roiRect.size().height;
        double coverageWidth = (double) roiRect.size().width / (double) gray.size().height;
        // cerr << "aspect=" << aspect << "; coverageWidth=" << coverageWidth << endl;
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
        roiRect += Point(p, p);
        // pad the bounding box since we applied erosions and now need
        // to re-grow it
        int pX = (roiRect.x + roiRect.size().width) * 0.03;
        int pY = (roiRect.y + roiRect.size().height) * 0.03;
        roiRect -= Point(pX, pY);
        roiRect += Size(pX * 2, pY * 2);
        // Ensure ROI is within image
        roiRect &= Rect(0, 0, image.size().width, image.size().height);
 
#ifdef DISPLAY_INTERMEDIATE_IMAGES
        // extract the ROI from the image and draw a bounding box
        // surrounding the MRZ
        Mat results = image.clone();
        rectangle(results, roiRect, Scalar(0, 255, 0), 2);
        // show the output images
        display_image("MRZ detection results", results);
#endif /* DISPLAY_INTERMEDIATE_IMAGES */
        Mat roiImage(image, roiRect);
#ifdef DISPLAY_INTERMEDIATE_IMAGES
        display_image("ROI", roiImage);
#endif /* DISPLAY_INTERMEDIATE_IMAGES */
        Mat roi_grey, roi_thresh;
        cvtColor(roiImage, roi_grey, COLOR_BGR2GRAY);
        threshold(roi_grey, roi_thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);
#ifdef DISPLAY_INTERMEDIATE_IMAGES
        display_image("ROI threshold", roi_thresh);
#endif /* DISPLAY_INTERMEDIATE_IMAGES */
#ifdef USE_TESSERACT
		vector<uchar> buf;
		imencode(".bmp", thresh, buf);
		string data_dir = getcwd();
		data_dir.append("/tessdata");
		cerr << "data dir: " << data_dir << endl;
		Recogniser tess("eng", &data_dir[0], MRZ_CHARS);
		tess.set_image_bmp(&buf[0]);
		tess.ocr();
#else /* ndef USE_TESSERACT */
		find_chars(roi_thresh);
#endif /* ndef USE_TESSERACT */
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
