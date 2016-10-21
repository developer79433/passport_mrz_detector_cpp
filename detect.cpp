#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <cerrno>

#include <opencv2/opencv.hpp>
#include <log4cpp/Category.hh>
#include <log4cpp/Appender.hh>
#include <log4cpp/FileAppender.hh>
#include <log4cpp/OstreamAppender.hh>
#include <log4cpp/Layout.hh>
#include <log4cpp/PatternLayout.hh>
#include <log4cpp/SimpleLayout.hh>
#include <log4cpp/Priority.hh>

#include "debug.h"
#include "ocr.h"
#include "mrz.h"
#include "RecogniserKNearest.h"

using namespace std;
using namespace cv;
using namespace ocr;

#define CHAR_SIZE_TOLERANCE 0.1
#define MRZ_LINE_SPACING 1.0
#define TRAINING_DATA_FILENAME "training.data"

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

void find_character_contours(const Mat &image, vector<vector<Point> > &characterContours, enum MRZ::mrz_type type = MRZ::mrz_type::UNKNOWN)
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
#if 0
            int expand = min(br.width, br.height) / 5;
            br -= Point(expand, expand);
            br += Size(2 * expand, 2 * expand);
#endif
            char_bboxes.push_back(br);
        } else {
        	// Not the right size
            // dump_rect("Rejected char", br);
        }
    });
}

static void assign_to_lines(const Mat &image, const vector<Rect> &char_bboxes, unsigned int num_lines, vector<vector<Rect> > &lines, vector<Rect> &indeterminate)
{
	for_each(char_bboxes.begin(), char_bboxes.end(), [image, num_lines, &lines, &indeterminate](const Rect &bbox) {
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
			lines[line_num].push_back(bbox);
		} else {
			indeterminate.push_back(bbox);
		}
	});
}

static float confidence_type(const Mat &image, unsigned int num_lines, unsigned int chars_per_line, const vector<Rect> &char_bboxes)
{
	if (char_bboxes.size() < num_lines * chars_per_line / 2) {
		return 0; // Less than 50% characters recognised
	}
	vector<vector<Rect> > lines(num_lines);
	vector<Rect> indeterminate;
	assign_to_lines(image, char_bboxes, lines.size(), lines, indeterminate);
	if (indeterminate.size() > char_bboxes.size() / 5) {
		return 0; // More than 20% of characters not aligned
	}
	unsigned int num_aligned = 0;
	for (unsigned int line_num = 0; line_num < num_lines; line_num++) {
		if (lines[line_num].size() > chars_per_line) {
			return 0; // Line too long
		}
		num_aligned += lines[line_num].size();
	}
	return static_cast<float>(num_aligned) / static_cast<float>(num_lines * chars_per_line);
}

static float confidence_type_1(const Mat &image, const vector<Rect> &char_bboxes)
{
	return confidence_type(image, MRZ::getLineCount(MRZ::mrz_type::TYPE_1), MRZ::getCharsPerLine(MRZ::mrz_type::TYPE_1), char_bboxes);
}

static float confidence_type_3(const Mat &image, const vector<Rect> &char_bboxes)
{
	return confidence_type(image, MRZ::getLineCount(MRZ::mrz_type::TYPE_3), MRZ::getCharsPerLine(MRZ::mrz_type::TYPE_3), char_bboxes);
}

static void fixup_missing_chars(const Mat &image, vector<Rect> &bboxes, enum MRZ::mrz_type type)
{
	// unsigned int num_expected = MRZ::getLineCount(type) * MRZ::getCharsPerLine(type);
	// TODO
}

static void find_chars(const Mat &image, vector<Rect> &bboxes)
{
	Rect borders = find_borders(image);
	// dump_rect("ROI border", borders);
	Mat cropped = image(borders);
	cropped = 255 - cropped;
	// display_image("Inverted cropped ROI", cropped);
	enum MRZ::mrz_type type = MRZ::mrz_type::UNKNOWN;
    find_character_bboxes(cropped, bboxes, type);
#if 0
    drawContours(cropped, characterContours, -1, Scalar(127, 127, 127));
    drawContourBoundingRects(cropped, characterContours, -1, Scalar(127, 127, 127));
    for_each(bboxes.begin(), bboxes.end(), [&image](const Rect &bbox) {
    	Mat character(cropped(bbox));
    	rectangle(cropped, bbox, Scalar(127, 127, 127));
    	// display_image("Character", character);
    });
#endif
    // TODO: sort(bboxes);
    float conf_type_1 = confidence_type_1(cropped, bboxes);
    float conf_type_3 = confidence_type_3(cropped, bboxes);
    if (conf_type_1 > max(conf_type_3, 0.75f)) {
    	cerr << "Looks like type 1" << endl;
    	type = MRZ::mrz_type::TYPE_1;
    } else if (conf_type_3 > max(conf_type_1, 0.75f)) {
    	cerr << "Looks like type 3" << endl;
    	type = MRZ::mrz_type::TYPE_3;
    } else {
    	cerr
		<< "Indeterminate type: " << conf_type_1 << " confidence Type 1, "
    	<< conf_type_3 << " confidence Type 3" << endl;
    }
    fixup_missing_chars(cropped, bboxes, type);
}

static void recognise_chars(const Mat &image, const Point &off, vector<Rect> &bboxes, string &text)
{
#if 0
	display_image("recognise_chars image", image);
#endif
    for_each(bboxes.begin(), bboxes.end(), [&image, off, &text](const Rect &bbox) {
    	Mat character(image(bbox));
    	char s[2] = {0, 0};
		RecogniserKNearest recogniser(TRAINING_DATA_FILENAME);
    	s[0] = recogniser.recognize(character, true);
    	text.append(s);
#if 0 || defined(DISPLAY_INTERMEDIATE_IMAGES)
    	cerr << "Recognised char: " << s[0] << endl;
    	display_image("Recognising", character);
#endif /* 0 || defined(DISPLAY_INTERMEDIATE_IMAGES) */
    });
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
        // Make it relative to original image again
        float scale = static_cast<float>(original.size().width) / static_cast<float>(image.size().width);
        roiRect.x *= scale;
        roiRect.y *= scale;
        roiRect.width *= scale;
        roiRect.height *= scale;
 
#ifdef DISPLAY_INTERMEDIATE_IMAGES
        // extract the ROI from the image and draw a bounding box
        // surrounding the MRZ
        Mat results = original.clone();
        rectangle(results, roiRect, Scalar(0, 255, 0), 2);
        // show the output images
        display_image("MRZ detection results", results);
#endif /* DISPLAY_INTERMEDIATE_IMAGES */
        Mat roiImage(original, roiRect);
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
		RecogniserTesseract tess("eng", &data_dir[0], MRZ_CHARS);
		tess.set_image_bmp(&buf[0]);
		tess.ocr();
#else /* ndef USE_TESSERACT */
		vector<Rect> char_bboxes;
		find_chars(roi_thresh, char_bboxes);
		string text;
		recognise_chars(original(roiRect), roiRect.tl(), char_bboxes, text);
		cerr << "Recognised text: " << text << endl;
#endif /* ndef USE_TESSERACT */
#if 1 || defined(DISPLAY_INTERMEDIATE_IMAGES)
    display_image("Original", original);
#endif /* 1 || DISPLAY_INTERMEDIATE_IMAGES */
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

int train(void)
{
	cout << getBuildInformation() << endl;
	Mat img = imread("ocrb.png");
	SlidingWindowCapture image_source(img, Size(70, 115), Point(70 + 2));
	RecogniserKNearest::learnOcr(image_source, MRZ::charset, TRAINING_DATA_FILENAME);

	return EXIT_SUCCESS;
}

int main(int argc, char *argv[])
{
	log4cpp::Appender *consoleAppender = new log4cpp::OstreamAppender("console", &std::cerr);
	log4cpp::Category& root = log4cpp::Category::getRoot();
	root.setPriority(log4cpp::Priority::getPriorityValue("DEBUG"));
	consoleAppender->setLayout(new log4cpp::SimpleLayout());
	root.addAppender(consoleAppender);
	train();
    return process_cmdline_args(argc, argv);
}
