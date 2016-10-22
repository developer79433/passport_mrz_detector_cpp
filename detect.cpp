#include <iostream>
#include <cstdlib>

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
#include "find_mrz.h"
#include "find_borders.h"
#include "getcwd.h"
#include "file_capture.h"
#include "sliding_window_capture.h"

using namespace std;
using namespace cv;
using namespace ocr;

#if 0
#define USE_TESSERACT
#endif
#define CHAR_SIZE_TOLERANCE 0.1
#define MRZ_LINE_SPACING 1.0
#define TRAINING_DATA_FILENAME "training.data"

#if 0
#define DISPLAY_INTERMEDIATE_IMAGES
#endif

static void calc_char_cell(const Size &mrz_size, Size &char_min, Size &char_max, enum MRZ::mrz_type type = MRZ::mrz_type::UNKNOWN)
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
    char_min = Size((double) mrz_size.width / (double) max_chars_per_line, (double) mrz_size.height / (double) max_lines);
    char_max = Size((double) mrz_size.width / (double) min_chars_per_line, (double) mrz_size.height / (double) min_lines);
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

static void find_character_bboxes(const Mat &image, vector<Rect> &char_bboxes, enum MRZ::mrz_type type = MRZ::mrz_type::UNKNOWN)
{
    vector<vector<Point> > contours;
    Mat work = image.clone();
    findContours(work, contours, RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE);
    Size char_min, char_max;
    calc_char_cell(image.size(), char_min, char_max, type);
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

static void assign_to_lines(const Size &image_size, const vector<Rect> &char_bboxes, vector<vector<Rect> > &lines, vector<Rect> &indeterminate)
{
	for_each(char_bboxes.begin(), char_bboxes.end(), [image_size, &lines, &indeterminate](const Rect &bbox) {
		unsigned int line_num;
		bool found = false;
		for (line_num = 0; line_num < lines.size(); line_num++) {
			int top = image_size.height * line_num / lines.size();
			int middle = image_size.height * (line_num + 0.5) / lines.size();
			int bottom = image_size.height * (line_num + 1) / lines.size();
			if (
				bbox.y >= top && bbox.y + bbox.height <= bottom &&
				abs((bbox.y + bbox.y + bbox.height) / 2 - middle) < abs(image_size.height / (3 * lines.size()))
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

static float confidence_type(const Size &image_size, vector<vector<Rect> > &lines, unsigned int chars_per_line, const vector<Rect> &char_bboxes, vector<Rect> &indeterminate)
{
	if (char_bboxes.size() < lines.size() * chars_per_line / 2) {
		return 0; // Less than 50% characters recognised
	}
	assign_to_lines(image_size, char_bboxes, lines, indeterminate);
	if (indeterminate.size() > char_bboxes.size() / 5) {
		return 0; // More than 20% of characters not aligned
	}
	unsigned int num_aligned = 0;
	for (unsigned int line_num = 0; line_num < lines.size(); line_num++) {
		if (lines[line_num].size() > chars_per_line) {
			return 0; // Line too long
		}
		num_aligned += lines[line_num].size();
	}
	return static_cast<float>(num_aligned) / static_cast<float>(lines.size() * chars_per_line);
}

static float confidence_type_1(const Size &image_size, vector<vector<Rect> > &lines, const vector<Rect> &char_bboxes, vector<Rect> &indeterminate)
{
	return confidence_type(image_size, lines, MRZ::getCharsPerLine(MRZ::mrz_type::TYPE_1), char_bboxes, indeterminate);
}

static float confidence_type_3(const Size &image_size, vector<vector<Rect> > &lines, const vector<Rect> &char_bboxes, vector<Rect> &indeterminate)
{
	return confidence_type(image_size, lines, MRZ::getCharsPerLine(MRZ::mrz_type::TYPE_3), char_bboxes, indeterminate);
}

static void fixup_missing_chars(const Mat &image, vector<vector<Rect> > &lines, enum MRZ::mrz_type type)
{
	// unsigned int num_expected = MRZ::getLineCount(type) * MRZ::getCharsPerLine(type);
	// TODO
}

static void sort_lines(vector<vector<Rect> > &lines)
{
	for_each(lines.begin(), lines.end(), [](vector<Rect> &line) {
		sort(line.begin(), line.end(), [](const Rect &r1, const Rect &r2) {
			return r1.x < r2.x;
		});
	});
}

static bool find_chars(const Mat &image, vector<vector<Rect> > &lines)
{
	Rect borders = find_borders(image);
	// dump_rect("ROI border", borders);
	Mat cropped = image(borders);
	cropped = 255 - cropped;
	// display_image("Inverted cropped ROI", cropped);
	enum MRZ::mrz_type type = MRZ::mrz_type::UNKNOWN;
	vector<Rect> bboxes;
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
	vector<Rect> indeterminate_type_1;
	vector<vector<Rect> > lines_type_1(MRZ::getLineCount(MRZ::mrz_type::TYPE_1));
    float conf_type_1 = confidence_type_1(cropped.size(), lines_type_1, bboxes, indeterminate_type_1);
	vector<Rect> indeterminate_type_3;
	vector<vector<Rect> > lines_type_3(MRZ::getLineCount(MRZ::mrz_type::TYPE_3));
    float conf_type_3 = confidence_type_3(cropped.size(), lines_type_3, bboxes, indeterminate_type_3);
    if (conf_type_1 > max(conf_type_3, 0.75f)) {
    	cerr << "Looks like type 1" << endl;
    	type = MRZ::mrz_type::TYPE_1;
    	lines = lines_type_1;
    } else if (conf_type_3 > max(conf_type_1, 0.75f)) {
    	cerr << "Looks like type 3" << endl;
    	type = MRZ::mrz_type::TYPE_3;
    	lines = lines_type_3;
    } else {
    	cerr
		<< "Indeterminate type: " << conf_type_1 << " confidence Type 1, "
    	<< conf_type_3 << " confidence Type 3" << endl;
    }

    if (type == MRZ::mrz_type::UNKNOWN) {
    	return false;
    }

    sort_lines(lines);
    fixup_missing_chars(cropped, lines, type);

    return true;
}

static void recognise_chars(const Mat &image, vector<vector<Rect> > &lines, string &text)
{
#if 0
	display_image("recognise_chars image", image);
#endif
	for_each(lines.begin(), lines.end(), [&image, &text](const vector<Rect> &line) {
	    for_each(line.begin(), line.end(), [&image, &text](const Rect &bbox) {
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
	});
}

static void process(Mat &original)
{
	Mat roiImage;
	if (!find_mrz(original, roiImage)) {
		cerr << "No MRZ found" << endl;
		return;
	}
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
		imencode(".bmp", roi_thresh, buf);
		string data_dir = getcwd();
		data_dir.append("/tessdata");
		cerr << "data dir: " << data_dir << endl;
		RecogniserTesseract tess("eng", &data_dir[0], MRZ::charset);
		tess.set_image_bmp(&buf[0]);
		tess.ocr();
#else /* ndef USE_TESSERACT */
		vector<vector<Rect> > lines;
		if (find_chars(roi_thresh, lines)) {
			string text;
			recognise_chars(roiImage, lines, text);
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
