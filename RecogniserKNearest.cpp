#include <iostream>
#include <fstream>

#include <log4cpp/Category.hh>
#include <log4cpp/Priority.hh>

#include "RecogniserKNearest.h"
#include "debug.h"

using namespace std;
using namespace cv;

namespace ocr {

unsigned int RecogniserKNearest::MAX_DISTANCE = 3000000; // 600000;

RecogniserKNearest::RecogniserKNearest(const char *filename)
{
    if (filename) {
        loadTrainingData(filename);
    }
}

RecogniserKNearest::~RecogniserKNearest()
{
}

/**
 * Prepare an image of a digit to work as a sample for the model.
 */
Mat RecogniserKNearest::prepareSample(const Mat& img, bool black_on_white) {
#if 0
    display_image("prepareSample input", img);
#endif
    Mat grey;
    if (img.channels() == 1) {
    	grey = img;
    } else {
    	cvtColor(img, grey, COLOR_BGR2GRAY);
    }
    if (black_on_white) {
    	grey = 255 - grey;
    }
    Mat roi, sample;
    resize(grey, roi, Size(14, 14));
    threshold(roi, roi, 0, 255, THRESH_BINARY | THRESH_OTSU);
#if 0
    display_image("prepareSample 10x10", roi);
#endif
    roi.reshape(1, 1).convertTo(sample, CV_32F);
#if 0
    display_image("prepareSample sample", sample);
#endif
    return sample;
}

/**
 * Learn a single digit.
 */
void RecogniserKNearest::learn(const Mat & img, char key) {
#if 0
    cerr << key << endl;
    imshow("Learn", img);
    waitKey(0);
#endif
    _responses.push_back(Mat(1, 1, CV_32F, (float) key - '0'));
    _samples.push_back(prepareSample(img));
}

/**
 * Learn a vector of digits.
 */
void RecogniserKNearest::learn(const std::vector<cv::Mat>& images, const std::string &answers) {
    std::vector<cv::Mat>::const_iterator it;
    std::string::const_iterator answer_it = answers.begin();
    for (it = images.begin(), answer_it = answers.begin();
            it < images.end(); ++it, ++answer_it) {
        learn(*it, *answer_it);
    }
}

/**
 * Initialize the model.
 */
void RecogniserKNearest::initModel() {
    _pModel = new CvKNearest(_samples, _responses);
}

/**
 * Recognize a single digit.
 */
char RecogniserKNearest::recognize(const Mat& img, bool black_on_white) {
#if 0
    cerr << "Sample count: " << _pModel->get_sample_count() << "; var count: " <<_pModel->get_var_count() << endl;
#endif
    char cres = '?';
    Mat results, neighborResponses, dists;
    Mat sample = prepareSample(img, black_on_white);
#if 0
    display_image("recognize sample", sample);
#endif
    float result = _pModel->find_nearest(
      sample, 2, results, neighborResponses, dists);

#if 0
    cerr << "results: " << results << endl;
    cerr << "neighborResponses: " << neighborResponses << endl;
    cerr << "dists: " << dists << endl;
#endif

    if (/*0 == int(neighborResponses.at<float>(0, 0) - neighborResponses.at<float>(0, 1))
          &&*/ dists.at<float>(0, 0) < MAX_DISTANCE) {

        cres = (int) result + '0';
        // rlog << log4cpp::Priority::DEBUG << "recog succes: " << cres;
    } else {
        // rlog << log4cpp::Priority::DEBUG << "recog fail: " << dists.at<float>(0, 0) << ", " << MAX_DISTANCE;
    }


    return cres;
}

/**
 * Save training data to file.
 */
void RecogniserKNearest::saveTrainingData(const char *filename)
{
    FileStorage fs(filename, FileStorage::WRITE);
    fs << "samples" << _samples;
    fs << "responses" << _responses;
    fs.release();
}

/**
 * Load training data from file and init model.
 */
void RecogniserKNearest::loadTrainingData(const char *filename)
{
    FileStorage fs(filename, FileStorage::READ);
    if (fs.isOpened()) {
        fs["samples"] >> _samples;
        fs["responses"] >> _responses;
        fs.release();

        initModel();
    }
}

void RecogniserKNearest::learnOcr(VideoCapture &pImageInput, const string &answers, const char *filename) {
    if (!pImageInput.isOpened()) {
        throw "Image source not open";
    }
    RecogniserKNearest ocr;
    string::const_iterator answer_iter = answers.begin();

    for (;;) {
        Mat img;
    	if (!pImageInput.read(img)) {
    		break;
    	}
        if (!img.data) {
            break;
        }
#if 0
        display_image("learnOcr", img);
#endif
        ocr.learn(img, *answer_iter);
        answer_iter++;
    }

    ocr.saveTrainingData(filename);
}

}
