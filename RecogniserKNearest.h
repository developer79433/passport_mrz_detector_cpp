#ifndef RECOGNISERKNEAREST_H_
#define RECOGNISERKNEAREST_H_

#include <opencv2/opencv.hpp>

#include "ocr.h"

namespace ocr {

class RecogniserKNearest : public Recogniser {
private:
    static unsigned int MAX_DISTANCE;
    cv::Mat prepareSample(const cv::Mat & img, bool black_on_white = false);
    void initModel();

    cv::Mat _samples;
    cv::Mat _responses;
    CvKNearest* _pModel;
public:
	RecogniserKNearest(const char *filename = NULL);
	virtual ~RecogniserKNearest();
    void learn(const cv::Mat & img, char key);
    void learn(const std::vector<cv::Mat> & images, const std::string &answers);
    char recognize(const cv::Mat & img, bool black_on_white = false);
    void saveTrainingData(const char *filename);
    void loadTrainingData(const char *filename);
    static void learnOcr(cv::VideoCapture &pImageInput, const std::string &answers, const char *filename);
};

}

#endif /* RECOGNISERKNEAREST_H_ */
