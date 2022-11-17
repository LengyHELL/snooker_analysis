#pragma once

#include <opencv2/opencv.hpp>
#include <fdeep/fdeep.hpp>

#include <string>
#include <vector>
#include <chrono>

#include <BallLabel.hpp>
#include <Template.hpp>
#include <Cut.hpp>
#include <Section.hpp>
#include <Ball.hpp>

struct BallIndex {
	int id;
	BallLabel label;

	BallIndex(const int& id, const BallLabel& label): id(id), label(label) {}

	bool operator<(const BallIndex& other) const {
		if (label == other.label) {
			return id < other.id;
		} else {
			return label < other.label;
		}
	}
};

struct BallMovement {
	float distance;
	int* currentId;
	int* previousId;

	BallMovement(const float& distance, int* currentId, int* previousId) : distance(distance), currentId(currentId), previousId(previousId) {}
};

class Recognition {
    const int width;
    const int height;
    const int circleRadius;

	int iterator = 0;

	std::vector<Ball> previousBalls;
	std::vector<Ball> balls;
    std::vector<cv::Point> quad;
	std::map<BallIndex, std::vector<cv::Point>> ballPaths;
	
    std::vector<Template> templates;
    fdeep::model model;

	static std::vector<cv::Point> contourToQuad(const std::vector<cv::Point>& contour);

	bool getRedBall(const int& id, Ball*& redBall);
	void findTable(const cv::Mat& image);
	bool cutAndWarp(const cv::Mat& image, cv::Mat& warpedImage);
	void findBalls(const cv::Mat& image);
	void setBallCuts(const cv::Mat& image);
	void labelBallsWithTM();
	void labelBallsWithNN();


public:
	cv::Mat debugFrameCanny;

	cv::Mat debugFrameMask;
	cv::Scalar lowerGreen = cv::Scalar(50, 50, 70);		// 40, 190, 50 -- 50, 0, 111
    cv::Scalar upperGreen = cv::Scalar(65, 255, 255);	// 65, 255, 255
	double kernelIterations = 1;
	double tableEpsilonRate = 5;

	cv::Mat debugFrameCircles;
	int minRadiusRate = 5;		// 6
	int maxRadiusRate = 12;		// 12
	int minDistanceRate = 12;	// 16
	int circlePerfectness = 10;	// 15
	int thresholdRate = 80;		// 80

	cv::Mat processedFramePath;

    Recognition();

	void processFrameWithNN(const cv::Mat& videoFrame);
	std::vector<cv::Point> getBallPath(const BallLabel& label, const int& id = 0) const;
};