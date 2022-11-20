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

class BallData {
private:
	int previousUpdateFrame = 0;

public:
	std::vector<cv::Point> path = std::vector<cv::Point>();
	std::vector<float> speed = std::vector<float>();
	float totalDistance = 0;

	void addPosition(const cv::Point& newPosition, const int& framePosition) {
		if (path.empty()) {
			speed.push_back(0);
            path.push_back(newPosition);
        }
		else {
			if (framePosition <= previousUpdateFrame) {
				return;
			}
			float distance = cv::norm(newPosition - path.back());
			float time = framePosition - previousUpdateFrame;

			totalDistance += distance;
			speed.push_back(distance / time);
			path.push_back(newPosition);
		}
		
		previousUpdateFrame = framePosition;
	}
};

struct BallMovement {
	float distance;
	int* currentId;
	int* previousId;

	BallMovement(const float& distance, int* currentId, int* previousId) : distance(distance), currentId(currentId), previousId(previousId) {}
};

class Recognition {
	const int trueWidth = 3658; // mm
    const int width = 1024;
    const int height = 512;
    const int circleRadius = 9;
	const int frameRate = 30; // fps

	int iterator = 0;

	std::vector<Ball> previousBalls;
	std::vector<Ball> balls;
    std::vector<cv::Point> quad;

	std::map<BallIndex, BallData> ballData;
	
    std::vector<Template> templates;
    fdeep::model model = fdeep::load_model("./recognizer_cpp/nn_models/classifier_with_none.json");

	static std::vector<cv::Point> contourToQuad(const std::vector<cv::Point>& contour);

	bool getRedBall(const int& id, Ball*& redBall, std::vector<Ball>& balls);
	void findTable(const cv::Mat& image);
	bool cutAndWarp(const cv::Mat& image, cv::Mat& warpedImage);
	void findBalls(const cv::Mat& image);
	void setBallCuts(const cv::Mat& image);
	void labelBallsWithTM();
	void labelBallsWithNN();

public:
	int processedFramePosition = 0;

	cv::Mat debugFrameCanny;

	cv::Mat debugFrameMask;
	cv::Scalar lowerGreen = cv::Scalar(50, 50, 70);		// 40, 190, 50 -- 50, 0, 111
    cv::Scalar upperGreen = cv::Scalar(65, 255, 255);	// 65, 255, 255
	double kernelIterations = 1;
	double tableEpsilonRate = 5;

	cv::Mat debugFrameCircles;
	int minRadiusRate = 5;		// 5
	int maxRadiusRate = 12;		// 12
	int minDistanceRate = 12;	// 12
	int circlePerfectness = 10;	// 10
	int circleThreshold = 40;	// 40
	int maxBallJump = 250;		// 250

	cv::Mat processedFramePath;
	int matchLimitRate = 50;
	int noneLimitRate = 100;

    Recognition();

	void processFrameWithNN(const cv::Mat& videoFrame);
	BallData getBallData(const BallLabel& label, const int& id = 0) const;
};