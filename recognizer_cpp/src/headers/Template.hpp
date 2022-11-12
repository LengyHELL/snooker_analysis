#pragma once

#include <opencv2/opencv.hpp>

#include <BallLabel.hpp>

struct Template {
	cv::Mat image;
	BallLabel label;

	Template(): image(), label(BallLabel::NONE) {}
	Template(const cv::Mat& image, const BallLabel& label): image(image), label(label) {}
};