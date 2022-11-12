#pragma once

#include <opencv2/opencv.hpp>
#include <string>

#include <BallLabel.hpp>
#include <Cut.hpp>

struct Ball {
	int id = 0;

	float x;
	float y;
	float radius;
	BallLabel label;
	Cut cut;

	Ball() : x(0), y(0), radius(0), label(BallLabel::NONE), cut(Cut()) {}
	Ball(
		const float& x,
		const float& y,
		const float& radius,
		const BallLabel& label = BallLabel::NONE,
		const Cut& cut = Cut()
	);
	Ball(
		const cv::Point& center,
		const float& radius,
		const BallLabel& label = BallLabel::NONE,
		const Cut& cut = Cut()
	);
	Ball(
		const cv::Vec3f& vec,
		const BallLabel& label = BallLabel::NONE,
		const Cut& cut = Cut()
	);

	cv::Point getCenter() const;
	cv::Rect getRect() const;
	cv::Point getTopLeft() const;
	std::string getLabelString() const;
};