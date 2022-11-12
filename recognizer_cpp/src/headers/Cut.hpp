#pragma once

#include <opencv2/opencv.hpp>

struct Cut {
	cv::Mat BGR;
	cv::Mat HSV;

	Cut(): BGR(), HSV() {}
	Cut(const cv::Mat& BGR, const cv::Mat& HSV): BGR(BGR), HSV(HSV) {}
};