#pragma once

#include <opencv2/opencv.hpp>

struct Section {
	float length;
	cv::Point start;
	cv::Point end;
 
	Section(): length(0), start(cv::Point()), end(cv::Point()) {}
	Section(const float& length, const cv::Point& start, const cv::Point& end): length(length), start(start), end(end) {}

    bool intersection(const Section& other, cv::Point& intersection) {
        float d = (start.x - end.x) * (other.start.y - other.end.y) - (start.y - end.y) * (other.start.x - other.end.x);
		if (std::abs(d) < 1e-8) {
			return false;
		}
		float t1 =  (start.x * end.y - start.y * end.x) * (other.start.x - other.end.x) -
                    (other.start.x * other.end.y - other.start.y * other.end.x) * (start.x - end.x);
		float t2 =  (start.x * end.y - start.y * end.x) * (other.start.y - other.end.y) -
                    (other.start.x * other.end.y - other.start.y * other.end.x) * (start.y - end.y);

		intersection = cv::Point(t1 / d, t2 / d);
		return true;
    }
};