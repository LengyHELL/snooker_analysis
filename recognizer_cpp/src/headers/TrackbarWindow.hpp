#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

template <typename T>
struct Trackbar {
    std::string name;
    T* value;

    Trackbar(const std::string& name, T* value): name(name), value(value) {}
};

template <typename T>
class TrackbarWindow {
    std::string windowName;
    std::vector<Trackbar<T>> trackbars;

    const cv::Mat blank = cv::Mat::zeros(cv::Size(300, 1),CV_8UC1);

public:
    TrackbarWindow(const std::string& windowName): windowName(windowName) {
        cv::namedWindow(windowName);
        cv::imshow(windowName, blank);
    }

    void addTrackbar(const std::string& trackbarName, const int& minValue, const int& maxValue, T& value) {
        trackbars.push_back(Trackbar<T>(trackbarName, &value));

        cv::createTrackbar(trackbarName, windowName, NULL, maxValue);
        cv::setTrackbarMin(trackbarName, windowName, minValue);
        cv::setTrackbarPos(trackbarName, windowName, value);

        cv::imshow(windowName, blank);
    }

    void updateTrackbars() {
        for (auto& trackbar : trackbars) {
            *trackbar.value = cv::getTrackbarPos(trackbar.name, windowName);
        }
    }
};