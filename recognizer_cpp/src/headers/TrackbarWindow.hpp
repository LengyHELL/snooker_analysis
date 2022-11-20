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

    bool findTrackbarByName(const std::string& name) const {
        for (const Trackbar<T>& trackbar: trackbars) {
            if (trackbar.name == name) {
                return true;
            }
        }

        return false;
    }

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
            int currentValue = cv::getTrackbarPos(trackbar.name, windowName);
            if (currentValue >= 0) {
                *trackbar.value = currentValue;
            }
        }
    }

    void saveTrackbars(const std::string& fileName, const bool& overwrite = false) {
        std::ofstream saveFile;

        if (overwrite) {
            saveFile.open(fileName);
        }
        else {
            saveFile.open(fileName, std::ios_base::app);
        }

        if (saveFile.is_open()) {
            for (const Trackbar<T>& trackbar : trackbars) {
                saveFile << trackbar.name << '=' << *trackbar.value << '\n';
            }

            saveFile.close();
        }
    }

    void loadTrackbars(const std::string& fileName) {
        std::ifstream loadFile(fileName);

        if (loadFile.is_open()) {
            std::string line = "";

            while(getline(loadFile, line)) {
                if (line.length() > 0) {
                    int separatorPos = line.find('=');
                    std::string key = line.substr(0, separatorPos);
                    std::string value = line.substr(separatorPos + 1);

                    if (findTrackbarByName(key)) {
                        cv::setTrackbarPos(key, windowName, std::stoi(value));
                        std::cerr << "setting trackbar '" << key << "' to " << std::stoi(value) << '\n';
                    }
                }
            }

            loadFile.close();
        }
    }
};