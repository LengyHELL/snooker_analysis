#include "headers/Recognition.hpp"
#include "headers/TrackbarWindow.hpp"

void drawFrameTime(cv::Mat image, const int& frameTime) {
	cv::putText(image, "Frametime:" + std::to_string(frameTime) + " ms", cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255));
}

void drawPath(cv::Mat& image, const std::vector<cv::Point>& path, const cv::Scalar& color = cv::Scalar(255, 0, 0)) {
	for (int i = 1; i < path.size(); i++) {
		cv::line(image, path[i - 1], path[i], color);
	}
}

TrackbarWindow<double> getHSVTrackbarWindow(Recognition& recognition) {
	TrackbarWindow<double> hsvTrackbars("hsv_trackbars");

	hsvTrackbars.addTrackbar("LG_H", 0, 255, recognition.lowerGreen[0]);
	hsvTrackbars.addTrackbar("LG_S", 0, 255, recognition.lowerGreen[1]);
	hsvTrackbars.addTrackbar("LG_V", 0, 255, recognition.lowerGreen[2]);

	hsvTrackbars.addTrackbar("UG_H", 0, 255, recognition.upperGreen[0]);
	hsvTrackbars.addTrackbar("UG_S", 0, 255, recognition.upperGreen[1]);
	hsvTrackbars.addTrackbar("UG_V", 0, 255, recognition.upperGreen[2]);

	hsvTrackbars.addTrackbar("KERNEL_ITER", 1, 10, recognition.kernelIterations);
	hsvTrackbars.addTrackbar("EPSILON_RATE", 0, 100, recognition.tableEpsilonRate);

	return hsvTrackbars;
}

TrackbarWindow<int> getCircleTrackbarWindow(Recognition& recognition) {
	TrackbarWindow<int> circleTrackbars("circle_trackbars");

	circleTrackbars.addTrackbar("MIN_RAD_RATE", 0, 20, recognition.minRadiusRate);
	circleTrackbars.addTrackbar("MAX_RAD_RATE", 2, 20, recognition.maxRadiusRate);
	circleTrackbars.addTrackbar("MIN_DIST_RATE", 2, 20, recognition.minDistanceRate);
	circleTrackbars.addTrackbar("PERFECTNESS", 1, 100, recognition.circlePerfectness);
	circleTrackbars.addTrackbar("THRESHOLD_RATE", 1, 100, recognition.thresholdRate);

	return circleTrackbars;
}

int main(int argc, char** argv) {
	if (argc != 2) {
		printf("usage: DisplayImage.out <Image_Path>\n");
		return -1;
	}
	
	cv::VideoCapture videoCapture(argv[1]);
	cv::Mat videoFrame;

	std::chrono::milliseconds duration;

	Recognition recognition;
	TrackbarWindow<double> hsvTrackbars = getHSVTrackbarWindow(recognition);
	TrackbarWindow<int> circleTrackbars = getCircleTrackbarWindow(recognition);

	int currentFrame = 0;
	const int maxFrames = 3;
	bool keyLock = false;

	int frameCounter = -1;

	while(videoCapture.read(videoFrame)) {
		/* frameCounter++;
		if ((frameCounter % 20) != 0) {
			continue;
		} */

		auto timerStart = std::chrono::high_resolution_clock::now();

		hsvTrackbars.updateTrackbars();
		circleTrackbars.updateTrackbars();

		recognition.processFrameWithNN(videoFrame);
		cv::Mat processedImage;

		switch(currentFrame) {
			case 0: default:
				processedImage = recognition.processedFramePath;
				drawPath(processedImage, recognition.getBallPath(BallLabel::RED, 10));
				break;
			case 1: processedImage = recognition.debugFrameCanny; break;
			case 2: processedImage = recognition.debugFrameMask; break;
			case 3: processedImage = recognition.debugFrameCircles; break;
		}

		drawFrameTime(processedImage, duration.count());

		cv::Mat resizedImage;
		cv::resize(processedImage, resizedImage, cv::Size(1400, 700));
		cv::imshow("snooker recognition", resizedImage);

		char key = cv::waitKey(1);
		if (key == 'q') {
			cv::destroyAllWindows();
			break;
		}
		else if (key == 'a') {
			currentFrame--;
			if (currentFrame < 0) {
				currentFrame = 0;
			}
		}
		else if (key == 'd') {
			currentFrame++;
			if (currentFrame > maxFrames) {
				currentFrame = maxFrames;
			}
		}

		auto timerStop = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(timerStop - timerStart);
	}

	return 0;
}
