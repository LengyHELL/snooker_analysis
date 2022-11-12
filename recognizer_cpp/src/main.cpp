#include "headers/Recognition.hpp"

void drawFrameTime(cv::Mat image, const int& frameTime) {
	cv::putText(image, "Frametime:" + std::to_string(frameTime) + " ms", cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255));
}

void drawPath(cv::Mat& image, const std::vector<cv::Point>& path, const cv::Scalar& color = cv::Scalar(255, 0, 0)) {
	for (int i = 1; i < path.size(); i++) {
		cv::line(image, path[i - 1], path[i], color);
	}
}

void createTrackbars(cv::Scalar lower, cv::Scalar upper, int kernelIterations) {
	cv::createTrackbar("LG_H", "trackbars", NULL, 255);
	cv::createTrackbar("LG_S", "trackbars", NULL, 255);
	cv::createTrackbar("LG_V", "trackbars", NULL, 255);

	cv::createTrackbar("UG_H", "trackbars", NULL, 255);
	cv::createTrackbar("UG_S", "trackbars", NULL, 255);
	cv::createTrackbar("UG_V", "trackbars", NULL, 255);


	cv::setTrackbarPos("LG_H", "trackbars", lower[0]);
	cv::setTrackbarPos("LG_S", "trackbars", lower[1]);
	cv::setTrackbarPos("LG_V", "trackbars", lower[2]);

	cv::setTrackbarPos("UG_H", "trackbars", upper[0]);
	cv::setTrackbarPos("UG_S", "trackbars", upper[1]);
	cv::setTrackbarPos("UG_V", "trackbars", upper[2]);

	
	cv::createTrackbar("KERNEL_ITER", "trackbars", NULL, 10);
	cv::setTrackbarPos("KERNEL_ITER", "trackbars", kernelIterations);
	cv::setTrackbarMin("KERNEL_ITER", "trackbars", 1);
}

cv::Scalar getHSVTrackbar(bool lower) {
	if (lower) {
		return cv::Scalar(
			cv::getTrackbarPos("LG_H", "trackbars"),
			cv::getTrackbarPos("LG_S", "trackbars"),
			cv::getTrackbarPos("LG_V", "trackbars")
		);
	}
	else {
		return cv::Scalar(
			cv::getTrackbarPos("UG_H", "trackbars"),
			cv::getTrackbarPos("UG_S", "trackbars"),
			cv::getTrackbarPos("UG_V", "trackbars")
		);
	}
}

int getIteratorTrackbar() {
	return cv::getTrackbarPos("KERNEL_ITER", "trackbars");
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

	cv::namedWindow("trackbars");
	createTrackbars(recognition.lowerGreen, recognition.upperGreen, recognition.kernelIterations);

	cv::Mat blank = cv::Mat::zeros(cv::Size(300, 1),CV_8UC1);
	cv::imshow("trackbars", blank);

	while(videoCapture.read(videoFrame)) {
		auto timerStart = std::chrono::high_resolution_clock::now();

		recognition.lowerGreen = getHSVTrackbar(true);
		recognition.upperGreen = getHSVTrackbar(false);
		recognition.kernelIterations = getIteratorTrackbar();

		cv::Mat processedImage = recognition.processFrameWithNN(videoFrame);

		drawFrameTime(processedImage, duration.count());

		drawPath(processedImage, recognition.getBallPath(BallLabel::RED, 10));

		cv::Mat resizedImage;
		cv::resize(processedImage, resizedImage, cv::Size(1400, 700));
		cv::imshow("warped image", resizedImage);

		if ((cv::waitKey(1) & 0xFF) == 'q') {
			cv::destroyAllWindows();
			break;
		}

		auto timerStop = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(timerStop - timerStart);
	}

	return 0;
}
