#include "headers/Recognition.hpp"


void showImageWithContours(cv::Mat image, const std::vector<std::vector<cv::Point>>& contours, int index = -1) {
	cv::drawContours(image, contours, index, cv::Scalar(255, 0, 0));
	cv::imshow("test", image);
}

void drawFrameTime(cv::Mat image, const int& frameTime) {
	cv::putText(image, "Frametime:" + std::to_string(frameTime) + " ms", cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255));
}

void drawPath(cv::Mat& image, const std::vector<cv::Point>& path, const cv::Scalar& color = cv::Scalar(255, 0, 0)) {
	for (int i = 1; i < path.size(); i++) {
		cv::line(image, path[i - 1], path[i], color);
	}
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

	int lowerGreen[3] = {40, 190, 50};
	int upperGreen[3] = {65, 255, 255};

	cv::createTrackbar("LG_H", "trackbars", &lowerGreen[0], 255);
	cv::createTrackbar("LG_S", "trackbars", &lowerGreen[1], 255);
	cv::createTrackbar("LG_V", "trackbars", &lowerGreen[2], 255);

	cv::createTrackbar("UG_H", "trackbars", &upperGreen[0], 255);
	cv::createTrackbar("UG_S", "trackbars", &upperGreen[1], 255);
	cv::createTrackbar("UG_V", "trackbars", &upperGreen[2], 255);

	cv::Mat blank = cv::Mat::zeros(cv::Size(300, 1),CV_8UC1);
	cv::imshow("trackbars", blank);

	while(videoCapture.read(videoFrame)) {
		auto timerStart = std::chrono::high_resolution_clock::now();

		recognition.lowerGreen = cv::Scalar(lowerGreen[0], lowerGreen[1], lowerGreen[2]);
		recognition.upperGreen = cv::Scalar(upperGreen[0], upperGreen[1], upperGreen[2]);

		cv::Mat processedImage = recognition.processFrameWithNN(videoFrame);

		drawFrameTime(processedImage, duration.count());

		drawPath(processedImage, recognition.getBallPath(BallLabel::RED, 10));

		cv::imshow("warped image", processedImage);

		if ((cv::waitKey(1) & 0xFF) == 'q') {
			cv::destroyAllWindows();
			break;
		}

		auto timerStop = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(timerStop - timerStart);
	}

	return 0;
}
