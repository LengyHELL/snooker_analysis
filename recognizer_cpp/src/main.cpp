#include "headers/Recognition.hpp"
#include "headers/TrackbarWindow.hpp"

void drawFrameTime(cv::Mat image, const int& frameTime) {
	cv::putText(image, "Frametime:" + std::to_string(frameTime) + " ms", cv::Point(0, 15), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255));
}

void drawCurrentFrame(cv::Mat image, const int& currentFrame) {
	cv::putText(image, "Current frame:" + std::to_string(currentFrame), cv::Point(190, 15), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255));
}

void drawBallData(cv::Mat& image, const BallData& ballData, const cv::Scalar& pathColor = cv::Scalar(255, 0, 0)) {
	for (int i = 1; i < ballData.path.size(); i++) {
		cv::line(image, ballData.path[i - 1], ballData.path[i], pathColor, 2);
	}
	/*
	true: 3658 mm
    virtual: 1024 px;
	*/
	float distanceConverter = (3.658 / 1024);
	float trueDistance = ballData.totalDistance * distanceConverter; //m

	cv::putText(image, "Distance:" + std::to_string(trueDistance) + " m", cv::Point(0, 35), cv::FONT_HERSHEY_SIMPLEX, 0.6, pathColor);
	if (!ballData.speed.empty()) {
		float trueAverageSpeed = ballData.averageSpeed * (0.03 / distanceConverter);

		cv::putText(image, "Average speed:" + std::to_string(ballData.averageSpeed) + " m/s", cv::Point(0, 55), cv::FONT_HERSHEY_SIMPLEX, 0.6, pathColor);
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
	circleTrackbars.addTrackbar("THRESHOLD_RATE", 2, 255, recognition.circleThreshold);
	circleTrackbars.addTrackbar("MAX_BALL_JUMP", 20, 1000, recognition.maxBallJump);

	return circleTrackbars;
}

TrackbarWindow<int> getNNTrackbarWindow(Recognition& recognition) {
	TrackbarWindow<int> nnTrackbars("nn_trackbars");

	nnTrackbars.addTrackbar("MATCH_LIMIT_RATE", 20, 100, recognition.matchLimitRate);
	nnTrackbars.addTrackbar("NONE_LIMIT_RATE", 5, 100, recognition.noneLimitRate);

	return nnTrackbars;
}

void createLogFile(const std::string& fileName) {
	std::ofstream logFile(fileName);
	
	if (logFile.is_open()) {
		logFile << "FRAME,";

		for (int i = 0; i < 8; i++) {
			BallLabel label = static_cast<BallLabel>(i);

			std::string labelString = labelToString(label);
			for (char& c : labelString) {
				c = toupper(c);
			}

			if (label == BallLabel::RED) {
				for (int j = 0; j < 15; j++) {
					logFile << labelString << '_' << j << "_POSITION_X,";
					logFile << labelString << '_' << j << "_POSITION_Y,";
					logFile << labelString << '_' << j << "_DISTANCE,";
					logFile << labelString << '_' << j << "_TOTAL_DISTANCE,";
					logFile << labelString << '_' << j << "_SPEED,";
					logFile << labelString << '_' << j << "_AVERAGE_SPEED";

					if (i < 7 || j < 14) {
						logFile << ',';
					}
				}
			}
			else {
				logFile << labelString << "_POSITION_X,";
				logFile << labelString << "_POSITION_Y,";
				logFile << labelString << "_DISTANCE,";
				logFile << labelString << "_TOTAL_DISTANCE,";
				logFile << labelString << "_SPEED,";
				logFile << labelString << "_AVERAGE_SPEED";

				if (i < 7) {
					logFile << ',';
				}
			}
		}
		logFile << '\n';

		logFile.close();
	}
}

void appendLogFile(const std::string& fileName, const Recognition& recognition) {
	std::ofstream logFile(fileName, std::ios_base::app);

	if (logFile.is_open()) {
		logFile << recognition.processedFramePosition << ',';

		for (int i = 0; i < 8; i++) {
			BallLabel label = static_cast<BallLabel>(i);

			if (label == BallLabel::RED) {
				for (int j = 0; j < 15; j++) {
					const BallData ballData = recognition.getBallData(label, j);
					if (ballData.path.empty()) {
						logFile << ",,,,,";
					}
					else {
						logFile << ballData.path.back().x << ',';
						logFile << ballData.path.back().y << ',';
						logFile << ballData.distance << ',';
						logFile << ballData.totalDistance << ',';
						logFile << ballData.speed.back() << ',';
						logFile << ballData.averageSpeed;
					}

					if (i < 7 || j < 14) {
						logFile << ',';
					}
				}
			}
			else {
				const BallData ballData = recognition.getBallData(label, 0);
				if (ballData.path.empty()) {
					logFile << ",,,,,";
				}
				else {
					logFile << ballData.path.back().x << ',';
					logFile << ballData.path.back().y << ',';
					logFile << ballData.distance << ',';
					logFile << ballData.totalDistance << ',';
					logFile << ballData.speed.back() << ',';
					logFile << ballData.averageSpeed;
				}

				if (i < 7) {
					logFile << ',';
				}
			}
		}

		logFile << "\n";

		logFile.close();
	}
}

int main(int argc, char** argv) {
	const int fixedArgs = 2;

	if (argc < fixedArgs) {
		printf("usage: snooker_analysis <video_path> | optional: -start_frame=<frame> -end_frame=<frame> -enable_logging\n");
		return -1;
	}
	
	int startFrame = 0;
	int endFrame = -1;
	bool enableLogging = false;

	for (int i = fixedArgs; i < argc; i++) {
		if (argv[i][0] == '-') {
			std::string arg = argv[i];
			int sep = arg.find('=');
			std::string tag = arg.substr(1, sep - 1);
			std::string value = arg.substr(sep + 1);

			if (tag == "start_frame") {
				startFrame = std::stoi(value);
			}
			else if (tag == "end_frame") {
				endFrame = std::stoi(value);
			}
			else if (tag == "enable_logging") {
				enableLogging = true;
			}
		}
		else {
			std::cerr << "Unable to process argument '" << argv[i] << "'";
		}
	}

	cv::VideoCapture videoCapture(argv[1]);
	videoCapture.set(cv::CAP_PROP_POS_FRAMES, startFrame);
	cv::Mat videoFrame;

	std::chrono::milliseconds duration;

	Recognition recognition;
	TrackbarWindow<double> hsvTrackbars = getHSVTrackbarWindow(recognition);
	TrackbarWindow<int> circleTrackbars = getCircleTrackbarWindow(recognition);
	TrackbarWindow<int> mainTrackbars("main_trackbars");
	TrackbarWindow<int> nnTrackbars = getNNTrackbarWindow(recognition);

	int shownBallColor = 5;
	int shownBallId = 0;
	mainTrackbars.addTrackbar("SHOWN_BALL_COLOR", 0, 7, shownBallColor);
	mainTrackbars.addTrackbar("SHOWN_BALL_ID", 0, 14, shownBallId);

	hsvTrackbars.loadTrackbars("recognizer_cpp/config.txt");
	circleTrackbars.loadTrackbars("recognizer_cpp/config.txt");
	mainTrackbars.loadTrackbars("recognizer_cpp/config.txt");
	nnTrackbars.loadTrackbars("recognizer_cpp/config.txt");

	int currentFrame = 0;
	const int maxFrames = 3;
	bool keyLock = false;

	bool stop = false;
	bool pause = false;
	bool nextFrame = false;

	if (enableLogging) {
		createLogFile("log.csv");
	}

	while(!stop) {
		if (!pause || nextFrame) {
			if (!videoCapture.read(videoFrame)) {
				stop = true;
				continue;
			}
		}
		auto timerStart = std::chrono::high_resolution_clock::now();

		hsvTrackbars.updateTrackbars();
		circleTrackbars.updateTrackbars();
		mainTrackbars.updateTrackbars();
		nnTrackbars.updateTrackbars();

		if (!pause || nextFrame) {
			recognition.processFrameWithNN(videoFrame);

			if (enableLogging) {
				appendLogFile("log.csv", recognition);
			}
		}
		cv::Mat processedImage;

		switch(currentFrame) {
			case 0: default:
				processedImage = recognition.processedFramePath.clone();
				drawBallData(processedImage, recognition.getBallData(static_cast<BallLabel>(shownBallColor), shownBallId));
				break;
			case 1: processedImage = recognition.debugFrameCanny; break;
			case 2: processedImage = recognition.debugFrameMask; break;
			case 3: processedImage = recognition.debugFrameCircles; break;
		}

		drawFrameTime(processedImage, duration.count());
		drawCurrentFrame(processedImage, recognition.processedFramePosition);

		cv::Mat resizedImage;
		cv::resize(processedImage, resizedImage, cv::Size(1400, 700));
		cv::imshow("snooker recognition", resizedImage);

		if (nextFrame) {
			nextFrame = false;
		}

		char key = cv::waitKey(1);
		if (key == 'q' || key == 's' || ((recognition.processedFramePosition > endFrame) && (endFrame > 0))) {
			cv::destroyAllWindows();

			if (key == 's') {
				//first save must be overwrite
				hsvTrackbars.saveTrackbars("recognizer_cpp/config.txt", true);
				circleTrackbars.saveTrackbars("recognizer_cpp/config.txt");
				mainTrackbars.saveTrackbars("recognizer_cpp/config.txt");
				nnTrackbars.saveTrackbars("recognizer_cpp/config.txt");
			}
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
		else if (int(key) == 32) {
			pause = !pause;
		}
		else if ((key == 'w') && pause) {
			nextFrame = true;
		}
		else if (key == 'e') {
			cv::imwrite("screenshot.png", resizedImage);
		}

		recognition.processedFramePosition = videoCapture.get(cv::CAP_PROP_POS_FRAMES);
		auto timerStop = std::chrono::high_resolution_clock::now();
		duration = std::chrono::duration_cast<std::chrono::milliseconds>(timerStop - timerStart);
	}

	return 0;
}
