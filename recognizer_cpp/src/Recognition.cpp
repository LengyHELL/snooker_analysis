#include <Recognition.hpp>
#include <filesystem>

void showImageWithContours(cv::Mat image, const std::vector<std::vector<cv::Point>>& contours, int index = -1) {
	cv::drawContours(image, contours, index, cv::Scalar(255, 0, 0));
	cv::imshow("test", image);
}

cv::Mat loadImage(std::string location) {
    return cv::imread(location, cv::IMREAD_COLOR);
}

bool areaComparator(const std::vector<cv::Point>& lhs, const std::vector<cv::Point>& rhs) {
    return cv::contourArea(lhs) > cv::contourArea(rhs);
}

bool sectionComparator(const Section& lhs, const Section& rhs) {
    return lhs.length > rhs.length;
}

bool distanceComparator(const BallMovement& lhs, const BallMovement& rhs) {
    return lhs.distance < rhs.distance;
}

Recognition::Recognition(): width(1024), height(512), circleRadius(9), model(fdeep::load_model("./recognizer_cpp/nn_models/classifier_with_none.json")) {
    templates.push_back(Template(loadImage("./recognizer_cpp/templates/black_ball_hd.png"), BallLabel::BLACK));
    templates.push_back(Template(loadImage("./recognizer_cpp/templates/blue_ball_hd.png"), BallLabel::BLUE));
    templates.push_back(Template(loadImage("./recognizer_cpp/templates/brown_ball_hd.png"), BallLabel::BROWN));
    templates.push_back(Template(loadImage("./recognizer_cpp/templates/green_ball_hd.png"), BallLabel::GREEN));
    templates.push_back(Template(loadImage("./recognizer_cpp/templates/pink_ball_hd.png"), BallLabel::PINK));
    templates.push_back(Template(loadImage("./recognizer_cpp/templates/red_ball_hd.png"), BallLabel::RED));
    templates.push_back(Template(loadImage("./recognizer_cpp/templates/white_ball_hd.png"), BallLabel::WHITE));
    templates.push_back(Template(loadImage("./recognizer_cpp/templates/yellow_ball_hd.png"), BallLabel::YELLOW));

    for (int i = 0; i < 8; i++) {
        BallLabel label = static_cast<BallLabel>(i);
        if (label == BallLabel::RED) {
            for (int j = 0; j < 15; j++) {
                ballData.insert(std::pair<BallIndex, BallData>(BallIndex(j, label), BallData()));
            }
        }
        else {
            ballData.insert(std::pair<BallIndex, BallData>(BallIndex(0, label), BallData()));
        }
    }
}

std::vector<cv::Point> Recognition::contourToQuad(const std::vector<cv::Point>& contour) {
    std::vector<Section> lengths;

    for (int i = 0; i < contour.size(); i++) {
        cv::Point start, end;

        if (i - 1 < 0) { start = contour[contour.size() - 1]; }
        else { start = contour[i - 1]; }
        end = contour[i];

        lengths.push_back(Section(cv::norm(start - end), start, end));
    }

    std::sort(lengths.begin(), lengths.end(), sectionComparator);
    lengths = std::vector<Section>(lengths.begin(), lengths.begin() + 4);

    std::vector<cv::Point> lengthPoints;
    for (const auto& length : lengths) {
        lengthPoints.push_back(length.start);
        lengthPoints.push_back(length.end);
    }
    cv::RotatedRect rect = cv::minAreaRect(lengthPoints);
    
    cv::Point2f points[4];
    rect.points(points);

    std::vector<cv::Point> pointsVec;
    for (int i = 0; i < 4; i++) {
        pointsVec.push_back(points[i]);
    }

    return pointsVec;

    /* Section temp = lengths[1];
    lengths[1] = lengths[2];
    lengths[2] = temp;

    std::vector<cv::Point> quadPoints;

    for (int i = 0; i < lengths.size(); i++) {
        cv::Point intersectionPoint;
        Section first, second;

        if (i - 1 < 0) { first = lengths[lengths.size() - 1]; }
        else { first = lengths[i - 1]; }
        second = lengths[i];

        if (!first.intersection(second, intersectionPoint)) {
            return std::vector<cv::Point>();
        }

        if (intersectionPoint.x < 0 || intersectionPoint.y < 0) {
            return std::vector<cv::Point>();
        }

        quadPoints.push_back(intersectionPoint);
    }

    return quadPoints; */
}

bool Recognition::getRedBall(const int& id, Ball*& redBall, std::vector<Ball>& balls) {
    for (auto& ball : balls) {
        if (ball.label == BallLabel::RED && ball.id == id) {
            redBall = &ball;
            return true;
        }
    }

    return false;
}

void Recognition::findTable(const cv::Mat& image) {
    cv::Mat hsv, mask, result, imageGray, imageThreshold, imageCanny;
    std::vector<std::vector<cv::Point>> contours;

    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    cv::inRange(hsv, lowerGreen, upperGreen, mask);
    cv::bitwise_and(image, image, result, mask);

    debugFrameMask = result;

    cv::cvtColor(result, imageGray, cv::COLOR_BGR2GRAY);
    float threshold = cv::threshold(imageGray, imageThreshold, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    cv::Canny(imageGray, imageCanny, 1.0 * threshold, 0.5 * threshold);

    float firstKernelData[] = {0, 0, 1, 0, 0,
                               0, 0, 1, 0, 0,
                               1, 1, 1, 1, 1,
                               0, 0, 1, 0, 0,
                               0, 0, 1, 0, 0};
    
    float secondKernelData[] = {1, 0, 0, 0, 1,
                                0, 1, 0, 1, 0,
                                0, 0, 1, 0, 0,
                                0, 1, 0, 1, 0,
                                1, 0, 0, 0, 1};
    
    cv::Mat firstKernel(5, 5, CV_8U, firstKernelData);
    cv::Mat secondKernel(5, 5, CV_8U, secondKernelData);

    cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size( 3, 3 ));
    cv::dilate(imageCanny, imageCanny, element, cv::Point(-1, -1), kernelIterations);
    
    cv::morphologyEx(imageCanny, imageCanny, cv::MORPH_CLOSE, firstKernel, cv::Point(-1, -1), kernelIterations, cv::BORDER_REPLICATE);
    cv::morphologyEx(imageCanny, imageCanny, cv::MORPH_CLOSE, secondKernel, cv::Point(-1, -1), kernelIterations, cv::BORDER_REPLICATE);

    debugFrameCanny = imageCanny;

    cv::findContours(imageCanny, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    for(auto& c : contours) {
        std::vector<cv::Point> hull;
        cv::convexHull(c, hull);
        c = hull;
    }

    std::sort(contours.begin(), contours.end(), areaComparator);

    if (contours.size() <= 0) {
        quad.clear();
    }
    else {
        double epsilon = cv::arcLength(contours[0], true) * (tableEpsilonRate / 1000);
        cv::approxPolyDP(contours[0], contours[0], epsilon, true);
        cv::drawContours(debugFrameMask, contours, 0, cv::Scalar(0, 128, 255), 2);

        quad = contourToQuad(contours[0]);
    }
}

bool Recognition::cutAndWarp(const cv::Mat& image, cv::Mat& warpedImage) {
    if (quad.size() != 4) {
        return false;
    }

    std::vector<cv::Point> arrangedPoints;
    std::vector<int> sums, diffs;

    for (auto& point : quad) {
        sums.push_back(point.x + point.y);
        diffs.push_back(point.y - point.x);
    }

    std::vector<cv::Point2f> sourceArray, destinationArray;
    sourceArray.push_back(quad[std::min_element(sums.begin(), sums.end()) - sums.begin()]);
    sourceArray.push_back(quad[std::min_element(diffs.begin(), diffs.end()) - diffs.begin()]);
    sourceArray.push_back(quad[std::max_element(sums.begin(), sums.end()) - sums.begin()]);
    sourceArray.push_back(quad[std::max_element(diffs.begin(), diffs.end()) - diffs.begin()]);

    destinationArray.push_back(cv::Point(0, 0));
    destinationArray.push_back(cv::Point(width, 0));
    destinationArray.push_back(cv::Point(width, height));
    destinationArray.push_back(cv::Point(0, height));

    cv::Mat M = cv::getPerspectiveTransform(sourceArray, destinationArray);
    cv::warpPerspective(image, warpedImage, M, cv::Size(width, height));

    return true;
}

void Recognition::findBalls(const cv::Mat& image) {
    cv::Mat result, imageGray, imageHough;
    result = image.clone();

    cv::cvtColor(image, imageGray, cv::COLOR_BGR2GRAY);

    float minRadius = circleRadius * (float(minRadiusRate) / 10);
    float maxRadius = circleRadius * (float(maxRadiusRate) / 10);
    float minDistance = circleRadius * (float(minDistanceRate) / 10);

    std::vector<cv::Vec3f> vecs;
    cv::HoughCircles(imageGray, vecs, cv::HOUGH_GRADIENT, 1, minDistance, circleThreshold, circlePerfectness, minRadius, maxRadius);

    balls.clear();
    for (auto& vec : vecs) {

        int borderSize = 1 * circleRadius;
        if (
            (vec[0] > borderSize) &&
            (vec[0] < image.cols - borderSize) &&
            (vec[1] > borderSize) &&
            (vec[1] < image.rows - borderSize)
        ) {
            vec[2] = circleRadius;
            balls.push_back(vec);
            cv::circle(result, cv::Point(vec[0], vec[1]), vec[2], cv::Scalar(255, 0, 255));
        }
    }

    debugFrameCircles = result;
}

void Recognition::setBallCuts(const cv::Mat& image) {
    cv::Mat imageHSV;

    cv::cvtColor(image, imageHSV, cv::COLOR_BGR2HSV);

    for (auto& ball : balls) {
        cv::Rect cutRect = ball.getRect();
        if (cutRect.x < 0) {
            cutRect.x = 0;
        }
        if (cutRect.x + cutRect.width >= image.cols) {
            cutRect.x = (image.cols - cutRect.width) - 1;
        }
        if (cutRect.y < 0) {
            cutRect.y = 0;
        }
        if (cutRect.y + cutRect.height >= image.rows) {
            cutRect.y = (image.rows - cutRect.height) - 1;
        }
        ball.cut = Cut(image(cutRect), imageHSV(cutRect));
    }
}

void Recognition::labelBallsWithTM() {
    for (auto& ball : balls) {
        std::vector<float> results;

        for (const auto& temp : templates) {
            if (ball.cut.BGR.size != temp.image.size) {
                return;
            }

            cv::Mat result;
            float resultRows = ball.cut.BGR.rows - temp.image.rows + 1;
            float resultCols = ball.cut.BGR.cols - temp.image.cols + 1;
            result.create(resultRows, resultCols, CV_32FC1);

            cv::matchTemplate(ball.cut.BGR, temp.image, result, cv::TM_CCORR_NORMED);

            double maxValue;
            cv::minMaxLoc(result, NULL, &maxValue);
            results.push_back(maxValue);
        }

        ball.label = templates[std::max_element(results.begin(), results.end()) - results.begin()].label;
    }
}

void Recognition::labelBallsWithNN() {
    fdeep::internal::tensors_vec inputs;
    float matchLimit = float(matchLimitRate) / 100;
    float noneLimit = float(noneLimitRate) / 100;

    if (balls.empty()) {
        return;
    }

    for (auto& ball : balls) {
        std::vector<cv::Mat> channels = std::vector<cv::Mat>{ball.cut.BGR, ball.cut.HSV};
        cv::Mat cutArray;
        cv::merge(channels, cutArray);

        inputs.push_back({fdeep::tensor_from_bytes(
            cutArray.ptr(),
            static_cast<std::size_t>(cutArray.rows),
            static_cast<std::size_t>(cutArray.cols),
            static_cast<std::size_t>(cutArray.channels()),
            0.0f,
            1.0f
        )});
    }

    std::vector<std::vector<float>> results;
    for (const auto& result : model.predict_multi(inputs, true)) {
        results.push_back(result[0].to_vector());
    }

    for (int type = 0; type < 8; type++) {
        BallLabel ballLabel = static_cast<BallLabel>(type);

        if (ballLabel != BallLabel::RED) {
            std::function<bool(const std::vector<float>&, const std::vector<float>&)> comparator;
            switch(ballLabel) {
                case BLACK:
                    comparator = [](const auto& lhs, const auto& rhs) { return lhs[0] < rhs[0]; };
                    break;
                case BLUE:
                    comparator = [](const auto& lhs, const auto& rhs) { return lhs[1] < rhs[1]; };
                    break;
                case BROWN:
                    comparator = [](const auto& lhs, const auto& rhs) { return lhs[2] < rhs[2]; };
                    break;
                case GREEN:
                    comparator = [](const auto& lhs, const auto& rhs) { return lhs[3] < rhs[3]; };
                    break;
                case PINK:
                    comparator = [](const auto& lhs, const auto& rhs) { return lhs[4] < rhs[4]; };
                    break;
                case WHITE:
                    comparator = [](const auto& lhs, const auto& rhs) { return lhs[6] < rhs[6]; };
                    break;
                case YELLOW:
                    comparator = [](const auto& lhs, const auto& rhs) { return lhs[7] < rhs[7]; };
                    break;
            }
            
            int maxIndex = std::max_element(results.begin(), results.end(), comparator) - results.begin();
            if ((results[maxIndex][type] > matchLimit) && (results[maxIndex][BallLabel::NONE] < noneLimit)) {
                balls[maxIndex].label = static_cast<BallLabel>(type);
            }
        }
    }
    
    int currentId = 0;
    for (int i = 0; i < results.size(); i++) {
        if ((results[i][BallLabel::RED] > matchLimit) && (balls[i].label == BallLabel::NONE)) {
            balls[i].label = BallLabel::RED;
            balls[i].id = currentId;
            currentId++;
        }
    }
}

void Recognition::processFrameWithNN(const cv::Mat& videoFrame) {
    findTable(videoFrame);

    cv::Mat imageWarped;

    if (!cutAndWarp(videoFrame, imageWarped)) {
        processedFramePath = videoFrame;
        return;
    }

    findBalls(imageWarped);
    setBallCuts(imageWarped);
    /* labelBallsWithTM();
    for (const auto& ball : balls) {
        switch (ball.label) {
            case BallLabel::BLACK:
                cv::imwrite("misc/dataset5/black/" + std::to_string(iterator) + ".png", ball.cut.BGR);
                break;
            case BallLabel::BLUE:
                cv::imwrite("misc/dataset5/blue/" + std::to_string(iterator) + ".png", ball.cut.BGR);
                break;
            case BallLabel::BROWN:
                cv::imwrite("misc/dataset5/brown/" + std::to_string(iterator) + ".png", ball.cut.BGR);
                break;
            case BallLabel::GREEN:
                cv::imwrite("misc/dataset5/green/" + std::to_string(iterator) + ".png", ball.cut.BGR);
                break;
            case BallLabel::PINK:
                cv::imwrite("misc/dataset5/pink/" + std::to_string(iterator) + ".png", ball.cut.BGR);
                break;
            case BallLabel::RED:
                cv::imwrite("misc/dataset5/red/" + std::to_string(iterator) + ".png", ball.cut.BGR);
                break;
            case BallLabel::WHITE:
                cv::imwrite("misc/dataset5/white/" + std::to_string(iterator) + ".png", ball.cut.BGR);
                break;
            case BallLabel::YELLOW:
                cv::imwrite("misc/dataset5/yellow/" + std::to_string(iterator) + ".png", ball.cut.BGR);
                break;
            default:
                break;
        }
        iterator++;
    } */
    labelBallsWithNN();

    // max cue ball speed 15 mm/ms -- 500mm at 30fps, 250mm at 60fps
    std::vector<BallMovement> movements;

    for (auto& ball : balls) {
        for (auto& previousBall : previousBalls) {
            if (ball.label == BallLabel::RED && previousBall.label == BallLabel::RED) {
                float d = cv::norm(ball.getCenter() - previousBall.getCenter());
                movements.push_back(BallMovement(d, &ball.id, &previousBall.id));
            }
            else if (ball.label == previousBall.label) {
                // checking for jump spikes in non red balls
                if (cv::norm(previousBall.getCenter() - ball.getCenter()) > maxBallJump) {
                    ball.x = previousBall.x;
                    ball.y = previousBall.y;
                }
            }
        }
    }

    std::sort(movements.begin(), movements.end(), distanceComparator);

    std::vector<int*> setPrevious;
    std::vector<int*> setCurrent;
    for (const auto& movement : movements) {
        if (std::count(setPrevious.begin(), setPrevious.end(), movement.previousId) <= 0 &&
            std::count(setCurrent.begin(), setCurrent.end(), movement.currentId) <= 0) {

            Ball* withCurrentId;
            Ball* withPreviousId;

            // switching ids in balls vector
            if (getRedBall(*movement.currentId, withCurrentId, balls)) {
                if (getRedBall(*movement.previousId, withPreviousId, balls)) {
                    withPreviousId->id = *movement.currentId;
                }

                withCurrentId->id = *movement.previousId;

                // checking for jump spikes in red balls
                Ball* previousBall;

                if (((movement.distance) > maxBallJump) &&
                    getRedBall(*movement.currentId, previousBall, previousBalls)) {

                    withCurrentId->x = previousBall->x;
                    withCurrentId->y = previousBall->y;
                }
            }


            // switching ids in movements vector
            for (int i = 0; i < movements.size(); i++) {
                if (*movements[i].currentId == *movement.previousId) {
                    *movements[i].currentId = *movement.currentId;
                }
            }

            *movement.currentId = *movement.previousId;

            setPrevious.push_back(movement.previousId);
            setCurrent.push_back(movement.currentId);
        }
    }

    previousBalls = balls;

    for (const auto& ball : balls) {
        if (ballData[{ball.id, ball.label}].path.empty()) {
            ballData[{ball.id, ball.label}].path.push_back(ball.getCenter());
        }
        else if (*ballData[{ball.id, ball.label}].path.end() != ball.getCenter()) {
            ballData[{ball.id, ball.label}].path.push_back(ball.getCenter());
        }
    }

    for (const auto& ball : balls) {
        if (ball.label == BallLabel::NONE) {
            continue;
        }

        cv::rectangle(imageWarped, ball.getRect(), cv::Scalar(0, 0, 255));
        cv::putText(imageWarped, ball.getLabelString(), ball.getTopLeft(), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255));
    }

    processedFramePath = imageWarped;
}

std::vector<cv::Point> Recognition::getBallPath(const BallLabel& label, const int& id) const {
    if (ballData.count({id, label})) {
        return ballData.at({id, label}).path;
    }
    else {
        return std::vector<cv::Point>();
    }
}