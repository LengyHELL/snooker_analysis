#include <Ball.hpp>

std::string labelToString(const BallLabel& label) {
    switch (label) {
    case BallLabel::BLACK:
        return "black";
    case BallLabel::BLUE:
        return "blue";
    case BallLabel::BROWN:
        return "brown";
    case BallLabel::GREEN:
        return "green";
    case BallLabel::PINK:
        return "pink";
    case BallLabel::RED:
        return "red";
    case BallLabel::WHITE:
        return "white";
    case BallLabel::YELLOW:
        return "yellow";
    default:
        return "";
    }
}

Ball::Ball(
    const float& x,
    const float& y,
    const float& radius,
    const BallLabel& label,
    const Cut& cut
): x(x), y(y), radius(radius), label(label), cut(cut) {}

Ball::Ball(
    const cv::Point& center,
    const float& radius,
    const BallLabel& label,
    const Cut& cut
): x(center.x), y(center.y), radius(radius), label(label), cut(cut) {}

Ball::Ball(
    const cv::Vec3f& vec,
    const BallLabel& label,
    const Cut& cut
): x(vec[0]), y(vec[1]), radius(vec[2]), label(label), cut(cut) {}

cv::Point Ball::getCenter() const {
    return cv::Point(x, y);
}

cv::Rect Ball::getRect() const {
    return cv::Rect(x - radius, y - radius, radius * 2, radius * 2);
}

cv::Point Ball::getTopLeft() const {
    return cv::Point(x - radius, y - radius);
}

std::string Ball::getLabelString() const {
    if (label == BallLabel::RED) {
        return labelToString(label) + '_' + std::to_string(id);
    }
    else {
        return labelToString(label);
    }
}