#include <iostream>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

std::vector<std::vector<cv::Point>> simplifyContoursByArea(std::vector<std::vector<cv::Point>> inContours, int noTop);

int main(int argc, char** argv) {
    cv::Mat orgImg = cv::imread(argv[1], 1);
    if (orgImg.empty()) {
        printf("Empty image.\n");
        return 1;
    }

    cv::Mat grayImg;
    cv::cvtColor(orgImg, grayImg, CV_BGR2GRAY);

    cv::Mat dilateImg;
    cv::dilate(grayImg, dilateImg, cv::Mat(), cv::Point(-1, -1), 1, 1, 1);

    cv::Mat edgeDilateImg;
    cv::Canny(dilateImg, edgeDilateImg, 50, 100, 3);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edgeDilateImg, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    contours = simplifyContoursByArea(contours, 0);

    cv::Mat contoursImg = cv::Mat::zeros(grayImg.size(), CV_8UC1);
    for (size_t i = 0; i < contours.size(); i++) {
        cv::drawContours(contoursImg, contours, i, 255, CV_FILLED);
    }

    cv::Mat mainImg;
    cv::bitwise_and(contoursImg, grayImg, mainImg);

    cv::Mat edgeMainImg;
    cv::Canny(mainImg, edgeMainImg, 100, 200, 3);
    cv::dilate(edgeMainImg, edgeMainImg, cv::Mat(), cv::Point(-1, -1), 1, 1, 1);
    std::vector<std::vector<cv::Point>> contours2;
    std::vector<cv::Vec4i> hierarchy2;
    cv::findContours(edgeMainImg, contours2, hierarchy2, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
    std::cout << contours2.size();
    if (contours2.size() > 81) {
        contours2 = simplifyContoursByArea(contours2, 81);
    }
    contours2.erase(contours2.begin(), contours2.begin() + 1);

    std::vector<cv::Point> centerP;
    for (unsigned int i = 0; i < contours2.size(); i++) {
        cv::Point P_mean;
        for (unsigned int j = 0; j < contours2[i].size(); j++) {
            P_mean += contours2[i][j];
        }
        P_mean.x /= contours2[i].size();
        P_mean.y /= contours2[i].size();
        centerP.push_back(P_mean);
    }

    cv::Mat contoursImg2 = cv::Mat::zeros(grayImg.size(), CV_8UC1);

    cv::Mat outImg = orgImg.clone();
    for (unsigned int i = 0; i < contours2.size(); i++) {
        cv::drawContours(outImg, contours2, i, 255, 2);
        cv::circle(outImg, centerP[i], 1, cv::Scalar(0, 0, 255), 4, 8, 0);
        std::ostringstream temp;
        temp << i + 1;
        std::string str = temp.str();
    }

    cv::imshow("orgImg", orgImg);
    cv::imshow("outImg", outImg);
    cv::waitKey();

    return 0;
}

std::vector<std::vector<cv::Point>> simplifyContoursByArea(std::vector<std::vector<cv::Point>> inContours, int noTop) {
    if (inContours.size() == 0) {
        return inContours;
    } else {
        std::vector<double> areaInContours;
        for (unsigned int i = 0; i < inContours.size(); i++) {
            areaInContours.push_back(contourArea(inContours[i]));
        }

        std::sort(areaInContours.begin(), areaInContours.end());
        std::reverse(areaInContours.begin(), areaInContours.end());

        ///Remove noise contours
        std::vector<std::vector<cv::Point>> outContours;
        for (unsigned int i = 0; i < inContours.size(); i++) {
            if (cv::contourArea(inContours[i]) >= areaInContours[noTop]) {
                outContours.push_back(inContours[i]);
            }
        }
        return outContours;
    }
}
