#include <iostream>
#include <dirent.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat orgImg, appliedImg, normImg, outImg;
std::vector<std::vector<cv::Point> > vales;

void CallBackFunc(int event, int x, int y, int flags, void* userdata);

int main() {
    std::string dirName = "input_images/";
    DIR* pDir;
    pDir = opendir(dirName.c_str());
    if (pDir == NULL) {
        std::cout << "Directory not found." << std::endl;
        return 1;
    }

    struct dirent* pDirent;
    std::vector<std::string> img_paths;
    while ((pDirent = readdir(pDir)) != NULL) {
        if (strcmp(pDirent->d_name, ".") == 0 || strcmp(pDirent->d_name, "..") == 0) {
            continue;
        }
        std::string imgPath = dirName;
        imgPath.append(pDirent->d_name);
        img_paths.push_back(imgPath);
    }
    closedir(pDir);

    std::sort(img_paths.begin(), img_paths.end());
    for (size_t idx = 0; idx < img_paths.size(); idx++) {
        std::cout << img_paths[idx] << std::endl;
        orgImg = cv::imread(img_paths[idx], 1);
        if (orgImg.empty()) {
            break;
        }

        appliedImg = orgImg.clone();
        normImg = cv::Mat::zeros(orgImg.size(), CV_8UC1);

        for (int i = 0; i < normImg.rows; i += 50) {
            for (int j = 0; j < normImg.cols; j += 50) {
                normImg.at<uchar>(i, j) = 255;
            }
        }

        //line(normImg, cv::Point(normImg.cols - 2, 1), cv::Point(1, normImg.rows - 2) , 255, 1);
        outImg = cv::Mat::zeros(orgImg.size(), CV_8UC3);
        cv::namedWindow("appliedImg", 1);
        imshow("appliedImg", appliedImg);
        imshow("outImg", outImg);
        cv::setMouseCallback("appliedImg", CallBackFunc, NULL);

        cv::waitKey();
    }

    return 0;
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        std::vector<cv::Point> temp;
        temp.push_back(cv::Point(x, y));
        vales.push_back(temp);
    } else if (event == cv::EVENT_LBUTTONUP && vales.size() > 0) {
        if (vales[vales.size() - 1].size() == 1) {
            circle(appliedImg, vales[vales.size() - 1][0], 1, cv::Scalar(0, 0, 255), -1);
            circle(normImg, vales[vales.size() - 1][0], 1, 255, 1);
        }
    } else if (event == cv::EVENT_MOUSEMOVE && flags == cv::EVENT_FLAG_LBUTTON && vales.size() > 0) {
        vales[vales.size() - 1].push_back(cv::Point(x, y));

        if (vales[vales.size() - 1].size() > 1) {
            line(appliedImg, vales[vales.size() - 1][vales[vales.size() - 1].size() - 2], vales[vales.size() - 1][vales[vales.size() - 1].size() - 1], cv::Scalar(0, 0, 255), 1);
            line(normImg, vales[vales.size() - 1][vales[vales.size() - 1].size() - 2], vales[vales.size() - 1][vales[vales.size() - 1].size() - 1], 255, 1);
        }
    }

    if (event == cv::EVENT_LBUTTONUP) {
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::Mat contoursImg = normImg.clone();
        findContours(contoursImg, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

        int noObjects = contours.size();

        cv::Mat markers = cv::Mat::zeros(normImg.size(), CV_32SC1);
        for (int i = 0; i < noObjects; i++) {
            drawContours(markers, contours, i, cv::Scalar::all(i + 1), -1);
        }

        watershed(orgImg, markers);

        std::vector<cv::Vec3b> colors;
        for (int i = 0; i < noObjects; i++) {
            int b = cv::theRNG().uniform(0, 255);
            int g = cv::theRNG().uniform(0, 255);
            int r = cv::theRNG().uniform(0, 255);

            colors.push_back(cv::Vec3b((uchar)b, (uchar)g, (uchar)r));
        }

        for (int i = 0; i < markers.rows; i++) {
            for (int j = 0; j < markers.cols; j++) {
                int index = markers.at<int>(i, j);
                if (index > 0 && index <= noObjects) {
                    outImg.at<cv::Vec3b>(i, j) = colors[index - 1];
                } else {
                    outImg.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
                }
            }
        }

        cv::imshow("outImg", outImg);
    }
    cv::imshow("normImg", normImg);
    cv::imshow("appliedImg", appliedImg);
}
