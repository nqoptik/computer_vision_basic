#include <iostream>
#include <dirent.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

const int noAPR = 6;  ///angles per  radius
const float PI = 3.14159265f;
const int lowThreshold = 70;

int allCircles[270][260][240][2];

std::vector<cv::Vec3i> findCircle(cv::Mat);
std::vector<std::vector<cv::Point>> simplifyContours(std::vector<std::vector<cv::Point>>);
std::vector<cv::Vec3i> findPositiveCircles(cv::Mat, float);
std::vector<cv::Vec3i> findStrictCircle(cv::Mat, std::vector<cv::Vec3i>);

int main() {
    std::string dirName = "input_hough_circle_images/";
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
    for (size_t i = 0; i < img_paths.size(); i++) {
        std::cout << img_paths[i] << std::endl;
        cv::Mat orgImg = cv::imread(img_paths[i], 1);
        if (orgImg.empty()) {
            break;
        }

        cv::Mat bilateralMat;
        cv::bilateralFilter(orgImg, bilateralMat, 5, 21, 21);

        std::vector<cv::Vec3i> strictCircle;
        strictCircle = findCircle(bilateralMat);

        ///Draw strict circle
        cv::Mat strictImg = orgImg.clone();
        for (unsigned int i = 0; i < strictCircle.size(); i++) {
            cv::circle(strictImg, cv::Point(strictCircle[i][1], strictCircle[i][0]), 1, cv::Scalar(0, 0, 255), 3, 8, 0);
            cv::circle(strictImg, cv::Point(strictCircle[i][1], strictCircle[i][0]), strictCircle[i][2], cv::Scalar(0, 0, 255), 1, 8, 0);
        }

        cv::imshow("strictImg", strictImg);
        cv::imshow("bilateralFilterMat", bilateralMat);
        cv::waitKey();
    }

    return 0;
}

std::vector<cv::Vec3i> findCircle(cv::Mat orgImg) {
    cv::Mat hsvImg;
    cv::cvtColor(orgImg, hsvImg, CV_BGR2HSV);

    ///Just use only v channel
    cv::Mat vImg;
    vImg.create(hsvImg.size(), hsvImg.depth());
    int from_To[] = {2, 0};
    cv::mixChannels(&hsvImg, 1, &vImg, 1, from_To, 1);

    ///Resize threshold image to 50*x
    cv::Mat resizeThresholdImg;
    float rate = vImg.cols / 50.0f;

    ///Find edge of resize threshold image
    cv::Mat edgeResizeImg;
    cv::Canny(vImg, edgeResizeImg, lowThreshold, lowThreshold * 2, 3);
    cv::dilate(edgeResizeImg, edgeResizeImg, cv::Mat(), cv::Point(-1, -1), 1, 1, 1);
    cv::resize(edgeResizeImg, edgeResizeImg, cv::Size(50, round(vImg.rows / rate)));

    cv::imshow("vImg", vImg);
    cv::imshow("edgeResizeImg", edgeResizeImg);

    ///Find positive circles

    std::vector<cv::Vec3i> positiveCircles;
    positiveCircles = findPositiveCircles(edgeResizeImg, rate);

    cv::Mat possitiveImg = orgImg.clone();
    for (unsigned int i = 0; i < positiveCircles.size(); i++) {
        cv::circle(possitiveImg, cv::Point(positiveCircles[i][1], positiveCircles[i][0]), 1, cv::Scalar(0, 0, 255), 3, 8, 0);
        cv::circle(possitiveImg, cv::Point(positiveCircles[i][1], positiveCircles[i][0]), positiveCircles[i][2], cv::Scalar(0, 0, 255), 1, 8, 0);
    }

    cv::imshow("possitiveImg", possitiveImg);

    ///Find edge of threshold image
    cv::Mat edgeImg;
    cv::Canny(vImg, edgeImg, 90, 90 * 2, 3);

    cv::imshow("edgeImg", edgeImg);

    ///Find strict cirlce
    std::vector<cv::Vec3i> strictCircle;
    strictCircle = findStrictCircle(edgeImg, positiveCircles);

    return strictCircle;
}

std::vector<std::vector<cv::Point>> simplifyContours(std::vector<std::vector<cv::Point>> inContours) {
    if (inContours.size() == 0) {
        return inContours;
    } else {
        std::vector<unsigned int> sizeInContours;
        for (unsigned int i = 0; i < inContours.size(); i++) {
            sizeInContours.push_back(inContours[i].size());
        }

        std::sort(sizeInContours.begin(), sizeInContours.end());
        std::reverse(sizeInContours.begin(), sizeInContours.end());

        ///Remove noise contours
        std::vector<std::vector<cv::Point>> outContours;
        for (unsigned int i = 0; i < inContours.size(); i++) {
            if (inContours[i].size() >= sizeInContours[MIN(3, inContours.size() - 1)]) {
                outContours.push_back(inContours[i]);
            }
        }
        return outContours;
    }
}

std::vector<cv::Vec3i> findPositiveCircles(cv::Mat contoursResizeImg, float rate) {
    std::vector<cv::Vec3i> roughCircles;

    ///Declare const
    ///i ~ Oy: iMin -> iMax
    int iMin = round(2 * contoursResizeImg.rows / 5);
    int iMax = round(3 * contoursResizeImg.rows / 5);
    int iStep = 1;

    ///j ~ Ox: jMin -> jMax
    int jMin = 20;
    int jMax = 30;
    int jStep = 1;

    ///r: rMin -> rMax
    int rMin = 19;
    int rMax = 26;
    int rStep = 1;

    ///Calculate weight of each circle
    for (int i = iMin; i <= iMax; i += iStep) {
        for (int j = jMin; j <= jMax; j += jStep) {
            for (int r = rMin; r <= rMax; r += rStep) {
                int noAngle = r * noAPR;
                int iAngleMin = noAngle / 12;
                int iAngleMax = 3 * noAngle / 4;
                float angleStep = 2 * PI / noAngle;
                for (int iAngle = iAngleMin; iAngle < iAngleMax; iAngle++) {
                    float angle = iAngle * angleStep;
                    int iP = i + round(r * cos(angle));
                    int jP = j + round(r * sin(angle));
                    if (iP >= 0 && iP < contoursResizeImg.rows && jP >= 0 && jP < contoursResizeImg.cols) {
                        if (contoursResizeImg.at<uchar>(iP, jP) > 100) {
                            allCircles[i][j][r][0]++;
                        }
                    }
                }
            }
        }
    }

    ///Save weight of each circle to vertor noPoint
    std::vector<int> noPoint;
    for (int i = iMin; i <= iMax; i += iStep) {
        for (int j = jMin; j <= jMax; j += jStep) {
            for (int r = rMin; r <= rMax; r += rStep) {
                noPoint.push_back(allCircles[i][j][r][0]);
            }
        }
    }

    ///Sort weight of each circle
    std::sort(noPoint.begin(), noPoint.end());
    std::reverse(noPoint.begin(), noPoint.end());

    ///Push positive circles to vector roughCircles
    for (int i = iMin; i <= iMax; i += iStep) {
        for (int j = jMin; j <= jMax; j += jStep) {
            for (int r = rMin; r <= rMax; r += rStep) {
                ///Push top 4 circles
                if (allCircles[i][j][r][0] >= noPoint[10] && allCircles[i][j][r][0] > 10) {
                    roughCircles.push_back(cv::Vec3i(i, j, r));
                }
            }
        }
    }

    ///Reset variable to default
    for (int i = iMin; i <= iMax; i += iStep) {
        for (int j = jMin; j <= jMax; j += jStep) {
            for (int r = rMin; r <= rMax; r += rStep) {
                allCircles[i][j][r][0] = 0;
            }
        }
    }

    ///Calculate real size of positive circles
    std::vector<cv::Vec3i> positiveCircles;
    for (unsigned int i = 0; i < roughCircles.size(); i++) {
        positiveCircles.push_back(cv::Vec3i(round(rate * roughCircles[i][0]), round(rate * roughCircles[i][1]), round(rate * roughCircles[i][2])));
    }
    return positiveCircles;
}

std::vector<cv::Vec3i> findStrictCircle(cv::Mat contoursImg, std::vector<cv::Vec3i> positiveCircles) {
    std::vector<cv::Vec3i> strictPositiveCircle;
    std::vector<int> weight;
    if (contoursImg.rows < 360 && contoursImg.cols < 350) {
        for (unsigned int k = 0; k < positiveCircles.size(); k++) {
            ///Declare const
            ///i ~ Oy: iMin -> iMax
            int iMin = positiveCircles[k][0] - 2;
            int iMax = positiveCircles[k][0] + 2;
            int iStep = 1;

            ///j ~ Ox: jMin -> jMax
            int jMin = positiveCircles[k][1] - 2;
            int jMax = positiveCircles[k][1] + 2;
            int jStep = 1;

            ///r: rMin -> rMax
            int rMin = positiveCircles[k][2] - 2;
            int rMax = positiveCircles[k][2] + 2;
            int rStep = 1;

            ///Calculate weight of each circle
            for (int i = iMin; i <= iMax; i += iStep) {
                for (int j = jMin; j <= jMax; j += jStep) {
                    for (int r = rMin; r <= rMax; r += rStep) {
                        int noAngle = r * noAPR;
                        int iAngleMin = noAngle / 12;
                        int iAngleMax = 3 * noAngle / 4;
                        float angleStep = 2 * PI / noAngle;
                        for (int iAngle = iAngleMin; iAngle < iAngleMax; iAngle++) {
                            float angle = iAngle * angleStep;
                            int iP = i + round(r * cos(angle));
                            int jP = j + round(r * sin(angle));
                            if (iP >= 0 && iP < contoursImg.rows && jP >= 0 && jP < contoursImg.cols) {
                                if (contoursImg.at<uchar>(iP, jP) > 100) {
                                    allCircles[i][j][r][0]++;
                                }
                            }
                        }
                    }
                }
            }

            ///Save weight of each circle to vertor noPoint
            std::vector<int> noPoint;
            for (int i = iMin; i <= iMax; i += iStep) {
                for (int j = jMin; j <= jMax; j += jStep) {
                    for (int r = rMin; r <= rMax; r += rStep) {
                        noPoint.push_back(allCircles[i][j][r][0]);
                    }
                }
            }

            ///Sort weight of each circle
            std::sort(noPoint.begin(), noPoint.end());
            std::reverse(noPoint.begin(), noPoint.end());

            ///Push the best positives circles to vector strictPositiveCircle
            for (int i = iMin; i <= iMax; i += iStep) {
                for (int j = jMin; j <= jMax; j += jStep) {
                    for (int r = rMin; r <= rMax; r += rStep) {
                        if (allCircles[i][j][r][0] >= noPoint[0] && allCircles[i][j][r][0] > 20) {
                            strictPositiveCircle.push_back(cv::Vec3i(i, j, r));
                            weight.push_back(allCircles[i][j][r][0]);
                        }
                    }
                }
            }

            ///Reset variable to default
            for (int i = iMin; i <= iMax; i += iStep) {
                for (int j = jMin; j <= jMax; j += jStep) {
                    for (int r = rMin; r <= rMax; r += rStep) {
                        allCircles[i][j][r][0] = 0;
                    }
                }
            }
        }
    }

    int iWeightMax = 0;
    std::vector<cv::Vec3i> strictCircle;

    if (weight.size() > 0) {
        for (unsigned int i = 0; i < weight.size(); i++) {
            if (weight[iWeightMax] < weight[i]) {
                iWeightMax = i;
            }
        }
        strictCircle.push_back(strictPositiveCircle[iWeightMax]);
    }
    return strictCircle;
}
