#include <iostream>
#include <dirent.h>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

void isDetected(bool& isSuccess, cv::Mat orgImg, cv::Point& point1, cv::Point& point2, cv::Point& center, float& radius);
std::vector<cv::Vec4i> simplifyLines(std::vector<cv::Vec4i>);
void drawLines(cv::Mat, std::vector<cv::Vec4i>);
void thinning(cv::Mat&);
std::vector<cv::Point> findIntersects(cv::Mat, std::vector<cv::Vec4i>);
void drawIntersects(cv::Mat, std::vector<cv::Point>);
std::vector<std::vector<cv::Point>> clusterIntersects(std::vector<cv::Point>);
std::vector<cv::Point> findCenterOfCluster(std::vector<std::vector<cv::Point>>, float);
void drawResults(cv::Mat, std::vector<cv::Point>, float, int);
int distanceBox(cv::Point, cv::Point);
int distanceEuclid(cv::Point, cv::Point);
cv::Point intersectPoint(cv::Point, cv::Point, cv::Point, cv::Point);
int scalarProduct(cv::Point, cv::Point, cv::Point, cv::Point);
float cosTwoLines(cv::Point, cv::Point, cv::Point, cv::Point);
void thinningIteration(cv::Mat&, int);
void processGetCirclesFindContours(cv::Mat originMat, cv::Mat cropOriginMat, std::vector<float> vectorContourAreaValues, cv::Point2f& center, float& radius);
bool checkResult(std::vector<cv::Point> centers, cv::Point2f centerCircle, float radiusCircle);

int main() {
    std::string dirName = "input_cross_light_images/";
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

        bool isSuccess;
        cv::Point point1, point2, center;
        float radius;
        isDetected(isSuccess, orgImg, point1, point2, center, radius);

        line(orgImg, point1, point2, 255, 2);
        circle(orgImg, center, 2, cv::Scalar(0, 0, 255), 3, 8, 0);
        circle(orgImg, center, radius, cv::Scalar(0, 0, 255), 2, 8, 0);
        cv::imshow("orgImg", orgImg);
        cv::waitKey(0);
    }

    return 0;
}

void isDetected(bool& isSuccess, cv::Mat orgImg, cv::Point& point1, cv::Point& point2, cv::Point& center, float& radius) {
    isSuccess = true;
    cv::Mat yellowImg;
    cvtColor(orgImg, yellowImg, CV_BGR2YCrCb);
    yellowImg = yellowImg - cv::Scalar(0, 255, 255);

    ///Fix size image
    cv::Mat resizeImg;
    float rate = orgImg.cols / 160.0f;
    resize(yellowImg, resizeImg, cv::Size(160, cvRound(orgImg.rows / rate)));

    ///Convert to hsv image
    cv::Mat hsvImg;
    cvtColor(resizeImg, hsvImg, CV_BGR2HSV);

    ///Choose v channel
    cv::Mat vImg;
    vImg.create(hsvImg.size(), hsvImg.depth());
    int from_To[] = {2, 0};
    mixChannels(&hsvImg, 1, &vImg, 1, from_To, 1);

    ///Threshold v channel
    cv::Mat thresholdImg;
    threshold(vImg, thresholdImg, 120, 255, CV_THRESH_BINARY);

    ///Erode threshold image
    cv::Mat erodeImg;
    erode(thresholdImg, erodeImg, cv::Mat(), cv::Point(-1, -1), 2, 1, 1);

    ///Use canny to find edge of threshold image
    cv::Mat edgeImg;
    Canny(erodeImg, edgeImg, 120, 240, 3);
    thinning(erodeImg);

    ///Merge thinImg and edgeImg
    for (int i = 0; i < edgeImg.rows; i++) {
        for (int j = 0; j < edgeImg.cols; j++) {
            if (erodeImg.at<uchar>(i, j) > 100) {
                edgeImg.at<uchar>(i, j) = 255;
            }
        }
    }

    cv::imshow("edgeImg", edgeImg);
    ///Find lines on edge image
    std::vector<cv::Vec4i> lines;
    HoughLinesP(edgeImg, lines, 1, CV_PI / 180, 20, 50, 10);

    ///Remove bad lines
    lines = simplifyLines(lines);

    ///Find good intersects
    std::vector<cv::Point> intersects;
    cv::Mat lineImg = cv::Mat::zeros(edgeImg.size(), CV_8UC1);
    drawLines(lineImg, lines);

    cv::imshow("lineImg", lineImg);
    intersects = findIntersects(lineImg, lines);

    ///Cluster intersects
    std::vector<std::vector<cv::Point>> intersectsByCluster;
    intersectsByCluster = clusterIntersects(intersects);

    ///Find center of each cluster
    std::vector<cv::Point> centers;
    centers = findCenterOfCluster(intersectsByCluster, rate);

    if (centers.size() != 2) {
        isSuccess = false;
    } else {
        if (centers[0].x < centers[1].x) {
            point1 = centers[0];
            point2 = centers[1];
        } else {
            point1 = centers[1];
            point2 = centers[0];
        }
    }

    cv::Mat blurImg;
    medianBlur(orgImg, blurImg, 3);

    cvtColor(orgImg, blurImg, CV_RGB2HSV);
    std::vector<cv::Mat> channels;
    split(blurImg, channels);

    std::vector<cv::Mat> setThres(channels.size());

    for (int i = 0; i < 3; i++) {
        threshold(channels[i], setThres[i], 150, 255, CV_THRESH_OTSU);
    }
    bitwise_not(setThres[0], setThres[0]);

    cv::Mat result = cv::Mat::zeros(channels[0].size(), CV_8UC1);
    bitwise_and(setThres[0], setThres[2], result);

    cv::Mat element1 = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3), cv::Point(1, 1));
    cv::Mat element2 = getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9), cv::Point(1, 1));
    cv::Mat img3;
    erode(result, img3, element1);
    dilate(img3, img3, element2);

    std::vector<float> vectorFloat;
    vectorFloat.insert(vectorFloat.end(), 200);
    vectorFloat.insert(vectorFloat.end(), 255);
    vectorFloat.insert(vectorFloat.end(), 40);
    vectorFloat.insert(vectorFloat.end(), 30);
    vectorFloat.insert(vectorFloat.end(), 140);
    vectorFloat.insert(vectorFloat.end(), 0.65);
    vectorFloat.insert(vectorFloat.end(), 25);

    cv::Point2f centerCircle;
    float radiusCircle;

    processGetCirclesFindContours(orgImg, img3, vectorFloat, centerCircle, radiusCircle);

    if (radiusCircle <= 0) {
        isSuccess = false;
    } else {
        center = centerCircle;
        radius = radiusCircle;
    }
}

std::vector<cv::Vec4i> simplifyLines(std::vector<cv::Vec4i> lines) {
    for (unsigned int i = 0; i < lines.size(); i++) {
        float x = (float)(lines[i][2] - lines[i][0]);
        float y = (float)(lines[i][3] - lines[i][1]);

        if (x == 0) {
            lines.erase(lines.begin() + i, lines.begin() + i + 1);
            i--;
        } else {
            float absTan = abs(y / x);

            if (absTan > 4 || absTan < 0.25f) {
                lines.erase(lines.begin() + i, lines.begin() + i + 1);
                i--;
            }
        }
    }
    return lines;
}

void drawLines(cv::Mat lineImg, std::vector<cv::Vec4i> lines) {
    for (unsigned int i = 0; i < lines.size(); i++) {
        line(lineImg, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), 255, 1, CV_AA);
    }
}

std::vector<cv::Point> findIntersects(cv::Mat lineImg, std::vector<cv::Vec4i> lines) {
    std::vector<cv::Point> intersects;
    if (lines.size() > 3) {
        for (unsigned int i = 0; i < lines.size() - 1; i++) {
            for (unsigned int j = i + 1; j < lines.size(); j++) {
                float cosTL = cosTwoLines(cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Point(lines[j][0], lines[j][1]), cv::Point(lines[j][2], lines[j][3]));

                if (cosTL < 0.6f && cosTL > -0.6f) {
                    cv::Point inter;
                    inter = intersectPoint(cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Point(lines[j][0], lines[j][1]), cv::Point(lines[j][2], lines[j][3]));
                    if (inter.y > 0 && inter.y < lineImg.rows && inter.x > 0 && inter.x < lineImg.cols) {
                        int dXL1 = abs(lines[i][2] - lines[i][0]);
                        int dYL1 = abs(lines[i][3] - lines[i][1]);
                        int dXL2 = abs(lines[j][2] - lines[j][0]);
                        int dYL2 = abs(lines[j][3] - lines[j][1]);
                        int xMinL1 = MIN(lines[i][0], lines[i][2]) - dXL1;
                        int xMaxL1 = MAX(lines[i][0], lines[i][2]) + dXL1;
                        int yMinL1 = MIN(lines[i][1], lines[i][3]) - dYL1;
                        int yMaxL1 = MAX(lines[i][1], lines[i][3]) + dYL1;
                        int xMinL2 = MIN(lines[j][0], lines[j][2]) - dXL2;
                        int xMaxL2 = MAX(lines[j][0], lines[j][2]) + dXL2;
                        int yMinL2 = MIN(lines[j][1], lines[j][3]) - dYL2;
                        int yMaxL2 = MAX(lines[j][1], lines[j][3]) + dYL2;
                        int xMin = MAX(xMinL1, xMinL2);
                        int xMax = MIN(xMaxL1, xMaxL2);
                        int yMin = MAX(yMinL1, yMinL2);
                        int yMax = MIN(yMaxL1, yMaxL2);

                        if (inter.x > xMin && inter.x < xMax && inter.y > yMin && inter.y < yMax) {
                            intersects.push_back(inter);
                        }
                    }
                }
            }
        }
    }

    return intersects;
}

void drawIntersects(cv::Mat intersectsImg, std::vector<cv::Point> intersects) {
    for (unsigned int i = 0; i < intersects.size(); i++) {
        intersectsImg.at<uchar>(intersects[i].y, intersects[i].x) = 255;
    }
}

std::vector<std::vector<cv::Point>> clusterIntersects(std::vector<cv::Point> intersects) {
    std::vector<std::vector<cv::Point>> intersectsByCluster;

    if (intersects.size() > 5) {
        std::vector<cv::Point> firstCluster;
        firstCluster.push_back(intersects[0]);
        intersectsByCluster.push_back(firstCluster);

        for (unsigned int i = 1; i < intersects.size(); i++) {
            int indexCluster = -1;
            int distancePointToCluster = 15;

            for (unsigned int j = 0; j < intersectsByCluster.size(); j++) {
                for (unsigned int k = 0; k < intersectsByCluster[j].size(); k++) {
                    int distanceTemp = distanceEuclid(intersects[i], intersectsByCluster[j][k]);

                    if (distanceTemp < distancePointToCluster) {
                        distancePointToCluster = distanceTemp;
                        indexCluster = j;
                    }
                }
            }

            if (indexCluster == -1) {
                std::vector<cv::Point> clusterTemp;
                clusterTemp.push_back(intersects[i]);
                intersectsByCluster.push_back(clusterTemp);
            } else {
                intersectsByCluster[indexCluster].push_back(intersects[i]);
            }
        }
    }

    std::vector<std::vector<cv::Point>> outCluster;
    std::vector<unsigned int> sizeCluster;
    for (unsigned int i = 0; i < intersectsByCluster.size(); i++) {
        sizeCluster.push_back(intersectsByCluster[i].size());
    }
    sort(sizeCluster.begin(), sizeCluster.end());
    reverse(sizeCluster.begin(), sizeCluster.end());
    if (intersectsByCluster.size() < 3) {
        return intersectsByCluster;
    } else {
        for (unsigned int i = 0; i < intersectsByCluster.size(); i++) {
            if (intersectsByCluster[i].size() >= sizeCluster[1]) {
                outCluster.push_back(intersectsByCluster[i]);
            }
        }
        return outCluster;
    }
}

std::vector<cv::Point> findCenterOfCluster(std::vector<std::vector<cv::Point>> intersectsByCluster, float rate) {
    std::vector<cv::Point> centers;

    for (unsigned int i = 0; i < intersectsByCluster.size(); i++) {
        unsigned int sizeCluster = intersectsByCluster[i].size();

        if (sizeCluster > 5) {
            int xCenter = 0;
            int yCenter = 0;
            for (unsigned int j = 0; j < sizeCluster; j++) {
                xCenter += intersectsByCluster[i][j].x;
                yCenter += intersectsByCluster[i][j].y;
            }

            xCenter = cvRound(rate * xCenter / sizeCluster);
            yCenter = cvRound(rate * yCenter / sizeCluster);

            centers.push_back(cv::Point(xCenter, yCenter));
        }
    }

    return centers;
}

void drawResults(cv::Mat orgImg, std::vector<cv::Point> centers, float rate, int color) {
    for (unsigned int i = 0; i < centers.size(); i++) {
        for (int x = -2; x <= 2; x++) {
            for (int y = -2; y <= 2; y++) {
                orgImg.at<cv::Vec3b>(cvRound(centers[i].y) + x, cvRound(centers[i].x) + y)[color] = 255;
                orgImg.at<cv::Vec3b>(cvRound(centers[i].y) + x, cvRound(centers[i].x) + y)[(color + 1) % 3] = 0;
                orgImg.at<cv::Vec3b>(cvRound(centers[i].y) + x, cvRound(centers[i].x) + y)[(color + 2) % 3] = 0;
            }
        }
    }
}

int distanceBox(cv::Point p1, cv::Point p2) {
    return abs(p1.x - p2.x) + abs(p1.y - p2.y);
}

int distanceEuclid(cv::Point p1, cv::Point p2) {
    return (int)sqrtf((float)((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y)));
}

cv::Point intersectPoint(cv::Point line1_p1, cv::Point line1_p2, cv::Point line2_p1, cv::Point line2_p2) {
    cv::Point v1 = line1_p2 - line1_p1;
    cv::Point v2 = line2_p2 - line2_p1;
    float a1 = (float)-v1.y;
    float b1 = (float)v1.x;
    float c1 = a1 * line1_p1.x + b1 * line1_p1.y;
    float a2 = (float)-v2.y;
    float b2 = (float)v2.x;
    float c2 = a2 * line2_p1.x + b2 * line2_p1.y;
    float delta = a1 * b2 - a2 * b1;
    float deltaX = c1 * b2 - c2 * b1;
    float deltaY = a1 * c2 - a2 * c1;

    if (delta != 0) {
        cv::Point p;
        p.x = cvRound(deltaX / delta);
        p.y = cvRound(deltaY / delta);
        return p;
    } else {
        return cv::Point(-1, -1);
    }
}

int scalarProduct(cv::Point line1_p1, cv::Point line1_p2, cv::Point line2_p1, cv::Point line2_p2) {
    cv::Point v1 = line1_p2 - line1_p1;
    cv::Point v2 = line2_p2 - line2_p1;
    return v1.x * v2.x + v1.y * v2.y;
}

float cosTwoLines(cv::Point line1_p1, cv::Point line1_p2, cv::Point line2_p1, cv::Point line2_p2) {
    float upper = (float)scalarProduct(line1_p1, line1_p2, line2_p1, line2_p2);
    float under = sqrtf((float)((line1_p1.x - line1_p2.x) * (line1_p1.x - line1_p2.x) + (line1_p1.y - line1_p2.y) * (line1_p1.y - line1_p2.y))) *
                  sqrtf((float)((line2_p1.x - line2_p2.x) * (line2_p1.x - line2_p2.x) + (line2_p1.y - line2_p2.y) * (line2_p1.y - line2_p2.y)));
    return upper / under;
}

void thinningIteration(cv::Mat& im, int iter) {
    cv::Mat marker = cv::Mat::zeros(im.size(), CV_8UC1);
    for (int i = 1; i < im.rows - 1; i++) {
        for (int j = 1; j < im.cols - 1; j++) {
            if (im.at<uchar>(i, j) == 1) {
                uchar p2 = im.at<uchar>(i - 1, j);
                uchar p3 = im.at<uchar>(i - 1, j + 1);
                uchar p4 = im.at<uchar>(i, j + 1);
                uchar p5 = im.at<uchar>(i + 1, j + 1);
                uchar p6 = im.at<uchar>(i + 1, j);
                uchar p7 = im.at<uchar>(i + 1, j - 1);
                uchar p8 = im.at<uchar>(i, j - 1);
                uchar p9 = im.at<uchar>(i - 1, j - 1);

                int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                        (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                        (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                        (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);

                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

                int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);

                int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

                if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                    marker.at<uchar>(i, j) = 1;
            }
        }
    }

    for (int i = 1; i < im.rows - 1; i++) {
        int j = 0;
        if (im.at<uchar>(i, j) == 1) {
            uchar p2 = im.at<uchar>(i - 1, j);
            uchar p3 = im.at<uchar>(i - 1, j + 1);
            uchar p4 = im.at<uchar>(i, j + 1);
            uchar p5 = im.at<uchar>(i + 1, j + 1);
            uchar p6 = im.at<uchar>(i + 1, j);
            uchar p7 = 0;
            uchar p8 = 0;
            uchar p9 = 0;

            int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                    (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                    (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                    (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);

            int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);

            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                marker.at<uchar>(i, j) = 1;
        }
    }

    for (int j = 1; j < im.cols - 1; j++) {
        int i = 0;
        if (im.at<uchar>(i, j) == 1) {
            uchar p2 = 0;
            uchar p3 = 0;
            uchar p4 = im.at<uchar>(i, j + 1);
            uchar p5 = im.at<uchar>(i + 1, j + 1);
            uchar p6 = im.at<uchar>(i + 1, j);
            uchar p7 = im.at<uchar>(i + 1, j - 1);
            uchar p8 = im.at<uchar>(i, j - 1);
            uchar p9 = 0;

            int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                    (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                    (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                    (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);

            int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);

            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                marker.at<uchar>(i, j) = 1;
        }
    }

    for (int i = 1; i < im.rows - 1; i++) {
        int j = im.cols - 1;
        {
            if (im.at<uchar>(i, j) == 1) {
                uchar p2 = im.at<uchar>(i - 1, j);
                uchar p3 = 0;
                uchar p4 = 0;
                uchar p5 = 0;
                uchar p6 = im.at<uchar>(i + 1, j);
                uchar p7 = im.at<uchar>(i + 1, j - 1);
                uchar p8 = im.at<uchar>(i, j - 1);
                uchar p9 = im.at<uchar>(i - 1, j - 1);

                int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                        (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                        (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                        (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);

                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

                int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);

                int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

                if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                    marker.at<uchar>(i, j) = 1;
            }
        }
    }

    for (int j = 1; j < im.cols - 1; j++) {
        int i = im.rows - 1;
        if (im.at<uchar>(i, j) == 1) {
            uchar p2 = im.at<uchar>(i - 1, j);
            uchar p3 = im.at<uchar>(i - 1, j + 1);
            uchar p4 = im.at<uchar>(i, j + 1);
            uchar p5 = 0;
            uchar p6 = 0;
            uchar p7 = 0;
            uchar p8 = im.at<uchar>(i, j - 1);
            uchar p9 = im.at<uchar>(i - 1, j - 1);

            int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) +
                    (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) +
                    (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                    (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);

            int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;

            int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);

            int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);

            if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                marker.at<uchar>(i, j) = 1;
        }
    }

    im &= ~marker;
}

void thinning(cv::Mat& im) {
    im /= 255;

    cv::Mat prev = cv::Mat::zeros(im.size(), CV_8UC1);
    cv::Mat diff;

    int i = 0;
    const int max_count = 1;

    do {
        thinningIteration(im, 0);
        thinningIteration(im, 1);
        absdiff(im, prev, diff);
        im.copyTo(prev);
        if (i >= max_count)
            break;
        i++;
    } while (countNonZero(diff) > 0);

    im *= 255;
}

void processGetCirclesFindContours(cv::Mat originMat, cv::Mat cropOriginMat, std::vector<float> vectorContourAreaValues, cv::Point2f& center, float& radius) {
    std::vector<std::vector<cv::Point>> contours;

    cv::Mat clone = originMat.clone();

    findContours(cropOriginMat, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

    std::vector<std::vector<cv::Point>> contour_after;

    for (unsigned int i = 0; i < contours.size(); i++) {
        cv::Mat test = cv::Mat::zeros(cropOriginMat.size(), CV_8UC1);

        drawContours(test, contours, i, cv::Scalar(255, 255, 255), CV_FILLED);

        cv::Mat binary;

        threshold(test, binary, vectorContourAreaValues[0], vectorContourAreaValues[1], CV_THRESH_BINARY);

        cv::Scalar mean_clone;

        mean_clone = mean(clone, binary);

        int r = (int)mean_clone[2], g = (int)mean_clone[1], b = (int)mean_clone[0];

        if ((b - r >= (int)vectorContourAreaValues[2]) && (b - g >= (int)vectorContourAreaValues[3]) && (b >= (int)vectorContourAreaValues[4])) {
            std::vector<cv::Point> cont_tmp = contours[i];

            contour_after.push_back(cont_tmp);
        }
    }

    for (unsigned int i = 0; i < contour_after.size(); i++) {
        cv::Point2f center1;

        float radius1;

        std::vector<cv::Point> hull;

        minEnclosingCircle(contour_after[i], center1, radius1);

        convexHull(contour_after[i], hull, false);

        float area_Circle = (float)(CV_PI * radius1 * radius1);

        float ratio = (float)(contourArea(hull) / area_Circle);

        if (ratio > vectorContourAreaValues[5] && area_Circle > CV_PI * vectorContourAreaValues[6] * vectorContourAreaValues[6]) {
            center.x = center1.x;

            center.y = center1.y;

            radius = radius1;
        }
    }
}

bool checkResult(std::vector<cv::Point> centers, cv::Point2f centerCircle, float radiusCircle) {
    if (centers.size() != 2) {
        return false;
    }

    if (radiusCircle > 120 || radiusCircle < 60) {
        return false;
    }

    cv::Point centerC = centerCircle;

    float cosTL = cosTwoLines(centerC, centers[0], centerC, centers[1]);

    if (cosTL < 0.75 && cosTL > -0.75) {
        return false;
    }

    int distance0 = distanceEuclid(centerC, centers[0]);
    int distance1 = distanceEuclid(centerC, centers[1]);
    int distanceMax;
    int distanceMin;

    if (distance0 < distance1) {
        distanceMax = distance1;
        distanceMin = distance0;
    } else {
        distanceMax = distance0;
        distanceMin = distance1;
    }

    float rateDistances = (float)distanceMax / (float)distanceMin;

    if (rateDistances < 1.35f || rateDistances > 1.8f) {
        return false;
    }

    float rateRPD = (float)radiusCircle / (float)distanceMin;

    if (rateRPD > 0.58f) {
        return false;
    }

    int dx = centers[0].x - centers[1].x;
    int dy = centers[0].y - centers[1].y;

    if (dx == 0) {
        return false;
    }

    float absTanLOx = abs((float)dy / (float)dx);

    if (absTanLOx > 2.0f || absTanLOx < 0.1f) {
        return false;
    }

    return true;
}
