#include <iostream>
#include <cmath>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

using namespace cv;
using namespace std;

int const noAngle = 100;
int const pMax = 414;
int const Hough = 60;
float const PI = 3.14159265f;
float const stepAngle = PI / noAngle;
int allLines[noAngle][pMax];
vector<int> pGood;
vector<float> angleGood;
vector<int> start;
vector<int> destination;

void drawLine(Mat orgImg, Mat aftImg, int p, float angle, int color);
void threshold(Mat inImg, Mat outImg, int thresh);
void findAllLines(Mat image);
void findGoodLines();
void drawGoodLines(Mat orgImg, Mat aftImg);
int main() {
    Mat orgImg = imread("Hough.png", 0);
    threshold(orgImg, orgImg, 200);
    findAllLines(orgImg);
    Mat aftImg = Mat::zeros(orgImg.rows, orgImg.cols, CV_8UC1);
    //  aftImg = orgImg.clone();
    findGoodLines();
    drawGoodLines(orgImg, aftImg);
    imshow("Original Image", orgImg);
    imshow("After Image", aftImg);
    waitKey(0);
    return 1;
}

void threshold(Mat inImg, Mat outImg, int thresh) {
    for (int i = 0; i < inImg.rows; i++) {
        for (int j = 0; j < inImg.cols; j++) {
            if (inImg.at<uchar>(i, j) < thresh) {
                outImg.at<uchar>(i, j) = 0;
            } else {
                outImg.at<uchar>(i, j) = 255;
            }
        }
    }
}

void findAllLines(Mat image) {
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if (image.at<uchar>(i, j) == 0) {
                for (int iAngle = 0; iAngle < noAngle; iAngle++) {
                    float angle = iAngle * stepAngle;
                    int p = (int)floor((i * cos(angle) + j * sin(angle)));
                    if (p >= 0) {
                        allLines[iAngle][p]++;
                    }
                }
            }
        }
    }
}

void drawLine(Mat orgImg, Mat aftImg, int p, float angle, int color) {
    if (angle > PI / 4 && angle < 3 * PI / 4) {
        for (int i = 1; i < aftImg.rows - 1; i++) {
            int j = (int)((p - i * cos(angle)) / sin(angle));
            if (j >= 0 && j < aftImg.cols - 2) {
                int countN = 0;
                for (int iDelta = -1; iDelta < 2; iDelta++) {
                    for (int jDelta = 0; jDelta < 3; jDelta++) {
                        if (orgImg.at<uchar>(i + iDelta, j + jDelta) == 0) {
                            countN++;
                        }
                    }
                }

                if (countN > 1)
                    aftImg.at<uchar>(i, j) = color;
            }
        }
    } else {
        for (int j = 0; j < aftImg.cols - 2; j++) {
            int i = (int)((p - j * sin(angle)) / cos(angle));
            if (i > 0 && i < aftImg.rows - 1) {
                int countN = 0;
                for (int iDelta = -1; iDelta < 2; iDelta++) {
                    for (int jDelta = 0; jDelta < 3; jDelta++) {
                        if (orgImg.at<uchar>(i + iDelta, j + jDelta) == 0) {
                            countN++;
                        }
                    }
                }

                if (countN > 1)
                    aftImg.at<uchar>(i, j) = color;
            }
        }
    }
}

void findGoodLines() {
    for (int iAngle = 0; iAngle < noAngle; iAngle++) {
        for (int p = 0; p < pMax; p++) {
            if (allLines[iAngle][p] > Hough) {
                pGood.push_back(p);
                angleGood.push_back(iAngle * stepAngle);
            }
        }
    }
}

void drawGoodLines(Mat orgImg, Mat aftImg) {
    for (unsigned int i = 0; i < pGood.size(); i++) {
        drawLine(orgImg, aftImg, pGood[i], angleGood[i], 255);
    }
}
