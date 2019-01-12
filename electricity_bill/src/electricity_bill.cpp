#include <iostream>
#include <vector>
#include <ctime>
#include <numeric>
#include <fstream>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/video/tracking.hpp>

float cvEuclidDistf(cv::Point p1, cv::Point p2);
void cvFindNearestPoint(std::vector<cv::Point> sequences, cv::Point root, cv::Point& result);
enum cvSelectTopContours_Mode {
    CV_SELECT_CONTOUR_AREA = 0,
    CV_SELECT_CONTOUR_SIZE = 1
};
void cvSelectTopContours(std::vector<std::vector<cv::Point>> src, std::vector<std::vector<cv::Point>>& dst, int top, int mode, int minSize, double minArea);

using namespace std;

void executeMain(cv::Mat orgImage, cv::Mat image, vector<cv::Mat>& informations);
void getBillContour(cv::Mat image, vector<vector<cv::Point>>& contours);
void getCornersFromContour(vector<cv::Point> sequences, vector<cv::Point>& rawCorners);
void getRawVertices(cv::Mat filledRawContour, vector<cv::Point> rawCorners, int noHrzPieces, vector<cv::Point>& rawTops, vector<cv::Point>& rawBots, int noVtcPieces, vector<cv::Point>& rawLefts, vector<cv::Point>& rawRights);
void getquadVertices(vector<cv::Point> quadCorners, int noHrzPieces, int noVtcPieces, vector<cv::Point>& quadTops, vector<cv::Point>& quadBots, vector<cv::Point>& quadLefts, vector<cv::Point>& quadRights);
void warpToMedialBill(cv::Mat rawBill, cv::Mat& medialBill, vector<cv::Point> rawTops, vector<cv::Point> rawBots, vector<cv::Point> quadTops, vector<cv::Point> quadBots, vector<cv::Mat>& medialMatrices);
void warpToQuadrangleBill(cv::Mat medialBill, cv::Mat& quadBill, vector<cv::Point> rawLefts, vector<cv::Point> rawRights, vector<cv::Point> quadLefts, vector<cv::Point> quadRights, vector<cv::Mat>& quadMatrices);
void warpToRectangleBill(cv::Mat quadBill, cv::Mat& rectBill, vector<cv::Point> quadCorners, cv::Mat& rectMatrix);
void setRectCheckPoints(vector<cv::Point2f>& rectPoints);
void getOriginalPoints(vector<cv::Point2f> rectPoints, vector<cv::Point2f>& orgPoints, cv::Mat rectMatrix, vector<cv::Mat> quadMatrices, vector<cv::Mat> medialMatrices, cv::Mat rectBill, cv::Mat image, cv::Mat orgImage);
void getOriginalRects(vector<cv::Point2f>& orgPoints, vector<vector<cv::Point2f>>& orgQuads, vector<cv::Rect>& orgRects);
void getInformations(cv::Mat orgImage, vector<cv::Mat>& informations, vector<vector<cv::Point2f>> orgQuads, vector<cv::Rect> orgRects);
void getMovingDistances(vector<cv::Mat> checkBoxes, vector<float>& movingDistances);
void setRectPoints(vector<cv::Point2f>& rectPoints, vector<float> movingDistances);
void writeInformationBoxes(vector<vector<cv::Mat>> informationBoxes);

int main() {
    std::vector<std::string> filenames;
    cv::glob("input_images/00*.png", filenames);
    ///Get informations from bill
    vector<vector<cv::Mat>> informationBoxes;
    int idx = 0;
    vector<cv::Mat> resizedImages;
    for (size_t i = 0; i < filenames.size(); ++i) {
        cv::Mat orgImage, image;
        orgImage = cv::imread(filenames[i]);
        if (orgImage.empty())
            break;
        vector<cv::Mat> informations;
        resize(orgImage, image, cv::Size(cvRound(1000 * orgImage.cols / (float)orgImage.rows), 1000), 0, 0, cv::INTER_CUBIC);
        resizedImages.push_back(image);
        executeMain(orgImage, image, informations);
        informationBoxes.push_back(informations);
        idx++;
    }

    for (unsigned int i = 0; i < informationBoxes.size(); i++) {
        cv::imshow("Resized image", resizedImages[i]);
        for (unsigned int j = 0; j < informationBoxes[i].size(); j++) {
            string str = "information box ";
            str.append(std::to_string(j));
            cv::imshow(str, informationBoxes[i][j]);
        }
        cv::waitKey();
    }
    ///Write information boxes
    writeInformationBoxes(informationBoxes);

    return 0;
}

float cvEuclidDistf(cv::Point p1, cv::Point p2) {
    int dx = p2.x - p1.x;
    int dy = p2.y - p1.y;
    return sqrt((float)(dx * dx + dy * dy));
}

void cvFindNearestPoint(std::vector<cv::Point> sequences, cv::Point root, cv::Point& result) {
    if (sequences.size() == 0) {
        return;
    }

    result = sequences[0];
    float minDist = cvEuclidDistf(sequences[0], root);
    for (unsigned int i = 1; i < sequences.size(); i++) {
        float dist = cvEuclidDistf(sequences[i], root);
        if (dist < minDist) {
            result = sequences[i];
            minDist = dist;
        }
    }
}

void cvSelectTopContours(std::vector<std::vector<cv::Point>> src, std::vector<std::vector<cv::Point>>& dst, int top, int mode, int minSize, double minArea) {
    dst.clear();
    int sizeOfSrc = src.size();
    if (sizeOfSrc != 0) {
        if (mode == CV_SELECT_CONTOUR_SIZE) {
            std::vector<int> src_sizes;
            for (int i = 0; i < sizeOfSrc; i++) {
                src_sizes.push_back(src[i].size());
            }
            std::vector<int> src_sortedSizes = src_sizes;
            sort(src_sortedSizes.begin(), src_sortedSizes.end());
            int minSize_Idx = MAX(sizeOfSrc - MAX(top, 1), 0);
            minSize = MAX(src_sortedSizes[minSize_Idx], minSize);
            for (int i = 0; i < sizeOfSrc; i++) {
                if (src_sizes[i] >= minSize) {
                    dst.push_back(src[i]);
                }
            }
        } else {
            std::vector<double> src_areas;
            for (int i = 0; i < sizeOfSrc; i++) {
                src_areas.push_back(contourArea(src[i]));
            }
            std::vector<double> src_sortedAreas = src_areas;
            sort(src_sortedAreas.begin(), src_sortedAreas.end());
            int minArea_Idx = MAX(sizeOfSrc - MAX(top, 1), 0);
            minArea = MAX(src_sortedAreas[minArea_Idx], minArea);
            for (int i = 0; i < sizeOfSrc; i++) {
                if (src_areas[i] >= minArea) {
                    dst.push_back(src[i]);
                }
            }
        }
    }
}

void getBillContour(cv::Mat image, vector<vector<cv::Point>>& contour) {
    cv::Mat redChannel(image.size(), CV_8UC1);
    int from_To[] = {2, 0};
    mixChannels(&image, 1, &redChannel, 1, from_To, 1);

    cv::Mat binaryImg;
    adaptiveThreshold(redChannel, binaryImg, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, CV_THRESH_BINARY, 95, 0);
    erode(binaryImg, binaryImg, cv::Mat(), cv::Point(-1, -1), 2, 1, 1);
    dilate(binaryImg, binaryImg, cv::Mat(), cv::Point(-1, -1), 3, 1, 1);

    vector<cv::Vec4i> hierarchy;
    findContours(binaryImg, contour, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
    cvSelectTopContours(contour, contour, 1, CV_SELECT_CONTOUR_AREA, 0, 100);
}

void getCornersFromContour(vector<cv::Point> sequences, vector<cv::Point>& rawCorners) {
    cv::Rect boundRect = boundingRect(sequences);
    cv::Point temp;
    cvFindNearestPoint(sequences, boundRect.tl(), temp);
    rawCorners[0] = temp;
    cvFindNearestPoint(sequences, cv::Point(boundRect.x + boundRect.width, boundRect.y), temp);
    rawCorners[1] = temp;
    cvFindNearestPoint(sequences, boundRect.br(), temp);
    rawCorners[2] = temp;
    cvFindNearestPoint(sequences, cv::Point(boundRect.x, boundRect.y + boundRect.height), temp);
    rawCorners[3] = temp;
}

void getRawVertices(cv::Mat filledRawContour, vector<cv::Point> rawCorners, int noHrzPieces, vector<cv::Point>& rawTops, vector<cv::Point>& rawBots, int noVtcPieces, vector<cv::Point>& rawLefts, vector<cv::Point>& rawRights) {
    rawTops.push_back(rawCorners[0]);
    rawBots.push_back(rawCorners[3]);
    rawLefts.push_back(rawCorners[0]);
    rawRights.push_back(rawCorners[1]);
    for (int i = 1; i < noHrzPieces; i++) {
        cv::Point temp;
        temp.x = cvRound(((noHrzPieces - i) * rawCorners[0].x + i * rawCorners[1].x) / (float)noHrzPieces);
        int yMid = (rawCorners[0].y + rawCorners[3].y) / 2;

        for (int y = 0; y < yMid; y++) {
            if (filledRawContour.at<cv::Vec3b>(y, temp.x)[0] == 0 && filledRawContour.at<cv::Vec3b>(y + 1, temp.x)[0] != 0) {
                temp.y = y + 1;
                break;
            }
        }
        rawTops.push_back(temp);

        temp.x = cvRound(((noHrzPieces - i) * rawCorners[3].x + i * rawCorners[2].x) / (float)noHrzPieces);
        for (int y = yMid; y < filledRawContour.rows; y++) {
            if (filledRawContour.at<cv::Vec3b>(y - 1, temp.x)[0] != 0 && filledRawContour.at<cv::Vec3b>(y, temp.x)[0] == 0) {
                temp.y = y - 1;
                break;
            }
        }
        rawBots.push_back(temp);
    }

    for (int i = 1; i < noVtcPieces; i++) {
        cv::Point temp;
        temp.y = cvRound(((noVtcPieces - i) * rawCorners[0].y + i * rawCorners[3].y) / (float)noVtcPieces);
        int xMid = (rawCorners[0].x + rawCorners[1].x) / 2;

        for (int x = 0; x < xMid; x++) {
            if (filledRawContour.at<cv::Vec3b>(temp.y, x)[0] == 0 && filledRawContour.at<cv::Vec3b>(temp.y, x + 1)[0] != 0) {
                temp.x = x + 1;
                break;
            }
        }
        rawLefts.push_back(temp);

        temp.y = cvRound(((noVtcPieces - i) * rawCorners[1].y + i * rawCorners[2].y) / (float)noVtcPieces);

        for (int x = xMid; x < filledRawContour.cols; x++) {
            if (filledRawContour.at<cv::Vec3b>(temp.y, x - 1)[0] != 0 && filledRawContour.at<cv::Vec3b>(temp.y, x)[0] == 0) {
                temp.x = x - 1;
                break;
            }
        }
        rawRights.push_back(temp);
    }

    rawTops.push_back(rawCorners[1]);
    rawBots.push_back(rawCorners[2]);
    rawLefts.push_back(rawCorners[3]);
    rawRights.push_back(rawCorners[2]);
}

void getquadVertices(vector<cv::Point> quadCorners, int noHrzPieces, int noVtcPieces, vector<cv::Point>& quadTops, vector<cv::Point>& quadBots, vector<cv::Point>& quadLefts, vector<cv::Point>& quadRights) {
    quadTops.push_back(quadCorners[0]);
    quadBots.push_back(quadCorners[3]);
    quadLefts.push_back(quadCorners[0]);
    quadRights.push_back(quadCorners[1]);

    for (int i = 1; i < noHrzPieces; i++) {
        cv::Point temp = (noHrzPieces - i) * quadCorners[0] + i * quadCorners[1];
        temp.x = cvRound(temp.x / (float)noHrzPieces);
        temp.y = cvRound(temp.y / (float)noHrzPieces);
        quadTops.push_back(temp);

        temp = (noHrzPieces - i) * quadCorners[3] + i * quadCorners[2];
        temp.x = cvRound(temp.x / (float)noHrzPieces);
        temp.y = cvRound(temp.y / (float)noHrzPieces);
        quadBots.push_back(temp);
    }

    for (int i = 1; i < noVtcPieces; i++) {
        cv::Point temp = (noVtcPieces - i) * quadCorners[0] + i * quadCorners[3];
        temp.x = cvRound(temp.x / (float)noVtcPieces);
        temp.y = cvRound(temp.y / (float)noVtcPieces);
        quadLefts.push_back(temp);

        temp = (noVtcPieces - i) * quadCorners[1] + i * quadCorners[2];
        temp.x = cvRound(temp.x / (float)noVtcPieces);
        temp.y = cvRound(temp.y / (float)noVtcPieces);
        quadRights.push_back(temp);
    }

    quadTops.push_back(quadCorners[1]);
    quadBots.push_back(quadCorners[2]);
    quadLefts.push_back(quadCorners[3]);
    quadRights.push_back(quadCorners[2]);
}

void warpToMedialBill(cv::Mat rawBill, cv::Mat& medialBill, vector<cv::Point> rawTops, vector<cv::Point> rawBots, vector<cv::Point> quadTops, vector<cv::Point> quadBots, vector<cv::Mat>& medialMatrices) {
    medialMatrices.clear();
    for (unsigned int i = 0; i < rawTops.size() - 1; i++) {
        vector<cv::Point2f> prePoints(4), curPoints(4);
        prePoints[0] = rawTops[i];
        prePoints[1] = rawTops[i + 1];
        prePoints[2] = rawBots[i + 1];
        prePoints[3] = rawBots[i];

        curPoints[0] = quadTops[i];
        curPoints[1] = quadTops[i + 1];
        curPoints[2] = quadBots[i + 1];
        curPoints[3] = quadBots[i];

        medialMatrices.push_back(getPerspectiveTransform(curPoints, prePoints));

        cv::Rect boundRect = boundingRect(curPoints);
        if (i == 0) {
            int delta = MIN(15, boundRect.x);
            boundRect.x -= delta;
            boundRect.width += delta;
        }

        if (i == rawTops.size() - 2) {
            int delta = MIN(15, rawBill.cols - boundRect.x - boundRect.width);
            boundRect.width += delta;
        }

        curPoints[0] -= cv::Point2f(boundRect.tl().x, boundRect.tl().y);
        curPoints[1] -= cv::Point2f(boundRect.tl().x, boundRect.tl().y);
        curPoints[2] -= cv::Point2f(boundRect.tl().x, boundRect.tl().y);
        curPoints[3] -= cv::Point2f(boundRect.tl().x, boundRect.tl().y);

        cv::Mat P = getPerspectiveTransform(prePoints, curPoints);
        cv::Mat piece = cv::Mat::zeros(rawBill.size(), CV_8UC3);

        vector<cv::Point> pieceCorners;

        if (i == 0) {
            pieceCorners.push_back(cv::Point(0, 0));
            pieceCorners.push_back(cv::Point(rawTops[1].x, 0));
            pieceCorners.push_back(rawTops[1]);
            pieceCorners.push_back(rawBots[1]);
            pieceCorners.push_back(cv::Point(rawBots[1].x, rawBill.rows - 1));
            pieceCorners.push_back(cv::Point(0, rawBill.rows - 1));
        }

        if (i == rawTops.size() - 2) {
            pieceCorners.push_back(cv::Point(rawTops[i].x, 0));
            pieceCorners.push_back(cv::Point(rawBill.cols - 1, 0));
            pieceCorners.push_back(cv::Point(rawBill.cols - 1, rawBill.rows - 1));
            pieceCorners.push_back(cv::Point(rawBots[i].y, rawBill.rows - 1));
            pieceCorners.push_back(rawBots[i]);
            pieceCorners.push_back(rawTops[i]);
        }

        if (i != 0 && i != rawTops.size() - 2) {
            pieceCorners.push_back(rawTops[i]);
            pieceCorners.push_back(rawTops[i + 1]);
            pieceCorners.push_back(rawBots[i + 1]);
            pieceCorners.push_back(rawBots[i]);
        }

        vector<vector<cv::Point>> vPieceCorners(1);
        vPieceCorners[0] = pieceCorners;
        drawContours(piece, vPieceCorners, 0, cv::Scalar(255, 255, 255), CV_FILLED);

        bitwise_and(piece, rawBill, piece);

        cv::Mat boundRectImg;
        warpPerspective(piece, boundRectImg, P, boundRect.size(), cv::INTER_CUBIC);
        cv::Mat medialPiece = cv::Mat::zeros(medialBill.size(), CV_8UC3);
        boundRectImg.copyTo(medialPiece(boundRect));
        medialBill = max(medialPiece, medialBill);
    }
}

void warpToQuadrangleBill(cv::Mat medialBill, cv::Mat& quadBill, vector<cv::Point> rawLefts, vector<cv::Point> rawRights, vector<cv::Point> quadLefts, vector<cv::Point> quadRights, vector<cv::Mat>& quadMatrices) {
    quadMatrices.clear();
    for (unsigned int i = 0; i < rawLefts.size() - 1; i++) {
        vector<cv::Point2f> prePoints(4), curPoints(4);
        prePoints[0] = rawLefts[i];
        prePoints[1] = rawLefts[i + 1];
        prePoints[2] = rawRights[i + 1];
        prePoints[3] = rawRights[i];

        curPoints[0] = quadLefts[i];
        curPoints[1] = quadLefts[i + 1];
        curPoints[2] = quadRights[i + 1];
        curPoints[3] = quadRights[i];

        quadMatrices.push_back(getPerspectiveTransform(curPoints, prePoints));

        cv::Rect boundRect = boundingRect(curPoints);
        if (i == 0) {
            int delta = MIN(15, boundRect.y);
            boundRect.y -= delta;
            boundRect.height += delta;
        }
        if (i == rawLefts.size() - 2) {
            int delta = MIN(15, medialBill.rows - boundRect.y - boundRect.height);
            boundRect.height += delta;
        }

        curPoints[0] -= cv::Point2f(boundRect.tl().x, boundRect.tl().y);
        curPoints[1] -= cv::Point2f(boundRect.tl().x, boundRect.tl().y);
        curPoints[2] -= cv::Point2f(boundRect.tl().x, boundRect.tl().y);
        curPoints[3] -= cv::Point2f(boundRect.tl().x, boundRect.tl().y);

        cv::Mat P = getPerspectiveTransform(prePoints, curPoints);
        cv::Mat piece = cv::Mat::zeros(medialBill.size(), CV_8UC3);

        vector<cv::Point> pieceCorners(4);
        pieceCorners[0] = rawLefts[i];
        pieceCorners[1] = rawLefts[i + 1];
        pieceCorners[2] = rawRights[i + 1];
        pieceCorners[3] = rawRights[i];

        vector<vector<cv::Point>> vPieceCorners(1);
        vPieceCorners[0] = pieceCorners;
        drawContours(piece, vPieceCorners, 0, cv::Scalar(255, 255, 255), CV_FILLED);

        bitwise_and(piece, medialBill, piece);

        cv::Mat boundRectImg;
        warpPerspective(piece, boundRectImg, P, boundRect.size(), cv::INTER_CUBIC);
        cv::Mat quadPiece = cv::Mat::zeros(quadBill.size(), CV_8UC3);
        boundRectImg.copyTo(quadPiece(boundRect));
        quadBill = max(quadBill, quadPiece);
    }
}

void warpToRectangleBill(cv::Mat quadBill, cv::Mat& rectBill, vector<cv::Point> quadCorners, cv::Mat& rectMatrix) {
    cv::Rect boundRect = cv::Rect(0, 0, 1000, 600);
    vector<cv::Point2f> prePoints(4), curPoints(4);
    prePoints[0] = quadCorners[0];
    prePoints[1] = quadCorners[1];
    prePoints[2] = quadCorners[2];
    prePoints[3] = quadCorners[3];

    curPoints[0] = cv::Point2f(0, 0);
    curPoints[1] = cv::Point2f((float)boundRect.width, 0);
    curPoints[2] = cv::Point2f((float)boundRect.width, (float)boundRect.height);
    curPoints[3] = cv::Point2f(0, (float)boundRect.height);

    rectMatrix = getPerspectiveTransform(curPoints, prePoints);

    cv::Mat P = getPerspectiveTransform(prePoints, curPoints);
    warpPerspective(quadBill, rectBill, P, boundRect.size(), cv::INTER_CUBIC);
}

void setRectCheckPoints(vector<cv::Point2f>& rectCheckPoints) {
    //check 1
    rectCheckPoints[0] = cv::Point2f(260, 228);
    rectCheckPoints[1] = cv::Point2f(270, 228);
    rectCheckPoints[2] = cv::Point2f(270, 268);
    rectCheckPoints[3] = cv::Point2f(260, 268);

    //check 2
    rectCheckPoints[4] = cv::Point2f(260, 228);
    rectCheckPoints[5] = cv::Point2f(270, 228);
    rectCheckPoints[6] = cv::Point2f(270, 268);
    rectCheckPoints[7] = cv::Point2f(260, 268);

    //check 3
    rectCheckPoints[8] = cv::Point2f(525, 538);
    rectCheckPoints[9] = cv::Point2f(535, 538);
    rectCheckPoints[10] = cv::Point2f(535, 578);
    rectCheckPoints[11] = cv::Point2f(525, 578);

    //check 4
    rectCheckPoints[12] = cv::Point2f(645, 128);
    rectCheckPoints[13] = cv::Point2f(655, 128);
    rectCheckPoints[14] = cv::Point2f(655, 168);
    rectCheckPoints[15] = cv::Point2f(645, 168);

    //check 5
    rectCheckPoints[16] = cv::Point2f(960, 428);
    rectCheckPoints[17] = cv::Point2f(970, 428);
    rectCheckPoints[18] = cv::Point2f(970, 468);
    rectCheckPoints[19] = cv::Point2f(960, 468);
}

void getOriginalPoints(vector<cv::Point2f> rectPoints, vector<cv::Point2f>& orgPoints, cv::Mat rectMatrix, vector<cv::Mat> quadMatrices, vector<cv::Mat> medialMatrices, cv::Mat rectBill, cv::Mat image, cv::Mat orgImage) {
    vector<int> quadIdx;
    for (unsigned int i = 0; i < rectPoints.size(); i++) {
        quadIdx.push_back(cvFloor(quadMatrices.size() * rectPoints[i].y / rectBill.rows));
    }

    vector<int> medialIdx;
    for (unsigned int i = 0; i < rectPoints.size(); i++) {
        medialIdx.push_back(cvFloor(medialMatrices.size() * rectPoints[i].x / rectBill.cols));
    }

    vector<cv::Point2f> quadPoints;
    vector<cv::Point2f> medialPoints;
    vector<cv::Point2f> rawPoints;
    perspectiveTransform(rectPoints, quadPoints, rectMatrix);

    for (unsigned int i = 0; i < quadPoints.size(); i++) {
        int idx = quadIdx[i];
        vector<cv::Point2f> quadPoint(1), medialPoint;
        quadPoint[0] = quadPoints[i];
        perspectiveTransform(quadPoint, medialPoint, quadMatrices[idx]);
        medialPoints.push_back(medialPoint[0]);
    }

    for (unsigned int i = 0; i < medialPoints.size(); i++) {
        int idx = medialIdx[i];
        vector<cv::Point2f> medialPoint(1), rawPoint;
        medialPoint[0] = medialPoints[i];
        perspectiveTransform(medialPoint, rawPoint, medialMatrices[idx]);
        rawPoints.push_back(rawPoint[0]);
    }

    orgPoints.clear();
    for (unsigned int i = 0; i < rawPoints.size(); i++) {
        cv::Point2f orgPoint;
        orgPoint.x = rawPoints[i].x * orgImage.cols / image.cols;
        orgPoint.y = rawPoints[i].y * orgImage.rows / image.rows;
        orgPoints.push_back(orgPoint);
    }
}

void getOriginalRects(vector<cv::Point2f>& orgPoints, vector<vector<cv::Point2f>>& orgQuads, vector<cv::Rect>& orgRects) {
    for (unsigned int i = 0; i < orgPoints.size(); i += 4) {
        vector<cv::Point2f> orgQuad;
        for (int j = 0; j < 4; j++) {
            orgQuad.push_back(orgPoints[i + j]);
        }
        cv::Rect r = boundingRect(orgQuad);
        orgQuads.push_back(orgQuad);
        orgRects.push_back(r);
    }
}

void getInformations(cv::Mat orgImage, vector<cv::Mat>& informations, vector<vector<cv::Point2f>> orgQuads, vector<cv::Rect> orgRects) {
    informations.clear();
    for (unsigned int i = 0; i < orgRects.size(); i++) {
        orgRects[i].x = 0;
        orgRects[i].y = 0;

        vector<cv::Point2f> prePoints(4), curPoints(4);
        prePoints[0] = orgQuads[i][0];
        prePoints[1] = orgQuads[i][1];
        prePoints[2] = orgQuads[i][2];
        prePoints[3] = orgQuads[i][3];

        curPoints[0] = cv::Point2f(0, 0);
        curPoints[1] = cv::Point2f((float)orgRects[i].width, 0);
        curPoints[2] = cv::Point2f((float)orgRects[i].width, (float)orgRects[i].height);
        curPoints[3] = cv::Point2f(0, (float)orgRects[i].height);

        cv::Mat P = getPerspectiveTransform(prePoints, curPoints);
        cv::Mat information;
        warpPerspective(orgImage, information, P, orgRects[i].size(), cv::INTER_CUBIC);
        informations.push_back(information);
    }
}

void getMovingDistances(vector<cv::Mat> checkBoxes, vector<float>& movingDistances) {
    movingDistances.clear();
    for (unsigned int i = 0; i < checkBoxes.size(); i++) {
        cv::Mat binaryImg, thresholdImg;
        cvtColor(checkBoxes[i], binaryImg, CV_BGR2GRAY);
        threshold(binaryImg, thresholdImg, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);
        dilate(thresholdImg, thresholdImg, cv::Mat(), cv::Point(-1, -1), 2, 1, 1);

        vector<vector<cv::Point>> contour;
        vector<cv::Vec4i> hierarchy;
        findContours(thresholdImg, contour, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
        cvSelectTopContours(contour, contour, 1, CV_SELECT_CONTOUR_AREA, 0, 1);
        if (contour.size() != 0) {
            cv::Rect boundRect = boundingRect(contour[0]);
            float originalDist = (boundRect.y + boundRect.height / 2.0f) - checkBoxes[i].rows / 2.0f;
            float rectDist = originalDist / checkBoxes[i].rows * 41;
            movingDistances.push_back(rectDist);
        } else {
            movingDistances.push_back(0);
        }
    }
}

void setRectPoints(vector<cv::Point2f>& rectPoints, vector<float> movingDistances) {
    //box 1
    rectPoints[0] = cv::Point2f(250, 160 + movingDistances[0]);
    rectPoints[1] = cv::Point2f(390, 160 + movingDistances[0]);
    rectPoints[2] = cv::Point2f(390, 190 + movingDistances[0]);
    rectPoints[3] = cv::Point2f(250, 190 + movingDistances[0]);

    //box 2
    rectPoints[4] = cv::Point2f(300, 195 + movingDistances[1]);
    rectPoints[5] = cv::Point2f(405, 195 + movingDistances[1]);
    rectPoints[6] = cv::Point2f(405, 225 + movingDistances[1]);
    rectPoints[7] = cv::Point2f(300, 225 + movingDistances[1]);

    //box 3
    rectPoints[8] = cv::Point2f(415, 492 + movingDistances[2]);
    rectPoints[9] = cv::Point2f(455, 492 + movingDistances[2]);
    rectPoints[10] = cv::Point2f(455, 520 + movingDistances[2]);
    rectPoints[11] = cv::Point2f(415, 520 + movingDistances[2]);

    //box 4
    rectPoints[12] = cv::Point2f(540, 152 + movingDistances[3]);
    rectPoints[13] = cv::Point2f(660, 152 + movingDistances[3]);
    rectPoints[14] = cv::Point2f(660, 182 + movingDistances[3]);
    rectPoints[15] = cv::Point2f(540, 182 + movingDistances[3]);

    //box 5
    rectPoints[16] = cv::Point2f(715, 416 + movingDistances[4]);
    rectPoints[17] = cv::Point2f(960, 416 + movingDistances[4]);
    rectPoints[18] = cv::Point2f(960, 447 + movingDistances[4]);
    rectPoints[19] = cv::Point2f(715, 447 + movingDistances[4]);
}

void executeMain(cv::Mat orgImage, cv::Mat image, vector<cv::Mat>& informations) {
    double t1 = clock();
    ///Get bill contour
    vector<vector<cv::Point>> contour;
    getBillContour(image, contour);
    if (contour.size() != 1)
        return;

    ///Approx to raw contour and quadrangle contour
    vector<vector<cv::Point>> rawContour(1), quadContour(1);
    approxPolyDP(contour[0], rawContour[0], 5, true);
    approxPolyDP(contour[0], quadContour[0], 35, true);

    ///Fill raw contour
    cv::Mat filledRawContour = cv::Mat::zeros(image.size(), CV_8UC3);
    drawContours(filledRawContour, rawContour, 0, cv::Scalar(255, 255, 255), CV_FILLED);

    ///Get raw corners
    vector<cv::Point> rawCorners(4);
    getCornersFromContour(rawContour[0], rawCorners);

    ///Get raw top vertices and raw bot vertices
    vector<cv::Point> rawTops, rawBots, rawLefts, rawRights;
    int noHrzPieces = 10;
    int noVtcPieces = 6;
    getRawVertices(filledRawContour, rawCorners, noHrzPieces, rawTops, rawBots, noVtcPieces, rawLefts, rawRights);

    ///Get quadrangle corners
    vector<cv::Point> quadCorners(4);
    getCornersFromContour(quadContour[0], quadCorners);

    ///Get quadrangle vertices
    vector<cv::Point> quadTops, quadBots, quadLefts, quadRights;
    getquadVertices(quadCorners, noHrzPieces, noVtcPieces, quadTops, quadBots, quadLefts, quadRights);

    double t2 = clock();
    ///Warp to medial bill
    cv::Mat rawBill;
    bitwise_and(filledRawContour, image, rawBill);
    cv::Mat medialBill = cv::Mat::zeros(image.size(), CV_8UC3);
    vector<cv::Mat> medialMatrices;
    warpToMedialBill(rawBill, medialBill, rawTops, rawBots, quadTops, quadBots, medialMatrices);

    double t3 = clock();
    ///Warp to quadrangle bill
    cv::Mat quadBill = cv::Mat::zeros(medialBill.size(), CV_8UC3);
    vector<cv::Mat> quadMatrices;
    warpToQuadrangleBill(medialBill, quadBill, rawLefts, rawRights, quadLefts, quadRights, quadMatrices);

    double t4 = clock();
    ///Warp to rectangle bill
    cv::Mat rectBill = cv::Mat::zeros(quadBill.size(), CV_8UC3);
    cv::Mat rectMatrix;
    warpToRectangleBill(quadBill, rectBill, quadCorners, rectMatrix);

    double t5 = clock();
    ///Set rectangle check points
    vector<cv::Point2f> rectCheckPoints(20);
    setRectCheckPoints(rectCheckPoints);

    ///Get original check points
    vector<cv::Point2f> orgCheckPoints;
    getOriginalPoints(rectCheckPoints, orgCheckPoints, rectMatrix, quadMatrices, medialMatrices, rectBill, image, orgImage);

    ///Get original check quads, rectangles
    vector<vector<cv::Point2f>> orgCheckQuads;
    vector<cv::Rect> orgCheckRects;
    getOriginalRects(orgCheckPoints, orgCheckQuads, orgCheckRects);

    ///Get check boxes information
    vector<cv::Mat> checkBoxes;
    getInformations(orgImage, checkBoxes, orgCheckQuads, orgCheckRects);

    ///Get moving distance
    vector<float> movingDistances;
    getMovingDistances(checkBoxes, movingDistances);

    ///Set rectangle points
    vector<cv::Point2f> rectPoints(20);
    setRectPoints(rectPoints, movingDistances);

    ///Get original points
    vector<cv::Point2f> orgPoints;
    getOriginalPoints(rectPoints, orgPoints, rectMatrix, quadMatrices, medialMatrices, rectBill, image, orgImage);

    ///Get original check quads, rectangles
    vector<vector<cv::Point2f>> orgQuads;
    vector<cv::Rect> orgRects;
    getOriginalRects(orgPoints, orgQuads, orgRects);

    ///Get billing information
    getInformations(orgImage, informations, orgQuads, orgRects);

    double t6 = clock();

    cout << "cv::Size image: " << image.size() << endl;
    cout << "Pre process: " << t2 - t1 << endl;
    cout << "Warp to medial bill: " << t3 - t2 << endl;
    cout << "Warp to quad bill: " << t4 - t3 << endl;
    cout << "Warp to rectangle bill: " << t5 - t4 << endl;
    cout << "Get billing informations: " << t6 - t5 << endl;
    cout << "Total time: " << t6 - t1 << endl;
}

void writeInformationBoxes(vector<vector<cv::Mat>> informationBoxes) {
    for (unsigned int i = 0; i < informationBoxes.size(); i++) {
        for (unsigned int j = 0; j < informationBoxes[i].size(); j++) {
            string str = "output_images_";
            str.append(std::to_string(j));
            str.append("/");
            if (i < 10) {
                str.append("0");
            }
            str.append(std::to_string(i));
            str.append(".png");
            imwrite(str, informationBoxes[i][j]);
        }
    }
}
