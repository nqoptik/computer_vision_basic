#include <iostream>
#include <dirent.h>
#include <fftw3.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

const float PI = 3.14159265f;

std::vector<cv::Vec4i> findDirection(cv::Mat orgImg, std::vector<float>& centerAngles);

///EuclidDistance
float EuclidDistance(cv::Point p1, cv::Point p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return sqrtf(dx * dx + dy * dy);
}

///triagleArea
float triagleArea(cv::Point p1, cv::Point p2, cv::Point p3) {
    float a = EuclidDistance(p1, p2);
    float b = EuclidDistance(p2, p3);
    float c = EuclidDistance(p3, p1);
    float upper = sqrtf((float)(a + b + c) * (a + b - c) * (a + c - b) * (b + c - a));
    return upper / 4;
}

int main() {
    std::string dirName = "rotated_images/";
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

        std::vector<cv::Vec4i> centerLines;

        std::vector<float> centerAngles;
        centerLines = findDirection(orgImg, centerAngles);

        for (unsigned int i = 0; i < centerLines.size(); i++) {
            if (i == 0) {
                line(orgImg, cv::Point(centerLines[i][0], centerLines[i][1]), cv::Point(centerLines[i][2], centerLines[i][3]), cv::Scalar(255, 0, 0), 2);
            } else if (i == 1) {
                line(orgImg, cv::Point(centerLines[i][0], centerLines[i][1]), cv::Point(centerLines[i][2], centerLines[i][3]), cv::Scalar(0, 255, 0), 2);
            } else if (i == 2) {
                line(orgImg, cv::Point(centerLines[i][0], centerLines[i][1]), cv::Point(centerLines[i][2], centerLines[i][3]), cv::Scalar(0, 0, 255), 2);
            } else {
                line(orgImg, cv::Point(centerLines[i][0], centerLines[i][1]), cv::Point(centerLines[i][2], centerLines[i][3]), cv::Scalar(255, 255, 255), 2);
            }
            std::cout << "angle: " << centerAngles[i] * 180 / PI << std::endl;
        }

        imshow("orgImg", orgImg);
        cv::waitKey();
    }
    closedir(pDir);

    return 0;
}

std::vector<cv::Vec4i> findDirection(cv::Mat orgImg, std::vector<float>& centerAngles) {
    cv::Mat img1;
    cv::Mat img2;
    uchar* img1_data;
    uchar* img2_data;

    fftw_complex* data_in;
    fftw_complex* fft;
    fftw_complex* ifft;
    fftw_plan plan_f;

    int width, height, step;
    int i, j, k;

    cv::cvtColor(orgImg, img1, CV_BGR2GRAY);
    cv::resize(img1, img1, cv::Size(img1.cols / 2, img1.rows / 2));

    // create new image for IFFT result
    img2 = cv::Mat::zeros(img1.size(), img1.type());

    // get image properties
    width = img1.size().width;
    height = img1.size().height;
    step = img1.step;
    img1_data = (uchar*)img1.data;
    img2_data = (uchar*)img2.data;

    // initialize arrays for fftw operations
    data_in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * width * height);
    fft = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * width * height);
    ifft = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * width * height);

    // create plans
    plan_f = fftw_plan_dft_2d(height, width, data_in, fft, FFTW_FORWARD, FFTW_ESTIMATE);

    // load img1's data to fftw input
    for (i = 0, k = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            data_in[k][0] = (double)img1_data[i * step + j];
            data_in[k][1] = 0.0;
            k++;
        }
    }

    // perform FFT
    fftw_execute(plan_f);

    // normalize IFFT result
    for (i = 0; i < (width * height); i++) {
        ifft[i][0] = sqrt(sqrt(fft[i][0] * fft[i][0] + fft[i][1] * fft[i][1]));
    }

    // copy IFFT result to img2's data
    for (i = 0, k = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            img2_data[i * step + j] = (uchar)ifft[k++][0];
        }
    }

    int cx = img2.cols / 2;
    int cy = img2.rows / 2;

    cv::Mat q0(img2, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(img2, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(img2, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(img2, cv::Rect(cx, cy, cx, cy));

    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
    normalize(img2, img2, 0, 255, CV_MINMAX);

    cv::Mat ucharMagImg;

    cv::threshold(img2, ucharMagImg, 100, 255, CV_THRESH_BINARY);

    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(ucharMagImg, lines, 1, CV_PI / 180, 65, 40, 20);

    for (unsigned int i = 0; i < lines.size(); i++) {
        float S, d;
        S = triagleArea(cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Point(cx, cy));
        d = EuclidDistance(cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]));
        float h = S / d;

        if (h > 2) {
            lines.erase(lines.begin() + i, lines.begin() + i + 1);
            i--;
        }
    }

    std::vector<float> atanLines;
    for (unsigned int i = 0; i < lines.size(); i++) {
        if (lines[i][0] == lines[i][2]) {
            atanLines.push_back(PI / 2);
        } else {
            float dx, dy;
            if (lines[i][0] < lines[i][2]) {
                dx = (float)(lines[i][2] - lines[i][0]);
                dy = (float)(lines[i][1] - lines[i][3]);
            } else {
                dx = (float)(lines[i][0] - lines[i][2]);
                dy = (float)(lines[i][3] - lines[i][1]);
            }

            float tanlines;
            tanlines = dy / dx;
            atanLines.push_back(atan(tanlines));
        }
    }

    for (unsigned int i = 0; i < lines.size(); i++) {
        if (atanLines[i] < PI / 6 && atanLines[i] > -PI / 6) {
            lines[i] *= 2;
        } else if (atanLines[i] < PI / 4 && atanLines[i] > -PI / 4) {
            lines[i] *= 1.5;
        } else if (atanLines[i] < PI / 3 && atanLines[i] > -PI / 3) {
            lines[i] *= 1.3;
        }
    }

    std::vector<std::vector<cv::Vec4i>> linesbyCluster;
    std::vector<std::vector<float>> atanbyCluster;

    if (lines.size() > 0) {
        std::vector<cv::Vec4i> firstLineCluster;
        firstLineCluster.push_back(lines[0]);
        linesbyCluster.push_back(firstLineCluster);

        std::vector<float> firstAtanCluster;
        firstAtanCluster.push_back(atanLines[0]);
        atanbyCluster.push_back(firstAtanCluster);

        for (unsigned int i = 1; i < lines.size(); i++) {
            int indexCluster = -1;
            float angleFromLineToCluster = 0.3f;

            for (unsigned int j = 0; j < linesbyCluster.size(); j++) {
                for (unsigned int k = 0; k < linesbyCluster[j].size(); k++) {
                    float angleTemp = MIN(abs(atanLines[i] - atanbyCluster[j][k]), PI - abs(atanLines[i] - atanbyCluster[j][k]));

                    if (angleTemp < angleFromLineToCluster) {
                        angleFromLineToCluster = angleTemp;
                        indexCluster = j;
                    }
                }
            }

            if (indexCluster == -1) {
                std::vector<cv::Vec4i> lineClusterTemp;
                lineClusterTemp.push_back(lines[i]);
                linesbyCluster.push_back(lineClusterTemp);

                std::vector<float> atanClusterTemp;
                atanClusterTemp.push_back(atanLines[i]);
                atanbyCluster.push_back(atanClusterTemp);
            } else {
                linesbyCluster[indexCluster].push_back(lines[i]);
                atanbyCluster[indexCluster].push_back(atanLines[i]);
            }
        }
    }

    std::vector<cv::Vec4i> centerLines;

    for (unsigned int i = 0; i < linesbyCluster.size(); i++) {
        int p1x = 0;
        int p1y = 0;
        int p2x = 0;
        int p2y = 0;
        for (unsigned int j = 0; j < linesbyCluster[i].size(); j++) {
            if (linesbyCluster[i][j][0] <= linesbyCluster[i][j][2]) {
                p1x += linesbyCluster[i][j][0];
                p1y += linesbyCluster[i][j][1];
                p2x += linesbyCluster[i][j][2];
                p2y += linesbyCluster[i][j][3];
            } else {
                p1x += linesbyCluster[i][j][2];
                p1y += linesbyCluster[i][j][3];
                p2x += linesbyCluster[i][j][0];
                p2y += linesbyCluster[i][j][1];
            }
        }

        p1x /= linesbyCluster[i].size();
        p1y /= linesbyCluster[i].size();
        p2x /= linesbyCluster[i].size();
        p2y /= linesbyCluster[i].size();

        cv::Vec4i lineTemp;
        lineTemp[0] = p1x;
        lineTemp[1] = p1y;
        lineTemp[2] = p2x;
        lineTemp[3] = p2y;
        centerLines.push_back(lineTemp);
    }

    std::vector<int> centerDistance;

    for (unsigned int i = 0; i < centerLines.size(); i++) {
        for (unsigned int j = i + 1; j < centerLines.size(); j++) {
            float d1 = EuclidDistance(cv::Point(centerLines[i][0], centerLines[i][1]), cv::Point(centerLines[i][2], centerLines[i][3]));
            float d2 = EuclidDistance(cv::Point(centerLines[j][0], centerLines[j][1]), cv::Point(centerLines[j][2], centerLines[j][3]));

            if (d1 < d2) {
                std::swap(centerLines[i], centerLines[j]);
            }
        }
    }

    for (unsigned int i = 0; i < centerLines.size(); i++) {
        if (centerLines[i][0] == centerLines[i][2]) {
            centerAngles.push_back(PI / 2);
        } else {
            float dx, dy;
            if (centerLines[i][0] < centerLines[i][2]) {
                dx = (float)(centerLines[i][2] - centerLines[i][0]);
                dy = (float)(centerLines[i][1] - centerLines[i][3]);
            } else {
                dx = (float)(centerLines[i][0] - centerLines[i][2]);
                dy = (float)(centerLines[i][3] - centerLines[i][1]);
            }

            float tanlines;
            tanlines = dy / dx;
            centerAngles.push_back(atan(tanlines));
        }
    }
    return centerLines;
}
