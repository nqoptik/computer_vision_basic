/**
 * @file hough_line.cpp
 * @author Nguyen Quang <nguyenquang.emailbox@gmail.com>
 * @brief The Hough line transform for line detection.
 * @since 0.0.1
 * 
 * @copyright Copyright (c) 2015, Nguyen Quang, all rights reserved.
 * 
 */

#include <cmath>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>

class HoughLine
{
private:
    int no_angle;
    int p_max;
    int hough;
    float step_angle;
    cv::Mat all_lines;

public:
    HoughLine();
    ~HoughLine();
    void detect(cv::Mat img);
};

HoughLine::HoughLine()
{
    no_angle = 360;
    p_max = 414;
    step_angle = M_PI / no_angle;
    all_lines = cv::Mat::zeros(cv::Size(p_max, no_angle), CV_8UC1);
}

HoughLine::~HoughLine()
{
}

void HoughLine::detect(cv::Mat img)
{
    cv::threshold(img, img, 200, 255, CV_THRESH_BINARY);
    for (int r = 0; r < img.rows; r++)
    {
        for (int c = 0; c < img.cols; c++)
        {
            if (img.data[r * img.cols + c] == 255)
            {
                for (int i = 0; i < no_angle; i++)
                {
                    float angle = i * step_angle;
                    int p = std::floor(r * cos(angle) + c * sin(angle));
                    if (p > 0)
                    {
                        all_lines.data[i * all_lines.cols + p]++;
                    }
                }
            }
        }
    }
    cv::imshow("threshold", all_lines);
    cv::imshow("img", img);
    cv::waitKey();
}

/**
 * @brief The main function.
 * 
 * @param[in] argc The argument count.
 * @param[in] argv The argument vector.
 * @return The status value.
 * @since 0.0.1
 */
int main(int argc, char** argv)
{
    if (argc != 2)
    {
        printf("To run hough line detection, type ./hough_line <image_file>\n");
        return 1;
    }
    cv::Mat img = cv::imread(argv[1], 0);
    HoughLine hough_line;
    hough_line.detect(img);
    return 0;
}
