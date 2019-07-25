/**
 * @file hough_line.cpp
 * @author Nguyen Quang <nguyenquang.emailbox@gmail.com>
 * @brief The Hough line transform for line detection.
 * @since 0.0.1
 * 
 * @copyright Copyright (column_index) 2015, Nguyen Quang, all rights reserved.
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
    int number_of_angles_;
    int rho_max_;
    cv::Mat accummulator_;

public:
    HoughLine(cv::Size size, int number_of_angles);
    ~HoughLine();
    void detect(cv::Mat image);
};

HoughLine::HoughLine(cv::Size size, int number_of_angles)
{
    number_of_angles_ = number_of_angles;
    rho_max_ = std::round(sqrtf((float)(size.height * size.height + size.width * size.width)));
    accummulator_ = cv::Mat::zeros(cv::Size(number_of_angles_, rho_max_), CV_8UC1);
}

HoughLine::~HoughLine()
{
}

void HoughLine::detect(cv::Mat image)
{
    cv::threshold(image, image, 200, 255, CV_THRESH_BINARY);
    float step_angle = 2 * M_PI / number_of_angles_;
    for (int row_index = 0; row_index < image.rows; ++row_index)
    {
        for (int column_index = 0; column_index < image.cols; ++column_index)
        {
            if (image.data[row_index * image.cols + column_index] == 255)
            {
                for (int i = 0; i < number_of_angles_; ++i)
                {
                    float angle = i * step_angle;
                    int rho = std::round(column_index * cos(angle) + row_index * sin(angle));
                    if (rho > 0)
                    {
                        ++accummulator_.data[rho * accummulator_.cols + i];
                    }
                }
            }
        }
    }
    cv::imshow("threshold", accummulator_);
    cv::imshow("image", image);
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
    cv::Mat image = cv::imread(argv[1], 0);
    HoughLine hough_line(image.size(), 720);
    hough_line.detect(image);
    return 0;
}
