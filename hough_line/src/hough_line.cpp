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

/**
 * @brief A struct to store the data of a line in the Polar coordinate system.
 * 
 * @since 0.0.1
 * 
 */
struct Line
{
    float rho;
    float theta;
};

/**
 * @brief A class to detect lines in an image using the Hough transform.
 * 
 * @since 0.0.1
 * 
 */
class HoughLine
{
private:
    float delta_theta_;
    int accumulator_threshold_;

public:
    /**
     * @brief Construct a new HoughLine object.
     * 
     * @param[in] delta_theta The angle resolution of the accumulator in radians.
     * @param[in] accumulator_threshold The accumulator threshold parameter, only those lines get enough votes will be returned.
     * @since 0.0.1
     */
    HoughLine(const float& delta_theta, const int& accumulator_threshold);

    /**
     * @brief Destroy the HoughLine object.
     * 
     * @since 0.0.1
     * 
     */
    ~HoughLine();

    /**
     * @brief Detect lines in the given image.
     * 
     * @param[in] image The input image.
     * @return The vector of lines detected.
     * @since 0.0.1
     */
    std::vector<Line> detect_lines(const cv::Mat& image) const;
};

HoughLine::HoughLine(const float& delta_theta, const int& accumulator_threshold)
    : delta_theta_(delta_theta),
      accumulator_threshold_(accumulator_threshold)
{
}

HoughLine::~HoughLine()
{
}

std::vector<Line> HoughLine::detect_lines(const cv::Mat& image) const
{
    int theta_index_max = std::round(2 * M_PI / delta_theta_);
    int rho_index_max = std::round(sqrtf((float)(image.rows * image.rows + image.cols * image.cols)));
    cv::Mat accumulator = cv::Mat::zeros(cv::Size(theta_index_max, rho_index_max), CV_16UC1);
    ushort* accumulator_data = (ushort*)accumulator.data;

    // Run the accumulator
    for (int row_index = 0; row_index < image.rows; ++row_index)
    {
        for (int column_index = 0; column_index < image.cols; ++column_index)
        {
            // The edge pixels are black
            if (image.data[row_index * image.cols + column_index] == 0)
            {
                for (float theta = 0; theta < 2 * M_PI; theta += delta_theta_)
                {
                    int rho = std::round(column_index * cos(theta) + row_index * sin(theta));
                    if (rho >= 0)
                    {
                        int theta_index = std::round(theta / delta_theta_);
                        ++accumulator_data[rho * accumulator.cols + theta_index];
                    }
                }
            }
        }
    }

    // Run the peak selection algorithm
    std::vector<Line> lines;
    while (true)
    {
        // Find the maximum cell in the accumulator
        double max_cell_value;
        cv::Point max_cell_location;
        cv::minMaxLoc(accumulator, nullptr, &max_cell_value, nullptr, &max_cell_location);
        if ((int)max_cell_value < accumulator_threshold_)
        {
            break;
        }

        // Zero out the area around the maximum cell that belongs to the same line
        for (int row_index = std::max(max_cell_location.y - 10, 0); row_index < std::min(max_cell_location.y + 10, accumulator.rows); ++row_index)
        {
            for (int column_index = std::max(max_cell_location.x - 10, 0); column_index < std::min(max_cell_location.x + 10, accumulator.cols); ++column_index)
            {
                accumulator_data[row_index * accumulator.cols + column_index] = 0;
            }
        }
        float rho = max_cell_location.y;
        float theta = max_cell_location.x * delta_theta_;
        lines.push_back(Line{rho, theta});
    }
    return lines;
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
        printf("To run the Hough line detection, type ./hough_line <image_file>\n");
        return 1;
    }
    cv::Mat image = cv::imread(argv[1], 0);
    if (image.empty())
    {
        printf("The input image is empty.\n");
        return 1;
    }
    cv::threshold(image, image, 200, 255, CV_THRESH_BINARY);

    // Apply the Hough line detection
    HoughLine hough_line(M_PI / 180, 100);
    std::vector<Line> lines = hough_line.detect_lines(image);
    for (size_t i = 0; i < lines.size(); ++i)
    {
        std::cout << "(rho, theta) = (" << lines[i].rho << ", " << lines[i].theta << ")\n";
    }
    return 0;
}
