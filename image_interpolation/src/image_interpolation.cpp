/**
 * @file sudoku_detection.cpp
 * @author Nguyen Quang <nguyenquang.emailbox@gmail.com>
 * @brief The image interpolation implementation.
 * @since 0.0.1
 * 
 * @copyright Copyright (c) 2015, Nguyen Quang, all rights reserved.
 * 
 */

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

struct Pixel
{
    float column;
    float row;
    Pixel(const float column = 0, const float row = 0)
        : column(column), row(row)
    {
    }
};

void resize_image(const cv::Mat& source,
                  cv::Mat& destination,
                  const cv::Size& size)
{
    std::vector<Pixel> pixel_map;
    size_t map_size = size.width * size.height;
    pixel_map.reserve(map_size);
    for (size_t i = 0; i < map_size; ++i)
    {
        int column_new = i % size.width;
        int row_new = i / size.width;
        float column = (float)(source.cols - 1) / (size.width - 1) * column_new;
        float row = (float)(source.rows - 1) / (size.height - 1) * row_new;
        pixel_map.emplace_back(Pixel(column, row));
    }

    destination.create(size, CV_8UC1);
    for (int destination_column = 0; destination_column < destination.cols; ++destination_column)
    {
        for (int destination_row = 0; destination_row < destination.rows; ++destination_row)
        {
            int destination_index = destination_row * destination.cols + destination_column;
            int source_column = std::round(pixel_map[destination_index].column);
            int source_row = std::round(pixel_map[destination_index].row);
            int source_index = source_row * source.cols + source_column;
            destination.data[destination_index] = source.data[source_index];
        }
    }
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        printf("To run the image interpolation, type ./image_interpolation <image_file>\n");
        return 1;
    }
    cv::Mat image = cv::imread(argv[1], 0);
    if (image.empty())
    {
        printf("The input image is empty.\n");
        return 1;
    }

    cv::Mat resized_image;
    resize_image(image, resized_image, cv::Size(image.cols * 1.7, image.rows * 1.7));

    cv::imshow("image", image);
    cv::imshow("resized_image", resized_image);
    cv::waitKey(0);
    return 0;
}
