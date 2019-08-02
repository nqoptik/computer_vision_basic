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

enum Method : uchar
{
    NEAREST = 0,
    BILINEAR = 1,
    BICUBIC = 2
};

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
                  const cv::Size& size, Method method = BILINEAR)
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
    if (method == NEAREST)
    {
        for (int destination_row = 0; destination_row < destination.rows; ++destination_row)
        {
            for (int destination_column = 0; destination_column < destination.cols; ++destination_column)
            {
                int destination_index = destination_row * destination.cols + destination_column;
                int source_row = std::round(pixel_map[destination_index].row);
                int source_column = std::round(pixel_map[destination_index].column);
                int source_index = source_row * source.cols + source_column;
                destination.data[destination_index] = source.data[source_index];
            }
        }
    }
    else if (method == BILINEAR)
    {
        for (int destination_row = 0; destination_row < destination.rows; ++destination_row)
        {
            for (int destination_column = 0; destination_column < destination.cols; ++destination_column)
            {
                int destination_index = destination_row * destination.cols + destination_column;
                float source_row = pixel_map[destination_index].row;
                float source_column = pixel_map[destination_index].column;
                int source_top = std::floor(source_row);
                int source_bottom = std::ceil(source_row);
                int source_left = std::floor(source_column);
                int source_right = std::ceil(source_column);
                int source_top_left_index = source_top * source.cols + source_left;
                int source_top_right_index = source_top * source.cols + source_right;
                int source_bottom_left_index = source_bottom * source.cols + source_left;
                int source_bottom_right_index = source_bottom * source.cols + source_right;
                if (source_top == source_bottom)
                {
                    if (source_bottom < destination.rows - 1)
                    {
                        ++source_bottom;
                    }
                    else
                    {
                        --source_top;
                    }
                }
                if (source_left == source_right)
                {
                    if (source_right < destination.cols - 1)
                    {
                        ++source_right;
                    }
                    else
                    {
                        --source_left;
                    }
                }

                destination.data[destination_index] = (source_bottom - source_row) * ((source_column - source_left) * source.data[source_top_right_index] + (source_right - source_column) * source.data[source_top_left_index]) +
                                                      (source_row - source_top) * ((source_column - source_left) * source.data[source_bottom_right_index] + (source_right - source_column) * source.data[source_bottom_left_index]);
            }
        }
    }
    else if (method == BICUBIC)
    {
    }
    else
    {
        printf("Invalid method!\n");
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

    float ratio = 2.3;
    cv::Mat resized_image_nearest;
    resize_image(image, resized_image_nearest, cv::Size(image.cols * ratio, image.rows * ratio), NEAREST);

    cv::Mat resized_image_bilinear;
    resize_image(image, resized_image_bilinear, cv::Size(image.cols * ratio, image.rows * ratio), BILINEAR);

    cv::imshow("image", image);
    cv::imshow("resized_image_nearest", resized_image_nearest);
    cv::imshow("resized_image_bilinear", resized_image_bilinear);
    cv::waitKey(0);
    return 0;
}
