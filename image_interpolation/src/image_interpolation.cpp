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

int get_cubic(int point_0, int point_1, int point_2, int point_3, float x)
{
    float a = -0.5 * point_0 + 1.5 * point_1 - 1.5 * point_2 + 0.5 * point_3;
    float b = point_0 - 2.5 * point_1 + 2 * point_2 - 0.5 * point_3;
    float c = -0.5 * point_0 + 0.5 * point_2;
    float d = point_1;
    int value = d + x * (c + x * (b + x * a));
    if (value < 0)
    {
        value = 0;
    }
    if (value > 255)
    {
        value = 255;
    }
    return value;
}

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
                if (source_top > source.cols - 2)
                {
                    source_top = source.cols - 2;
                }
                int source_bottom = source_top + 1;
                int source_left = std::floor(source_column);
                if (source_left > source.cols - 2)
                {
                    source_left = source.cols - 2;
                }
                int source_right = source_left + 1;
                int source_top_left_index = source_top * source.cols + source_left;
                int source_top_right_index = source_top * source.cols + source_right;
                int source_bottom_left_index = source_bottom * source.cols + source_left;
                int source_bottom_right_index = source_bottom * source.cols + source_right;

                destination.data[destination_index] = (source_bottom - source_row) * ((source_column - source_left) * source.data[source_top_right_index] + (source_right - source_column) * source.data[source_top_left_index]) +
                                                      (source_row - source_top) * ((source_column - source_left) * source.data[source_bottom_right_index] + (source_right - source_column) * source.data[source_bottom_left_index]);
            }
        }
    }
    else
    {
        for (int destination_row = 0; destination_row < destination.rows; ++destination_row)
        {
            for (int destination_column = 0; destination_column < destination.cols; ++destination_column)
            {
                int destination_index = destination_row * destination.cols + destination_column;
                float source_row = pixel_map[destination_index].row;
                float source_column = pixel_map[destination_index].column;
                int source_x_0 = std::floor(source_column) - 1;
                if (source_x_0 < 0)
                {
                    source_x_0 = 0;
                }
                if (source_x_0 > source.cols - 4)
                {
                    source_x_0 = source.cols - 4;
                }
                int source_x_1 = source_x_0 + 1;
                int source_x_2 = source_x_0 + 2;
                int source_x_3 = source_x_0 + 3;
                int source_y_0 = std::floor(source_row) - 1;
                if (source_y_0 < 0)
                {
                    source_y_0 = 0;
                }
                if (source_y_0 > source.rows - 4)
                {
                    source_y_0 = source.rows - 4;
                }
                int source_y_1 = source_y_0 + 1;
                int source_y_2 = source_y_0 + 2;
                int source_y_3 = source_y_0 + 3;
                int source_x_y_0 = get_cubic(source.data[source_y_0 * source.cols + source_x_0], source.data[source_y_0 * source.cols + source_x_1], source.data[source_y_0 * source.cols + source_x_2], source.data[source_y_0 * source.cols + source_x_3], source_column - std::floor(source_column));
                int source_x_y_1 = get_cubic(source.data[source_y_1 * source.cols + source_x_0], source.data[source_y_1 * source.cols + source_x_1], source.data[source_y_1 * source.cols + source_x_2], source.data[source_y_1 * source.cols + source_x_3], source_column - std::floor(source_column));
                int source_x_y_2 = get_cubic(source.data[source_y_2 * source.cols + source_x_0], source.data[source_y_2 * source.cols + source_x_1], source.data[source_y_2 * source.cols + source_x_2], source.data[source_y_2 * source.cols + source_x_3], source_column - std::floor(source_column));
                int source_x_y_3 = get_cubic(source.data[source_y_3 * source.cols + source_x_0], source.data[source_y_3 * source.cols + source_x_1], source.data[source_y_3 * source.cols + source_x_2], source.data[source_y_3 * source.cols + source_x_3], source_column - std::floor(source_column));
                int source_value = get_cubic(source_x_y_0, source_x_y_1, source_x_y_2, source_x_y_3, source_row - std::floor(source_row));
                destination.data[destination_index] = (uchar)source_value;
            }
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

    float ratio = 2.3;
    cv::Mat resized_image_nearest;
    resize_image(image, resized_image_nearest, cv::Size(image.cols * ratio, image.rows * ratio), NEAREST);

    cv::Mat resized_image_bilinear;
    resize_image(image, resized_image_bilinear, cv::Size(image.cols * ratio, image.rows * ratio), BILINEAR);

    cv::Mat resized_image_bicubic;
    resize_image(image, resized_image_bicubic, cv::Size(image.cols * ratio, image.rows * ratio), BICUBIC);

    cv::imshow("image", image);
    cv::imshow("resized_image_nearest", resized_image_nearest);
    cv::imshow("resized_image_bilinear", resized_image_bilinear);
    cv::imshow("resized_image_bicubic", resized_image_bicubic);
    cv::waitKey(0);
    return 0;
}
