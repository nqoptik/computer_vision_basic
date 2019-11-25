/**
 * @file sudoku_detection.cpp
 * @author Nguyen Quang <nqoptik@gmail.com>
 * @brief The image interpolation implementation.
 * @since 0.0.1
 * 
 * @copyright Copyright (c) 2015, Nguyen Quang, all rights reserved.
 * 
 */

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

/**
 * @brief The interpolation methods.
 * 
 * @since 0.0.1
 * 
 */
enum Method : uchar
{
    NEAREST = 0,
    BILINEAR = 1,
    BICUBIC = 2
};

/**
 * @brief The pixel struct.
 * 
 * @since 0.0.1
 * 
 */
struct Pixel
{
    float column;
    float row;
    Pixel(const float column = 0, const float row = 0)
        : column(column), row(row)
    {
    }
};

/**
 * @brief The interpolation fuction for 1D data.
 * 
 * @param[in] point_0 The pixel value of the first point.
 * @param[in] point_1 The pixel value of the second point.
 * @param[in] point_2 The pixel value of the third point.
 * @param[in] point_3 The pixel value of the fourth point.
 * @param[in] x The position of the interpolation point.
 * @return The interpolated value.
 * @since 0.0.1
 */
uchar get_cubic_interpolation(uchar point_0, uchar point_1, uchar point_2, uchar point_3, float x)
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

/**
 * @brief The resize image function.
 * 
 * @param[in] source The input image.
 * @param[out] destination The resized image.
 * @param[in] size The desired resolution.
 * @param[in] method The interpolation method.
 * @since 0.0.1
 */
void resize_image(const cv::Mat& source,
                  cv::Mat& destination,
                  const cv::Size& size, Method method = BILINEAR)
{
    // Calculate the map from source image to destination image
    std::vector<Pixel> pixel_map;
    size_t map_size = size.width * size.height;
    pixel_map.reserve(map_size);
    for (size_t i = 0; i < map_size; ++i)
    {
        int column_new = i % size.width;
        int row_new = i / size.width;
        float column = ((float)(source.cols) / (size.width) * (column_new + 0.5)) - 0.5;
        float row = ((float)(source.rows) / (size.height) * (row_new + 0.5)) - 0.5;
        pixel_map.emplace_back(Pixel(column, row));
    }

    // Create storage for the destination image
    int number_of_channels = source.channels();
    int image_type = source.type();
    destination.create(size, image_type);

    if (method == NEAREST)
    {
        // The nearest neighbour interpolation
        for (int destination_row = 0; destination_row < destination.rows; ++destination_row)
        {
            for (int destination_column = 0; destination_column < destination.cols; ++destination_column)
            {
                int destination_index = destination_row * destination.cols + destination_column;
                int source_row = std::round(pixel_map[destination_index].row);
                int source_column = std::round(pixel_map[destination_index].column);
                int source_index = source_row * source.cols + source_column;
                for (int i = 0; i < number_of_channels; ++i)
                {
                    // Assign to the closest pixel
                    destination.data[destination_index * number_of_channels + i] = source.data[source_index * number_of_channels + i];
                }
            }
        }
    }
    else if (method == BILINEAR)
    {
        // The bilinear interpolation
        for (int destination_row = 0; destination_row < destination.rows; ++destination_row)
        {
            for (int destination_column = 0; destination_column < destination.cols; ++destination_column)
            {
                int destination_index = destination_row * destination.cols + destination_column;
                float source_row = pixel_map[destination_index].row;
                float source_column = pixel_map[destination_index].column;

                int source_top, source_bottom;
                if (std::floor(source_row) == -1)
                {
                    // The top margin
                    source_top = 0;
                    source_bottom = 0;
                }
                else if (std::floor(source_row) == source.rows - 1)
                {
                    // The bottom margin
                    source_top = source.rows - 1;
                    source_bottom = source.rows - 1;
                }
                else
                {
                    source_top = std::floor(source_row);
                    source_bottom = source_top + 1;
                }

                int source_left, source_right;
                if (std::floor(source_column) == -1)
                {
                    // The left margin
                    source_left = 0;
                    source_right = 0;
                }
                else if (std::floor(source_column) == source.cols - 1)
                {
                    // The right margin
                    source_left = source.cols - 1;
                    source_right = source.cols - 1;
                }
                else
                {
                    source_left = std::floor(source_column);
                    source_right = source_left + 1;
                }

                int source_top_left_index = source_top * source.cols + source_left;
                int source_top_right_index = source_top * source.cols + source_right;
                int source_bottom_left_index = source_bottom * source.cols + source_left;
                int source_bottom_right_index = source_bottom * source.cols + source_right;
                float d_x = source_column - source_left;
                float d_y = source_row - source_top;
                for (int i = 0; i < number_of_channels; ++i)
                {
                    // Assign to the weighted average
                    destination.data[destination_index * number_of_channels + i] = (1 - d_y) * ((1 - d_x) * source.data[source_top_left_index * number_of_channels + i] + d_x * source.data[source_top_right_index * number_of_channels + i]) +
                                                                                   d_y * ((1 - d_x) * source.data[source_bottom_left_index * number_of_channels + i] + d_x * source.data[source_bottom_right_index * number_of_channels + i]);
                }
            }
        }
    }
    else
    {
        // The bicubic interpolation
        for (int destination_row = 0; destination_row < destination.rows; ++destination_row)
        {
            for (int destination_column = 0; destination_column < destination.cols; ++destination_column)
            {
                int destination_index = destination_row * destination.cols + destination_column;
                float source_row = pixel_map[destination_index].row;
                float source_column = pixel_map[destination_index].column;

                int source_x_0, source_x_1, source_x_2, source_x_3;
                if (std::floor(source_column) == -1)
                {
                    // The top margin
                    source_x_0 = 0;
                    source_x_1 = 0;
                    source_x_2 = 0;
                    source_x_3 = 1;
                }
                else if (std::floor(source_column) == 0)
                {
                    // The second top margin
                    source_x_0 = 0;
                    source_x_1 = 0;
                    source_x_2 = 1;
                    source_x_3 = 2;
                }
                else if (std::floor(source_column) == source.cols - 2)
                {
                    // The second bottom margin
                    source_x_0 = source.cols - 3;
                    source_x_1 = source.cols - 2;
                    source_x_2 = source.cols - 1;
                    source_x_3 = source.cols - 1;
                }
                else if (std::floor(source_column) == source.cols - 1)
                {
                    // The bottom margin
                    source_x_0 = source.cols - 2;
                    source_x_1 = source.cols - 1;
                    source_x_2 = source.cols - 1;
                    source_x_3 = source.cols - 1;
                }
                else
                {
                    source_x_0 = std::floor(source_column) - 1;
                    source_x_1 = source_x_0 + 1;
                    source_x_2 = source_x_0 + 2;
                    source_x_3 = source_x_0 + 3;
                }

                int source_y_0, source_y_1, source_y_2, source_y_3;
                if (std::floor(source_row) == -1)
                {
                    // The left margin
                    source_y_0 = 0;
                    source_y_1 = 0;
                    source_y_2 = 0;
                    source_y_3 = 1;
                }
                else if (std::floor(source_row) == 0)
                {
                    // The second left margin
                    source_y_0 = 0;
                    source_y_1 = 0;
                    source_y_2 = 1;
                    source_y_3 = 2;
                }
                else if (std::floor(source_row) == source.rows - 2)
                {
                    // The second right margin
                    source_y_0 = source.rows - 3;
                    source_y_1 = source.rows - 2;
                    source_y_2 = source.rows - 1;
                    source_y_3 = source.rows - 1;
                }
                else if (std::floor(source_row) == source.rows - 1)
                {
                    // The right margin
                    source_y_0 = source.rows - 2;
                    source_y_1 = source.rows - 1;
                    source_y_2 = source.rows - 1;
                    source_y_3 = source.rows - 1;
                }
                else
                {
                    source_y_0 = std::floor(source_row) - 1;
                    source_y_1 = source_y_0 + 1;
                    source_y_2 = source_y_0 + 2;
                    source_y_3 = source_y_0 + 3;
                }
                float d_x = source_column - std::floor(source_column);
                float d_y = source_row - std::floor(source_row);
                for (int i = 0; i < number_of_channels; ++i)
                {
                    // Calculate the cubic interpolation for each row
                    uchar source_x_y_0 = get_cubic_interpolation(source.data[(source_y_0 * source.cols + source_x_0) * number_of_channels + i],
                                                                 source.data[(source_y_0 * source.cols + source_x_1) * number_of_channels + i],
                                                                 source.data[(source_y_0 * source.cols + source_x_2) * number_of_channels + i],
                                                                 source.data[(source_y_0 * source.cols + source_x_3) * number_of_channels + i],
                                                                 d_x);
                    uchar source_x_y_1 = get_cubic_interpolation(source.data[(source_y_1 * source.cols + source_x_0) * number_of_channels + i],
                                                                 source.data[(source_y_1 * source.cols + source_x_1) * number_of_channels + i],
                                                                 source.data[(source_y_1 * source.cols + source_x_2) * number_of_channels + i],
                                                                 source.data[(source_y_1 * source.cols + source_x_3) * number_of_channels + i],
                                                                 d_x);
                    uchar source_x_y_2 = get_cubic_interpolation(source.data[(source_y_2 * source.cols + source_x_0) * number_of_channels + i],
                                                                 source.data[(source_y_2 * source.cols + source_x_1) * number_of_channels + i],
                                                                 source.data[(source_y_2 * source.cols + source_x_2) * number_of_channels + i],
                                                                 source.data[(source_y_2 * source.cols + source_x_3) * number_of_channels + i],
                                                                 d_x);
                    uchar source_x_y_3 = get_cubic_interpolation(source.data[(source_y_3 * source.cols + source_x_0) * number_of_channels + i],
                                                                 source.data[(source_y_3 * source.cols + source_x_1) * number_of_channels + i],
                                                                 source.data[(source_y_3 * source.cols + source_x_2) * number_of_channels + i],
                                                                 source.data[(source_y_3 * source.cols + source_x_3) * number_of_channels + i],
                                                                 d_x);
                    // Calculate the cubic interpolation for the final column
                    uchar source_value = get_cubic_interpolation(source_x_y_0, source_x_y_1, source_x_y_2, source_x_y_3, d_y);
                    destination.data[destination_index * number_of_channels + i] = source_value;
                }
            }
        }
    }
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
    if (argc != 3)
    {
        printf("To run the image interpolation, type ./image_interpolation <image_file> <scale>\n");
        return 1;
    }

    // Read the image
    cv::Mat image = cv::imread(argv[1]);
    if (image.empty())
    {
        printf("The input image is empty.\n");
        return 1;
    }

    // Get the destination size
    std::string ratio_string = argv[2];
    float ratio = std::stoi(ratio_string);
    if (ratio <= 0)
    {
        printf("Ratio has to be greater than 0\n");
    }
    cv::Size destination_size(image.cols * ratio, image.rows * ratio);

    // Nearest neighbour interpolation
    cv::Mat resized_image_nearest;
    resize_image(image, resized_image_nearest, destination_size, NEAREST);

    // Bilinear interpolation
    cv::Mat resized_image_bilinear;
    resize_image(image, resized_image_bilinear, destination_size, BILINEAR);

    // Bicubic interpolation
    cv::Mat resized_image_bicubic;
    resize_image(image, resized_image_bicubic, destination_size, BICUBIC);

    // Display the images
    cv::imshow("image", image);
    cv::imshow("resized_image_nearest", resized_image_nearest);
    cv::imshow("resized_image_bilinear", resized_image_bilinear);
    cv::imshow("resized_image_bicubic", resized_image_bicubic);
    cv::waitKey(0);
    return 0;
}
