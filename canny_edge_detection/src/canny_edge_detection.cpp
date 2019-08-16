#include <cmath>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

int const Th = 110;
int const T1 = 80;

Mat gaussFilter(Mat image)
{
    Mat gauImg = image.clone();
    for (int i = 2; i < image.rows - 2; i++)
    {
        for (int j = 2; j < image.cols - 2; j++)
        {
            int gau0 = 2 * image.at<uchar>(i - 2, j - 2) + 4 * image.at<uchar>(i - 2, j - 1) + 5 * image.at<uchar>(i - 2, j) + 4 * image.at<uchar>(i - 2, j + 1) + 2 * image.at<uchar>(i - 2, j + 2);
            int gau1 = 4 * image.at<uchar>(i - 1, j - 2) + 9 * image.at<uchar>(i - 1, j - 1) + 12 * image.at<uchar>(i - 1, j) + 9 * image.at<uchar>(i - 1, j + 1) + 4 * image.at<uchar>(i - 1, j + 2);
            int gau2 = 5 * image.at<uchar>(i, j - 2) + 12 * image.at<uchar>(i, j - 1) + 15 * image.at<uchar>(i, j) + 12 * image.at<uchar>(i, j + 1) + 5 * image.at<uchar>(i, j + 2);
            int gau3 = 4 * image.at<uchar>(i + 1, j - 2) + 9 * image.at<uchar>(i + 1, j - 1) + 12 * image.at<uchar>(i + 1, j) + 9 * image.at<uchar>(i + 1, j + 1) + 4 * image.at<uchar>(i + 1, j + 2);
            int gau4 = 2 * image.at<uchar>(i + 2, j - 2) + 4 * image.at<uchar>(i + 2, j - 1) + 5 * image.at<uchar>(i + 2, j) + 4 * image.at<uchar>(i + 2, j + 1) + 2 * image.at<uchar>(i + 2, j + 2);
            int gau = (gau0 + gau1 + gau2 + gau3 + gau4) / 159;
            gauImg.at<uchar>(i, j) = gau;
        }
    }
    return gauImg;
}

int main()
{
    Mat orgImg = imread("hanoi.png", 0);
    Mat gauImg = gaussFilter(orgImg);

    Mat aftImg = Mat::zeros(gauImg.rows, gauImg.cols, CV_8UC1);
    for (int i = 1; i < aftImg.rows - 1; i++)
    {
        for (int j = 1; j < aftImg.cols - 1; j++)
        {
            int Gx = gauImg.at<uchar>(i + 1, j - 1) + 2 * gauImg.at<uchar>(i + 1, j) + gauImg.at<uchar>(i + 1, j + 1) - gauImg.at<uchar>(i - 1, j - 1) - 2 * gauImg.at<uchar>(i - 1, j) - gauImg.at<uchar>(i - 1, j + 1);
            int Gy = gauImg.at<uchar>(i - 1, j + 1) + 2 * gauImg.at<uchar>(i, j + 1) + gauImg.at<uchar>(i + 1, j + 1) - gauImg.at<uchar>(i - 1, j - 1) - 2 * gauImg.at<uchar>(i, j - 1) - gauImg.at<uchar>(i + 1, j - 1);
            aftImg.at<uchar>(i, j) = (int)(sqrtf((float)(Gx * Gx) + (float)(Gy * Gy)));
        }
    }

    for (int i = 1; i < aftImg.rows - 1; i++)
    {
        for (int j = 1; j < aftImg.cols - 1; j++)
        {
            int Gx = gauImg.at<uchar>(i + 1, j - 1) + 2 * gauImg.at<uchar>(i + 1, j) + gauImg.at<uchar>(i + 1, j + 1) - gauImg.at<uchar>(i - 1, j - 1) - 2 * gauImg.at<uchar>(i - 1, j) - gauImg.at<uchar>(i - 1, j + 1);
            int Gy = gauImg.at<uchar>(i - 1, j + 1) + 2 * gauImg.at<uchar>(i, j + 1) + gauImg.at<uchar>(i + 1, j + 1) - gauImg.at<uchar>(i - 1, j - 1) - 2 * gauImg.at<uchar>(i, j - 1) - gauImg.at<uchar>(i + 1, j - 1);
            if (Gx == 0)
            {
                if (aftImg.at<uchar>(i, j) < aftImg.at<uchar>(i, j - 1) || aftImg.at<uchar>(i, j) < aftImg.at<uchar>(i, j + 1))
                {
                    aftImg.at<uchar>(i, j) = 0;
                }
            }
            else
            {
                if (atanf((float)Gy / (float)Gx) >= 3 * M_PI / 8 || atanf((float)Gy / (float)Gx) <= -3 * M_PI / 8)
                {
                    if (aftImg.at<uchar>(i, j) < aftImg.at<uchar>(i, j - 1) || aftImg.at<uchar>(i, j) < aftImg.at<uchar>(i, j + 1))
                    {
                        aftImg.at<uchar>(i, j) = 0;
                    }
                }

                else if (atanf((float)Gy / (float)Gx) >= -M_PI / 8 || atanf((float)Gy / (float)Gx) <= M_PI / 8)
                {
                    if (aftImg.at<uchar>(i, j) < aftImg.at<uchar>(i - 1, j) || aftImg.at<uchar>(i, j) < aftImg.at<uchar>(i - 1, j))
                    {
                        aftImg.at<uchar>(i, j) = 0;
                    }
                }

                else if (atanf((float)Gy / (float)Gx) > -3 * M_PI / 8 && atanf((float)Gy / (float)Gx) < -M_PI / 8)
                {
                    if (aftImg.at<uchar>(i, j) < aftImg.at<uchar>(i - 1, j + 1) || aftImg.at<uchar>(i, j) < aftImg.at<uchar>(i - 1, j - 1))
                    {
                        aftImg.at<uchar>(i, j) = 0;
                    }
                }

                else if (atanf((float)Gy / (float)Gx) > M_PI / 8 || atanf((float)Gy / (float)Gx) < 3 * M_PI / 8)
                {
                    if (aftImg.at<uchar>(i, j) < aftImg.at<uchar>(i - 1, j - 1) || aftImg.at<uchar>(i, j) < aftImg.at<uchar>(i - 1, j + 1))
                    {
                        aftImg.at<uchar>(i, j) = 0;
                    }
                }
            }
        }
    }

    for (int i = 1; i < aftImg.rows - 1; i++)
    {
        for (int j = 1; j < aftImg.cols - 1; j++)
        {
            if (aftImg.at<uchar>(i, j) > Th)
            {
                aftImg.at<uchar>(i, j) = 255;
            }
            else if ((aftImg.at<uchar>(i, j) < T1))
            {
                aftImg.at<uchar>(i, j) = 0;
            }
            else
            {
                if (aftImg.at<uchar>(i - 1, j - 1) < Th && aftImg.at<uchar>(i - 1, j) < Th && aftImg.at<uchar>(i - 1, j + 1) < Th && aftImg.at<uchar>(i, j - 1) < Th && aftImg.at<uchar>(i, j + 1) < Th && aftImg.at<uchar>(i + 1, j - 1) < Th && aftImg.at<uchar>(i + 1, j) < Th && aftImg.at<uchar>(i + 1, j + 1) < Th)
                {
                    aftImg.at<uchar>(i, j) = 0;
                }
            }
        }
    }

    imshow("Original Image", orgImg);
    imshow("Gauss Image", gauImg);
    imshow("After Image", aftImg);
    waitKey(0);
    return 0;
}
