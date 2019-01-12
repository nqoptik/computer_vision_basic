#include <iostream>
#include <dirent.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

std::vector<std::vector<cv::Point> > foreGround;

void CallBackFunc(int event, int x, int y, int flags, void* userdata);

int main() {
    std::string dirName = "input_images/";
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

    cv::Mat image, appliedImg;
    std::sort(img_paths.begin(), img_paths.end());
    for (size_t idx = 0; idx < img_paths.size(); idx++) {
        std::cout << img_paths[idx] << std::endl;
        image = cv::imread(img_paths[idx], 1);
        if (image.empty()) {
            break;
        }

        appliedImg = image.clone();
        foreGround.erase(foreGround.begin(), foreGround.end());

        cv::namedWindow("appliedImg", 1);
        cv::imshow("appliedImg", image);
        cv::setMouseCallback("appliedImg", CallBackFunc, NULL);
        cv::waitKey();

        cv::Rect rectangle;

        cv::Mat1b result(image.rows, image.cols);  // segmentation result (4 possible values)
        result.setTo(cv::GC_PR_BGD);

        for (unsigned int i = 0; i < foreGround.size(); i++) {
            for (unsigned int j = 0; j < foreGround[i].size(); j++) {
                line(result, foreGround[i][j], foreGround[i][(j + 1) % foreGround[i].size()], cv::GC_PR_FGD, 10);
            }
        }

        cv::Mat bgModel, fgModel;  // the models (internally used)

        // GrabCut segmentation
        cv::grabCut(image,                   // input image
                    result,                  // segmentation result
                    rectangle,               // rectangle containing foreground
                    bgModel, fgModel,        // models
                    1,                       // number of iterations
                    cv::GC_INIT_WITH_MASK);  // use rectangle

        // Get the pixels marked as likely foreground
        compare(result, cv::GC_PR_FGD, result, cv::CMP_EQ);

        // Generate output image
        cv::Mat foreground(image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
        image.copyTo(foreground, result);  // bg pixels not copied

        // draw rectangle on original image
        cv::imshow("Segmented Image", foreground);
        cv::waitKey();
    }

    return 0;
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
    if (event == cv::EVENT_LBUTTONDOWN) {
        std::vector<cv::Point> temp;
        temp.push_back(cv::Point(x, y));
        foreGround.push_back(temp);
    } else if (event == cv::EVENT_MOUSEMOVE && flags == cv::EVENT_FLAG_LBUTTON && foreGround.size() > 0) {
        foreGround[foreGround.size() - 1].push_back(cv::Point(x, y));
    }
}
