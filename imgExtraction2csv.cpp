#include "csv_util.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

int main() {
    std::string directory_path = "C:/Users/Shi Zhang/My Drive/CS/NEU Align/Courses/2023 Fall/5330/Project02/olympus";
    char filename[] = "C:/Users/Shi Zhang/My Drive/CS/NEU Align/Courses/2023 Fall/5330/Project02/img_database.csv";

    // Delete the existing file before writing new data
    if (fs::exists(filename)) {
        std::remove(filename); // use remove to delete the file
    }

    for (const auto& entry : fs::directory_iterator(directory_path)) {
        std::string image_path = entry.path().string();
        std::string extension = entry.path().extension().string();

        if (extension != ".jpg" && extension != ".jpeg" && extension != ".png") {
            std::cerr << "Skipping non-image file: " << image_path << std::endl;
            continue;
        }

        cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Could not read the image: " << image_path << std::endl;
            continue;
        }

        int center_x = img.cols / 2;
        int center_y = img.rows / 2;
        cv::Rect roi(center_x - 4, center_y - 4, 9, 9);
        cv::Mat feature_mat = img(roi).clone();

        std::vector<float> features;

        // double for loop for the center of the image
        for (int i = 0; i < feature_mat.rows; i++) {
            for (int j = 0; j < feature_mat.cols; j++) {
                cv::Vec3b pixel = feature_mat.at<cv::Vec3b>(i, j); // assuming the image is 3-channel (BGR)
                for (int k = 0; k < 3; k++) { // iterate over the three channels
                    features.push_back(static_cast<float>(pixel[k]));
                }
            }
        }

        /* debug print for csv writing
        *  std::cout << "Writing feature vector for " << image_path << ": ";
        for (const auto& feature : features) {
            std::cout << feature << " ";
        }
        std::cout << std::endl;
        */

        // Append to the new file or create it if it does not exist
        append_image_data_csv(filename, const_cast<char*>(image_path.c_str()), features);
        features.clear(); // Clear the features for the next image
    }

    return 0;
}
