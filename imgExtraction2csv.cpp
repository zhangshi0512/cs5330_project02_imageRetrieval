// imgExtraction2csv.cpp
// This program extracts features from images in a specified directory and saves these features 
// along with image paths into a CSV file for further analysis.

// CS 5330 Computer Vision
// Fall 2023 Dec 9th
// Author: Shi Zhang

// Import statements
#include "csv_util.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

int main() {
    // Set the directory path and output CSV file name
    std::string directory_path = "C:/Users/Shi Zhang/My Drive/CS/NEU Align/Courses/2023 Fall/5330/Project02/olympus";
    char filename[] = "C:/Users/Shi Zhang/My Drive/CS/NEU Align/Courses/2023 Fall/5330/Project02/img_database.csv";

    // Delete the existing file to avoid appending to old data
    if (fs::exists(filename)) {
        std::remove(filename); // Delete the file if it exists
    }

    // Iterate through each file in the directory
    for (const auto& entry : fs::directory_iterator(directory_path)) {
        std::string image_path = entry.path().string();
        std::string extension = entry.path().extension().string();

        // Skip files that are not images
        if (extension != ".jpg" && extension != ".jpeg" && extension != ".png") {
            std::cerr << "Skipping non-image file: " << image_path << std::endl;
            continue;
        }

        // Read the image from the path
        cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "Could not read the image: " << image_path << std::endl;
            continue;
        }

        // Define a region of interest at the center of the image
        int center_x = img.cols / 2;
        int center_y = img.rows / 2;
        cv::Rect roi(center_x - 4, center_y - 4, 9, 9);
        cv::Mat feature_mat = img(roi).clone();

        std::vector<float> features;

        // Extract pixel values from the region of interest
        for (int i = 0; i < feature_mat.rows; i++) {
            for (int j = 0; j < feature_mat.cols; j++) {
                cv::Vec3b pixel = feature_mat.at<cv::Vec3b>(i, j); // Assuming a 3-channel (BGR) image
                for (int k = 0; k < 3; k++) {
                    features.push_back(static_cast<float>(pixel[k]));
                }
            }
        }

        // Write the extracted features and image path to the CSV file
        append_image_data_csv(filename, const_cast<char*>(image_path.c_str()), features);
        features.clear(); // Clear the feature vector for the next image
    }

    return 0;
}
