#include "csv_util.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>
#include <string>

float computeDistance(const std::vector<float>& feature1, const std::vector<float>& feature2) {
    if (feature1.size() != feature2.size()) {
        std::cerr << "Error: Feature vectors must have the same length!" << std::endl;
        return -1;
    }

    float sumSquaredDifferences = 0;
    for (size_t i = 0; i < feature1.size(); i++) {
        float diff = feature1[i] - feature2[i];
        sumSquaredDifferences += diff * diff;
    }

    return sumSquaredDifferences;
}

// Function to compute 2D histogram
std::vector<float> computeHistogram(const cv::Mat& img, int bins) {
    std::vector<float> hist(bins * bins, 0);

    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);
            int sum = pixel[0] + pixel[1] + pixel[2];

            if (sum == 0) continue; // Skip this pixel if the sum is 0 to avoid division by zero

            float r = pixel[2] / (float)sum;
            float g = pixel[1] / (float)sum;

            int binR = std::min((int)(r * bins), bins - 1);
            int binG = std::min((int)(g * bins), bins - 1);

            hist[binR * bins + binG] += 1;
        }
    }

    // Normalize the histogram
    float sum = 0;
    for (const auto& val : hist) sum += val;
    for (auto& val : hist) val /= sum;

    return hist;
}


// Function to compute histogram intersection
float histogramIntersection(const std::vector<float>& h1, const std::vector<float>& h2) {
    float intersection = 0;
    for (size_t i = 0; i < h1.size(); i++) {
        intersection += std::min(h1[i], h2[i]);
    }
    return intersection;
}

// Function to replace all occurrences of a substring
std::string replaceAll(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();
    }
    return str;
}

int main() {
    char filename[] = "C:/Users/Shi Zhang/My Drive/CS/NEU Align/Courses/2023 Fall/5330/Project02/img_database.csv";
    std::string baseline_target_image_path = "C:/Users/Shi Zhang/My Drive/CS/NEU Align/Courses/2023 Fall/5330/Project02/olympus/pic.1016.jpg";
    std::string histogram_target_image_path = "C:/Users/Shi Zhang/My Drive/CS/NEU Align/Courses/2023 Fall/5330/Project02/olympus/pic.0164.jpg";
    int N = 3;


    /*Task 1 Baseline Matching*/
    cv::Mat baseline_target_img = cv::imread(baseline_target_image_path, cv::IMREAD_COLOR);
    if (baseline_target_img.empty()) {
        std::cerr << "Could not read the baseline target image: " << baseline_target_image_path << std::endl;
        return -1;
    }

    int center_x = baseline_target_img.cols / 2;
    int center_y = baseline_target_img.rows / 2;
    cv::Rect roi(center_x - 4, center_y - 4, 9, 9);
    cv::Mat feature_mat = baseline_target_img(roi).clone();

    std::vector<float> target_features;

    for (int i = 0; i < feature_mat.rows; i++) {
        for (int j = 0; j < feature_mat.cols; j++) {
            cv::Vec3b& pixel = feature_mat.at<cv::Vec3b>(i, j);
            for (int k = 0; k < 3; k++) {
                target_features.push_back(static_cast<float>(pixel[k]));
            }
        }
    }

    std::cout << "Feature vector length: " << target_features.size() << std::endl;

    std::vector<char*> filenames;
    std::vector<std::vector<float>> data;
    read_image_data_csv(filename, filenames, data);

    // iterate over all read feature vectors and print them
    /*
    * debug print for csv reading
    *   for (size_t i = 0; i < data.size(); i++) {
        std::cout << "Read feature vector for " << filenames[i] << ": ";
        for (const auto& feature : data[i]) {
            std::cout << feature << " ";
        }
        std::cout << std::endl;
    }
    */

    std::vector<std::pair<float, std::string>> distances;
    for (size_t i = 0; i < data.size(); i++) {
        std::string compare_filename = replaceAll(std::string(filenames[i]), "\\", "/");
        if (baseline_target_image_path != compare_filename) {
            float distance = computeDistance(target_features, data[i]);
            if (distance != -1)
                distances.push_back({ distance, compare_filename });
        }
    }

    std::sort(distances.begin(), distances.end());

    std::cout << "Top " << N << " Matches for " << baseline_target_image_path << " are: " << std::endl;
    for (int i = 0; i < std::min(N, (int)distances.size()); i++) {
        std::cout << distances[i].second << " with distance: " << distances[i].first << std::endl;
    }

    distances.clear();

    /*Task 2 Histogram Matching*/
    cv::Mat histogram_target_img = cv::imread(histogram_target_image_path, cv::IMREAD_COLOR);
    if (histogram_target_img.empty()) {
        std::cerr << "Could not read the histogram target image: " << histogram_target_image_path << std::endl;
        return -1;
    }

    // Compute the histogram for the target image
    std::vector<float> target_histogram = computeHistogram(histogram_target_img, 16);

    // Compute histograms for each image in the database and compute the histogram intersection
    for (size_t i = 0; i < data.size(); i++) {
        std::string compare_filename = replaceAll(std::string(filenames[i]), "\\", "/");
        if (histogram_target_image_path != compare_filename) {
            cv::Mat img = cv::imread(compare_filename, cv::IMREAD_COLOR);
            std::vector<float> histogram = computeHistogram(img, 16);

            float distance = histogramIntersection(target_histogram, histogram);
            distances.push_back({ distance, compare_filename });
        }
    }

    // Sort the distances in descending order as we are using histogram intersection
    std::sort(distances.begin(), distances.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
        });

    std::cout << "Top " << N << " Histogram Matches for " << histogram_target_image_path << " are: " << std::endl;
    for (int i = 0; i < std::min(N, (int)distances.size()); i++) {
        std::cout << distances[i].second << " with histogram intersection: " << distances[i].first << std::endl;
    }

    return 0;
}
