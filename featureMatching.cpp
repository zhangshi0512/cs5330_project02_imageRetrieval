// featureMatching.cpp
// This program implements various image feature extraction and matching techniques using histograms, 
// texture analysis, and custom feature vectors, primarily utilizing OpenCV for image processing.

// CS 5330 Computer Vision
// Fall 2023 Dec 9th
// Author: Shi Zhang

// Import statements
#include "csv_util.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <numeric>

// Function to compute Euclidean distance between two feature vectors
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

// Function to compute 2D histogram for color distribution in an image
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


// Function to compute histogram intersection between two histograms
float histogramIntersection(const std::vector<float>& h1, const std::vector<float>& h2) {
    float intersection = 0;
    for (size_t i = 0; i < h1.size(); i++) {
        intersection += std::min(h1[i], h2[i]);
    }
    return intersection;
}

// Function to compute 3D histogram for color distribution in an image
std::vector<float> compute3DHistogram(const cv::Mat& img, int bins) {
    int bins3D = bins * bins * bins;
    std::vector<float> hist(bins3D, 0);

    for (int y = 0; y < img.rows; y++) {
        for (int x = 0; x < img.cols; x++) {
            cv::Vec3b pixel = img.at<cv::Vec3b>(y, x);

            int binR = std::min((int)(pixel[2] * bins / 256.0), bins - 1);
            int binG = std::min((int)(pixel[1] * bins / 256.0), bins - 1);
            int binB = std::min((int)(pixel[0] * bins / 256.0), bins - 1);

            hist[binR * bins * bins + binG * bins + binB] += 1;
        }
    }

    // Normalize the histogram
    float sum = 0;
    for (const auto& val : hist) sum += val;
    for (auto& val : hist) val /= sum;

    return hist;
}

// Function to compute texture histogram using Sobel operator
std::vector<float> computeTextureHistogram(const cv::Mat& img, int bins) {
    cv::Mat gray, grad_x, grad_y, magnitude;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Compute gradients along x and y using Sobel operator
    cv::Sobel(gray, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(gray, grad_y, CV_32F, 0, 1, 3);

    // Compute magnitude of gradients
    cv::magnitude(grad_x, grad_y, magnitude);

    // Compute histogram of gradient magnitudes
    std::vector<float> hist(bins, 0);
    float max_magnitude = 255.0; // Maximum possible magnitude with Sobel operator

    // Use pointer access for improved performance
    for (int y = 0; y < magnitude.rows; y++) {
        const float* magRow = magnitude.ptr<float>(y);
        for (int x = 0; x < magnitude.cols; x++) {
            float mag = magRow[x];  // Accessing pixel value using pointer
            int binIdx = std::min(int(mag * bins / max_magnitude), bins - 1);
            hist[binIdx] += 1;
        }
    }

    // Normalize the histogram
    float sum = 0;
    for (const auto& val : hist) sum += val;
    for (auto& val : hist) val /= sum;

    return hist;
}

// Function to compute the co-occurrence matrix from a grayscale image
cv::Mat computeGLCM(const cv::Mat& gray, int dx, int dy, int levels) {
    cv::Mat glcm(levels, levels, CV_32F, cv::Scalar(0));

    for (int y = 0; y < gray.rows - dy; y++) {
        for (int x = 0; x < gray.cols - dx; x++) {
            int val1 = gray.at<uchar>(y, x);
            int val2 = gray.at<uchar>(y + dy, x + dx);
            glcm.at<float>(val1, val2) += 1.0;
        }
    }

    // Normalize the GLCM
    glcm /= (gray.rows * gray.cols);

    return glcm;
}

// Function to compute features from the co-occurrence matrix
std::vector<float> computeGLCMFeatures(const cv::Mat& glcm) {
    float energy = 0.0, entropy = 0.0, contrast = 0.0, homogeneity = 0.0, maxProb = 0.0;

    for (int i = 0; i < glcm.rows; i++) {
        for (int j = 0; j < glcm.cols; j++) {
            float val = glcm.at<float>(i, j);

            energy += val * val;
            if (val > 0) entropy -= val * std::log(val);
            contrast += (i - j) * (i - j) * val;
            homogeneity += val / (1.0 + std::abs(i - j));
            if (val > maxProb) maxProb = val;
        }
    }

    return { energy, entropy, contrast, homogeneity, maxProb };
}


// Function to compute histograms of Laws filter responses
std::vector<float> computeLawsFeatures(const cv::Mat& gray) {
    // Define 1D masks for Laws' texture energy measures
    cv::Mat L5 = (cv::Mat_<float>(1, 5) << 1, 4, 6, 4, 1);
    cv::Mat E5 = (cv::Mat_<float>(1, 5) << -1, -2, 0, 2, 1);
    cv::Mat S5 = (cv::Mat_<float>(1, 5) << -1, 0, 2, 0, -1);
    cv::Mat W5 = (cv::Mat_<float>(1, 5) << -1, 2, 0, -2, 1);
    cv::Mat R5 = (cv::Mat_<float>(1, 5) << 1, -4, 6, -4, 1);

    // Pre-allocate space for responses
    std::vector<cv::Mat> responses(5);

    // Compute responses for each mask
    cv::filter2D(gray, responses[0], CV_32F, L5);
    cv::filter2D(gray, responses[1], CV_32F, E5);
    cv::filter2D(gray, responses[2], CV_32F, S5);
    cv::filter2D(gray, responses[3], CV_32F, W5);
    cv::filter2D(gray, responses[4], CV_32F, R5);

    // Compute the histogram for each response
    // For simplicity, we'll compute the mean of each response and use it as a feature
    std::vector<float> features;
    for (const auto& response : responses) {
        cv::Scalar mean, stddev;
        cv::meanStdDev(response, mean, stddev);
        features.push_back(mean[0]);
    }

    return features;
}


// Function to compute histograms of Gabor filter responses
std::vector<float> computeGaborFeatures(const cv::Mat& gray) {
    // Define Gabor filter parameters
    int kernel_size = 31;
    double pos_sigma = 2.5;
    double pos_lm = 50.0;
    double pos_th = 0;
    double pos_psi = 90;
    cv::Mat dest;
    cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), pos_sigma, pos_th, pos_lm, 0.5, pos_psi, CV_32F);
    cv::filter2D(gray, dest, CV_32F, kernel);

    // Compute the histogram for the response
    // For simplicity, we'll compute the mean and standard deviation of the response and use them as features
    cv::Scalar mean, stddev;
    cv::meanStdDev(dest, mean, stddev);

    return { static_cast<float>(mean[0]), static_cast<float>(stddev[0]) };

}

// Function to compute the histogram of gradient orientations using Canny edge detector
std::vector<float> computeEdgeHistogram(const cv::Mat& img, int bins) {
    cv::Mat gray, edges, dx, dy;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 50, 150);

    // Compute gradient orientations
    cv::Sobel(edges, dx, CV_32F, 1, 0, 3);
    cv::Sobel(edges, dy, CV_32F, 0, 1, 3);
    cv::Mat mag, angle;
    cv::cartToPolar(dx, dy, mag, angle);

    std::vector<float> histogram(bins, 0);
    for (int i = 0; i < angle.rows; i++) {
        for (int j = 0; j < angle.cols; j++) {
            int bin = int(angle.at<float>(i, j) * bins / (2 * CV_PI));
            histogram[bin]++;
        }
    }

    // Normalize the histogram
    float sum = std::accumulate(histogram.begin(), histogram.end(), 0.0f);
    for (auto& val : histogram) val /= sum;

    return histogram;
}

// Function to compute the custom feature vector for an image
std::vector<float> computeCustomFeatureVector(const cv::Mat& img) {
    // Compute 3D color histogram
    std::vector<float> colorHist = compute3DHistogram(img, 8);

    // Convert image to grayscale for texture and shape features
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Compute GLCM features
    cv::Mat glcm = computeGLCM(gray, 1, 0, 256);
    std::vector<float> glcmFeatures = computeGLCMFeatures(glcm);

    // Compute Gabor features
    std::vector<float> gaborFeatures = computeGaborFeatures(gray);

    // Compute edge histogram
    std::vector<float> edgeHist = computeEdgeHistogram(img, 8);

    // Combine all features into a single feature vector
    std::vector<float> customFeatureVector;
    customFeatureVector.insert(customFeatureVector.end(), colorHist.begin(), colorHist.end());
    customFeatureVector.insert(customFeatureVector.end(), glcmFeatures.begin(), glcmFeatures.end());
    customFeatureVector.insert(customFeatureVector.end(), gaborFeatures.begin(), gaborFeatures.end());
    customFeatureVector.insert(customFeatureVector.end(), edgeHist.begin(), edgeHist.end());

    return customFeatureVector;
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
    std::string histogram_target_image_path = "C:/Users/Shi Zhang/My Drive/CS/NEU Align/Courses/2023 Fall/5330/Project02/olympus/pic.0535.jpg";
    std::string multiHistogram_target_image_path = "C:/Users/Shi Zhang/My Drive/CS/NEU Align/Courses/2023 Fall/5330/Project02/olympus/pic.0535.jpg";
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

    std::cout << std::endl;

    /*Task 2 2D Histogram Matching*/
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

    std::cout << "Top " << N << " 2D Histogram Matches for " << histogram_target_image_path << " are: " << std::endl;
    for (int i = 0; i < std::min(N, (int)distances.size()); i++) {
        std::cout << distances[i].second << " with histogram intersection: " << distances[i].first << std::endl;
    }

    distances.clear();

    std::cout << std::endl;

    /*Task 2 3D Histogram Matching*/
    cv::Mat histogram3D_target_img = cv::imread(histogram_target_image_path, cv::IMREAD_COLOR);
    if (histogram3D_target_img.empty()) {
        std::cerr << "Could not read the 3D histogram target image: " << histogram_target_image_path << std::endl;
        return -1;
    }

    // Compute the 3D histogram for the target image
    std::vector<float> target3DHistogram = compute3DHistogram(histogram3D_target_img, 8); // 8 bins for each channel

    distances.clear(); // Clear the distances vector

    // Compute 3D histograms for each image in the database and compute the histogram intersection
    for (size_t i = 0; i < data.size(); i++) {
        std::string compare_filename = replaceAll(std::string(filenames[i]), "\\", "/");
        if (histogram_target_image_path != compare_filename) {
            cv::Mat img = cv::imread(compare_filename, cv::IMREAD_COLOR);
            std::vector<float> histogram = compute3DHistogram(img, 8); // 8 bins for each channel

            float distance = histogramIntersection(target3DHistogram, histogram);
            distances.push_back({ distance, compare_filename });
        }
    }

    // Sort the distances in descending order as we are using histogram intersection
    std::sort(distances.begin(), distances.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
        });

    std::cout << "Top " << N << " 3D Histogram Matches for " << histogram_target_image_path << " are: " << std::endl;
    for (int i = 0; i < std::min(N, (int)distances.size()); i++) {
        std::cout << distances[i].second << " with 3D histogram intersection: " << distances[i].first << std::endl;
    }

    distances.clear();

    std::cout << std::endl;

    /*Task 3 Multi-histogram Matching*/
    cv::Mat multiHistogram_target_img = cv::imread(multiHistogram_target_image_path, cv::IMREAD_COLOR);
    if (multiHistogram_target_img.empty()) {
        std::cerr << "Could not read the multi-histogram target image: " << multiHistogram_target_image_path << std::endl;
        return -1;
    }

    int half_height = multiHistogram_target_img.rows / 2;
    cv::Rect top_roi(0, 0, multiHistogram_target_img.cols, half_height); // Top half of the image
    cv::Rect bottom_roi(0, half_height, multiHistogram_target_img.cols, half_height); // Bottom half of the image

    // Compute the 3D histograms for the target image's top and bottom halves
    std::vector<float> top_histogram = compute3DHistogram(multiHistogram_target_img(top_roi), 8); // 8 bins for each channel
    std::vector<float> bottom_histogram = compute3DHistogram(multiHistogram_target_img(bottom_roi), 8); // 8 bins for each channel

    distances.clear();

    for (size_t i = 0; i < data.size(); i++) {
        std::string compare_filename = replaceAll(std::string(filenames[i]), "\\", "/");
        if (multiHistogram_target_image_path != compare_filename) {
            cv::Mat img = cv::imread(compare_filename, cv::IMREAD_COLOR);

            std::vector<float> compare_top_histogram = compute3DHistogram(img(top_roi), 8); // 8 bins for each channel
            std::vector<float> compare_bottom_histogram = compute3DHistogram(img(bottom_roi), 8); // 8 bins for each channel

            float distance_top = histogramIntersection(top_histogram, compare_top_histogram);
            float distance_bottom = histogramIntersection(bottom_histogram, compare_bottom_histogram);

            float combined_distance = 0.5 * distance_top + 0.5 * distance_bottom;
            distances.push_back({ combined_distance, compare_filename });
        }
    }

    std::sort(distances.begin(), distances.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
        });

    std::cout << "Top " << N << " Multi-histogram Matches for " << multiHistogram_target_image_path << " are: " << std::endl;
    for (int i = 0; i < std::min(N, (int)distances.size()); i++) {
        std::cout << distances[i].second << " with combined histogram intersection: " << distances[i].first << std::endl;
    }

    std::cout << std::endl;

    /*Task 4 Texture and Color Matching*/
    std::string textureColor_target_image_path = "C:/Users/Shi Zhang/My Drive/CS/NEU Align/Courses/2023 Fall/5330/Project02/olympus/pic.0535.jpg";
    cv::Mat textureColor_target_img = cv::imread(textureColor_target_image_path, cv::IMREAD_COLOR);
    if (textureColor_target_img.empty()) {
        std::cerr << "Could not read the texture and color target image: " << textureColor_target_image_path << std::endl;
        return -1;
    }

    // Compute the 3D color histogram for the target image
    std::vector<float> targetColorHistogram = compute3DHistogram(textureColor_target_img, 8); // 8 bins for each channel

    // Compute the texture histogram for the target image
    std::vector<float> targetTextureHistogram = computeTextureHistogram(textureColor_target_img, 8); // 8 bins for magnitude

    distances.clear();

    for (size_t i = 0; i < data.size(); i++) {
        std::string compare_filename = replaceAll(std::string(filenames[i]), "\\", "/");
        if (textureColor_target_image_path != compare_filename) {
            cv::Mat img = cv::imread(compare_filename, cv::IMREAD_COLOR);

            std::vector<float> compareColorHistogram = compute3DHistogram(img, 8); // 8 bins for each channel
            std::vector<float> compareTextureHistogram = computeTextureHistogram(img, 8); // 8 bins for magnitude

            float distance_color = histogramIntersection(targetColorHistogram, compareColorHistogram);
            float distance_texture = histogramIntersection(targetTextureHistogram, compareTextureHistogram);

            float combined_distance = 0.5 * distance_color + 0.5 * distance_texture;
            distances.push_back({ combined_distance, compare_filename });
        }
    }

    std::sort(distances.begin(), distances.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
        });

    std::cout << "Top " << N << " Texture and Color Matches for " << textureColor_target_image_path << " are: " << std::endl;
    for (int i = 0; i < std::min(N, (int)distances.size()); i++) {
        std::cout << distances[i].second << " with combined texture and color intersection: " << distances[i].first << std::endl;
    }

    std::cout << std::endl;

    /*Task 4 Extension Co-occurrence Matrices, Laws and Gabor Matching*/
    std::string extension_target_image_path = textureColor_target_image_path;  // This is the same path as the one used for Task 4.
    cv::Mat extension_target_img = cv::imread(extension_target_image_path, cv::IMREAD_GRAYSCALE);  // Convert to grayscale
    if (extension_target_img.empty()) {
        std::cerr << "Could not read the target image for Task 4 Extension: " << extension_target_image_path << std::endl;
        return -1;
    }

    // Compute GLCM features for the target image
    cv::Mat target_glcm = computeGLCM(extension_target_img, 1, 0, 256);  // 1 offset in x, 0 in y, 256 levels
    std::vector<float> target_glcm_features = computeGLCMFeatures(target_glcm);

    // Compute Laws' texture features for the target image
    std::vector<float> target_laws_features = computeLawsFeatures(extension_target_img);

    // Compute Gabor features for the target image
    std::vector<float> target_gabor_features = computeGaborFeatures(extension_target_img);

    // Concatenate all features together for the target image
    std::vector<float> target_features_extension;
    target_features_extension.insert(target_features_extension.end(), target_glcm_features.begin(), target_glcm_features.end());
    target_features_extension.insert(target_features_extension.end(), target_laws_features.begin(), target_laws_features.end());
    target_features_extension.insert(target_features_extension.end(), target_gabor_features.begin(), target_gabor_features.end());

    distances.clear();

    for (size_t i = 0; i < data.size(); i++) {
        std::string compare_filename = replaceAll(std::string(filenames[i]), "\\", "/");
        if (extension_target_image_path != compare_filename) {
            cv::Mat img = cv::imread(compare_filename, cv::IMREAD_GRAYSCALE);  // Convert to grayscale

            // Compute GLCM features for the image
            cv::Mat img_glcm = computeGLCM(img, 1, 0, 256);
            std::vector<float> img_glcm_features = computeGLCMFeatures(img_glcm);

            // Compute Laws' texture features for the image
            std::vector<float> img_laws_features = computeLawsFeatures(img);

            // Compute Gabor features for the image
            std::vector<float> img_gabor_features = computeGaborFeatures(img);

            // Concatenate all features together for the image
            std::vector<float> img_features_extension;
            img_features_extension.insert(img_features_extension.end(), img_glcm_features.begin(), img_glcm_features.end());
            img_features_extension.insert(img_features_extension.end(), img_laws_features.begin(), img_laws_features.end());
            img_features_extension.insert(img_features_extension.end(), img_gabor_features.begin(), img_gabor_features.end());

            // Compute distance between target and image features
            float distance = computeDistance(target_features_extension, img_features_extension);
            distances.push_back({ distance, compare_filename });
        }
    }

    std::sort(distances.begin(), distances.end());

    std::cout << "Top " << N << " Matches (GLCM, Laws, Gabor) for " << extension_target_image_path << " are: " << std::endl;
    for (int i = 0; i < std::min(N, (int)distances.size()); i++) {
        std::cout << distances[i].second << " with distance: " << distances[i].first << std::endl;
    }

    std::cout << std::endl;

    // Clean up memory
    distances.clear();

    /*Task 5 Custom Design Feature Matching */
    std::vector<std::vector<float>> customFeatureDatabase(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        std::string image_path = replaceAll(std::string(filenames[i]), "\\", "/");
        cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
        customFeatureDatabase[i] = computeCustomFeatureVector(img);
    }

    // Define the query images for evaluation
    std::vector<std::string> queryImages = {
        "C:/Users/Shi Zhang/My Drive/CS/NEU Align/Courses/2023 Fall/5330/Project02/olympus/pic.0343.jpg",
        "C:/Users/Shi Zhang/My Drive/CS/NEU Align/Courses/2023 Fall/5330/Project02/olympus/pic.0346.jpg"
    };

    for (const auto& queryImagePath : queryImages) {
        cv::Mat queryImage = cv::imread(queryImagePath, cv::IMREAD_COLOR);
        std::vector<float> queryFeatures = computeCustomFeatureVector(queryImage);

        std::vector<std::pair<float, std::string>> distances;
        for (size_t i = 0; i < customFeatureDatabase.size(); i++) {
            std::string compare_filename = replaceAll(std::string(filenames[i]), "\\", "/");
            if (queryImagePath != compare_filename) {
                float distance = computeDistance(queryFeatures, customFeatureDatabase[i]);
                distances.push_back({ distance, compare_filename });
            }
        }

        std::sort(distances.begin(), distances.end());

        std::cout << "Top 10 Custom Feature Matches for " << queryImagePath << " are: " << std::endl;
        for (int i = 0; i < 10; i++) {
            std::cout << distances[i].second << " with distance: " << distances[i].first << std::endl;
        }

        std::cout << "\nLeast similar images are: " << std::endl;
        for (int i = distances.size() - 10; i < distances.size(); i++) {
            std::cout << distances[i].second << " with distance: " << distances[i].first << std::endl;
        }

        std::cout << std::endl;
    }


    return 0;
}
