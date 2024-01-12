#pragma once
#ifndef CALCHIST_H
#define CALCHIST_H

#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>

std::unordered_map<char, std::vector<int>> calcHist(const cv::Mat& img);

cv::Mat drawHist(const std::unordered_map<char, std::vector<int>>& hist);

#endif // !CALCHIST_H
