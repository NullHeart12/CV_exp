#include "calcHist.h"

std::unordered_map<char, std::vector<int>> calcHist(const cv::Mat& img)
{
	int rows = img.rows;
	int cols = img.cols;

	std::unordered_map<char, std::vector<int>> hist;
	hist['r'] = std::vector<int>(256, 0);
	hist['g'] = std::vector<int>(256, 0);
	hist['b'] = std::vector<int>(256, 0);

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j ++)
		{
			cv::Vec3b cur = img.at<cv::Vec3b>(i, j);
			hist['b'][cur[0]]++;
			hist['g'][cur[1]]++;
			hist['r'][cur[2]]++;
		}
	}
	return hist;
}

cv::Mat drawHist(const std::unordered_map<char, std::vector<int>>& hist)
{
	cv::Scalar color[3] = { cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255) };
	std::vector<int> B= hist.at('b');
	std::vector<int> G = hist.at('g');
	std::vector<int> R = hist.at('r');

	int max = INT_MIN;
	int min = INT_MAX;
	for (int i = 0; i < B.size(); i++)
	{
		if (B[i] > max)
			max = B[i];
		if (G[i] > max)
			max = G[i];
		if (R[i] > max)
			max = R[i];

		if (B[i] < min)
			min = B[i];
		if (G[i] < min)
			min = G[i];
		if (R[i] < min)
			min = R[i];
	}

	int diff = max - min;
	for (int i = 0; i < B.size(); i++)
	{
		B[i] = (B[i] - min)  * 255 / diff;
		G[i] = (G[i] - min) * 255 / diff;
		R[i] = (R[i] - min) * 255 / diff;
	}

	cv::Mat result = cv::Mat::zeros(265, B.size() + 10, CV_8UC3);
	for (int i = 0; i < B.size(); i++)
	{
		cv::line(result, cv::Point(i, 265 - B[i]), cv::Point(i, 265 - B[i - 1 >= 0 ? i - 1 : 0]), color[0]);
		cv::line(result, cv::Point(i, 265 - G[i]), cv::Point(i, 265 - G[i - 1 >= 0 ? i -1 : 0]), color[1]);
		cv::line(result, cv::Point(i, 265 - R[i]), cv::Point(i, 265 - R[i - 1 >= 0 ? i - 1 : 0]), color[2]);
	}

	return result;
}