#include "LBP.h"

cv::Mat LBP(const cv::Mat& img)
{
	int rows = img.rows;
	int cols = img.cols;

	cv::Mat result = img.clone();

	for (int i = 1; i < rows - 1; i++)
	{
		for (int j = 1; j < cols - 1; j++)
		{
			uchar cur = img.at<uchar>(i, j);
			uchar code = 0;
			code |= (img.at<uchar>(i - 1, j - 1) > cur) << 7;
			code |= (img.at<uchar>(i - 1, j) > cur) << 6;
			code |= (img.at<uchar>(i - 1, j + 1) > cur) << 5;
			code |= (img.at<uchar>(i, j + 1) > cur) << 4;
			code |= (img.at<uchar>(i + 1, j + 1) > cur) << 3;
			code |= (img.at<uchar>(i + 1, j) > cur) << 2;
			code |= (img.at<uchar>(i + 1, j - 1) > cur) << 1;
			code |= (img.at<uchar>(i, j - 1) > cur);
			result.at<uchar>(i, j) = code;
		}
	}
	return result;
}