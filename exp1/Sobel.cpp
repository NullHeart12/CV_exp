#include "Sobel.h"

cv::Mat Sobel(const cv::Mat& img)
{
	int rows = img.rows;
	int cols = img.cols;

	cv::Mat result = img.clone();

	uchar* data = img.data;
	for (int i = 1; i < rows - 1; i++)
	{
		for (int j = 1; j < cols - 1; j++)
		{
			int index = i * cols + j;
			int index1 = (i - 1) * cols + j - 1;
			int index2 = (i - 1) * cols + j;
			int index3 = (i - 1) * cols + j + 1;
			int index4 = i * cols + j - 1;
			int index5 = i * cols + j + 1;
			int index6 = (i + 1) * cols + j - 1;
			int index7 = (i + 1) * cols + j;
			int index8 = (i + 1) * cols + j + 1;

			int gx = data[index1] + 2 * data[index2] + data[index3] - data[index6] - 2 * data[index7] - data[index8];
			int gy = data[index1] + 2 * data[index4] + data[index6] - data[index3] - 2 * data[index5] - data[index8];

			int sum = abs(gx) + abs(gy);
			if (sum > 255)
				sum = 255;
			result.data[index] = sum;
		}
	}
	return result;
}