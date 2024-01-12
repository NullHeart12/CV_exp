#include "preset.h"

cv::Mat preset(const cv::Mat& img)
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
			int index3 = (i - 1) * cols + j + 1;
			int index4 = i * cols + j - 1;
			int index5 = i * cols + j + 1;
			int index6 = (i + 1) * cols + j - 1;
			int index8 = (i + 1) * cols + j + 1;

			int gy = data[index1] + 2 * data[index4] + data[index6] - data[index3] - 2 * data[index5] - data[index8];

			result.data[index] = abs(gy);
		}
	}
	return result;
}