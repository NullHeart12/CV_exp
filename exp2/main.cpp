#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat segment(const cv::Mat& img)
{
	int rows = img.rows;
	int cols = img.cols;

	std::vector<cv::Point> points = {
		cv::Point(0, rows), cv::Point(cols, rows), cv::Point(cols / 2, rows * 0.6)//, cv::Point(0, rows * 0.7), cv::Point(cols, rows * 0.7)
	};
	cv::Mat mask = cv::Mat::zeros(rows, cols, CV_8UC1);
	cv::fillPoly(mask, points, 255);

	cv::Mat segment;
	cv::bitwise_and(img, img, segment, mask);

	return segment;
}

int main()
{
	//std::string path = "./image/road.png";
	//std::string path = "./image/road2.png";
	//std::string path = "./image/road3.png";
	std::string path = "./image/road4.png";
	cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
	cv::Mat clo = cv::imread(path, cv::IMREAD_COLOR);
	if (img.empty())
	{
		std::cout << "read image failed" << std::endl;
		system("pause");
		return 0;
	}

	cv::Mat edges;
	int lowThreshold =100, highThreshold = 200;
	int kernel_size = 3;
	cv::Canny(img, edges, lowThreshold, highThreshold, kernel_size);
	//cv::imshow("edges", edges);
	cv::imwrite("./image/edges.jpg", edges);
	
	cv::Mat segment_result = segment(edges);
	//cv::imshow("segment", segment_result);
	cv::imwrite("./image/segment.jpg", segment_result);

	std::vector<cv::Vec4i> result;
	cv::HoughLinesP(segment_result, result, 1, CV_PI / 180, 20, 30, 10);

	for (int i = 0; i < result.size(); i++)
	{
		cv::Vec4i line = result[i];
		cv::line(clo, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255, 0, 0), 2);
	}

	cv::imshow("result", clo);
	cv::imwrite("./image/result.jpg", clo);

	cv::waitKey(0);
	return 0;
}