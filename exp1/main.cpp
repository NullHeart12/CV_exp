#include "Sobel.h"
#include "preset.h"
#include "calcHist.h"
#include "LBP.h"

int main()
{
	std::string path = "./image/snow.jpg";
	//std::string path = "./image/leaves.jpg";
	//std::string path = "./image/test.png";
	//std::string path = "./image/man.png";
	cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
	if (img.empty())
	{
		std::cout << "read image failed" << std::endl;
		system("pause");
		return 0;
	}

	cv::Mat Sobel_result = Sobel(img);
	cv::imshow("Sobel", Sobel_result);
	cv::imwrite("./image/Sobel.jpg", Sobel_result);

	cv::Mat preset_result = preset(img);
	cv::imshow("preset", preset_result);
	cv::imwrite("./image/preset.jpg", preset_result);

	cv::Mat color_img= cv::imread(path, cv::IMREAD_COLOR);
	if (color_img.empty())
	{
		std::cout << "read image failed" << std::endl;
		cv::waitKey(0);
		return 0;
	}

	std::unordered_map<char, std::vector<int>> hist = calcHist(color_img);
	cv::Mat hist_img = drawHist(hist);
	cv::imshow("hist", hist_img);
	cv::imwrite("./image/hist.jpg", hist_img);

	cv::Mat LBP_result = LBP(img);
	cv::imshow("LBP", LBP_result);
	cv::imwrite("./image/LBP.jpg", LBP_result);

	cv::waitKey(0);
	return 0;
}