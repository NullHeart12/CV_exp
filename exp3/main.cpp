#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <filesystem>
#include <vector>
#include <queue>

//little endian transform
int reverseInt(int i)
{
	uchar c1, c2, c3, c4;
	c1 = i & 255;
	c2 = (i >> 8) & 255;
	c3 = (i >> 16) & 255;
	c4 = (i >> 24) & 255;

	i = (c1 << 24) + (c2 << 16) + (c3 << 8) + c4;
	return i;
}

//read labels
cv::Mat read_labels(std::string path)
{
	int magic_number = 0;
	int number_of_items = 0;

	std::ifstream file(path, std::ios::binary);
	if (!file.is_open())
	{
		std::cout << "read label failed" << std::endl;
		system("pause");
		return cv::Mat();
	}

	file.read((char*)&magic_number, sizeof(magic_number));
	file.read((char*)&number_of_items, sizeof(number_of_items));
	magic_number = reverseInt(magic_number);
	number_of_items = reverseInt(number_of_items);
	std::cout << "magic_number: " << magic_number << std::endl;
	std::cout << "number_of_items: " << number_of_items << std::endl;

	cv::Mat labels = cv::Mat::zeros(number_of_items, 1, CV_32SC1);
	for (int i = 0; i < number_of_items; i++)
	{
		uchar temp = 0;
		file.read((char*)&temp, sizeof(temp));
		labels.at<int>(i, 0) = temp;
	}

	file.close();

	return labels;
}

//read images
cv::Mat read_images(std::string path)
{
	int magic_number = 0;
	int number_of_items = 0;
	int rows = 0, cols = 0;

	std::ifstream file(path, std::ios::binary);
	if (!file.is_open())
	{
		std::cout << "read image failed" << std::endl;
		system("pause");
		return cv::Mat();
	}

	file.read((char*)&magic_number, sizeof(magic_number));
	file.read((char*)&number_of_items, sizeof(number_of_items));
	file.read((char*)&rows, sizeof(rows));
	file.read((char*)&cols, sizeof(cols));
	magic_number = reverseInt(magic_number);
	number_of_items = reverseInt(number_of_items);
	rows = reverseInt(rows);
	cols = reverseInt(cols);
	std::cout << "magic_number: " << magic_number << std::endl;
	std::cout << "number_of_items: " << number_of_items << std::endl;
	std::cout << "rows: " << rows << std::endl;
	std::cout << "cols: " << cols << std::endl;

	cv::Mat images = cv::Mat::zeros(number_of_items, rows * cols, CV_32FC1);
	for (int i = 0; i < number_of_items; i++)
	{
		for (int j = 0; j < rows * cols; j++)
		{
			uchar temp = 0;
			file.read((char*)&temp, sizeof(temp));
			images.at<float>(i, j) = float(temp) / 255.0;
		}
	}

	file.close();

	return images;
}

//one hot
cv::Mat one_hot(cv::Mat labels, int kinds_num)
{
	cv::Mat one_hot = cv::Mat::zeros(labels.rows, kinds_num, CV_32FC1);
	for (int i = 0; i < labels.rows; i++)
	{
		int index = labels.at<int>(i, 0);
		one_hot.at<float>(i, index) = 1.0;
	}

	return one_hot;
}

void train(std::string model_path, std::string train_images_path, std::string train_labels_path, std::string test_images_path, std::string test_labels_path)
{
	//the first part: prepare train data
	std::cout << "*****start read data*****" << std::endl;

	cv::Mat train_labels = read_labels(train_labels_path);
	cv::Mat train_images = read_images(train_images_path);
	cv::Mat test_labels = read_labels(test_labels_path);
	cv::Mat test_images = read_images(test_images_path);
	cv::Mat train_labels_one_hot = one_hot(train_labels, 10);
	//cv::Mat test_labels_one_hot = one_hot(test_labels, 10);

	std::cout << "*****read data finished*****\n" << std::endl;

	//the second part: build ann model and train
	std::cout << "*****start train*****" << std::endl;

	cv::Ptr<cv::ml::ANN_MLP> ann = cv::ml::ANN_MLP::create();
	cv::Mat layerSize = (cv::Mat_<int>(1, 3) << 784, 64, 10);
	ann->setLayerSizes(layerSize);
	ann->setTrainMethod(cv::ml::ANN_MLP::BACKPROP, 0.001, 0.1);
	ann->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1.0, 1.0);
	ann->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 10, 0.0001));

	cv::Ptr<cv::ml::TrainData> train_data = cv::ml::TrainData::create(train_images, cv::ml::ROW_SAMPLE, train_labels_one_hot);
	ann->train(train_data);

	std::cout << "*****train finished*****\n" << std::endl;

	//the third part: predict and calculate accuracy
	std::cout << "*****start predict*****" << std::endl;

	cv::Mat pre_out = cv::Mat::zeros(test_images.rows, 10, CV_32FC1);
	float ret = ann->predict(test_images, pre_out);

	int equal_nums = 0;
	for (int i = 0; i < pre_out.rows; i++)
	{
		cv::Mat temp = pre_out.rowRange(i, i + 1);
		double maxVal = 0;
		cv::Point maxPoint;
		cv::minMaxLoc(temp, NULL, &maxVal, NULL, &maxPoint);
		int max_index = maxPoint.x;
		int test_index = test_labels.at<int>(i, 0);
		if (max_index == test_index)
			equal_nums++;
	}

	float acc = float(equal_nums) / float(pre_out.rows);
	std::cout << "accuracy on test data set: " << acc * 100 << "%" << std::endl;
	ann->save(model_path);

	std::cout << "*****predict finished*****\n" << std::endl;
}

void down(cv::Mat& img)
{
	int nr = img.rows;
	int nc = img.cols;
	for (int y = 0; y < nr; y++)
	{
		uchar* data = img.ptr<uchar>(y);
		for (int x = 0; x < nc; x++)
		{
			if (data[x] < 50)
				data[x] = 0;
		}
	}
}

cv::Mat binaryzation(cv::Mat img, int threshold) {
	cv::Mat binary;
	cv::threshold(img, binary, threshold, 255, cv::THRESH_BINARY);

	return binary;
}

int bfs(cv::Mat& binary, cv::Mat& labels)
{
	int label = 0;
	std::queue<std::pair<int, int>> q;

	int nr = binary.rows;
	int nc = binary.cols;

	for (int y = 0; y < nr; y++)
	{
		uchar* data = binary.ptr<uchar>(y);
		for (int x = 0; x < nc; x++)
		{
			if (data[x] == 255 && labels.at<int>(y, x) == 0)
			{
				label++;
				labels.at<int>(y, x) = label;
				q.push({ x,y });

				while (!q.empty())
				{
					std::pair<int, int> front = q.front();
					q.pop();
					int sub_x = front.first;
					int sub_y = front.second;

					if (sub_y - 1 >= 0 && binary.at<uchar>(sub_y - 1, sub_x) == 255 && labels.at<int>(sub_y - 1, sub_x) == 0)
					{
						labels.at<int>(sub_y - 1, sub_x) = label;
						q.push({ sub_x,sub_y - 1 });
					}
					if (sub_y + 1 < nr && binary.at<uchar>(sub_y + 1, sub_x) == 255 && labels.at<int>(sub_y + 1, sub_x) == 0)
					{
						labels.at<int>(sub_y + 1, sub_x) = label;
						q.push({ sub_x,sub_y + 1 });
					}
					if (sub_x - 1 >= 0 && binary.at<uchar>(sub_y, sub_x - 1) == 255 && labels.at<int>(sub_y, sub_x - 1) == 0)
					{
						labels.at<int>(sub_y, sub_x - 1) = label;
						q.push({ sub_x - 1,sub_y });
					}
					if (sub_x + 1 < nc && binary.at<uchar>(sub_y, sub_x + 1) == 255 && labels.at<int>(sub_y, sub_x + 1) == 0)
					{
						labels.at<int>(sub_y, sub_x + 1) = label;
						q.push({ sub_x + 1,sub_y });
					}
				}

			}
		}
	}
	return label;
}

std::vector<std::vector<int>> get_boundary(const cv::Mat& labels, int labels_num)
{
	std::vector<std::vector<int>> boundary(labels_num, {INT_MAX,INT_MIN, INT_MAX, INT_MIN});
	int nr = labels.rows;
	int nc = labels.cols;

	for (int y = 0; y < nr; y++)
	{
		const int* data = labels.ptr<int>(y);
		for (int x = 0; x < nc; x++)
		{
			int label = data[x];
			if (label != 0)
			{
				if (x < boundary[label - 1][0] )
					boundary[label - 1][0] = x;
				if (x > boundary[label - 1][1])
					boundary[label - 1][1] = x;
				if (y < boundary[label - 1][2])
					boundary[label - 1][2] = y;
				if (y > boundary[label - 1][3])
					boundary[label - 1][3] = y;
			}
		}
	}

	return boundary;
}

std::vector<int> get_order(const cv::Mat& labels, int labels_num)
{
	std::vector<int> order(labels_num, 0);
	int nr = labels.rows;
	int nc = labels.cols;

	int y = nr / 2;
	const int* data = labels.ptr<int>(y);
	int pos = 0;
	for (int x = 0; x < nc; x++)
	{
		int label = data[x];
		if (label != 0 && (pos == 0 || label - 1 != order[pos - 1]))
			order[pos++] = label - 1;
	}

	return order;
}

std::vector<cv::Mat> cut(cv::Mat img, std::vector<std::vector<int>> boundary, std::vector<int> order)
{
	std::vector<cv::Mat> result;
	for (int i = 0; i < order.size(); i++)
	{
		int cur_left = boundary[order[i]][0];
		int cur_right = boundary[order[i]][1];
		int cur_top = boundary[order[i]][2];
		int cur_bottom = boundary[order[i]][3];
		cv::Mat temp = img(cv::Range(cur_top, cur_bottom + 1), cv::Range(cur_left, cur_right + 1));

		result.push_back(temp);
	}

	return result;
}

std::vector<cv::Mat> resize_(std::vector<cv::Mat> imgs, int width, int height)
{
	std::vector<cv::Mat> result;
	for (int i = 0; i < imgs.size(); i++)
	{
		cv::Mat temp;
		//cv::resize(imgs[i], temp, cv::Size(width, height), 0, 0, cv::INTER_NEAREST);
		int len = std::max(imgs[i].cols, imgs[i].rows);
		cv::resize(imgs[i], imgs[i], cv::Size(0, 0), 18.0 / len, 18.0 / len);
		int top = (height - imgs[i].rows) / 2;
		int bottom = height - imgs[i].rows - top;
		int left = (width - imgs[i].cols) / 2;
		int right = width - imgs[i].cols - left;
		cv::copyMakeBorder(imgs[i], temp, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0));
		
		result.push_back(temp);
	}

	return result;
}

std::vector<cv::Mat> read_frag()
{
	std::vector<cv::Mat> result;
	for (int i = 0; i < 10; i++)
	{
		std::string path = "./image/student_num/" + std::to_string(i) + ".jpg";
		cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
		if (img.empty())
		{
			std::cout << "read image failed" << std::endl;
			system("pause");
			return std::vector<cv::Mat>();
		}
		result.push_back(img);
	}

	return result;
}

void joint()
{
	std::vector<cv::Mat> imgs = read_frag();
	int width = imgs[0].cols;
	int height = imgs[0].rows;
	cv::Mat result = cv::Mat::zeros(height, width * imgs.size(), CV_8UC1);
	for (int i = 0; i < imgs.size(); i++)
	{
		cv::Mat temp = imgs[i];
		temp.copyTo(result.colRange(i * width, (i + 1) * width));
	}

	cv::imwrite("./image/joint.jpg", result);
}

int main()
{
	bool change_since_last_time = 1;
	std::string model_path = "./model/mnist_ann.xml";
	std::string train_images_path = "./image/train-images.idx3-ubyte";
	std::string train_labels_path = "./image/train-labels.idx1-ubyte";
	std::string test_images_path = "./image/t10k-images.idx3-ubyte";
	std::string test_labels_path = "./image/t10k-labels.idx1-ubyte";
	if (!std::filesystem::exists(model_path) || change_since_last_time)
		train(model_path, train_images_path, train_labels_path, test_images_path, test_labels_path);

	//joint();

	std::string pic_path = "./image/test11.jpg";
	cv::Mat img = cv::imread(pic_path, cv::IMREAD_GRAYSCALE);
	if (img.empty())
	{
		std::cout << "read image failed" << std::endl;
		system("pause");
		return 0;
	}
	img = 255 - img;
	down(img);

	cv::Mat binary = binaryzation(img, 100);
	cv::Mat labels = cv::Mat::zeros(binary.size(), CV_32SC1);
	int labels_num = bfs(binary, labels);
	std::vector<std::vector<int>> boundary = get_boundary(labels, labels_num);
	std::vector<int> order = get_order(labels, labels_num);
	std::vector<cv::Mat> imgs = cut(img, boundary, order);
	std::vector<cv::Mat> resize_imgs = resize_(imgs, 28, 28);

	cv::Ptr<cv::ml::ANN_MLP> ann = cv::ml::StatModel::load<cv::ml::ANN_MLP>(model_path);
	long long result = 0;

	for (int i = 0; i < resize_imgs.size(); i++)
	{
		cv::Mat cur_img = resize_imgs[i];
		cv::Mat img_show = cur_img.clone();
		cur_img.convertTo(cur_img, CV_32F);
		cur_img = cur_img / 255.0;
		cv::Mat pre_img = cur_img.reshape(1, 1);

		cv::Mat pre_out;
		float ret = ann->predict(pre_img, pre_out);
		double maxVal = 0;
		cv::Point maxPoint;
		cv::minMaxLoc(pre_out, NULL, &maxVal, NULL, &maxPoint);
		int max_index = maxPoint.x;
		result = result * 10 + max_index;

		std::cout << "max_index:" << max_index << "; ";
		std::cout << "maxVal:" << maxVal << std::endl;
		cv::imshow("img", img_show);
		cv::imwrite("./image/out/" + std::to_string(i) + ".jpg", img_show);
		cv::waitKey(0);
	}

	std::cout << "result: " << result << std::endl;

	return 0;
}