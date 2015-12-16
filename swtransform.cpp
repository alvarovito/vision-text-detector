#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <unordered_map>
#include "DisjointSets.h"
#include "Component.h"

#define PI 3.14159265


using namespace cv;

/// Global variables
Mat gradient_x, abs_grad_x, gradient_y, abs_grad_y;
Mat grad;

int strokeMedianFilter(Mat *SWTimage, std::vector<std::vector<Point2d>> *saved_rays);
int strokeWidth(int i, int j, short g_x, short g_y, Mat *SWTimage, std::vector<std::vector<Point2d>> *saved_rays, Mat *edge_image);
int getSteps(int g_x, int g_y, float& step_x, float& step_y);
Mat SWTransform(Mat* edge_image);
std::vector<Component> legallyConnectedComponents(Mat* SWTimage);
std::vector<Component> filterComponents(Mat * SWTimage, std::vector<Component> components);
std::vector<std::vector<Component>> connectedPairs(std::vector<Component> valid_components);

/** @function main */

int main(int argc, char** argv)
{
	Mat src, src_gray;
	/// Load an image
	//src = imread("road_signs.png");
	src = imread("australiasigns.jpg");

	if (!src.data)
	{
		std::cout << "Couldn't read the image" << std::endl;
		return -1;
	}

	/// Convert the image to grayscale
	cvtColor(src, src_gray, CV_BGR2GRAY);
	Mat filtered;
	// Reduce noise with a 3x3 gaussian kernel
	blur(src_gray, filtered, Size(3, 3));
	/// Canny detector
	Mat detected_edges;
	Canny(filtered, detected_edges, 100, 300, 3);
	//gradients
	Mat gradxx;
	Sobel(filtered, gradient_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	gradient_x.copyTo(gradxx);
	//Scharr(filtered, gradient_x, CV_16S, 1, 0, 3);
	convertScaleAbs(gradient_x, abs_grad_x);
	Sobel(filtered, gradient_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	//Scharr(filtered, gradient_y, CV_16S, 0, 1, 3);
	convertScaleAbs(gradient_y, abs_grad_y);
	/// Total Gradient (approximate)
	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

	Mat SWTimage = SWTransform(&detected_edges);


	std::vector < Component > components;
	components = legallyConnectedComponents(&SWTimage);
	std::vector < Component > valid_components;
	valid_components = filterComponents(&SWTimage, components);
	std::vector<std::vector<Component>> words;
	words = connectedPairs(valid_components);

	Mat components_image = cv::Mat(src.rows, src.cols, CV_8UC3, cv::Scalar(0, 0, 0));

	int count = 0;
	for (std::vector<std::vector<Component>>::iterator it1 = words.begin(); it1 != words.end(); ++it1){
		for (std::vector<Component>::iterator it2 = it1->begin(); it2 != it1->end(); ++it2){
			int color0 = rand() % 256;
			int color1 = rand() % 256;
			int color2 = rand() % 256;
			for (std::vector<Point2d>::iterator it3 = it2->points.begin(); it3 != it2->points.end(); ++it3){
				components_image.at<cv::Vec3b>(it3->y, it3->x)[0] = color0;
				components_image.at<cv::Vec3b>(it3->y, it3->x)[1] = color1;
				components_image.at<cv::Vec3b>(it3->y, it3->x)[2] = color2;
			}
			count++;
		}
	}
	std::cout << "Components after filtering" << count << std::endl;

	////////////////////////////////////Rotate certain angle componenents///////////////////////////////////////
	/*for (std::vector<Component>::iterator it1 = valid_components.begin(); it1 != valid_components.end(); ++it1){
		int color0 = rand() % 256;
		int color1 = rand() % 256;
		int color2 = rand() % 256;
		std::vector<Point2d> rotated_component = it1->rotate_component(270 * PI / 180);
		for (std::vector<Point2d>::iterator it2 = rotated_component.begin(); it2 != rotated_component.end(); ++it2){
			if (it2->x >= 0 && it2->x < components_image.cols && it2->y >= 0 && it2->y < components_image.rows){
				components_image.at<cv::Vec3b>(it2->y, it2->x)[0] = color0;
				components_image.at<cv::Vec3b>(it2->y, it2->x)[1] = color1;
				components_image.at<cv::Vec3b>(it2->y, it2->x)[2] = color2;
			}
		}
	}*/
	
	////////////////////////////////////Before filtering components//////////////
	//count = 0; 
	//Mat components_image_before = cv::Mat(src.rows, src.cols, CV_8UC3, cv::Scalar(0, 0, 0));
	//for (std::vector<Component>::iterator it1 = components.begin(); it1 != components.end(); ++it1){
	//	int color0 = rand() % 256;
	//	int color1 = rand() % 256;
	//	int color2 = rand() % 256;
	//	for (std::vector<Point2d>::iterator it2 = it1->points.begin(); it2 != it1->points.end(); ++it2){
	//		components_image_before.at<cv::Vec3b>(it2->y, it2->x)[0] = color0;
	//		components_image_before.at<cv::Vec3b>(it2->y, it2->x)[1] = color1;
	//		components_image_before.at<cv::Vec3b>(it2->y, it2->x)[2] = color2;
	//	}
	//	count++;
	//}
	//std::cout << "Components before filtering" << count << std::endl;

	

	////////////////////////Prettier SWTimage/////////////////
	//SWTimage.convertTo(SWTimage, CV_8U, 1.0);
	//uchar *ptr4;
	//for (int i = 0; i < SWTimage.rows; i++){
	//	ptr4 = SWTimage.ptr<uchar>(i);
	//	for (int j = 0; j < SWTimage.cols; j++){
	//		if (ptr4[j] == 0){
	//			ptr4[j] = 255;
	//		}
	//		else
	//			ptr4[j] = (int)(ptr4[j] * 3 );
	//	}
	//}

	///////////////////Boxes over original image////////////////
	//for (std::vector<Component>::iterator it = valid_components.begin(); it != valid_components.end(); it++){
	//	Point vertex1;
	//	vertex1.x = it->minx;
	//	vertex1.y = it->miny;
	//	Point vertex2;
	//	vertex2.x = it->maxx;
	//	vertex2.y = it->maxy;
	//	rectangle(src, vertex1, vertex2, Scalar(0, 0, 255));
	//}

	//Create windo canvas to show img
	namedWindow("original", CV_WINDOW_AUTOSIZE);
	namedWindow("gray", CV_WINDOW_AUTOSIZE);
	namedWindow("filtered", CV_WINDOW_AUTOSIZE);
	namedWindow("edges", CV_WINDOW_AUTOSIZE);
	namedWindow("gradientx", CV_WINDOW_AUTOSIZE);
	namedWindow("gradienty", CV_WINDOW_AUTOSIZE);
	namedWindow("grad", CV_WINDOW_AUTOSIZE);
	namedWindow("SWT", CV_WINDOW_AUTOSIZE);
	namedWindow("connected_components", CV_WINDOW_AUTOSIZE);
	//Show image in the name of the window
	imshow("original", src);
	imshow("gray", src_gray);
	imshow("filtered", filtered);
	imshow("edges", detected_edges);
	imshow("gradientx", abs_grad_x);
	imshow("gradienty", abs_grad_y);
	imshow("grad", grad);
	imshow("SWT", SWTimage);
	imshow("connected_components", components_image);

	/// Wait until user exit program by pressing a key
	waitKey(0);

	return 0;
}

std::vector<Component> filterComponents(Mat * SWTimage, std::vector<Component> components){
	std::vector<Component> valid_components;
	for (std::vector<Component>::iterator it = components.begin(); it != components.end(); it++){
		it->getStats(SWTimage);
		if (it->isValid())
			valid_components.push_back(*it);
	}
	return valid_components;
}

std::vector<std::vector<Component>> connectedPairs(std::vector<Component> valid_components){
	DisjointSets words(valid_components.size());

	for (int i = 0; i < valid_components.size(); i++){
		for (int j = 0; j < valid_components.size(); j++){
			if (valid_components[i].getCenter().x < valid_components[j].getCenter().x){
				if (valid_components[j].getCenter().y < valid_components[i].maxy){
					if (valid_components[j].getCenter().y > valid_components[i].miny){
						if (valid_components[j].isSimilar(valid_components[i])){
							words.Union(i, j);
						}
					}
				}
			}
		}
	}
	typedef std::unordered_map<int, std::vector<Component>> componentmap;
	componentmap map_words;

	for (int i = 0; i < valid_components.size(); i++){
		int word = words.FindSet(i);
		map_words[word].push_back(valid_components[i]);
	}

	std::vector<std::vector<Component>> words_vector;
	for (componentmap::iterator it = map_words.begin(); it != map_words.end(); it++){
		if (it->second.size() > 3)
			words_vector.push_back(it->second);
	}
	
	//for (std::vector<Component>::iterator it = valid_components.begin(); it != valid_components.end(); it++){
	//	bool matched = false;
	//	std::vector<Component> canidates = valid_components;
	//	while (canidates.empty() != 0 && matched != false){
	//		if (canidates.back().getCenter().x > it->getCenter().x){
	//			if (canidates.back().getCenter().y < it->maxy && canidates.back().getCenter().y > it->miny){
	//				if (it->isSimilar(canidates.back())){
	//					pair_letters.push_back(*it);
	//					matched = true;
	//				}	
	//			}
	//		}
	//		canidates.pop_back();
	//	}/*
	//	for (std::vector<Component>::iterator it2 = valid_components.begin(); it2 != valid_components.end(); it2++){
	//		if (it2->getCenter().x > it->getCenter().x){
	//			if (it2->getCenter().y < it->maxy && it2->getCenter().y > it->miny){
	//				if (it->isSimilar(*it2))
	//					pair_letters.push_back(*it);
	//			}
	//		}
	//	}*/
	//}
	return words_vector;
}

Mat SWTransform(Mat* edge_image){

	// accept only char and one channel type matrices
	CV_Assert(edge_image->depth() != sizeof(uchar));
	CV_Assert(edge_image->channels() == 1); 

	int nRows = edge_image->rows;
	int nCols = edge_image->cols;
	
	Mat SWTimage = cv::Mat(nRows, nCols, CV_32FC1, cv::Scalar(-1.0));
	std::vector<std::vector<Point2d>> saved_rays;
	/*if (I.isContinuous())
	{
	nCols *= nRows;
	nRows = 1;
	}*/

	int i, j;
	uchar* p;
	short g_x, g_y;
	for (i = 0; i < nRows; ++i)
	{
		p = edge_image->ptr<uchar>(i);
		for (j = 0; j < nCols; ++j)
		{	
			//found a edge
			if (p[j] == 255){
				//if (i == 22 && j == 138){
				//		int a; //Debugging code. Used to access specific pixel
				//}
				//gradient on the edge. Negative gradient to find dark text over bright background.
				g_x = -(gradient_x.at<short>(i, j));
				g_y = -(gradient_y.at<short>(i, j));
				strokeWidth(j, i, g_x, g_y, &SWTimage, &saved_rays, edge_image);
			}
		}
	}

	strokeMedianFilter(&SWTimage, &saved_rays);

	return SWTimage;
}

std::vector<Component> legallyConnectedComponents(Mat* SWTimage){
	Mat labels = cv::Mat(SWTimage->rows, SWTimage->cols, CV_16U, cv::Scalar(0));
	DisjointSets equivalent_connections(1); //created with one element (background)
	short next_label = 1;

	float* ptr;
	for (int i = 0; i < SWTimage->rows; i++){
		ptr = SWTimage->ptr<float>(i);
		for (int j = 0; j < SWTimage->cols; j++){
			if (ptr[j] > 0){
				std::vector<int> neighbour_labels;
				// check pixel to the west, west-north, north, north-east
				if (j - 1 >= 0) { //if not out of image
					float west = SWTimage->at<float>(i, j - 1);
					if (west > 0){
						//if (ptr[j] / west <= 1.5 || west / ptr[j] <= 1.5){
						if (max(ptr[j], west) / min(ptr[j], west) <= 2.0){
							int west_label = labels.at<short>(i, j - 1);
							if (west_label != 0)
								neighbour_labels.push_back(west_label);
						}
					}
				}
				if (i - 1 >= 0) { //if not out of image
					if (j - 1 >= 0){ //if not out of image
						float west_north = SWTimage->at<float>(i - 1, j - 1);
						if (west_north > 0){
							//if (ptr[j] / west_north <= 1.5 || west_north / ptr[j] <= 1.5){
							if (max(ptr[j], west_north) / min(ptr[j], west_north) <= 2.0){
								int west_north_label = labels.at<short>(i - 1, j - 1);
								if (west_north_label != 0)
									neighbour_labels.push_back(west_north_label);
							}
						}
					}
					float north = SWTimage->at<float>(i - 1, j);
					if (north > 0){
						//if (ptr[j] / north <= 1.5 || north / ptr[j] <= 1.5){
						if (max(ptr[j], north) / min(ptr[j], north) <= 2.0){
							int north_label = labels.at<short>(i - 1, j);
							if (north_label != 0)
								neighbour_labels.push_back(north_label);
						}
					}
					if (j + 1 < SWTimage->cols){ //if not out of image
						float north_east = SWTimage->at<float>(i - 1, j + 1);
						if (north_east > 0){
							//if (ptr[j] / north_east <= 1.5 || north_east / ptr[j] <= 1.5){
							if (max(ptr[j], north_east) / min(ptr[j], north_east) <= 2.0){
								int north_east_label = labels.at<short>(i - 1, j + 1);
								if (north_east_label != 0)
									neighbour_labels.push_back(north_east_label);
							}
						}
					}
				}
				//if neighbours have no label. Set new label(component)
				if (neighbour_labels.empty()){
					labels.at<short>(i, j) = next_label;
					next_label++;
					equivalent_connections.AddElements(1);
				}
				else{//find minium label in connected neighbours. Save the equivalent connections for the second pass
					int minimum = neighbour_labels[0];
					if (neighbour_labels.size() > 1){
						for (std::vector<int>::iterator it = neighbour_labels.begin(); it != neighbour_labels.end(); it++){
							minimum = std::min(minimum, *it);
							equivalent_connections.Union(equivalent_connections.FindSet(neighbour_labels[0]), equivalent_connections.FindSet(*it));
						}
					}
					labels.at<short>(i, j) = minimum;
				}

			}
		}
	}
	//second pass. Assign component label from the equivalent connections list
	short * ptr2;
	typedef std::unordered_map<short, std::vector<Point2d>> componentmap;
	componentmap map;

	for (int i = 0; i < labels.rows; i++){
		ptr2 = labels.ptr<short>(i);
		for (int j = 0; j < labels.cols; j++){
			if (ptr2[j] != 0){
				ptr2[j] = equivalent_connections.FindSet(ptr2[j]);
				Point2d point;
				point.x = j;
				point.y = i;
				map[ptr2[j]].push_back(point);
			}
		}
	}

	std::vector<Component> components;
	for (componentmap::iterator it = map.begin(); it != map.end(); it++){
		components.push_back(Component(it->second));
	}


	return components;
}

int strokeMedianFilter(Mat *SWTimage, std::vector<std::vector<Point2d>> *saved_rays){
	/*Compute median of each ray. Values higher than the median are set to median value. This function is necessary to avoid
	bad stroke values on corners.*/

	float median;
	for (std::vector<std::vector<Point2d>>::iterator it1 = saved_rays->begin(); it1 != saved_rays->end(); ++it1){
		std::vector<float> swt_values;
		for (std::vector<Point2d>::iterator it2 = it1->begin(); it2 != it1->end(); ++it2){
			swt_values.push_back(SWTimage->at<float>(it2->y, it2->x));
		}
		std::sort(swt_values.begin(), swt_values.end());
		median = swt_values[round(swt_values.size() / 2)];
		for (std::vector<Point2d>::iterator it2 = it1->begin(); it2 != it1->end(); ++it2){
			SWTimage->at<float>(it2->y, it2->x) = std::min(median, SWTimage->at<float>(it2->y, it2->x));
		}
	}

	return 0;
}

int strokeWidth(int j, int i, short g_x, short g_y, Mat *SWTimage,
	std::vector<std::vector<Point2d>> *saved_rays, Mat *edge_image){
	/*This function follow the g_x/g_y gradient direction from the D(j,i) point until finding a another edge pint P. 
	Then both gradient direction are compared. If the dD = -dP +- PI/2 all the pixeles that connect D-P are labeld with the DP length*/

	float step_x, step_y;
	//get steps to move on the gradient direction
	getSteps(g_x, g_y, step_x, step_y);
	float sum_x = (float)j;
	float sum_y = (float)i;
	std::vector<Point2d> ray;
	Point2d point;
	point.x = j; point.y = i;
	ray.push_back(point);

	while (true){
		sum_x += step_x;
		sum_y += step_y;

		int x = (int)round(sum_x);
		int y = (int)round(sum_y);
		
		//pixel out of image
		if (x < 0 || (x >= (*SWTimage).cols) || y < 0 || (y >= (*SWTimage).rows)) {
			return 0;
		}

		point.x = x;
		point.y = y;
		ray.push_back(point); 
		
		//matched a edge 
		if ((*edge_image).at<uchar>(y, x) > 0){
			short gx_new = -(gradient_x.at<short>(y, x));
			short gy_new = -(gradient_y.at<short>(y, x));

			//get angles 2PI to 4PI (to have always positive angles)
			float angle_ij = atan2(-(g_y), -(g_x)) + PI +2*PI;
			float angle_yx = atan2(-(gy_new), -(gx_new)) + PI +2*PI;

			// if dp = - dq +- PI/2
			if (angle_ij >= angle_yx) angle_ij = angle_ij - PI;
			else angle_ij = angle_ij + PI;
			if (angle_ij  < angle_yx + PI / 2  && angle_ij  > angle_yx - PI / 2 ){
				//(*SWTimage).at<float>(y, x) = 255;
				//(*SWTimage).at<float>(i, j) = 255;
				float length = sqrt((y-i)*(y-i) + (x-j)*(x-j));
				saved_rays->push_back(ray);
				while (!ray.empty()){
					int x_ray = ray.back().x;
					int y_ray = ray.back().y;
					if ((*SWTimage).at<float>(y_ray, x_ray) < 0)
						(*SWTimage).at<float>(y_ray, x_ray) = length;
					else
						(*SWTimage).at<float>(y_ray, x_ray) = std::min((*SWTimage).at<float>(y_ray, x_ray), length);
					ray.pop_back();
				}
				return 1;
			}
			return 0;
		}
	}
}

int getSteps(int g_x, int g_y, float& step_x, float& step_y){
	float absg_x = abs(g_x);
	float absg_y = abs(g_y);
	if (g_x == 0){
		step_x = 0;
		if (g_y > 0) step_y = 1;
		else step_y = -1;
		return 0;
	}
	if (g_y == 0){
		step_y = 0;
		if (g_x > 0) step_x = 1;
		else step_x = -1;
		return 0;
	}
	if (absg_x > absg_y){
		step_y = g_y / absg_x;
		if (g_x > 0) step_x = 1;
		else step_x = -1;
	}
	else{
		step_x = g_x / absg_y;
		if (g_y > 0) step_y = 1;
		else step_y = -1;
	}
	return 0;
}
