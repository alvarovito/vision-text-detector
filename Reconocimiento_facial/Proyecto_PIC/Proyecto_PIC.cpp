// Proyecto_PIC.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"

#define ESCAPE 27

#include "opencv2/opencv.hpp"
#include "vector"
#include <iostream>

using namespace cv;
using namespace std;

Mat calcHistogram(Mat img) {

	int histSize = 256;
	float range[] = { 0, 255 };
	const float *ranges[] = { range };

	Mat hist;
	calcHist(&img, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false);

	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC1, Scalar(0, 0, 0));

	for (int i = 1; i < histSize; i++) {

		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}
	return histImage;
}

void DrawMarker(Mat& image, Rect rect, string msg, int LINE_WIDTH)
{
	Rect r = rect;
	Scalar DETECT_COLOR = CV_RGB(0, 255, 0);

	line(image, Point(r.x, r.y), Point(r.x, r.y + LINE_WIDTH), Scalar(0,255,0), 3);
	line(image, Point(r.x, r.y), Point(r.x + LINE_WIDTH, r.y), Scalar(0, 255, 0), 3);

	line(image, Point(r.x + r.width, r.y), Point(r.x + r.width, r.y + LINE_WIDTH), Scalar(0, 255, 0), 3);
	line(image, Point(r.x + r.width, r.y), Point(r.x + r.width - LINE_WIDTH, r.y), Scalar(0, 255, 0), 3);

	line(image, Point(r.x, r.y + r.height), Point(r.x, r.y + r.height - LINE_WIDTH), Scalar(0, 255, 0), 3);
	line(image, Point(r.x, r.y + r.height), Point(r.x + LINE_WIDTH, r.y + r.height), Scalar(0, 255, 0), 3);

	line(image, Point(r.x + r.width, r.y + r.height), Point(r.x + r.width, r.y + r.height - LINE_WIDTH), Scalar(0, 255, 0), 3);
	line(image, Point(r.x + r.width, r.y + r.height), Point(r.x + r.width - LINE_WIDTH, r.y + r.height), Scalar(0, 255, 0), 3);

	int font = FONT_HERSHEY_DUPLEX;
	Size s = getTextSize(msg, font, 1, 1, 0);

	int x = rect.x + (rect.width - s.width) / 2;
	int y = rect.y + rect.height + s.height + 5;

	putText(image, msg, Point(x, y), font, 1, Scalar(0, 0, 255), 1, CV_AA);
}

bool training(Ptr<FaceRecognizer> model,map<int,string> &names) {

	vector<Mat> images;
	vector<int> labels;

	images.push_back(imread("Miguel/ROI1.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI2.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI3.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI4.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI5.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI6.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI7.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI8.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI9.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI10.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI11.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI12.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI13.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI14.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI15.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI16.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI17.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI18.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI19.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI20.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI21.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI22.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI23.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI24.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI25.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI26.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI27.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI28.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI29.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI30.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI31.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI32.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI33.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI34.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI35.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI36.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI37.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI38.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI39.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI40.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI41.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI42.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI43.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI44.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI45.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI46.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI47.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI48.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI49.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);
	images.push_back(imread("Miguel/ROI50.jpg", CV_LOAD_IMAGE_GRAYSCALE)); labels.push_back(0);

	model->train(images, labels);
	images.clear();
	names[0] = "Miguel";

	return true;
}

map <int, Rect> sort(map <int, Rect> ids, int lenght) {

	Rect Temp;

	for (int j = 0; j < lenght - 1; j++){

		if (ids[j].x > ids[j + 1].x) {

			Temp = ids[j];
			ids[j] = ids[j + 1];
			ids[j + 1] = Temp;
		}
	}

	return ids;
}

int main(int argc, char* argv[]) {

	vector<Rect> faces;
	bool holdImage = false;
	
	Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
	map<int, string> names;
	map<int, Rect> ids;
	bool init_id = true;
	int distthreshold = 10;
	Mat equal;
	int selected = -1;

	if (!training(model, names)) {
		cout << "Error in training" << endl;
		return -1;
	}

	CascadeClassifier face_cascade;

	if (!face_cascade.load("haarcascade_frontalface_alt.xml")) {
		cout << "Cannot load face xml!" << endl;
		return -1;
	}

	VideoCapture inputvideo(0);

	if (!inputvideo.isOpened())
		return -1;

	namedWindow("Face Detection", WINDOW_AUTOSIZE);

	for (;;) {

		Mat frame;
		Mat workingImage;
		Mat image;
		Mat selection;
		inputvideo >> frame;
		int id = -1;
		double confidence = 0.0;
		ids.clear();
		Rect temp;
		if (frame.empty())
			break;

		frame.copyTo(image);
		cvtColor(image, workingImage, COLOR_BGR2GRAY);
		equalizeHist(workingImage, workingImage);

		face_cascade.detectMultiScale(workingImage, faces, 1.1, 8, CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_CANNY_PRUNING, Size(50, 50), Size(300, 300));

		int n = faces.size();
		Mat rois[5];
		Mat nface;

		for (size_t i = 0;i < faces.size(); i++) {

			ids[i] = faces[i];

			if (i < 5) {

				rois[i] = Mat(image, Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height)).clone();
				resize(rois[i], rois[i], Size(image.cols / 5, image.rows / 5));
			}
		}
		
		ids = sort(ids, ids.size());

		for (size_t i = 0;i < faces.size(); i++) {

			if (i == selected && faces.size() > selected) {

				selection = Mat(image, ids[selected]);
				cvtColor(selection, nface, CV_BGR2GRAY);
				resize(nface, nface, Size(100, 100));
				equalizeHist(nface, nface);
				resize(selection, selection, Size(image.cols *0.23,image.rows * 0.23));
				selection.copyTo(image.rowRange(image.rows - selection.rows, image.rows).colRange(0, selection.cols));

				model->set("threshold", 80);
				model->predict(nface, id, confidence);

				if (id >= 0) {

					string msg = names[id] + " : " + to_string((int)confidence);
					putText(image, msg, Point(0, image.rows - selection.rows - 10), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 255), 1, CV_AA);

				}
			}

			DrawMarker(image, ids[i], "" + to_string(i), 10);
		}

		int keypressed = waitKey(10);

		if (keypressed == ESCAPE) {

			workingImage.release();
			frame.release();
			break;
		}
		else if (keypressed == 32) {

				/*Mat ROI = Mat(workingImage, faces[0]);
				//resize(ROI, ROI, Size(100, 100));
				Mat equal;
				//equalizeHist(ROI, equal);
				equal = imread("Miguel/ROI14.jpg");
				GaussianBlur(equal, equal, Size(5, 5), 0, 0);
				imwrite("Miguel/ROI44.jpg", equal);
				equal = imread("Miguel/ROI15.jpg");
				GaussianBlur(equal, equal, Size(5, 5), 0, 0);
				imwrite("Miguel/ROI45.jpg", equal);
				equal = imread("Miguel/ROI16.jpg");
				GaussianBlur(equal, equal, Size(5, 5), 0, 0);
				imwrite("Miguel/ROI46.jpg", equal);
				equal = imread("Miguel/ROI17.jpg");
				GaussianBlur(equal, equal, Size(5, 5), 0, 0);
				imwrite("Miguel/ROI47.jpg", equal);
				equal = imread("Miguel/ROI18.jpg");
				GaussianBlur(equal, equal, Size(5, 5), 0, 0);
				imwrite("Miguel/ROI48.jpg", equal);
				equal = imread("Miguel/ROI19.jpg");
				GaussianBlur(equal, equal, Size(5, 5), 0, 0);
				imwrite("Miguel/ROI49.jpg", equal);
				equal = imread("Miguel/ROI20.jpg");
				GaussianBlur(equal, equal, Size(5, 5), 0, 0);
				imwrite("Miguel/ROI50.jpg", equal);*/

			if (holdImage)
				holdImage = false;
			else
				holdImage = true;
		}
		else if (keypressed >= 48 && keypressed < 57 && faces.size() > keypressed - 48)		// 0 = 48 y 9 = 57
			selected = keypressed - 48;

		for (int i = 0;i < size(rois); i++) {

			if (holdImage && rois[i].data) {

				rois[i].copyTo(image.rowRange(image.rows - rois[i].rows*(5 - i), rois[i].rows*(i + 1)).colRange(image.cols - rois[i].cols, image.cols));
			}
		}

		imshow("Face Detection", image);

		waitKey(10);

		workingImage.release();
		frame.release();
	}

	destroyAllWindows();

	return 0;
}