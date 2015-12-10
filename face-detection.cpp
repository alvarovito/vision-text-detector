// Practica_PIC.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"

#define ESCAPE 27

#include "opencv/cv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

#include <iostream>

using namespace cv;
using namespace std;

Mat calcHistogram(Mat img) {

	namedWindow("histogram", CV_WINDOW_AUTOSIZE);

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

int main(int argc, char* argv[]) {

	vector<Rect> faces;
	vector <int> numdetections;
	bool holdImage = true;

	CascadeClassifier face_cascade;

	if (!face_cascade.load("haarcascade_frontalface_alt.xml")) {
		cout << "Cannot load face xml!" << endl;
		return -1;
	}

	VideoCapture inputvideo(0);

	if (!inputvideo.isOpened())
		return -1;

	namedWindow("Face Detection", WINDOW_AUTOSIZE);
	//namedWindow("Working Image", WINDOW_AUTOSIZE);
	//namedWindow("Temp", WINDOW_AUTOSIZE);

	for (;;) {

		Mat frame;
		Mat workingImage;
		Mat image;
		inputvideo >> frame;

		if (frame.empty())
			break;

		frame.copyTo(image);
		cvtColor(image, workingImage, COLOR_BGR2GRAY);
		equalizeHist(workingImage, workingImage);

		//Mat histImage = calcHistogram(workingImage);
		//imshow("histogram", histImage);

		face_cascade.detectMultiScale(workingImage, faces, numdetections, 1.1, 8, CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_CANNY_PRUNING, Size(50,50), Size(300,300));

		int n = faces.size();
		Mat rois[5];

		for (size_t i = 0;i < faces.size(); i++) {

			if (i < 5) {

				rois[i] = Mat(image, Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height)).clone();
				resize(rois[i], rois[i], Size(image.cols / 5, image.rows / 5));
			}
		}

		for (size_t i = 0;i < faces.size(); i++) {

			rectangle(image,
				Point(faces[i].x, faces[i].y),
				Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height),
				Scalar(0, 255, 0), 2);
			putText(image, "Coincidences = " + std::to_string(numdetections[i]), Point(faces[i].x - faces[i].width * 0.25, faces[i].y), 2, 1, Scalar(0, 0, 255));
		}
		
		
		for (int i = 0;i < size(rois); i++) {

			if (holdImage && rois[i].data) {

				rois[i].copyTo(image.rowRange(image.rows - rois[i].rows*(5-i), rois[i].rows*(i+1)).colRange(image.cols - rois[i].cols, image.cols));
			}
		}

		imshow("Face Detection", image);
		//imshow("Working Image", workingImage);

		int keypressed = waitKey(10);

		if (keypressed == ESCAPE) {
			
			workingImage.release();
			frame.release();
			break;
		}

		workingImage.release();
		frame.release();
	}

	destroyAllWindows();

	return 0;
}
