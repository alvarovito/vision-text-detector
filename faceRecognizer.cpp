// Practica_PIC.cpp: define el punto de entrada de la aplicación de consola.
//

#include "stdafx.h"

#define ESCAPE 27

#include "opencv/cv.hpp"

#include <iostream>

using namespace cv;
using namespace std;

CascadeClassifier face_cascade;
CascadeClassifier lEyeDetector;
CascadeClassifier rEyeDetector;

float EYE_SX = 0.12f;
float EYE_SY = 0.17f;
float EYE_SW = 0.37f;
float EYE_SH = 0.36f;

double DESIRED_LEFT_EYE_Y = 0.14;
double DESIRED_LEFT_EYE_X = 0.19;

int FaceWidth = 100;
int FaceHeight = 100;

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

bool EyeDetection(Mat& image, Rect& face_detected, Rect& lEye, Rect& rEye){



	Mat face = image(face_detected);

	int leftX = cvRound(face.cols * EYE_SX);
	int topY = cvRound(face.rows * EYE_SY);
	int widthX = cvRound(face.cols * EYE_SW);
	int heightY = cvRound(face.rows * EYE_SH);
	int rightX = cvRound(face.cols * (1.0 - EYE_SX - EYE_SW));

	Mat topLeftOfFace = face(Rect(leftX, topY, widthX, heightY));
	Mat topRightOfFace = face(Rect(rightX, topY, widthX, heightY));

	vector<Rect> lEyeR, rEyeR;

	lEyeDetector.detectMultiScale(topLeftOfFace, lEyeR, 1.1, 3, CASCADE_DO_ROUGH_SEARCH);
	rEyeDetector.detectMultiScale(topRightOfFace, rEyeR, 1.1, 3, CASCADE_DO_ROUGH_SEARCH);

	if (lEyeR.size() == 1 && rEyeR.size() == 1)
	{
		lEye = lEyeR[0];
		rEye = rEyeR[0];

		lEye.x += leftX;
		lEye.y += topY;

		rEye.x += rightX;
		rEye.y += topY;

		return true;
	}

	return false;
}

void CropFace(Mat& face, Mat& warped, Rect leftEye, Rect rightEye)
{
	Point left = Point(leftEye.x + leftEye.width / 2, leftEye.y + leftEye.height / 2);
	Point right = Point(rightEye.x + rightEye.width / 2, rightEye.y + rightEye.height / 2);
	Point2f eyesCenter = Point2f((left.x + right.x) * 0.5f, (left.y + right.y) * 0.5f);

	// Get the angle between the 2 eyes.
	double dy = (right.y - left.y);
	double dx = (right.x - left.x);
	double len = sqrt(dx*dx + dy*dy);
	double angle = atan2(dy, dx) * 180.0 / CV_PI;

	// Hand measurements shown that the left eye center should ideally be at roughly (0.19, 

0.14) of a scaled face image.
	const double DESIRED_RIGHT_EYE_X = (1.0f - DESIRED_LEFT_EYE_X);

	// Get the amount we need to scale the image to be the desired fixed size we want.
	double desiredLen = (DESIRED_RIGHT_EYE_X - DESIRED_LEFT_EYE_X) * FaceWidth;
	double scale = desiredLen / len;

	// Get the transformation matrix for rotating and scaling the face to the desired angle 

& size.
	Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, scale);

	// Shift the center of the eyes to be the desired center between the eyes.
	rot_mat.at<double>(0, 2) += FaceWidth * 0.5f - eyesCenter.x;
	rot_mat.at<double>(1, 2) += FaceHeight * DESIRED_LEFT_EYE_Y - eyesCenter.y;

	warped = Mat(FaceHeight, FaceWidth, CV_8U, Scalar(128));

	warpAffine(face, warped, rot_mat, warped.size());
}

void DrawMarker(Mat& dst, Rect rect, string msg, int LINE_WIDTH)
{
	Rect r = rect;
	Scalar DETECT_COLOR = CV_RGB(0, 255, 0);

	line(dst, Point(r.x, r.y), Point(r.x, r.y + LINE_WIDTH), DETECT_COLOR, 3);
	line(dst, Point(r.x, r.y), Point(r.x + LINE_WIDTH, r.y), DETECT_COLOR, 3);

	line(dst, Point(r.x + r.width, r.y), Point(r.x + r.width, r.y + LINE_WIDTH), 

DETECT_COLOR, 3);
	line(dst, Point(r.x + r.width, r.y), Point(r.x + r.width - LINE_WIDTH, r.y), 

DETECT_COLOR, 3);

	line(dst, Point(r.x, r.y + r.height), Point(r.x, r.y + r.height - LINE_WIDTH), 

DETECT_COLOR, 3);
	line(dst, Point(r.x, r.y + r.height), Point(r.x + LINE_WIDTH, r.y + r.height), 

DETECT_COLOR, 3);

	line(dst, Point(r.x + r.width, r.y + r.height), Point(r.x + r.width, r.y + r.height - 

LINE_WIDTH), DETECT_COLOR, 3);
	line(dst, Point(r.x + r.width, r.y + r.height), Point(r.x + r.width - LINE_WIDTH, r.y + 

r.height), DETECT_COLOR, 3);

	int font = FONT_HERSHEY_DUPLEX;
	Size s = getTextSize(msg, font, 1, 1, 0);

	int x = (dst.cols - s.width) / 2;
	int y = rect.y + rect.height + s.height + 5;

	putText(dst, msg, Point(x, y), font, 1, CV_RGB(255, 0, 0), 1, CV_AA);
}

bool Init()
{

	if (!face_cascade.load("haarcascade_frontalface_alt_tree.xml"))
	{
		cout << "No se encuentra el archivo haarcascade_frontalface_alt_tree.xml" << 

endl;
		return false;
	}

	if (!lEyeDetector.load("haarcascade_eye_tree_eyeglasses.xml"))
	{
		cout << "No se encuentra el archivo haarcascade_eye_tree_eyeglasses.xml" << 

endl;
		return false;
	}

	if (!rEyeDetector.load("haarcascade_eye_tree_eyeglasses.xml"))
	{
		cout << "No se encuentra el archivo haarcascade_eye_tree_eyeglasses.xml" << 

endl;
		return false;
	}

	return true;
}

int main(int argc, char* argv[]) {

	vector<Rect> faces;
	vector <int> numdetections;
	bool holdImage = false;

	Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
	vector<Mat> rostros;
	vector<int> ids;
	map<int, string> names;

	bool entrenamiento = false;
	bool agregarRostro = false;
	bool entrenar = false;
	int identificador = 0, capCount = 0;

	string msg1 = "Reconocimiento Facial \n\n\t[E] Iniciar Entrenamiento \n\t[ESC] Salir

\n";
	string msg2 = "Reconocimiento Facial \n\n\t[A] Capturar Rostro \n\t[T] Finalizar 

Entrenamiento \n\t[ESC] Salir\n";
	cout << msg1;

	if (!Init()){
		cout << "error al cargas los archivos .xml" << endl;
		return 1;
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
		Rect lEye, rEye;
		inputvideo >> frame;

		if (frame.empty())
			break;

		frame.copyTo(image);
		cvtColor(image, workingImage, COLOR_BGR2GRAY);
		equalizeHist(workingImage, workingImage);

		//Mat histImage = calcHistogram(workingImage);
		//imshow("histogram", histImage);

		face_cascade.detectMultiScale(workingImage, faces, numdetections, 1.1, 8, 

CV_HAAR_SCALE_IMAGE | CV_HAAR_DO_CANNY_PRUNING, Size(50, 50), Size(300, 300));

		int n = faces.size();

		if (entrenamiento) {

			if (n == 1 && EyeDetection(workingImage, faces[0], lEye, rEye)) {

				Mat nface;
				CropFace(workingImage(faces[0]), nface, lEye, rEye);

				if (agregarRostro)
				{
					Mat fface;

					flip(nface, fface, 1);
					rostros.push_back(fface);
					ids.push_back(identificador);

					rostros.push_back(nface);
					ids.push_back(identificador);
					agregarRostro = false;

					capCount += 1;
					cout << "Se han capturado " << capCount << " Rostros" 

<< endl;
				}

				if (entrenar && rostros.size() >= 1)
				{
					model->update(rostros, ids);

					cout << "\nNombre de la persona: ";
					cin >> names[identificador];
					system("cls");

					entrenar = agregarRostro = entrenamiento = false;
					rostros.clear();
					ids.clear();
					identificador += 1;
					capCount = 0;

					cout << msg1;
				}

			}
		}

		Mat rois[5];

		for (size_t i = 0;i < n; i++) {

			if (i < 5) {

				rois[i] = Mat(image, Rect(faces[i].x, faces[i].y, faces

[i].width, faces[i].height)).clone();
				resize(rois[i], rois[i], Size(image.cols / 5, image.rows / 5));
			}
		}

			if (identificador >= 1)
			{

		for (size_t i = 0;i < n; i++) {

			if (identificador >= 1)
			{
				int id = -1;
				double confidence = 0.0;

				Mat nface;
				CropFace(workingImage(faces[i]), nface, lEye, rEye);

				//calquier confidence mayor que threshold id = -1
				//redicir o aumentar este valor segun nos convenga
				model->set("threshold", 70);
				model->predict(nface, id, confidence);

				if (id >= 0) {

					string msg = names[id] + " : " + to_string

((int)confidence);

					DrawMarker(image, faces[i], msg, 20);

				}
				else
					DrawMarker(image, faces[i], "???", 20);

			}
			else
				DrawMarker(image, faces[i], "???", 20);
		}

		//imshow("Face Detection", image);
		//imshow("Working Image", workingImage);

		switch (waitKey(30))
		{
		case 'T':
		case 't':
			entrenar = true;
			break;
		case 'A':
		case 'a':
			agregarRostro = entrenamiento;
			break;
		case 'E':
		case 'e':
			entrenamiento = true;
			system("cls");
			cout << msg2 << endl;
			break;
		
		case 32:

			if (holdImage)
				holdImage = false;
			else
				holdImage = true;
			break;

		default:
			break;
		
		case 27:
			return 0;
		}

		for (int i = 0;i < size(rois); i++) {

			if (holdImage && rois[i].data) {

				rois[i].copyTo(image.rowRange(image.rows - rois[i].rows*(5-i), 

rois[i].rows*(i+1)).colRange(image.cols - rois[i].cols, image.cols));
			}
		}

		imshow("Face Detection", image);

		waitKey(10);

		workingImage.release();
		frame.release();
	}

	return 0;
}
