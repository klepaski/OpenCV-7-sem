#include <fstream>
#include <sstream>
#include <iostream>
#include <string.h>
#include <opencv2/core.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/video.hpp>
#include "opencv2/optflow.hpp"
using namespace cv;
using namespace std;
using namespace dnn;
using namespace cv::motempl;
void Opticalflow(int argc, char **argv);
void rcnn();
void yolo();
static void  update_mhi(const Mat& img, Mat& dst, int diff_threshold);
Mat DiffImage(Mat t0, Mat t1, Mat t2)
{
	Mat d1, d2, diff;
	absdiff(t2, t1, d1);
	absdiff(t1, t0, d2);
	bitwise_or(d1, d2, diff);
	return diff;
}

const double MHI_DURATION = 5;
const double MAX_TIME_DELTA = 0.5;
const double MIN_TIME_DELTA = 0.05;

// ring image buffer
vector<Mat> buf;
int last = 0;

// temporary images
Mat mhi, orient, mask, segmask, zplane;
vector<Rect> regions;

float conf_threshold = 0.5;
// nms threshold
float nms = 0.2;
int width = 128;
int height = 128;

vector<string> classes;

// remove unnecessary bounding boxes
void remove_box(Mat&frame, const vector<Mat>&out);

// draw bounding boxes
void draw_box(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);

// get output layers
vector<String> getOutputsNames(const Net& net);

int main(int argc, char **argv)
{
	const string about =
		"This sample demonstrates Lucas-Kanade Optical Flow calculation.\n"
		"The example file can be downloaded from:\n"
		"  https://www.bogotobogo.com/python/OpenCV_Python/images/mean_shift_tracking/slow_traffic_small.mp4";
	const string keys =
		"{ h help |      | print this help message }"
		"{ @image | vtest.avi | path to image file }";
	CommandLineParser parser(argc, argv, keys);
	parser.about(about);
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	//string filename = "C:\\Users\\USER\\Downloads\\opencv\\sources\\samples\\data";
	string filename = "D:\\4\\олло\\Lab7\\Lab1\\slow_traffic.mp4";
	if (!parser.check())
	{
		parser.printErrors();
		return 0;
	}
	VideoCapture capture(filename);
	if (!capture.isOpened()) {
		cerr << "Unable to open file!" << endl;
		return 0;
	}
	// Create some random colors
	vector<Scalar> colors;
	RNG rng;
	for (int i = 0; i < 100; i++)
	{
		int r = rng.uniform(0, 256);
		int g = rng.uniform(0, 256);
		int b = rng.uniform(0, 256);
		colors.push_back(Scalar(r, g, b));
	}
	Mat old_frame, old_gray;
	vector<Point2f> p0, p1;
	// Take first frame and find corners in it
	capture >> old_frame;
	cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
	goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
	// Create a mask image for drawing purposes
	Mat mask = Mat::zeros(old_frame.size(), old_frame.type());
	
	
	//MOTION HISTORY 
	Mat frame2, frame3, frame4;
	VideoCapture cap2 = VideoCapture("C:\\Users\\USER\\Downloads\\opencv\\sources\\samples\\data\\vtest.avi");
	buf.resize(2);
	Mat image, motion;
	for (;;)
	{
		cap2 >> image;
		if (image.empty())
			break;

		update_mhi(image, motion, 30);
		imshow("Motion", motion);

		if (waitKey(10) >= 0)
			break;
	}
	destroyAllWindows();

	//2 TASK
	while (true) {
		Mat frame, frame_gray;
		capture >> frame;
		if (frame.empty())
			break;
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		// calculate optical flow
		vector<uchar> status;
		vector<float> err;
		TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
		calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15, 15), 2, criteria);
		vector<Point2f> good_new;
		for (uint i = 0; i < p0.size(); i++)
		{
			// Select good points
			if (status[i] == 1) {
				good_new.push_back(p1[i]);
				// draw the tracks
				line(mask, p1[i], p0[i], colors[i], 2);
				circle(frame, p1[i], 5, colors[i], -1);
			}
		}
		Mat img;
		add(frame, mask, img);
		imshow("Frame", img);
		int keyboard = waitKey(30);
		if (keyboard == 'q' || keyboard == 27)
			break;
		// Now update the previous frame and previous points
		old_gray = frame_gray.clone();
		p0 = good_new;

		char key = (char)cv::waitKey(30);
		if (key == 27) break;
	}
	destroyAllWindows();


	rcnn();
	destroyAllWindows();
}

void Opticalflow(int argc, char ** argv)
{
	const char* params
		= "{ help h         |           | Print usage }"
		"{ input          | vtest.avi | Path to a video or a sequence of image }"
		"{ algo           | MOG2      | Background subtraction method (KNN, MOG2) }";
	CommandLineParser parser(argc, argv, params);
	parser.about("This program shows how to use background subtraction methods provided by "
		" OpenCV. You can process both videos and images.\n");
	if (parser.has("help"))
	{
		//print help information
		parser.printMessage();
	}
	//create Background Subtractor objects
	Ptr<BackgroundSubtractor> pBackSub;
	if (parser.get<String>("algo") == "MOG2")
		pBackSub = createBackgroundSubtractorMOG2();
	else
		pBackSub = createBackgroundSubtractorKNN();
	VideoCapture capture("D:\\4\\олло\\Lab7\\Lab1\\slow_traffic.mp4");
	if (!capture.isOpened()) {
		//error in opening the video input
		cerr << "Unable to open: " << parser.get<String>("input") << endl;
	}
	Mat frame, fgMask;
	while (true) {
		capture >> frame;
		if (frame.empty())
			break;
		//update the background model
		pBackSub->apply(frame, fgMask);
		//get the frame number and write it on the current frame
		rectangle(frame, cv::Point(10, 2), cv::Point(100, 20),
			cv::Scalar(255, 255, 255), -1);
		stringstream ss;
		ss << capture.get(CAP_PROP_POS_FRAMES);
		string frameNumberString = ss.str();
		putText(frame, frameNumberString.c_str(), cv::Point(15, 15),
			FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		//show the current frame and the fg masks
		imshow("Frame", frame);
		imshow("FG Mask", fgMask);
		//get the input from the keyboard
		int keyboard = waitKey(30);
		if (keyboard == 'q' || keyboard == 27)
			break;
	}
	destroyAllWindows();
}


void rcnn()
{

	// List of tracker types in OpenCV 3.4.1
	string trackerTypes[8] = { "BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "GOTURN", "MOSSE", "CSRT" };
	// vector <string> trackerTypes(types, std::end(types));

	// Create a tracker
	string trackerType = trackerTypes[2];

	Ptr<Tracker> tracker;

#if (CV_MINOR_VERSION < 3)
	{
		tracker = Tracker::create(trackerType);
	}
#else
	{
		if (trackerType == "BOOSTING")
			tracker = TrackerBoosting::create();
		if (trackerType == "MIL")
			tracker = TrackerMIL::create();
		if (trackerType == "KCF")
			tracker = TrackerKCF::create();
		if (trackerType == "TLD")
			tracker = TrackerTLD::create();
		if (trackerType == "MEDIANFLOW")
			tracker = TrackerMedianFlow::create();
		if (trackerType == "GOTURN")
			tracker = TrackerGOTURN::create();
		if (trackerType == "MOSSE")
			tracker = TrackerMOSSE::create();
		if (trackerType == "CSRT")
			tracker = TrackerCSRT::create();
	}
#endif
	// Read video
	VideoCapture video("D:\\4\\олло\\Lab7\\Lab1\\slow_traffic.mp4");

	// Exit if video is not opened
	if (!video.isOpened())
	{
		cout << "Could not read video file" << endl;
	}

	// Read first frame 
	Mat frame;
	bool ok = video.read(frame);

	// Define initial bounding box 
	Rect2d bbox(287, 23, 86, 320);

	// Uncomment the line below to select a different bounding box 
	 bbox = selectROI(frame, false); 
	// Display bounding box. 
	rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);

	imshow("Tracking", frame);
	tracker->init(frame, bbox);

	while (video.read(frame))
	{
		// Start timer
		double timer = (double)getTickCount();

		// Update the tracking result
		bool ok = tracker->update(frame, bbox);

		// Calculate Frames per second (FPS)
		float fps = getTickFrequency() / ((double)getTickCount() - timer);

		if (ok)
		{
			// Tracking success : Draw the tracked object
			rectangle(frame, bbox, Scalar(255, 0, 0), 2, 1);
		}
		else
		{
			// Tracking failure detected.
			putText(frame, "Tracking failure detected", Point(100, 80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 2);
		}

		// Display tracker type on frame
		putText(frame, trackerType + " Tracker", Point(100, 20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50, 170, 50), 2);

		// Display frame.
		imshow("Tracking", frame);

		// Exit if ESC pressed.
		int k = waitKey(1);
		if (k == 27)
		{
			break;
		}
	}
}

static void  update_mhi(const Mat& img, Mat& dst, int diff_threshold)
{
	double timestamp = (double)clock() / CLOCKS_PER_SEC; // get current time in seconds
	Size size = img.size();
	int i, idx1 = last;
	Rect comp_rect;
	double count;
	double angle;
	Point center;
	double magnitude;
	Scalar color;

	// allocate images at the beginning or
	// reallocate them if the frame size is changed
	if (mhi.size() != size)
	{
		mhi = Mat::zeros(size, CV_32F);
		zplane = Mat::zeros(size, CV_8U);

		buf[0] = Mat::zeros(size, CV_8U);
		buf[1] = Mat::zeros(size, CV_8U);
	}

	cvtColor(img, buf[last], COLOR_BGR2GRAY); // convert frame to grayscale

	int idx2 = (last + 1) % 2; // index of (last - (N-1))th frame
	last = idx2;

	Mat silh = buf[idx2];
	absdiff(buf[idx1], buf[idx2], silh); // get difference between frames

	threshold(silh, silh, diff_threshold, 1, THRESH_BINARY); // and threshold it
	updateMotionHistory(silh, mhi, timestamp, MHI_DURATION); // update MHI

	// convert MHI to blue 8u image
	mhi.convertTo(mask, CV_8U, 255. / MHI_DURATION, (MHI_DURATION - timestamp)*255. / MHI_DURATION);

	Mat planes[] = { mask, zplane, zplane };
	merge(planes, 3, dst);

	// calculate motion gradient orientation and valid orientation mask
	calcMotionGradient(mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3);

	// segment motion: get sequence of motion components
	// segmask is marked motion components map. It is not used further
	regions.clear();
	segmentMotion(mhi, segmask, regions, timestamp, MAX_TIME_DELTA);

	// iterate through the motion components,
	// One more iteration (i == -1) corresponds to the whole image (global motion)
	for (i = -1; i < (int)regions.size(); i++) {

		if (i < 0) { // case of the whole image
			comp_rect = Rect(0, 0, size.width, size.height);
			color = Scalar(255, 255, 255);
			magnitude = 100;
		}
		else { // i-th motion component
			comp_rect = regions[i];
			if (comp_rect.width + comp_rect.height < 100) // reject very small components
				continue;
			color = Scalar(0, 0, 255);
			magnitude = 30;
		}
		// select component ROI
		Mat silh_roi = silh(comp_rect);
		Mat mhi_roi = mhi(comp_rect);
		Mat orient_roi = orient(comp_rect);
		Mat mask_roi = mask(comp_rect);

		// calculate orientation
		angle = calcGlobalOrientation(orient_roi, mask_roi, mhi_roi, timestamp, MHI_DURATION);
		angle = 360.0 - angle;  // adjust for images with top-left origin

		count = norm(silh_roi, NORM_L1);; // calculate number of points within silhouette ROI

		// check for the case of little motion
		if (count < comp_rect.width*comp_rect.height * 0.05)
			continue;

		// draw a clock with arrow indicating the direction
		center = Point((comp_rect.x + comp_rect.width / 2),
			(comp_rect.y + comp_rect.height / 2));

		circle(img, center, cvRound(magnitude*1.2), color, 3, 16, 0);
		line(img, center, Point(cvRound(center.x + magnitude * cos(angle*CV_PI / 180)),
			cvRound(center.y - magnitude * sin(angle*CV_PI / 180))), color, 3, 16, 0);
	}
}

void yolo()
{
	// get labels of all classes
	string classesFile = "coco.names";
	ifstream ifs(classesFile.c_str());
	string line;
	while (getline(ifs, line)) classes.push_back(line);

	// load model weights and architecture
	String configuration = "yolov3.cfg";
	String model = "yolov3.weights";

	// Load the network
	Net net = readNetFromDarknet(configuration, model);
	Mat frame, blob;

	// read the image
	frame = imread("eagle.jpg", IMREAD_COLOR);

	// convert image to blob
	blobFromImage(frame, blob, 1 / 255, cvSize(width, height), Scalar(0, 0, 0), true, false);
	net.setInput(blob);

	vector<Mat> outs;
	net.forward(outs, getOutputsNames(net));

	remove_box(frame, outs);

	vector<double> layersTimes;
	double freq = getTickFrequency() / 1000;
	double t = net.getPerfProfile(layersTimes) / freq;
	string label = format("Inference time for a frame : %.2f ms", t);
	putText(frame, label, Point(0, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255));

	Mat detectedFrame;
	frame.convertTo(detectedFrame, CV_8U);
	static const string kWinName = "Deep learning object detection in OpenCV";

	imshow(kWinName, frame);
	waitKey();
}

void remove_box(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t k = 0; k < outs.size(); k++)
	{
		float* data = (float*)outs[k].data;
		for (size_t i = 0; i < outs[k].total(); i += 7)
		{
			float confidence = data[i + 2];
			if (confidence > conf_threshold)
			{
				int left = (int)data[i + 3];
				int top = (int)data[i + 4];
				int right = (int)data[i + 5];
				int bottom = (int)data[i + 6];
				int width = right - left + 1;
				int height = bottom - top + 1;
				if (width <= 2 || height <= 2)
				{
					left = (int)(data[i + 3] * frame.cols);
					top = (int)(data[i + 4] * frame.rows);
					right = (int)(data[i + 5] * frame.cols);
					bottom = (int)(data[i + 6] * frame.rows);
					width = right - left + 1;
					height = bottom - top + 1;
				}
				classIds.push_back((int)(data[i + 1]) - 1);  // Skip 0th background class id.
				boxes.push_back(Rect(left, top, width, height));
				confidences.push_back(confidence);
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	for (size_t idx = 0; idx < boxes.size(); ++idx)
	{
		Rect box = boxes[idx];
		draw_box(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}

// Draw the predicted bounding box
void draw_box(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(255, 178, 50), 3);

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!classes.empty())
	{
		CV_Assert(classId < (int)classes.size());
		label = classes[classId] + ":" + label;
	}

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	rectangle(frame, Point(left, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 0), 1);
}

// Get the names of the output layers
vector<Mat> outs;
vector<String> getOutputsNames(const Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}
