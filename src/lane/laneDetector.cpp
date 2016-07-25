#include "utils.hpp"
#include "laneDetector.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
namespace lane{

using namespace cv;

LaneDetection::LaneDetection()
{
	// nothing special yet
}

LaneDetection::~LaneDetection()
{
	// nothing special yet
}

void LaneDetection::init()
{
	log("init");

	// size of the ground plane in pixels
	intStore["GROUND_W"]  = 2000;
	intStore["GROUND_H"] = 2800;


	// Quadratic ROI in image plane - from calibration!
	// 825 736 1125 732 1252 825 695 825
	intStore["S1X"] = 825;
	intStore["S1Y"] = 736;
	intStore["S2X"] = 1125;
	intStore["S2Y"] = 732;
	intStore["S3X"] = 1252;
	intStore["S3Y"] = 825;
	intStore["S4X"] = 695;
	intStore["S4Y"] = 825;

	// Rectangular projection ROI in ground plane
	// 1000 1800 1300 1800 1300 2500 1000 2500
	intStore["D1X"] = 1000;
	intStore["D1Y"] = 1800;
	intStore["D2X"] = 1300;
	intStore["D2Y"] = 1800;
	intStore["D3X"] = 1300;
	intStore["D3Y"] = 2500;
	intStore["D4X"] = 1000;
	intStore["D4Y"] = 2500;

	// ROI on the ground plane
	intStore["G_ROI_X"] = 0;
	intStore["G_ROI_Y"] = 1200;
	intStore["G_ROI_W"] = 2000;
	intStore["G_ROI_H"] = 1300;

	// Canny
	intStore["CANNY_HI"]  = 180;
	intStore["CANNY_LOW"] = 100;

	// AdaptiveThreshold
	intStore["ATH_BLOCK"] = 25;
	intStore["ATH_VALUE"] = 15;


	double dists[] = {16,2,77,29};
	std::vector<double> lineDistances(dists,dists + sizeof(dists)/sizeof(double) );
	initLaneModels(lineDistances);
}

void LaneDetection::preprocess()
{
	Mat input = imageStore["input"];
	imageStore["frame"] = input.clone();
}

void LaneDetection::projectFrameToGound()
{
	Mat input = imageStore["frame"];
	Mat inputROI = input.clone();
	Mat ground = cv::Mat::zeros(getInt("GROUND_H"), getInt("GROUND_W"), CV_8UC3);;
	Mat groundDebug;
	Point2f cvSrc[4];
	Point2f cvDst[4];

	cvSrc[0] = Point2f(getInt("S1X"), getInt("S1Y"));
	cvSrc[1] = Point2f(getInt("S2X"), getInt("S2Y"));
	cvSrc[2] = Point2f(getInt("S3X"), getInt("S3Y"));
	cvSrc[3] = Point2f(getInt("S4X"), getInt("S4Y"));

	cvDst[0] = Point2f(getInt("D1X"), getInt("D1Y"));
	cvDst[1] = Point2f(getInt("D2X"), getInt("D2Y"));
	cvDst[2] = Point2f(getInt("D3X"), getInt("D3Y"));
	cvDst[3] = Point2f(getInt("D4X"), getInt("D4Y"));

	int    interpolationMode = cv::INTER_NEAREST;
	bool   inverseMap = false;
	double borderValue = 0.0;
	int    borderMode = cv::BORDER_CONSTANT;

	// overlay ROI on inputROI
	{
		int lineThickness = 15;
		Scalar color(100, 100, 0);
		line(inputROI, cvSrc[0], cvSrc[1], color, lineThickness);
		line(inputROI, cvSrc[1], cvSrc[2], color, lineThickness);
		line(inputROI, cvSrc[2], cvSrc[3], color, lineThickness);
		line(inputROI, cvSrc[3], cvSrc[0], color, lineThickness);

	}


	// calculate projection matrix
	Mat warpTr = getPerspectiveTransform(cvSrc, cvDst);

	warpPerspective(input, groundDebug, warpTr, ground.size(),
			interpolationMode | (inverseMap ? cv::WARP_INVERSE_MAP : 0), borderMode, borderValue);


	Rect rect;
	{
		int x = getInt("G_ROI_X");
		int y = getInt("G_ROI_Y");
		int sx = getInt("G_ROI_W");
		int sy = getInt("G_ROI_H");

		// Validate X and Y parameters
		if (x < 0)
			x = 0;
		else if (x > ground.cols)
			x = ground.cols;
		if (y < 0)
			y = 0;
		else if (y > ground.rows)
			y = ground.rows;

		// Validate width and height parameters
		if (sx < 0)
			sx = 0;
		else if (x + sx > ground.cols)
			sx = ground.cols - x;
		if (sy < 0)
			sy = 0;
		else if (y + sy > ground.rows)
			sy = ground.rows - y;


		rect = cvRect(x, y, sx, sy);

		// copy ROI region only
		Mat h1 = ground(rect);
		Mat h2 = groundDebug(rect);
		h2.copyTo(h1);
	}

	// overlay projection ROI on groundDebug
	{
		int lineThickness = 15;
		Scalar color(0, 200, 0);
		line(groundDebug, cvDst[0], cvDst[1], color, lineThickness);
		line(groundDebug, cvDst[1], cvDst[2], color, lineThickness);
		line(groundDebug, cvDst[2], cvDst[3], color, lineThickness);
		line(groundDebug, cvDst[3], cvDst[0], color, lineThickness);
	}

	// overlay ROI on groundDebug
	{
		int lineThickness = 15;
		cv::Scalar color(200, 100, 0);
		rectangle(groundDebug, rect, color, lineThickness);
	}

	// store images
	imageStore["inputROI"] = inputROI;
	imageStore["ground"] = ground;
	imageStore["groundDebug"] = groundDebug;
}

void LaneDetection::displayLineModels()
{
	for(LineIterator line_ref = vertices(model).first; line_ref != vertices(model).second; ++line_ref)
	{

		std::cout << "line:" << model[*line_ref].r_ << std::endl;
	}
}

void LaneDetection::displayLaneModels()
{
	for(LaneIterator lane_ref = edges(model).first; lane_ref != edges(model).second; ++lane_ref)
	{
		std::cout << "lane:" << model[*lane_ref].r_ << std::endl;
	}

}


void LaneDetection::displayAll()
{
	// TODO: mechanism for selecting debug images
	for(ImgeStoreType::iterator it= imageStore.begin();it!=imageStore.end();it++)
	{
		namedWindow(it->first,WINDOW_OPENGL);
		imshow(it->first, it->second);
	}

	std::cout << "NN" << std::endl;
	displayLineModels();
	displayLaneModels();
}

int LaneDetection::process(cv::Mat input)
{
	imageStore["input"] = input;

	preprocess();
	projectFrameToGound();

	detectLineFeatures();
	updateLineModels();
	updateLaneModels();

	displayAll();

	return 0;
}

void LaneDetection::updateLineModels() {

}

void LaneDetection::updateLaneModels()
{

}

void LaneDetection::initLaneModels(std::vector<double> distances) {
	for(std::vector<double>::iterator it = distances.begin(); it != distances.end(); it++)
	{
		LaneProperty propLane(*it);
		LineProperty propLeftLine(*it-1);
		LineProperty propRightLine(*it+1);

		Line s= add_vertex(propLeftLine, model);
		Line t= add_vertex(propRightLine, model);
		add_edge(s,t,propLane, model);
	}
}

void LaneDetection::detectLineFeatures()
{
	Mat input = imageStore["ground"];

	Mat canny;
	Mat grayscale;
	Mat th;
	Mat features;
	Mat featuresDebug;

	int th1 = getInt("CANNY_HI");
	int th2 = getInt("CANNY_LOW");

	double maxValue    = 255;
	int adaptiveMethod = cv::ADAPTIVE_THRESH_MEAN_C;
	int thresholdType  = cv::THRESH_BINARY_INV;
	int blockSize      = getInt("ATH_BLOCK");
	double constant    = getInt("ATH_VALUE");

	int dilateSize = 3;

	Canny(input, canny, th1, th2);


	if (input.type() == CV_8UC3)
	{
		cv::cvtColor(input, grayscale, CV_RGB2GRAY);
	}
	else if (input.type() == CV_8UC1)
	{
		grayscale = input;
	}
	adaptiveThreshold(grayscale, th, maxValue, adaptiveMethod, thresholdType, blockSize, constant);

	add(canny, th, features);

	Mat element = getStructuringElement(cv::MORPH_ELLIPSE, Size(2 * dilateSize + 1, 2 * dilateSize + 1), Point(dilateSize, dilateSize));
	dilate(features, featuresDebug, element);

	imageStore["lineFeatures"]  = features;
	imageStore["featuresDebug"] = featuresDebug;

}

}

