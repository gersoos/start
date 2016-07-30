#include "utils.hpp"
#include "laneDetector.hpp"
#include "hough.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

using namespace cv;
using namespace std;

namespace lane{

// Implementing RANSAC to remove outlier lines
// Picking the best estimate having maximum number of inliers
// TO DO: Better implementation
vector<Point2f> ransac(vector<Point2f> data){
	double _ransacThres = 0.02;

	vector<Point2f> res;
	int maxInliers = 0;

	// Picking up the first sample
	for(int i = 0;i < data.size();i++){
		Point2f p1 = data[i];

		// Picking up the second sample
		for(int j = i + 1;j < data.size();j++){
			Point2f p2 = data[j];
			int n = 0;

			// Finding the total number of inliers
			for (int k = 0;k < data.size();k++){
				Point2f p3 = data[k];
				float normalLength = norm(p2 - p1);
				float distance = abs((float)((p3.x - p1.x) * (p2.y - p1.y) - (p3.y - p1.y) * (p2.x - p1.x)) / normalLength);
				if (distance < _ransacThres) n++;
			}

			// if the current selection has more inliers, update the result and maxInliers
			if (n > maxInliers) {
				res.clear();
				maxInliers = n;
				res.push_back(p1);
				res.push_back(p2);
			}

		}

	}

	return res;
}




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

	// Hough
	intStore["HOUGH_RHO"]       = 5;
	intStore["HOUGH_TH"]        = 1500;
	intStore["HOUGH_LINECOUNT"] = 340;


	double dists[] = {1000,1290,1550,1730};
	floatStore["LANE_WIDTH"]     = 250;
	floatStore["LANE_GATE"]      = 110;

	std::vector<double> lineDistances(dists,dists + sizeof(dists)/sizeof(double) );
	initLaneModels(lineDistances);

	vv.open("debug.avi", CV_FOURCC('M','J','P','G'),25,Size(intStore["GROUND_H"]/2,intStore["GROUND_W"]/2));
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
	warpTr = getPerspectiveTransform(cvSrc, cvDst);

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

/**
 * @brief LaneDetection::displayLineModels
 */
void LaneDetection::displayLineModels()
{
	/// Input: imageStore["frame"]
	Mat frame = imageStore["frame"];

	Mat linesFound = cv::Mat::zeros(getInt("GROUND_H"), getInt("GROUND_W"), CV_8UC3);
	Mat linesFoundImg = frame.clone();
	Mat result;

	/// Show the result
	for(LineIterator line_ref = vertices(model).first; line_ref != vertices(model).second; ++line_ref)
	{

		if (model[*line_ref].valid_)// model
		{
			float r = model[*line_ref].rMeas_;
			float t = model[*line_ref].fiMeas_;
			double cos_t = cos(t), sin_t = sin(t);
			double x0 = r*cos_t, y0 = r*sin_t;
			double alpha = 3000;

			Point pt1( cvRound(x0 + alpha*(-sin_t)), cvRound(y0 + alpha*cos_t) );
			Point pt2( cvRound(x0 - alpha*(-sin_t)), cvRound(y0 - alpha*cos_t) );
			line( linesFound, pt1, pt2, Scalar(0,255,0), 10, LINE_AA);
		}
		// model
		{
			float r = model[*line_ref].r_;
			float t = model[*line_ref].fi_;
			double cos_t = cos(t), sin_t = sin(t);
			double x0 = r*cos_t, y0 = r*sin_t;
			double alpha = 3000;

			Point pt1( cvRound(x0 + alpha*(-sin_t)), cvRound(y0 + alpha*cos_t) );
			Point pt2( cvRound(x0 - alpha*(-sin_t)), cvRound(y0 - alpha*cos_t) );

			int blue = 200 + model[*line_ref].score_; // fades to white
			if (blue > 190) {
				line( linesFound, pt1, pt2, Scalar(blue,0,0), 30, LINE_AA);
			}
		}


		{
			float r_mean = model[*line_ref].r_;
			Point pt1( r_mean, 100 );
			Point pt2( r_mean, 500 );
			line( imageStore["groundDebug"], pt1, pt2, Scalar(0,255,0), 15, LINE_AA);
		}
	}

	// project detections to image plane
	{
		int    interpolationMode = cv::INTER_LINEAR;
		bool   inverseMap = true;
		double borderValue = 0.0;
		int    borderMode = cv::BORDER_CONSTANT;

		warpPerspective(linesFound, linesFoundImg, warpTr, imageStore["frame"].size(),
				interpolationMode | (inverseMap ? cv::WARP_INVERSE_MAP : 0), borderMode, borderValue);
		add(linesFoundImg,frame,linesFoundImg);
		resize(linesFoundImg,result,Size(0,0),0.5,0.5);
	}
	imageStore["linesFoundImg"] = linesFoundImg;
	imageStore["result"] = result;

}

void LaneDetection::displayLaneModels()
{
	for(LaneIterator lane_ref = edges(model).first; lane_ref != edges(model).second; ++lane_ref)
	{
		{
			float r_mean = model[*lane_ref].r_;
			Point pt1( r_mean, 100 );
			Point pt2( r_mean, 500 );
			line( imageStore["groundDebug"], pt1, pt2, Scalar(0,0,255), 15, LINE_AA);
		}
	}

}


void LaneDetection::displayAll()
{
	displayLineModels();
	displayLaneModels();

	// TODO: mechanism for selecting debug images

	/*
	for(ImgeStoreType::iterator it= imageStore.begin();it!=imageStore.end();it++)
	{
		namedWindow(it->first,WINDOW_OPENGL);
		imshow(it->first, it->second);
	}
	*/

	namedWindow("ground",WINDOW_OPENGL);
	imshow("ground", imageStore["groundDebug"]);

	namedWindow("processed",WINDOW_OPENGL);
	imshow("processed", imageStore["result"]);

	{
		Mat result;
		resize(imageStore["groundDebug"],result,Size(intStore["GROUND_H"]/2,intStore["GROUND_W"]/2));
		vv << result;
	}
}

int LaneDetection::process(cv::Mat input)
{
	imageStore["input"] = input;

	//std::cout << "preprocess" << std::endl;
	preprocess();

	//std::cout << "project" << std::endl;
	projectFrameToGound();

	//std::cout << "extract" << std::endl;
	extractPointFeatures();

	//std::cout << "detect" << std::endl;
	detectLineFeatres();

	//std::cout << "update1" << std::endl;
	updateLineModels();

	//std::cout << "update2" << std::endl;
	updateLaneModels();

	//std::cout << "display" << std::endl;
	displayAll();

	return 0;
}

int LaneDetection::getNumberOfLanes()
{
	double mMin = 50000;
	double mMax = -50000;

	for(LineIterator line_ref = vertices(model).first; line_ref != vertices(model).second; ++line_ref)
	{
		if (model[*line_ref].r_ < mMin && model[*line_ref].score_ > 5)
		{
			mMin = model[*line_ref].r_;
		}
		if (model[*line_ref].r_ > mMax && model[*line_ref].score_ > 5)
		{
			mMax = model[*line_ref].r_;
		}
	}

	//std::cout << mMax-mMin << " " << mMax << " " << mMin << std::endl;
	return int( round (mMax - mMin) / floatStore["LANE_WIDTH"]) + 1;

}

Mat LaneDetection::getResultFrame()
{
	return imageStore["result"];
}

void LaneDetection::updateLineModels() {
	float gate = floatStore["LANE_GATE"];

	std::vector<Vec2f> lines;

	Mat frame = imageStore["frame"];

	// for all lines
	vector<bool> houghPaired(houghLines.size(),false);
	for(LineIterator line_ref = vertices(model).first; line_ref != vertices(model).second; ++line_ref)
	{
		vector<Point2f> left;

		double r_mean;
		double fi_mean;
		model[*line_ref].predict(r_mean, fi_mean); // predict the state of the next frame

		
		// gate all houghLines
		for(size_t i=0; i<houghLines.size(); i++)
		{
			if ( fabs(houghLines[i][0] - r_mean) < gate)
			{
				left.push_back(houghLines[i]);
				houghPaired[i] = true;
			}
		}

		if (left.size()>1)
		{
			vector<Point2f> leftR = ransac(left);
			if (leftR.size()>1)
			{
				// update!!!!!!
				double r  = ( leftR[0].x + leftR[1].x)/2;
				double fi = ( leftR[0].y + leftR[1].y)/2;

				model[*line_ref].correct(r,fi); // Correct the state of the next frame after obtaining the measurements

				lines.push_back(Vec2f(r,fi));
				lines.push_back(Vec2f(model[*line_ref].r_,model[*line_ref].fi_));



			}
			else
			{
				std::cout << "RANSAC" << std::endl;
			}
		}
		else
		{
			model[*line_ref].notFound();
		}



	}

	// TODO: if houghPaired is 0 - new lane???

}

void LaneDetection::updateLaneModels()
{
	float gate = floatStore["LANE_GATE"];

	for(LaneIterator lane_ref = edges(model).first; lane_ref != edges(model).second; ++lane_ref)
	{
		Line left  = source(*lane_ref, model);
		Line right = target(*lane_ref, model);
		if      ( model[right].hasScore() &&  model[right].hasScore())
		{
			if (model[right].r_ - model[left].r_ < gate)
			{
				std::cout << "dist: " << model[right].r_ << " " << model[left].r_ << std::endl;
				model[left ].reset();
				model[right].reset();
			}
		}
		else if (!model[right].hasScore() &&  model[right].hasScore() )
		{

		}
		else if ( model[right].hasScore() && !model[right].hasScore() )
		{

		}
		else {

		}

	}
}

void LaneDetection::initLaneModels(std::vector<double> distances) {

	if (distances.size()<2)
	{
		return;
	}

	std::vector<double>::iterator it1 = distances.begin();
	std::vector<double>::iterator it2 = distances.begin();
	it2++;
	LineProperty propLeftLine(*it1);
	Line s= add_vertex(propLeftLine,  model);
	Line t;

	for(; it1 != distances.end() && it2 != distances.end(); it1++,it2++)
	{
		LaneProperty propLane( (*it1 + *it2) / 2);
		LineProperty propRightLine(*it2);

		t = add_vertex(propRightLine, model);
		add_edge(s,t,propLane, model);
		s = t;
	}
}

void LaneDetection::extractPointFeatures()
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

	imageStore["pointFeatures"]  = features;
	imageStore["pointFeaturesDebug"] = featuresDebug;

}

void LaneDetection::detectLineFeatres()
{
	Mat input       = imageStore["pointFeatures"];
	Mat groundDebug = imageStore["groundDebug"];
	Mat hg;

	float rho = (float) getInt("HOUGH_RHO");
	float theta = CV_PI/180;
	int threshold = getInt("HOUGH_TH");
	int linesMax = getInt("HOUGH_LINECOUNT");
	float min_theta = -0.1;
	float max_theta = 0.1;

	houghLines.clear();
	HoughLinesStandard2(input, rho, theta, threshold, houghLines, linesMax, min_theta, max_theta, hg);

	/// Show the result
	for( size_t i = 0; i < houghLines.size(); i++ )
	{
		float r = houghLines[i][0], t = houghLines[i][1];
		double cos_t = cos(t), sin_t = sin(t);
		double x0 = r*cos_t, y0 = r*sin_t;
		double alpha = 3000;

		Point pt1( cvRound(x0 + alpha*(-sin_t)), cvRound(y0 + alpha*cos_t) );
		Point pt2( cvRound(x0 - alpha*(-sin_t)), cvRound(y0 - alpha*cos_t) );
		line( groundDebug, pt1, pt2, Scalar(255,0,0), 15, LINE_AA);
	}

	// dummy store
	imageStore["groundDebug"] = groundDebug;
}

}

