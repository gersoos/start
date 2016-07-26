#include "utils.hpp"
#include "laneDetector.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>


// sneak into opencv to access low level function

namespace cv{

struct LinePolar
{
	float rho;
	float angle;
};

struct hough_cmp_gt
{
	hough_cmp_gt(const int* _aux) : aux(_aux) {}
	bool operator()(int l1, int l2) const
	{
		return aux[l1] > aux[l2] || (aux[l1] == aux[l2] && l1 < l2);
	}
	const int* aux;
};

/*
Here image is an input raster;
step is it's step; size characterizes it's ROI;
rho and theta are discretization steps (in pixels and radians correspondingly).
threshold is the minimum number of pixels in the feature for it
to be a candidate for line. lines is the output
array of (rho, theta) pairs. linesMax is the buffer size (number of pairs).
Functions return the actual number of found lines.
*/
static void
HoughLinesStandard2( const Mat& img, float rho, float theta,
					int threshold, std::vector<Vec2f>& lines, int linesMax,
					double min_theta, double max_theta,Mat& img_out )
{
	int i, j;
	float irho = 1 / rho;

	CV_Assert( img.type() == CV_8UC1 );

	const uchar* image = img.ptr();
	int step = (int)img.step;
	int width = img.cols;
	int height = img.rows;

	if (max_theta < min_theta ) {
		CV_Error( CV_StsBadArg, "max_theta must be greater than min_theta" );
	}
	int numangle = cvRound((max_theta - min_theta) / theta);
	int numrho = cvRound(((width + height) * 2 + 1) / rho);

	AutoBuffer<int> _accum((numangle+2) * (numrho+2));
	std::vector<int> _sort_buf;
	AutoBuffer<float> _tabSin(numangle);
	AutoBuffer<float> _tabCos(numangle);
	int *accum = _accum;
	float *tabSin = _tabSin, *tabCos = _tabCos;

	memset( accum, 0, sizeof(accum[0]) * (numangle+2) * (numrho+2) );

	float ang = static_cast<float>(min_theta);
	for(int n = 0; n < numangle; ang += theta, n++ )
	{
		tabSin[n] = (float)(sin((double)ang) * irho);
		tabCos[n] = (float)(cos((double)ang) * irho);
	}

	// stage 1. fill accumulator
	for( i = 0; i < height; i++ )
		for( j = 0; j < width; j++ )
		{
			if( image[i * step + j] != 0 )
				for(int n = 0; n < numangle; n++ )
				{
					int r = cvRound( j * tabCos[n] + i * tabSin[n] );
					r += (numrho - 1) / 2;
					accum[(n+1) * (numrho+2) + r+1]++;
				}
		}

	// stage 2. find local maximums
	for(int r = 0; r < numrho; r++ )
		for(int n = 0; n < numangle; n++ )
		{
			int base = (n+1) * (numrho+2) + r+1;
			if( accum[base] > threshold &&
				accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
				accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2] )
				_sort_buf.push_back(base);
		}

	// stage 3. sort the detected lines by accumulator value
	std::sort(_sort_buf.begin(), _sort_buf.end(), hough_cmp_gt(accum));

	// stage 4. store the first min(total,linesMax) lines to the output buffer
	linesMax = std::min(linesMax, (int)_sort_buf.size());
	double scale = 1./(numrho+2);
	for( i = 0; i < linesMax; i++ )
	{
		LinePolar line;
		int idx = _sort_buf[i];
		int n = cvFloor(idx*scale) - 1;
		int r = idx - (n+1)*(numrho+2) - 1;
		line.rho = (r - (numrho - 1)*0.5f) * rho;
		line.angle = static_cast<float>(min_theta) + n * theta;
		lines.push_back(Vec2f(line.rho, line.angle));
	}

	img_out.create(numrho+2,numangle+2,CV_64FC1);
	for(int r = 0; r < numrho; r++ )
		for(int n = 0; n < numangle; n++ )
		{
			int base = (n+1) * (numrho+2) + r+1;
			img_out.at<float>(r,n)=accum[base];
		}


}

} // namespace


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


	double dists[] = {1170,1470,1880};
	std::vector<double> lineDistances(dists,dists + sizeof(dists)/sizeof(double) );
	initLaneModels(lineDistances);

	vv.open("out.avi", CV_FOURCC('M','J','P','G'),25,Size(1920/2,1080/2));
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

void LaneDetection::displayLineModels()
{
	for(LineIterator line_ref = vertices(model).first; line_ref != vertices(model).second; ++line_ref)
	{

		//std::cout << "line:" << model[*line_ref].r_ << std::endl;
	}
}

void LaneDetection::displayLaneModels()
{
	for(LaneIterator lane_ref = edges(model).first; lane_ref != edges(model).second; ++lane_ref)
	{
		//std::cout << "lane:" << model[*lane_ref].r_ << std::endl;
	}

}


void LaneDetection::displayAll()
{
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

	vv << imageStore["result"];

	//std::cout << "NN" << std::endl;
	displayLineModels();
	displayLaneModels();
}

int LaneDetection::process(cv::Mat input)
{
	imageStore["input"] = input;

	std::cout << "preprocess" << std::endl;
	preprocess();

	std::cout << "project" << std::endl;
	projectFrameToGound();

	std::cout << "extract" << std::endl;
	extractPointFeatures();

	std::cout << "detect" << std::endl;
	detectLineFeatres();

	std::cout << "update1" << std::endl;
	updateLineModels();

	std::cout << "update2" << std::endl;
	updateLaneModels();

	std::cout << "display" << std::endl;
	displayAll();

	return 0;
}

void LaneDetection::updateLineModels() {
	float gate = 130;

	std::vector<Vec2f> lines;

	Mat ground = imageStore["ground"];
	Mat frame = imageStore["frame"];
	Mat linesFound = cv::Mat::zeros(getInt("GROUND_H"), getInt("GROUND_W"), CV_8UC3);
	Mat linesFoundImg = frame.clone();
	Mat result;


	// for all lines
	for(LineIterator line_ref = vertices(model).first; line_ref != vertices(model).second; ++line_ref)
	{
		vector<Point2f> left;

		float r_mean = model[*line_ref].r_;
		// gate all houghLines
		for(size_t i=0; i<houghLines.size(); i++)
		{
			if ( fabs(houghLines[i][0] - r_mean) < gate)
			{
				left.push_back(houghLines[i]);
			}
		}

		if (left.size()>3 && left.size()<500)
		{
			vector<Point2f> leftR = ransac(left);
			if (leftR.size()>1)
			{
				lines.push_back(Vec2f((leftR[0].x + leftR[1].x)/2, (leftR[0].y + leftR[1].y)/2));
				// update!!!!!!

			}
		}


	}

	/// Show the result
	for( size_t i = 0; i < lines.size(); i++ )
	   {
		float r = lines[i][0], t = lines[i][1];
		double cos_t = cos(t), sin_t = sin(t);
		double x0 = r*cos_t, y0 = r*sin_t;
		double alpha = 3000;

		 Point pt1( cvRound(x0 + alpha*(-sin_t)), cvRound(y0 + alpha*cos_t) );
		 Point pt2( cvRound(x0 - alpha*(-sin_t)), cvRound(y0 - alpha*cos_t) );
		 line( ground, pt1, pt2, Scalar(0,255,0), 15, LINE_AA);
		 line( linesFound, pt1, pt2, Scalar(0,255,0), 15, LINE_AA);
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

	imageStore["ground"] = ground;
	imageStore["linesFoundImg"] = linesFoundImg;
	imageStore["result"] = result;
}

void LaneDetection::updateLaneModels()
{

}

void LaneDetection::initLaneModels(std::vector<double> distances) {
	for(std::vector<double>::iterator it = distances.begin(); it != distances.end(); it++)
	{
		LaneProperty propLane(*it);
		LineProperty propLeftLine(*it-150);
		LineProperty propRightLine(*it+150);

		Line s= add_vertex(propLeftLine, model);
		Line t= add_vertex(propRightLine, model);
		add_edge(s,t,propLane, model);
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

