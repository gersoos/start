#include "utils.hpp"
#include "laneDetector.hpp"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
}

void LaneDetection::preprocess()
{
	Mat input = imageStore["input"];

	Mat edges;
	cvtColor(input, edges, COLOR_BGR2GRAY);
	GaussianBlur(edges, edges, Size(7,7), 1.5, 1.5);
	Canny(edges, edges, 0, 30, 3);

	imageStore["output"] = edges;
}

void LaneDetection::displayAll()
{
	for(StoreType::iterator it= imageStore.begin();it!=imageStore.end();it++)
	{
		namedWindow(it->first,1);
		imshow(it->first, it->second);
	}
}

int LaneDetection::process(cv::Mat input)
{
	imageStore["input"] = input;

	preprocess();
	displayAll();

	return 0;
}


}

