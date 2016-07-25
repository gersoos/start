#include <opencv2/opencv.hpp>
#include "laneDetector.hpp"


int main(int, char**)
{
	cv::VideoCapture cap("/home/beegee/Downloads/GOPR5936PART.MP4"); // open the default camera
	if(!cap.isOpened())  // check if we succeeded
		return -1;

	lane::LaneDetection lane;
	lane.init();

	for(;;)
	{
		cv::Mat frame;
		cap >> frame; // get a new frame from camera

		lane.process(frame);
		if(cv::waitKey(30) >= 0) break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}
