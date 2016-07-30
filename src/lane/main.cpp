#include <opencv2/opencv.hpp>
#include "laneDetector.hpp"

#include <iostream>

// TODO: consider using smart pointers insted of object members
// TODO: use file based parameter initialization
// TODO: enhance doxygen and plantuml

/*! \mainpage A simple manual

Some general info.

This manual is divided in the following sections:
- \subpage intro
- \subpage object_diag   "Object Diagram"
- \subpage flowchart "Flowchart"

*/

//-----------------------------------------------------------

/*! \page intro Introduction
Lane Detection Experiment
Please check the GoogleDoc document
*/

//-----------------------------------------------------------

/*! \page object_diag Object Structure
Diagram is generated using plantuml

@startuml{objects.png}
title Objects

LaneDetector --> "model" Graph

@enduml


\see http://www.planttext.com/planttext

*/

//-----------------------------------------------------------

/*! \page flowchart High Level Flowchart
Diagram is generated using plantuml

\see http://www.planttext.com/planttext

*/

//-----------------------------------------------------------

/**
 * @brief main
 * @return
 */

int main(int, char**)
{
	int i = 0;
	cv::VideoCapture cap("/home/beegee/Downloads/GOPR5936PART.MP4"); // open the default camera
	if(!cap.isOpened())  // check if we succeeded
		return -1;

	cv::VideoWriter vv;
	vv.open("out.avi", CV_FOURCC('M','J','P','G'),25,cv::Size(1920/2,1080/2));

	lane::LaneDetection lane;
	lane.init();

	int FF = 0; // fast forward
	for(;;)
	{
		i++;

		cv::Mat frame;
		cap >> frame; // get a new frame from camera
		if (i < FF)
		{
			continue;
		}

		lane.process(frame);

		std::cout << i << ": " << lane.getNumberOfLanes() << std::endl;
		cv::Mat resultFrame = lane.getResultFrame();
		vv << resultFrame;

		if(cv::waitKey(30) >= 0) break;
		if (i>3100)
		{
			break;
		}
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;
}
