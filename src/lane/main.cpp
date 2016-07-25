#include <opencv2/opencv.hpp>
#include "laneDetector.hpp"


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

LaneDetector --> "lanes" LaneModel
LaneModel *-- "lines" LineModel

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
