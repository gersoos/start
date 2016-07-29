#include "laneModel.hpp"
#include <iostream>

using namespace cv;


namespace lane {

LineProperty::LineProperty(double r, double fi, int score):
	r_(r), fi_(fi), rReset_(r), rMeas_(0), fiMeas_(0), valid_(0), score_(score)
{
	init();
}

void LineProperty::predict(double& r,double& fi)
{
	Mat prediction = kalman.predict();
	r  = prediction.at<float>(0);
	fi = prediction.at<float>(1);
}

void LineProperty::notFound()
{
	valid_ = false;
	score_--;

	if (score_ < -10)
	{
		reset();
	}
}

void LineProperty::correct(double r, double fi)
{
	Mat_<float> measurement(2,1);
	measurement.setTo(Scalar(0));

	measurement.at<float>(0) = r;
	measurement.at<float>(1) = fi;

	Mat estimated = kalman.correct(measurement); // Correct the state of the next frame after obtaining the measurements

	r_  = estimated.at<float>(0);
	fi_ = estimated.at<float>(1);

	rMeas_  = r;
	fiMeas_ = fi;
	valid_  = true;

}

void LineProperty::reset()
{
	valid_ = false;
	r_  = rReset_;
	fi_ = 0;

	std::cout << "RESET" <<std::endl;
}

void LineProperty::init()
{
	kalman.init(2, 2, 0);
	//kalman = new KalmanFilter( 4, 4, 0 ); // 4 measurement and state parameters
	kalman.transitionMatrix = (Mat_<float>(2, 2) << 1,0,0,1);

	// Initialization
	kalman.statePre.at<float>(0) = r_; // r1
	kalman.statePre.at<float>(1) = fi_; // theta1

	kalman.statePost.at<float>(0)=r_;
	kalman.statePost.at<float>(1)=fi_;

	setIdentity(kalman.measurementMatrix);
	setIdentity(kalman.processNoiseCov, Scalar::all(1e-4));
	setIdentity(kalman.measurementNoiseCov, Scalar::all(1e-1));
	setIdentity(kalman.errorCovPost, Scalar::all(5));

}



} // namespace
