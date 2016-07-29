#ifndef LANE_LANE_MODEL_HPP
#define LANE_LANE_MODEL_HPP

#include <opencv2/video/tracking.hpp>

namespace lane {


/**
 * @brief The LineProperty class
 *
 * LineProperty is a pod VALUE type class
 */
class LineProperty {
public:
	LineProperty(double r = 0, double fi = 0, int score=0);
	double r_; ///< distance on ground plane - pixel
	double fi_; ///< orientation on ground plane - rad 0 is vertical

	double rReset_; ///< initialization value for r_ during reset()

	double rMeas_;
	double fiMeas_;
	bool valid_; ///< last <rMeas_, fiMeas_> is valid

	int score_;  ///< score

	void predict(double& r,double& fi); ///< predict
	void notFound(); ///< signal for empty gate range
	void correct(double r, double fi); ///< correct with measurement

	void reset(); ///< reset r_ value and tracking using init
	void init(); ///< initialize tracking
protected:
	cv::KalmanFilter kalman;
};

/**
 * @brief The LaneProperty class
 *
 * LanePropert is a pod VALUE type class
 */
class LaneProperty {
public:
	LaneProperty(double r = 0, double fi = 0, int score=0):
		r_(r), fi_(fi), score_(score) {}

	double r_; ///< distance on ground plane - pixel
	double fi_; ///< orientation on ground plane - rad 0 is vertical
	int score_;  ///< score
};


} // namespace

#endif
