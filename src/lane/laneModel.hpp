#ifndef LANE_LANE_MODEL_HPP
#define LANE_LANE_MODEL_HPP

namespace lane {


/**
 * @brief The LineProperty class
 *
 * LineProperty is a pod VALUE type class
 */
class LineProperty {
public:
	LineProperty(double r = 0, double fi = 0, int score=0):
		r_(r), fi_(fi), score_(score) {}

	double r_; ///< distance on ground plane - pixel
	double fi_; ///< orientation on ground plane - rad 0 is vertical
	int score_;  ///< score
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
