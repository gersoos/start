#ifndef LANE_LANE_MODEL_HPP
#define LANE_LANE_MODEL_HPP

#include <vector>

namespace lane {

/**
 * @brief The LineModel class
*/

class LineModel {
public:
	LineModel(double r, double fi = 0, int score=0):
		r_(r), fi_(fi), score_(score) {}

	double r_; ///< distance on ground plane - pixel
	double fi_; ///< orientation on ground plane - rad 0 is vertical
	int score_;  ///< score
};

/**
 * @brief The LaneModel class
 */
class LaneModel {
public:
	void initLineModel(std::vector<double> distances);
	void updateLineModel();

protected:
	typedef std::vector<LineModel> Lines;
	Lines lines;

};


} // namespace

#endif
