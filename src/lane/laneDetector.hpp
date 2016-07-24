#ifndef LANE_LANE_DETECTOR_HPP
#define LANE_LANE_DETECTOR_HPP

#include "utils.hpp"
#include <map>

namespace lane {
class LaneDetection : public Operation {

public:
	LaneDetection();
	virtual ~LaneDetection();
	virtual void init();
	virtual int process(cv::Mat input);

private:
	void preprocess(); ///< generates "filtered" image, uses "input"
	void displayAll();

	typedef std::map<std::string,cv::Mat> StoreType;
	StoreType imageStore;
};

} // namespace
#endif
