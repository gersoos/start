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
	void projectFrameToGound(); ///< projects input image using calibration data
	void displayAll();

	typedef std::map<std::string,cv::Mat> ImgeStoreType;
	ImgeStoreType imageStore;

	typedef std::map<std::string,float> FloatStoreType;
	FloatStoreType floatStore;

	typedef std::map<std::string,int> IntStoreType;
	IntStoreType intStore;

	int getInt(const std::string& key)
	{
		return intStore.at(key);
	}

	int getFloat(const std::string& key)
	{
		return intStore.at(key);
	}
};

} // namespace
#endif
