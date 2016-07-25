#ifndef LANE_LANE_DETECTOR_HPP
#define LANE_LANE_DETECTOR_HPP

#include "laneModel.hpp"

#include "utils.hpp"
#include <map>

#include <boost/graph/adjacency_list.hpp>


namespace lane {

/**
 * @brief The LaneDetection class
*/
class LaneDetection : public Operation {

public:
	LaneDetection();
	virtual ~LaneDetection();
	virtual void init();
	virtual int process(cv::Mat input);

private:
	/**
	 * @brief Graph
	 *
	 * Vertex: Line with LineProperty
	 * Edge: Lane with LaneProperty
	 *
	 * \see http://www.boost.org/doc/libs/1_55_0/libs/graph/doc/bundles.html
	 * \see http://www.boost.org/doc/libs/1_55_0/libs/graph/doc/adjacency_list.html
	 *
	 */
	typedef boost::adjacency_list<boost::mapS, boost::vecS, boost::directedS,
							 LineProperty, LaneProperty> Graph;

	typedef Graph::vertex_descriptor Line;
	typedef Graph::vertex_iterator LineIterator;

	typedef Graph::edge_descriptor Lane;
	typedef Graph::edge_iterator LaneIterator;



	Graph model;

	void preprocess(); ///< generates "filtered" image, uses "input"
	void projectFrameToGound(); ///< projects input image using calibration data
	void displayLineModels();
	void displayLaneModels();
	void displayAll();

	void initLaneModels(std::vector<double> distances);

	void detectLineFeatures();

	void updateLineModels();
	void updateLaneModels();


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
