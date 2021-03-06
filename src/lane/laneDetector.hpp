#ifndef LANE_LANE_DETECTOR_HPP
#define LANE_LANE_DETECTOR_HPP

#include "laneModel.hpp"

#include "utils.hpp"
#include <map>
#include <vector>
#include <opencv2/videoio.hpp>

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

	int getNumberOfLanes();
	cv::Mat getResultFrame();
private:
	/**
	 * @brief Graph
	 *
	 * Vertex: Line with LineProperty
	 * Edge: Lane with LaneProperty
	 *
	 * \see http://www.boost.org/doc/libs/1_55_0/libs/graph/doc/graph_concepts.html
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


	std::vector<cv::Vec2f> houghLines;
	cv::Mat warpTr;
	Graph model;

	cv::VideoWriter vv; ///< VideoWriter for debug stream

	void preprocess(); ///< generates spatio-temporal filtered image
	void projectFrameToGound(); ///< projects input image using calibration data
	void displayLineModels(); ///< generates images with overlay
	void displayLaneModels(); ///< generates images with overlay
	void displayAll(); ///< generates windows and display images

	void initLaneModels(std::vector<double> distances);

	void extractPointFeatures();
	void detectLineFeatres();

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
