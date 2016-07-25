#include "laneModel.hpp"

namespace lane {

void LaneModel::updateLineModel() {

}

void LaneModel::initLineModel(std::vector<double> distances) {
	for(std::vector<double>::iterator it = distances.begin(); it != distances.end(); it++)
	{
		lines.push_back(LineModel(*it) );
	}
}


} // namespace
