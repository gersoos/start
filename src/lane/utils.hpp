#ifndef LANE_UTILS_HPP
#define LANE_UTILS_HPP

#include <string>
#include <opencv2/core.hpp>

namespace lane{

class Operation{
public:
	Operation();
	virtual ~Operation();

protected:
	void log(const std::string& msg); ///< using std::cout
	void error(const std::string& msg); ///< using std::cout then throws msg

public:
	virtual void init() = 0;
	virtual int process(cv::Mat input) = 0; ///< Input is never updated
};

} // namespace

#endif // LANE_UTILS_HPP
