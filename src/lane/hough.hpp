#ifndef CV_HOUGH2_HPP
#define CV_HOUGH2_HPP

#include <opencv2/core/core.hpp>

// sneak into opencv
namespace cv{

/**
 * @brief HoughLinesStandard2
 * @param img image is an input raster
 * @param rho discretization steps  - in pixels
 * @param theta discretization steps in radians
 * @param threshold threshold is the minimum number of pixels in the feature for it
to be a candidate for line
 * @param lines the output array of (rho, theta) pairs
 * @param linesMax linesMax is the buffer size (number of pairs)
 * @param min_theta
 * @param max_theta
 * @param img_out sneak into the accumulator
 */
void
HoughLinesStandard2( const Mat& img, float rho, float theta,
					int threshold, std::vector<Vec2f>& lines, int linesMax,
					double min_theta, double max_theta,Mat& img_out );


} // namespace

#endif
