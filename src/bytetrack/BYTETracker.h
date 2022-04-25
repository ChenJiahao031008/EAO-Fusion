#ifndef _BYTE_STRACK_H_
#define _BYTE_STRACK_H_

#include "STrack.h"

namespace ORB_SLAM2
{
namespace BYTE_TRACK
{

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

class BYTETracker
{
public:
	BYTETracker(int frame_rate = 30, int track_buffer = 30);
	~BYTETracker();

	std::vector<STrack> update(const std::vector<Object>& objects);
	cv::Scalar get_color(int idx);

private:
	std::vector<STrack*> joint_stracks(std::vector<STrack*> &tlista, std::vector<STrack> &tlistb);
	std::vector<STrack> joint_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb);

	std::vector<STrack> sub_stracks(std::vector<STrack> &tlista, std::vector<STrack> &tlistb);
	void remove_duplicate_stracks(std::vector<STrack> &resa, std::vector<STrack> &resb, std::vector<STrack> &stracksa, std::vector<STrack> &stracksb);

	void linear_assignment(std::vector<std::vector<float> > &cost_matrix, int cost_matrix_size, int cost_matrix_size_size, float thresh,
		std::vector<std::vector<int> > &matches, std::vector<int> &unmatched_a, std::vector<int> &unmatched_b);
	std::vector<std::vector<float> > iou_distance(std::vector<STrack*> &atracks, std::vector<STrack> &btracks, int &dist_size, int &dist_size_size);
	std::vector<std::vector<float> > iou_distance(std::vector<STrack> &atracks, std::vector<STrack> &btracks);
	std::vector<std::vector<float> > ious(std::vector<std::vector<float> > &atlbrs, std::vector<std::vector<float> > &btlbrs);

	double lapjv(const std::vector<std::vector<float> > &cost, std::vector<int> &rowsol, std::vector<int> &colsol,
		bool extend_cost = false, float cost_limit = LONG_MAX, bool return_cost = true);

private:

	float track_thresh;
	float high_thresh;
	float match_thresh;
	int frame_id;
	int max_time_lost;

	std::vector<STrack> tracked_stracks;
	std::vector<STrack> lost_stracks;
	std::vector<STrack> removed_stracks;
	byte_kalman::KalmanFilter kalman_filter;
};
}
}

#endif
