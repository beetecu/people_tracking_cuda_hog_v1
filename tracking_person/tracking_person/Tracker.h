#pragma once

#include <array>

#include <vector>
#include <map>
#include <iostream>

#include "opencv2/cudabgsegm.hpp"
#include "opencv2/cudafilters.hpp"

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <iostream>

#include <stdlib.h>



using namespace std;
using namespace cv;
using namespace cv::cuda;


template<typename T, int C>
class CircularBuffer {
	int _start = 0;
	int _size = 0;
	std::array<T, C> data;
public:

};

class Tracker
{
	const int track_size = 15;
public:
	Tracker();
	Tracker(int x, int y, double speed);
	void addDetection(const cv::Rect& detection);
	void addDetections(const std::vector<cv::Rect>& detections);
	int queryDetectionTrack(const cv::Rect& detection) const;
	std::vector<cv::Rect> getTrack(int i) const;
	const cv::Rect& getDetection(int i) const;
	int size() const;
	std::vector<cv::Rect> getTracks() const;
	~Tracker();
private:
	int winx;
	int winy;
	double eps_mult;
	std::vector<bool> updated_tracks;
	std::vector<cv::Rect> tracks;
	std::vector<int> track_times;
};

