// tracking_person.cpp: define el punto de entrada de la aplicación de consola.
//


#include "stdafx.h"
#include "Tracker.h"

#include <cstdlib>
#include <string>

#include <cstdio>
#include <chrono>

#include <opencv2/objdetect.hpp>
#include <opencv2/cudaobjdetect.hpp>

#include <opencv2/videoio.hpp>

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

#include < iostream>    
#include "opencv2\objdetect\objdetect.hpp"
#include "opencv2\highgui\highgui.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\cudaobjdetect.hpp"
#include "opencv2\cudaimgproc.hpp"
#include "opencv2\cudawarping.hpp"


#ifdef _DEBUG               
#pragma comment(lib, "opencv_core300d.lib")       
#pragma comment(lib, "opencv_highgui300d.lib")    
#pragma comment(lib, "opencv_imgcodecs300d.lib")  
#pragma comment(lib, "opencv_objdetect300d.lib")  
#pragma comment(lib, "opencv_imgproc300d.lib")  
#pragma comment(lib, "opencv_videoio300d.lib")    
#pragma comment(lib, "opencv_cudaobjdetect300d.lib")  
#pragma comment(lib, "opencv_cudawarping300d.lib")
#pragma comment(lib, "opencv_cudaimgproc300d.lib")

#else       
#pragma comment(lib, "opencv_core300.lib")       
#pragma comment(lib, "opencv_highgui300.lib")    
#pragma comment(lib, "opencv_imgcodecs300.lib")    
#pragma comment(lib, "opencv_objdetect300.lib")  
#pragma comment(lib, "opencv_imgproc300.lib")  
#pragma comment(lib, "opencv_videoio300.lib")    
#pragma comment(lib, "opencv_cudaobjdetect300.lib")
#pragma comment(lib, "opencv_cudawarping300.lib")
#pragma comment(lib, "opencv_cudaimgproc300.lib")
#endif    


using namespace std;
using namespace cv;
using namespace cv::cuda;


void run(const std::string& video) {
	using namespace cv;
	using std::chrono::high_resolution_clock;
	using std::chrono::time_point;
	using std::chrono::duration;
	using std::chrono::milliseconds;
	using std::string;
	using std::chrono::duration_cast;
	using std::to_string;
	//using namespace std::literals;
	using std::vector;
	const int frameskip = 1;
	const int framedump = 30;
	VideoCapture cap(video);
	if (!cap.isOpened()) {
		fprintf(stderr, "Could not open video\n");
		std::exit(EXIT_FAILURE);
	}
	//cap.set(CV_CAP_PROP_POS_MSEC, 120000);
	VideoWriter output_tracking;
	Size videoSize{ (int)cap.get(CV_CAP_PROP_FRAME_WIDTH), (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT) };
	auto it = find(begin(video), end(video), '.');
	string outputVideo(begin(video), it);
	output_tracking.open(outputVideo + "_out_tracking.mkv", VideoWriter::fourcc('D', 'I', 'V', 'X'), cap.get(CV_CAP_PROP_FPS), videoSize, true);
	VideoWriter output;
	output.open(outputVideo + "_out.mkv", VideoWriter::fourcc('D', 'I', 'V', 'X'), cap.get(CV_CAP_PROP_FPS), videoSize, true);
    
	cv::Ptr< cv::cuda::HOG> hog = cv::cuda::HOG::create();
	//auto hog = HOG::create(cv::Size(64,128), cv::Size(16, 16), cv::Size(8, 8));
	//auto hog = cv::Ptr<cv::HOGDescriptor>(new cv::HOGDescriptor());
	Mat detector = hog->getDefaultPeopleDetector();
	hog->setSVMDetector(detector);
	hog->setNumLevels(9);
	hog->setHitThreshold(0);
	hog->setScaleFactor(1.02);
	hog->setGroupThreshold(1);

	namedWindow("Tracking", CV_WINDOW_AUTOSIZE);
	Mat img;
	Mat img_aux;
	Mat orig;
	cuda::GpuMat gpu_img;
	// set up clocks to time our function
	time_point<high_resolution_clock> start, end;
	duration<double> elapsed;
	Tracker tracker(360, 288, frameskip);
	int lastTracks = 0;
	vector<Rect> found;
	while (cap.read(img)) {
		start = high_resolution_clock::now();
		//cap >> img;
		if (!img.data) {
			continue;
		}
		long long frame = cap.get(CV_CAP_PROP_POS_FRAMES);
		orig = img;
		cv::cvtColor(img, img_aux, COLOR_BGR2GRAY);
		cv::resize(img_aux, img, Size(360, 288));
		if (!(frame % frameskip)) {
			gpu_img.upload(img);
			found = vector<Rect>{};
			assert(found.size() == 0);
			//hog.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.02, 2);
			//hog.detect(img, found, 0, Size(8, 8), Size(32, 32));
			hog->detectMultiScale(gpu_img, found);
			/*
			if (found.size() > 150) {
			found = vector<Rect>{};
			}*/
			auto ms = cap.get(CAP_PROP_POS_MSEC);
			auto s = ms / 1000;
			auto m = s / 60;
			ms = ms - int(s) * 1000;
			s = s - int(m) * 60;


			tracker.addDetections(found);
			end = high_resolution_clock::now();
			elapsed = end - start;
			fprintf(stderr, "Time: %d:%d:%d", (int)m, (int)s, (int)ms);
			fprintf(stderr, " MS/frame: %ld ms\n", duration_cast<milliseconds>(elapsed).count());
		}
		for (int i = 0; i < tracker.size(); ++i) {
			auto& r = tracker.getDetection(i);
			//putText(orig, to_string(i), r.tl() * 1.5, FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255, 255));
			rectangle(orig, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);

		}

		if (lastTracks != tracker.size()) {
			output_tracking << orig;
		}
		if (found.size()) {
			output << orig;
		}
		if (!(frame % framedump)) lastTracks = tracker.size();
		imshow("Tracking", orig);
		if (waitKey(20) >= 0) {
			break;
		}
	}
}

int main()
{

	
	using namespace cv;
	using std::chrono::high_resolution_clock;
	using std::chrono::time_point;
	using std::chrono::duration;
	using std::chrono::milliseconds;
	using std::string;
	using std::chrono::duration_cast;
	using std::to_string;
	//using namespace std::literals;
	using std::vector;
	const vector<string> inputs{
		"C:/tmp/terrace1.avi",
	};
	//cap.set(CV_CAP_PROP_POS_MSEC, 120000);
	for (auto& video : inputs) {
		run(video);
	}
	return EXIT_SUCCESS;
}
