#include <iostream>
#include <fstream>
#include <list>
#include <string>
#include <cinttypes>
using namespace std;

// Baumer SDK : camera SDK
#include "bgapi.hpp"

// OPENCV : display preview and save video 
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// Boost : parse config file / create thread for preview
#include "boost/program_options.hpp"
#include "boost/filesystem.hpp"
#include "boost/thread.hpp"
#include "boost/chrono.hpp"
namespace po = boost::program_options;
namespace fs = boost::filesystem;


// PARAMETERS-------------------------------------------------------------------
int preview;				
int subsample;
string result_dir;			

// Global variables-------------------------------------------------------------
int sys = 0;
int cam = 0;
BGAPI::System * pSystem = NULL;
BGAPI::Camera * pCamera = NULL;
BGAPI::Image ** pImage = NULL;
cv::VideoWriter writer;
ofstream file;
BGAPIX_TypeINT iTimeHigh, iTimeLow, iFreqHigh, iFreqLow;
BGAPIX_CameraImageFormat cformat; 
cv::Mat img_display;
uint64_t first_ts = 0;
uint64_t current_ts = 0;
uint64_t previous_ts = 0;
int roi_left = 0;	            
int roi_top = 0;
int roi_right = 0;
int roi_bottom = 0;
int height = 0;		            
int width = 0;			
int gainvalue = 0;
int gainmax = 0;
int exposurevalue = 0;
int exposuremax = 0;
int triggers = 0;
int fps = 0;
int packetsizevalue = 0;
int interpacketgapvalue = 0;
int fpsmax = 0;
int formatindex = 0;
int formatindexmax = 0;
uint32_t numbuffer = 0;
uint32_t exposuremax_slider = 0;
int fpsmax_slider = 0;

// memory buffer
list<cv::Mat> ImageList; // list of incoming images from camera
list<double> timeStampsList; // list of image timings
list<int> counterList; // list of image timings
list<unsigned int> HeaderFrameCountList; // list of header
list<unsigned int> HeaderTriggerCountList; // list of header
list<unsigned int> HeaderTimestampList; // list of header
list<int> hcounterList; // list of image timings
list<double> fpsList; // list of image timings

boost::mutex mtx_buffer;
boost::mutex mtx;

int read_config(int ac, char* av[]) {
	try {
		string config_file;

		// only on command line
		po::options_description generic("Generic options");
		generic.add_options()
			("help", "produce help message")
			("config,c", po::value<string>(&config_file)->default_value("behavior.cfg"),
				"configuration file")
			;

		// both on command line and config file
		po::options_description config("Configuration");
		config.add_options()
			("preview,p", po::value<int>(&preview)->default_value(1), "preview")
			("subsample,b", po::value<int>(&subsample)->default_value(1), "subsampling factor")
			("left", po::value<int>(&roi_left)->default_value(1), "ROI left")
			("top", po::value<int>(&roi_top)->default_value(1), "ROI top")
			("right", po::value<int>(&roi_right)->default_value(1), "ROI right")
			("bottom", po::value<int>(&roi_bottom)->default_value(1), "ROI bottom")
			("formatindex", po::value<int>(&formatindex)->default_value(0), "image format")
			("gain", po::value<int>(&gainvalue)->default_value(0), "gain")
			("exposure", po::value<int>(&exposurevalue)->default_value(3000), "exposure")
			("triggers", po::value<int>(&triggers)->default_value(0), "triggers")
			("fps", po::value<int>(&fps)->default_value(300), "fps")
			("result_dir,d", po::value<string>(&result_dir)->default_value(""), "result directory")
			("numbuffer,n", po::value<uint32_t>(&numbuffer)->default_value(100), "buffer size")
			("packetsize", po::value<int>(&packetsizevalue)->default_value(576), "buffer size")
			("exposuremax,e", po::value<uint32_t>(&exposuremax_slider)->default_value(3000), "max exposure slider")
			("fpsmax,f", po::value<int>(&fpsmax_slider)->default_value(300), "max fps slider")
			("interpacketgap,f", po::value<int>(&interpacketgapvalue)->default_value(0), "max fps slider")
			;

		po::options_description cmdline_options;
		cmdline_options.add(generic).add(config);

		po::options_description config_file_options;
		config_file_options.add(config);

		po::options_description visible("Allowed options");
		visible.add(generic).add(config);

		po::variables_map vm;
		store(po::command_line_parser(ac, av).options(cmdline_options).run(), vm);
		notify(vm);

		ifstream ifs(config_file.c_str());
		if (!ifs) {
			cout << "can not open config file: " << config_file << "\n";
			return 1;
		}
		else {
			store(parse_config_file(ifs, config_file_options), vm);
			notify(vm);
		}

		if (vm.count("help")) {
			cout << visible << "\n";
			return 2;
		}

		width = roi_right - roi_left;
		height = roi_bottom - roi_top;

		cout << "Running with following options " << endl
			<< "  Preview: " << preview << endl
			<< "  Subsample: " << subsample << endl
			<< "  Left: " << roi_left << endl
			<< "  Top: " << roi_top << endl
			<< "  Right: " << roi_right << endl
			<< "  Bottom: " << roi_bottom << endl
			<< "  Width: " << width << endl
			<< "  Height: " << height << endl
			<< "  Format Index: " << formatindex << endl
			<< "  Gain: " << gainvalue << endl
			<< "  Exposure: " << exposurevalue << endl
			<< "  Triggers: " << triggers << endl
			<< "  FPS: " << fps << endl
			<< "  Result directory: " << result_dir << endl
			<< "  Buffer size: " << numbuffer << endl
			<< "  Packet size: " << packetsizevalue << endl
			<< "  Inter Packet Gap: " << interpacketgapvalue << endl
			<< "  Max exposure slider: " << exposuremax_slider << endl
			<< "  Max fps slider: " << fpsmax_slider << endl;
			
	}
	catch (exception& e)
	{
		cout << e.what() << "\n";
		return 1;
	}
	return 0;
}
    
BGAPI_RESULT BGAPI_CALLBACK imageCallback(void * callBackOwner, BGAPI::Image* pCurrImage)
{
	cv::Mat img;
	cv::Mat img_resized;
	int swc;
	int hwc;
	int timestamplow = 0;
	int timestamphigh = 0;
	uint32_t timestamplow_u = 0;
	uint32_t timestamphigh_u = 0;
	BGAPI_RESULT res = BGAPI_RESULT_OK;

	unsigned char* imagebuffer = NULL;
	res = pCurrImage->get(&imagebuffer);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Image::get Errorcode: %d\n", res);
		return 0;
	}

	//TODO: print image counters somewhere
	res = pCurrImage->getNumber(&swc, &hwc);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Image::getNumber Errorcode: %d\n", res);
		return 0;
	}

	res = pCurrImage->getTimeStamp(&timestamphigh, &timestamplow);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Image::getTimeStamp Errorcode: %d\n", res);
		return 0;
	}
	timestamplow_u = timestamplow;
	timestamphigh_u = timestamphigh;
	if (swc == 0) {
		first_ts = (uint64_t) timestamphigh_u << 32 | timestamplow_u;
	}
	current_ts = (uint64_t) timestamphigh_u << 32 | timestamplow_u;
	double current_time = (double)(current_ts - first_ts) / (double)iFreqLow.current;
	double fps_hat = (double)iFreqLow.current / (double)(current_ts - previous_ts);
	previous_ts = current_ts;

	img = cv::Mat(cv::Size(width, height), CV_8U, imagebuffer);

	mtx_buffer.lock();
	// add current image and timestamp to buffers
	ImageList.push_back(img.clone()); // image memory buffer
	timeStampsList.push_back(current_time); // timing buffer
	counterList.push_back(swc);
	hcounterList.push_back(hwc);
	fpsList.push_back(fps_hat);
	mtx_buffer.unlock();

	res = ((BGAPI::Camera*)callBackOwner)->setImage(pCurrImage);
	if (res != BGAPI_RESULT_OK) {
		printf("setImage failed with %d\n", res);
	}
	return res;
}

static void trackbar_callback(int,void*) {

	BGAPI_RESULT res = BGAPI_RESULT_FAIL;
	BGAPI_FeatureState state; 
	BGAPIX_TypeROI roi;
	BGAPIX_TypeRangeFLOAT gain;
	BGAPIX_TypeRangeINT exposure;
	BGAPIX_TypeRangeFLOAT framerate;
	BGAPIX_TypeListINT imageformat;
	BGAPI_Resend resendvalues;
	BGAPIX_TypeRangeFLOAT sensorfreq;
	BGAPIX_TypeINT readouttime;
	BGAPIX_TypeRangeINT packetsize;
	BGAPIX_TypeINT tPacketDelay;

	state.cbSize = sizeof(BGAPI_FeatureState);
	roi.cbSize = sizeof(BGAPIX_TypeROI);
	gain.cbSize = sizeof(BGAPIX_TypeRangeFLOAT);
	exposure.cbSize = sizeof(BGAPIX_TypeRangeINT);
	framerate.cbSize = sizeof(BGAPIX_TypeRangeFLOAT);
	imageformat.cbSize = sizeof(BGAPIX_TypeListINT);
	resendvalues.cbSize = sizeof(BGAPI_Resend);
	sensorfreq.cbSize = sizeof(BGAPIX_TypeRangeFLOAT);
	readouttime.cbSize = sizeof(BGAPIX_TypeINT);
	packetsize.cbSize = sizeof(BGAPIX_TypeRangeINT);
	tPacketDelay.cbSize = sizeof(BGAPIX_TypeINT);

	// FORMAT INDEX : this goes first ?
	res = pCamera->setImageFormat(formatindex);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setImageFormat Errorcode: %d\n", res);
	}

	res = pCamera->getImageFormat(&state, &imageformat);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setImageFormat Errorcode: %d\n", res);
	}
	formatindex = imageformat.current;

	res = pCamera->getImageFormatDescription(formatindex, &cformat);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getImageFormatDescription Errorcode: %d\n", res);
	}

	// ROI
	// check dimensions 
	if ((roi_left + width > cformat.iSizeX) || (roi_top + height > cformat.iSizeY)) {
		printf("Image size is not compatible with selected format\n");
	}

	res = pCamera->setPartialScan(1, roi_left, roi_top, roi_left + width, roi_top + height);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setPartialScan Errorcode: %d\n", res);
	}

	res = pCamera->getPartialScan(&state, &roi);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getImageFormat Errorcode: %d\n", res);
	}

	roi_left = roi.curleft;
	roi_top = roi.curtop;
	roi_right = roi.curright;
	roi_bottom = roi.curbottom;
	width = roi.curright - roi.curleft;
	height = roi.curbottom - roi.curtop;

	// change size of display accordingly
	mtx.lock();
	img_display = cv::Mat(height / subsample, width / subsample, CV_8UC1);
	mtx.unlock();
	// change image size -> detach and reallocate images (only if using external buffer)  

	// GAIN
	res = pCamera->setGain(gainvalue);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setGain Errorcode: %d\n", res);
	}

	res = pCamera->getGain(&state, &gain);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setGain Errorcode: %d\n", res);
	}
	gainvalue = gain.current;

	// EXPOSURE 
	res = pCamera->setExposure(exposurevalue);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setExposure Errorcode: %d\n", res);
	}

	res = pCamera->getExposure(&state, &exposure);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setExposure Errorcode: %d\n", res);
	}
	exposurevalue = exposure.current;

	// TRIGGERS
	if (triggers) {
		res = pCamera->setTriggerSource(BGAPI_TRIGGERSOURCE_HARDWARE1);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTriggerSource Errorcode: %d\n", res);
		}

		res = pCamera->setTrigger(true);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTrigger Errorcode: %d\n", res);
		}

		res = pCamera->setTriggerActivation(BGAPI_ACTIVATION_RISINGEDGE);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTriggerActivation Errorcode: %d\n", res);
		}

		res = pCamera->setTriggerDelay(0);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTriggerDelay Errorcode: %d\n", res);
		}
	}
	else {
		res = pCamera->setTriggerSource(BGAPI_TRIGGERSOURCE_SOFTWARE);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTriggerSource Errorcode: %d\n", res);
		}

		res = pCamera->setTrigger(false);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTrigger Errorcode: %d\n", res);
		}

		// FPS: maybe do that only in preview mode without triggers ?
		res = pCamera->setFramesPerSecondsContinuous(fps);
		if (res != BGAPI_RESULT_OK) {
			printf("BGAPI::Camera::setFramesPerSecondsContinuous Errorcode: %d\n", res);
		}

		res = pCamera->getFramesPerSecondsContinuous(&state, &framerate);
		if (res != BGAPI_RESULT_OK) {
			printf("BGAPI::Camera::getFramesPerSecondsContinuous Errorcode: %d\n", res);
		}
		fps = framerate.current;
	}
	res = pCamera->getTrigger(&state);
	if (res != BGAPI_RESULT_OK)
	{
		printf("BGAPI::Camera::getTrigger Errorcode: %d\n", res);
	}
	triggers = state.bIsEnabled;

	res = pCamera->getReadoutTime(&state, &readouttime);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getReadoutTime Errorcode: %d\n", res);
	}
	cout << "Readout time: " << readouttime.current << endl;
	
	cv::setTrackbarMax("ROI height", "Controls", cformat.iSizeY);
	cv::setTrackbarMax("ROI width", "Controls", cformat.iSizeX);
	cv::setTrackbarMax("ROI left", "Controls", cformat.iSizeX);
	cv::setTrackbarMax("ROI top", "Controls", cformat.iSizeY);
}

void display_preview() {

	cv::namedWindow("Preview", cv::WINDOW_AUTOSIZE);
	if (preview) {
		cv::namedWindow("Controls", cv::WINDOW_NORMAL);
		cv::createTrackbar("ROI left", "Controls", &roi_left, cformat.iSizeX, trackbar_callback);
		cv::createTrackbar("ROI top", "Controls", &roi_top, cformat.iSizeY, trackbar_callback);
		cv::createTrackbar("ROI width", "Controls", &width, cformat.iSizeX, trackbar_callback);
		cv::createTrackbar("ROI height", "Controls", &height, cformat.iSizeY, trackbar_callback);
		cv::createTrackbar("Exposure", "Controls", &exposurevalue, exposuremax_slider, trackbar_callback);
		cv::createTrackbar("Gain", "Controls", &gainvalue, gainmax, trackbar_callback);
		cv::createTrackbar("FPS", "Controls", &fps, fpsmax_slider, trackbar_callback);
		cv::createTrackbar("Triggers", "Controls", &triggers, 1, trackbar_callback);
		cv::createTrackbar("Format Index", "Controls", &formatindex, formatindexmax, trackbar_callback);
	}
	while (true) {
		mtx.lock();
		cv::imshow("Preview", img_display);
		mtx.unlock();
		cv::waitKey(16); 
		boost::this_thread::sleep_for(boost::chrono::milliseconds(1)); // interruption point
	}
}

int setup_camera() {
	
	BGAPI_RESULT res = BGAPI_RESULT_FAIL;
	BGAPI_FeatureState state; 
	BGAPIX_TypeROI roi; 
	BGAPIX_TypeRangeFLOAT gain;
	BGAPIX_TypeRangeFLOAT framerate;
	BGAPIX_TypeRangeINT exposure;
	BGAPIX_TypeListINT imageformat;
	BGAPI_Resend resendvalues;
	BGAPIX_TypeRangeFLOAT sensorfreq;
	BGAPIX_TypeINT readouttime;
	BGAPIX_TypeRangeINT packetsize;
	BGAPIX_TypeINT tPacketDelay;
	BGAPIX_TypeListINT driverlist;

	cformat.cbSize = sizeof(BGAPIX_CameraImageFormat);
	state.cbSize = sizeof(BGAPI_FeatureState);
	iTimeHigh.cbSize = sizeof(BGAPIX_TypeINT);
	iTimeLow.cbSize = sizeof(BGAPIX_TypeINT);
	iFreqHigh.cbSize = sizeof(BGAPIX_TypeINT);
	iFreqLow.cbSize = sizeof(BGAPIX_TypeINT);
	roi.cbSize = sizeof(BGAPIX_TypeROI);
	gain.cbSize = sizeof(BGAPIX_TypeRangeFLOAT);
	exposure.cbSize = sizeof(BGAPIX_TypeRangeINT);
	framerate.cbSize = sizeof(BGAPIX_TypeRangeFLOAT);
	imageformat.cbSize = sizeof(BGAPIX_TypeListINT);
	resendvalues.cbSize = sizeof(BGAPI_Resend);
	sensorfreq.cbSize = sizeof(BGAPIX_TypeRangeFLOAT);
	readouttime.cbSize = sizeof(BGAPIX_TypeINT);
	packetsize.cbSize = sizeof(BGAPIX_TypeRangeINT);
	tPacketDelay.cbSize = sizeof(BGAPIX_TypeINT);
	driverlist.cbSize = sizeof(BGAPIX_TypeListINT);

	// Initializing the system--------------------------------------------------
	res = BGAPI::createSystem(sys, &pSystem);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::createSystem Errorcode: %d System index: %d\n", res, sys);
		return EXIT_FAILURE;
	}
	printf("Created system: System index %d\n", sys);

	res = pSystem->open();
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::System::open Errorcode: %d System index: %d\n", res, sys);
		return EXIT_FAILURE;
	}
	printf("System opened: System index %d\n", sys);

	res = pSystem->getGVSDriverModel(&state, &driverlist);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::System::getGVSDriverModel Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}
	cout << "Available driver models: " << endl;
	for (int i = 0; i < driverlist.length; i++) {
		cout << driverlist.array[i] << endl;
	}
	cout << "Current driver models: " << driverlist.current << endl;

	res = pSystem->createCamera(cam, &pCamera);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::System::createCamera Errorcode: %d Camera index: %d\n", res, cam);
		return EXIT_FAILURE;
	}
	printf("Created camera: Camera index %d\n", cam);

	res = pCamera->open();
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::open Errorcode: %d Camera index: %d\n", res, cam);
		return EXIT_FAILURE;
	}
	printf("Camera opened: Camera index %d\n", cam);

	// CAMERA FEATURES ------------------------------------------------------

	// FORMAT INDEX
	res = pCamera->getImageFormat(&state, &imageformat);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setImageFormat Errorcode: %d\n", res);
	}
	formatindexmax = imageformat.length;

	if ((formatindex < 0) || (formatindex > formatindexmax)) {
		printf("Image size is not compatible with selected format\n");
		return EXIT_FAILURE;
	}

	res = pCamera->setImageFormat(formatindex);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setImageFormat Errorcode: %d\n", res);
	}

	res = pCamera->getImageFormat(&state, &imageformat);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setImageFormat Errorcode: %d\n", res);
	}
	formatindex = imageformat.current;

	// ROI
	// check dimensions 
	res = pCamera->getImageFormatDescription(formatindex, &cformat);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getImageFormatDescription Errorcode: %d\n", res);
	}

	if ((roi_left + width > cformat.iSizeX) || (roi_top + height > cformat.iSizeY)) {
		printf("Image size is not compatible with selected format\n");
		return EXIT_FAILURE;
	}

	res = pCamera->setPartialScan(1, roi_left, roi_top, roi_left + width, roi_top + height);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setPartialScan Errorcode: %d\n", res);
	}

	res = pCamera->getPartialScan(&state, &roi);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getImageFormat Errorcode: %d\n", res);
	}

	roi_left = roi.curleft;
	roi_top = roi.curtop;
	roi_right = roi.curright;
	roi_bottom = roi.curbottom;
	width = roi.curright - roi.curleft;
	height = roi.curbottom - roi.curtop;

	// change size of display accordingly
	mtx.lock();
	img_display = cv::Mat(height/subsample, width / subsample, CV_8UC1);
	mtx.unlock();

	// GAIN
	res = pCamera->getGain(&state, &gain);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setGain Errorcode: %d\n", res);
	}
	gainmax = gain.maximum;

	if ((gainvalue < 0) || (gainvalue > gainmax)) {
		printf("Gain value is incorrect\n");
		return EXIT_FAILURE;
	}

	res = pCamera->setGain(gainvalue);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setGain Errorcode: %d\n", res);
	}

	res = pCamera->getGain(&state, &gain);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setGain Errorcode: %d\n", res);
	}
	gainvalue = gain.current;

	// EXPOSURE
	res = pCamera->getExposure(&state, &exposure);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setExposure Errorcode: %d\n", res);
	}
	exposuremax = exposure.maximum;

	if ((exposurevalue <= 0) || (exposurevalue > exposuremax)) {
		printf("Exposure value is incorrect\n");
		return EXIT_FAILURE;
	}

	res = pCamera->setExposure(exposurevalue);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setExposure Errorcode: %d\n", res);
	}

	res = pCamera->getExposure(&state, &exposure);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setExposure Errorcode: %d\n", res);
	}
	exposurevalue = exposure.current;

	// TRIGGERS
	if (triggers) {
		res = pCamera->setTriggerSource(BGAPI_TRIGGERSOURCE_HARDWARE1);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTriggerSource Errorcode: %d\n", res);
		}

		res = pCamera->setTrigger(true);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTrigger Errorcode: %d\n", res);
		}

		res = pCamera->setTriggerActivation(BGAPI_ACTIVATION_RISINGEDGE);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTriggerActivation Errorcode: %d\n", res);
		}

		res = pCamera->setTriggerDelay(0);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTriggerDelay Errorcode: %d\n", res);
		}
	}
	else {
		res = pCamera->setTriggerSource(BGAPI_TRIGGERSOURCE_SOFTWARE);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTriggerSource Errorcode: %d\n", res);
		}

		res = pCamera->setTrigger(false);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::Camera::setTrigger Errorcode: %d\n", res);
		}

		// FPS
		res = pCamera->getFramesPerSecondsContinuous(&state, &framerate);
		if (res != BGAPI_RESULT_OK) {
			printf("BGAPI::Camera::getFramesPerSecondsContinuous Errorcode: %d\n", res);
		}
		fpsmax = framerate.maximum;

		if ((fps <= 0) || (fps > fpsmax)) {
			printf("FPS continuous value is incorrect\n");
			return EXIT_FAILURE;
		}

		res = pCamera->setFramesPerSecondsContinuous(fps);
		if (res != BGAPI_RESULT_OK) {
			printf("BGAPI::Camera::setFramesPerSecondsContinuous Errorcode: %d\n", res);
		}

		res = pCamera->getFramesPerSecondsContinuous(&state, &framerate);
		if (res != BGAPI_RESULT_OK) {
			printf("BGAPI::Camera::getFramesPerSecondsContinuous Errorcode: %d\n", res);
		}
		fps = framerate.current;
	}
	res = pCamera->getTrigger(&state);
	if (res != BGAPI_RESULT_OK)
	{
		printf("BGAPI::Camera::getTrigger Errorcode: %d\n", res);
	}
	triggers = state.bIsEnabled;

	// READOUT
	res = pCamera->setReadoutMode(BGAPI_READOUTMODE_OVERLAPPED);
	if (res != BGAPI_RESULT_OK)
	{
		if (res == BGAPI_RESULT_FEATURE_NOTIMPLEMENTED) {
			printf("BGAPI::Camera::setReadoutMode not implemented, ignoring\n");
		}
		else {
			printf("BGAPI::Camera::setReadoutMode Errorcode: %d\n", res);
			return EXIT_FAILURE;
		}
	}

	// DIGITIZATION TAP
	res = pCamera->setSensorDigitizationTaps(BGAPI_SENSORDIGITIZATIONTAPS_SIXTEEN);
	if (res != BGAPI_RESULT_OK)
	{
		if (res == BGAPI_RESULT_FEATURE_NOTIMPLEMENTED) {
			printf("BGAPI::Camera::setSensorDigitizationTaps not implemented, ignoring\n");
		}
		else {
			printf("BGAPI::Camera::setSensorDigitizationTaps Errorcode: %d\n", res);
			return EXIT_FAILURE;
		}
	}

	// EXPOSURE MODE 
	// maybe change this to trigger width ?
	res = pCamera->setExposureMode(BGAPI_EXPOSUREMODE_TIMED);
	if (res != BGAPI_RESULT_OK)
	{
		if (res == BGAPI_RESULT_FEATURE_NOTIMPLEMENTED) {
			printf("BGAPI::Camera::setExposureMode not implemented, ignoring\n");
		}
		else {
			printf("BGAPI::Camera::setExposureMode Errorcode: %d\n", res);
			return EXIT_FAILURE;
		}
	}

	// TIME STAMPS
	res = pCamera->getTimeStamp(&state, &iTimeHigh, &iTimeLow, &iFreqHigh, &iFreqLow);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getTimeStamp Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}
	printf("Timestamps frequency [%d,%d]\n", iFreqHigh.current, iFreqLow.current);

	// For some reason this seems to freeze the hxg20nir
	/*
	res = pCamera->resetTimeStamp();
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::resetTimeStamp Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}*/

	res = pCamera->setFrameCounter(0, 0);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setFrameCounter Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}

	// Setting the right packet size is crucial for reliable performance
	// large packet size (7200 bytes) should be used for high-speed recording.
	// To allow the use of large packets, the network card must support 
	// "Jumbo frames" (this can be set in windows device manager)
	res = pCamera->setPacketSize(packetsizevalue);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::setPacketSize Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}

	// WARNING the minimum and maximum packet size seem to not always
	// reflect  the actual max size for the network card/camera.
	// Do not trust those values.
	res = pCamera->getPacketSize(&state, &packetsize);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getPacketSize Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}
	cout << "Packet size: "  << packetsize.current << ", Max: " << packetsize.maximum << ", Min: " << packetsize.minimum << endl;

	// set interpacket delay to optimize performance
	res = pCamera->setGVSPacketDelay(interpacketgapvalue);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getPacketSize Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}

	res = pCamera->getGVSPacketDelay(&state, &tPacketDelay);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getPacketSize Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}
	cout << "Packet delay: " << tPacketDelay.current << endl;



	// Resend algorithm: default values are probably fine
	res = pCamera->getGVSResendValues(&state, &resendvalues);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getGVSResendValues Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}
	cout << "Resend values: " << endl
		<< "\t MaxResendsPerImage: " << resendvalues.gigeresend.MaxResendsPerImage << endl
		<< "\t MaxResendsPerPacket: " << resendvalues.gigeresend.MaxResendsPerPacket << endl
		<< "\t FirstResendWaitPackets: " << resendvalues.gigeresend.FirstResendWaitPackets << endl
		<< "\t FirstResendWaitTime: " << resendvalues.gigeresend.FirstResendWaitTime << endl
		<< "\t NextResendWaitPackets: " << resendvalues.gigeresend.NextResendWaitPackets << endl
		<< "\t NextResendWaitTime: " << resendvalues.gigeresend.NextResendWaitTime << endl
		<< "\t FirstResendWaitPacketsDualLink: " << resendvalues.gigeresend.FirstResendWaitPacketsDualLink << endl
		<< "\t NextResendWaitPacketsDualLink: " << resendvalues.gigeresend.NextResendWaitPacketsDualLink << endl;

	res = pCamera->getDeviceClockFrequency(BGAPI_DEVICECLOCK_SENSOR, &state, &sensorfreq);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getDeviceClockFrequency Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}
	cout << "Sensor freq: " << sensorfreq.current << ", Max: " << sensorfreq.maximum << ", Min: " << sensorfreq.minimum << endl;

	res = pCamera->getReadoutTime(&state, &readouttime);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getReadoutTime Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}
	cout << "Readout time: " << readouttime.current << endl;

	// ALLOCATE BUFFERS 
	res = pCamera->setDataAccessMode(BGAPI_DATAACCESSMODE_QUEUEDINTERN, numbuffer);
	if (res != BGAPI_RESULT_OK)
	{
		printf("BGAPI::Camera::setDataAccessMode Errorcode %d\n", res);
		return EXIT_FAILURE;
	}

	// dynamic allocation
	pImage = new BGAPI::Image*[numbuffer];

	int i = 0;
	for (i = 0; i < numbuffer; i++)
	{
		res = BGAPI::createImage(&pImage[i]);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::createImage for Image %d Errorcode %d\n", i, res);
			break;
		}
	}
	printf("Images created successful!\n");

	for (i = 0; i < numbuffer; i++)
	{
		res = pCamera->setImage(pImage[i]);
		if (res != BGAPI_RESULT_OK)
		{
			printf("BGAPI::System::setImage for Image %d Errorcode %d\n", i, res);
			break;
		}
	}
	printf("Images allocated successful!\n");

	res = pCamera->registerNotifyCallback(pCamera, (BGAPI::BGAPI_NOTIFY_CALLBACK) &imageCallback);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::registerNotifyCallback Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

int run_camera()
{
	BGAPI_RESULT res = BGAPI_RESULT_FAIL;
	BGAPI_FeatureState state; state.cbSize = sizeof(BGAPI_FeatureState);
	BGAPIX_CameraStatistic statistics; statistics.cbSize = sizeof(BGAPIX_CameraStatistic);

    res = pCamera->setStart(true);
    if(res != BGAPI_RESULT_OK) {
        printf("BGAPI::Camera::setStart Errorcode: %d\n",res);
		return EXIT_FAILURE;
    }
	printf("Acquisition started\n");
    
    printf("\n\n=== ENTER TO STOP ===\n\n");
	int d;
	scanf("&d",&d);
    while ((d = getchar()) != '\n' && d != EOF)

    res = pCamera->setStart(false);
    if(res != BGAPI_RESULT_OK) {
        printf("BGAPI::Camera::setStart Errorcode: %d\n",res);
		return EXIT_FAILURE;
    }

	res = pCamera->getStatistic(&state, &statistics);
	if (res != BGAPI_RESULT_OK) {
		printf("BGAPI::Camera::getStatistic Errorcode: %d\n", res);
		return EXIT_FAILURE;
	}
	cout << endl << "Camera statistics:" << endl
		<< "  Received Frames Good: " << statistics.statistic[0] << endl
		<< "  Received Frames Corrupted: " << statistics.statistic[1] << endl
		<< "  Lost Frames: " << statistics.statistic[2] << endl
		<< "  Resend Requests: " << statistics.statistic[3] << endl
		<< "  Resend Packets: " << statistics.statistic[4] << endl
		<< "  Lost Packets: " << statistics.statistic[5] << endl
		<< "  Bandwidth: " << statistics.statistic[6] << endl 
		<< endl;		

	// release all resources ?

    res = pSystem->release();
    if(res != BGAPI_RESULT_OK) {
        printf( "BGAPI::System::release Errorcode: %d System index: %d\n", res,sys);
        return EXIT_FAILURE;
    }
    printf("System released: System index %d\n", sys);

	return EXIT_SUCCESS;
}

int exit_gracefully(int exitcode) {

	printf("\n\n=== ENTER TO CLOSE ===\n\n");
	scanf("&d");

	delete[] pImage;

	// Stop the program and release resources 
	if (!preview) {
		file.close();
	}

	cv::destroyAllWindows();
	return exitcode;
}

void process() {

	cv::Mat current_image;
	cv::Mat img_resized;
	BGAPI_ImageHeader header;
	double current_timing = 0;
	int swc = 0;
	int hwc = 0;
	double fps_hat = 0;

	while (true)
	{
		if (!ImageList.empty())
		{
			mtx_buffer.lock();
			current_image = ImageList.front();
			current_timing = timeStampsList.front();
			swc = counterList.front();
			hwc = hcounterList.front();
			fps_hat = fpsList.front();
			mtx_buffer.unlock();

			size_t buflen = ImageList.size();

			cv::resize(current_image, img_resized, cv::Size(), 1.0 / subsample, 1.0 / subsample);

			// compress image
			if (!preview) {
				file << swc << "\t" << setprecision(3) << std::fixed << 1000 * current_timing  << "\t" << hwc << std::endl;
				writer << img_resized;
			}

			// if you want to do online processing of the images, it should go here

			mtx.lock();
			img_resized.copyTo(img_display);
			mtx.unlock();

			if (((int)(current_timing) * 1000) % 100 == 0)
			{
				printf("FPS %.2f, Time elapsed : %d sec, Buffer size %zd \r", fps_hat, (int)(current_timing), buflen);
				fflush(stdout);
			}

			mtx_buffer.lock();
			ImageList.pop_front();
			timeStampsList.pop_front();
			counterList.pop_front();
			hcounterList.pop_front();
			fpsList.pop_front();
			mtx_buffer.unlock();
		}
		else
		{
			boost::this_thread::sleep_for(boost::chrono::milliseconds(1)); // interruption point
		}
	}
}

int main(int ac, char* av[])
{
	int retcode = 0;

	// read configuration files
	int read = 1;
	read = read_config(ac, av);
	if (read == 1) {
		printf("Problem parsing options, aborting");
		return exit_gracefully(1);
	}
	else if (read == 2) {
		return exit_gracefully(0);
	}

	retcode = setup_camera();
	if (retcode == EXIT_FAILURE) {
		return exit_gracefully(EXIT_FAILURE);
	}
	printf("Camera setup complete\n");

	if (!preview) {
		// Check if result directory exists
		fs::path dir(result_dir);
		if (!exists(dir)) {
			if (!fs::create_directory(dir)) {
				cout << "unable to create result directory, aborting" << endl;
				return exit_gracefully(1);
			}
		}

		// Get formated time string 
		time_t now;
		struct tm* timeinfo;
		char buffer[100];
		time(&now);
		timeinfo = localtime(&now);
		strftime(buffer, sizeof(buffer), "%Y_%m_%d_", timeinfo);
		string timestr(buffer);

		// Check if video file exists
		fs::path video;
		stringstream ss;
		int i = 0;
		do {
			ss << setfill('0') << setw(2) << i;
			video = dir / (timestr + ss.str() + ".avi");
			i++;
			ss.str("");
		} while (exists(video));
		char videoname[100];
		wcstombs(videoname, video.c_str(), 100);

		// Check if timestamps file exists
		fs::path ts = video.replace_extension("txt");
		if (exists(ts)) {
			printf("timestamp file exists already, aborting\n");
			return exit_gracefully(1);
		}

		writer.open(videoname, 
			cv::CAP_FFMPEG,
			cv::VideoWriter::fourcc('X', '2', '6', '4'), 
			fps, cv::Size(width, height), 
			{ cv::VIDEOWRITER_PROP_IS_COLOR, false,
			cv::VIDEOWRITER_PROP_HW_ACCELERATION, cv::VIDEO_ACCELERATION_ANY }
		);
		if (!writer.isOpened()) {
			printf("Problem opening Video writer, aborting\n");
			return exit_gracefully(1);
		}

		cout << "VideoWriter backend = " << writer.getBackendName() << endl
	         << "VideoWriter acceleration = " << writer.get(cv::VIDEOWRITER_PROP_HW_ACCELERATION) << endl
		     << "VideoWriter acceleration device = " << writer.get(cv::VIDEOWRITER_PROP_HW_DEVICE) << endl;

		// Create timestamps file
		file.open(ts.string());
	}
    
	img_display = cv::Mat(height / subsample, width / subsample, CV_8UC1);
	boost::thread bt(display_preview);
	boost::thread bt1(process);

	// launch acquisition 
	retcode = run_camera();
	if (retcode == EXIT_FAILURE) {
		return exit_gracefully(EXIT_FAILURE);
	}

	bt.interrupt();
	bt1.interrupt();

	// Stop the program 
	return exit_gracefully(EXIT_SUCCESS);
}
