#pragma once

#include <stdio.h>
#include <memory.h>
#include <math.h>


#include <chrono>
#include <iostream>
#include <fstream>

#include <NvInfer.h>
#include "NvInferPlugin.h"
#include <NvOnnxParser.h>
#include "opencv2/opencv.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "Texture.h"

#include "Framebuffer.h"
#include "Buffer.h"
#include "Shader.h"
#include "glhelper.h"
#include "Quad.h"

#include "find_peaks.hpp"
#include "refine_peaks.hpp"
#include "paf_score_graph.hpp"
#include "munkres.hpp"
#include "connect_parts.hpp"

#include "nlohmann/json.hpp"

#include <k4a/k4a.hpp>
#include <k4arecord/playback.hpp>


// includes, cuda

#include <cuda_runtime.h>
#include <cuda_gl_interop.h> // needs to be included after GL is included


#include <json.hpp>

class App
{

private:
	
	nvinfer1::ICudaEngine* mEngine;
	//std::unique_ptr<nvinfer1::IExecutionContext> context{nullptr};
	nvinfer1::IExecutionContext* context;
	// std::shared_ptr<nvinfer1::IRuntime> runtime;
	std::vector<void*> buffers_;

    int mNumber{0};             //!< The number to classify

	GLFWwindow* window_ = NULL;
	gl::Quad* quad;

	GLuint vao_;

	// KINECT
	uint32_t deviceCount;
	k4a_device_configuration_t config;
	k4a::calibration calibration;
	k4a::device dev;
	bool cameraRunning = false;
	k4a::capture capture;
	k4a::playback fileHandle;
	k4a::capture cap;
	int64_t fileLen;
	int64_t position;
	std::chrono::microseconds startTime;

	struct FrameProperties
	{
		int width;
		int height;
		int rate;
		int depthUnits;
	};

	struct FrameIntrinsics
	{
		float cx;
		float cy;
		float fx;
		float fy;

		float k1, k2, p1, p2, k3;
	};

	void init(k4a_depth_mode_t depthMode, k4a_color_resolution_t colorResolution);
	bool startCamera();
	bool getImages(k4a::image& depthBuffer, k4a::image& colorBuffer, k4a::image& infraImage);
	bool getIMU(k4a_imu_sample_t& imuSample);
	bool openFile(std::string filename);
	bool getImagesFromFile(k4a::image& depthBuffer, k4a::image& colorBuffer, k4a::image& infraImage);

	int64_t getLength() {
		return fileLen;
	}

	k4a::calibration getCalibration() {
		return calibration;
	}

	void setFilePos(int64_t pos) {
		position = pos;
		positionChanged = true;
	}

	void setPause(bool mode) {
		pause = mode;
	}

	void step() {
		stepForward = true;
	}


	bool positionChanged = false;
	bool pause = false;
	bool stepForward = false;

	k4a::image depthImage;
	k4a::image depthInColorSaceImage;
	k4a::image xyzImage;

	k4a::image colorImage;
	k4a::image infraImage;
	k4a_imu_sample_t imuValues;

	std::string k4a_mkv_filename_ = "./data/output1.mkv";
	bool frameReady = false;
	bool imuReady = false;

	// Color Buffer
	uint32_t color_width_rs_;
	uint32_t color_height_rs_;

	// Depth Buffer
	uint32_t depth_width_;
	uint32_t depth_height_;

	// Infrared Buffer
	uint32_t infrared_width_;
	uint32_t infrared_height_;


	gl::ShaderStorageBuffer<float>* color_buffer_input = NULL;
	gl::ShaderStorageBuffer<float>* cmap_buffer_output = NULL;
	gl::ShaderStorageBuffer<float>* paf_buffer_output = NULL;
	gl::ShaderStorageBuffer<float>* skeleton_keypoints = NULL;
	gl::ShaderStorageBuffer<float>* skeleton_links = NULL;

	
	gl::Texture::Ptr color_frame_;
	gl::Texture::Ptr depth_frame_;
	gl::Texture3D::Ptr cmap_frame_;
	gl::Texture3D::Ptr paf_frame_;

	cudaGraphicsResource* cuda_buffer_resource_input;
	cudaGraphicsResource* cuda_buffer_resource_output_cmap;
	cudaGraphicsResource* cuda_buffer_resource_output_paf;

	void* cuda_buffer_pointer_input;
	void* cuda_buffer_pointer_output_cmap;
	void* cuda_buffer_pointer_output_paf;

	int image_width_ = 256;
	int image_height_ = 256;

	int display_width_ = 1920;
	int display_height_ = 1080;

	int num_part_types = 18;
	int num_link_types = 21;

	std::map<std::string, const gl::Shader::Ptr> progs;

	// post-processing buffers
	int* mPeaks;
	int* mPeakCounts;
	int* mConnections;
	int* mObjects;
	int  mNumObjects;

	float* mRefinedPeaks;
	float* mScoreGraph;

	void* mAssignmentWorkspace;
	void* mConnectionWorkspace;

	const int C = 18;
	const int H = 64;
	const int W = 64;
	const int K = 21;
	const int M = 100;
	const int MAX_OBJECTS = 100;

	struct Topology
	{
		std::string category;
		std::vector<std::string> keypoints;
		int links[100 * 4];
		int numLinks;
	} topology;

	std::ofstream outputJsonFile;
	std::string jsonFileName = "./data/poses.json";




public:

	std::vector<const char*> args;

	App();
	~App();

	nvinfer1::ICudaEngine* createCudaEngine(const std::string& onnxFileName, nvinfer1::ILogger& logger);
	void copyPointersToBuffers(void* inputPtr, void* cmap_outputPtr, void* paf_outputPtr);
	void cudaInitBuffer(GLuint buffer_input_ID, size_t in_bufferSize, GLuint buffer_output_cmap_ID, size_t out_cmap_bufferSize, GLuint buffer_output_paf_ID, size_t out_paf_bufferSize);


	void mainLoop();

};


// Command line argument parser class

class CommandLineParser
{
public:
	struct CommandLineOption {
		std::vector<std::string> commands;
		std::string value;
		bool hasValue = false;
		std::string help;
		bool set = false;
	};
	std::unordered_map<std::string, CommandLineOption> options;
	CommandLineParser();
	void add(std::string name, std::vector<std::string> commands, bool hasValue, std::string help);
	void printHelp();
	void parse(std::vector<const char*> arguments);
	bool isSet(std::string name);
	std::string getValueAsString(std::string name, std::string defaultValue);
	int32_t getValueAsInt(std::string name, int32_t defaultValue);
};


CommandLineParser::CommandLineParser()
{
	add("help", { "--help" }, 0, "Show help");
	add("input", { "-i", "--input" }, 1, "set .mkv file for processing");
	add("output", { "-o", "--output" }, 1, "set location for output json (default ./data/REALSENSE_BAG_FILENAME/)");
}

void CommandLineParser::add(std::string name, std::vector<std::string> commands, bool hasValue, std::string help)
{
	options[name].commands = commands;
	options[name].help = help;
	options[name].set = false;
	options[name].hasValue = hasValue;
	options[name].value = "";
}

void CommandLineParser::printHelp()
{
	std::cout << "Available command line options:\n";
	for (auto option : options) {
		std::cout << " ";
		for (size_t i = 0; i < option.second.commands.size(); i++) {
			std::cout << option.second.commands[i];
			if (i < option.second.commands.size() - 1) {
				std::cout << ", ";
			}
		}
		std::cout << ": " << option.second.help << "\n";
	}
	std::cout << "Press any key to close...";
}

void CommandLineParser::parse(std::vector<const char*> arguments)
{
	bool printHelp = false;
	// Known arguments
	for (auto& option : options) {
		for (auto& command : option.second.commands) {
			for (size_t i = 0; i < arguments.size(); i++) {
				if (strcmp(arguments[i], command.c_str()) == 0) {
					option.second.set = true;
					// Get value
					if (option.second.hasValue) {
						if (arguments.size() > i + 1) {
							option.second.value = arguments[i + 1];
						}
						if (option.second.value == "") {
							printHelp = true;
							break;
						}
					}
				}
			}
		}
	}
	// Print help for unknown arguments or missing argument values
	if (printHelp) {
		options["help"].set = true;
	}
}

bool CommandLineParser::isSet(std::string name)
{
	return ((options.find(name) != options.end()) && options[name].set);
}

std::string CommandLineParser::getValueAsString(std::string name, std::string defaultValue)
{
	assert(options.find(name) != options.end());
	std::string value = options[name].value;
	return (value != "") ? value : defaultValue;
}

int32_t CommandLineParser::getValueAsInt(std::string name, int32_t defaultValue)
{
	assert(options.find(name) != options.end());
	std::string value = options[name].value;
	if (value != "") {
		char* numConvPtr;
		int32_t intVal = strtol(value.c_str(), &numConvPtr, 10);
		return (intVal > 0) ? intVal : defaultValue;
	}
	else {
		return defaultValue;
	}
	return int32_t();
}



int main(int argc, char** argv) {



	App app;

	for (int32_t i = 0; i < argc; i++) { app.args.push_back(argv[i]); };

	app.mainLoop();

	return 0;
}
