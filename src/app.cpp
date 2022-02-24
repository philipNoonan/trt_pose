#include "app.h"
#include <cuda_gl_interop.h> // needs to be included after GL is included



class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char *msg) noexcept override {
        using namespace std;
        std::string s;
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                s = "INTERNAL_ERROR";
                break;
            case Severity::kERROR:
                s = "ERROR";
                break;
            case Severity::kWARNING:
                s = "WARNING";
                break;
            case Severity::kINFO:
                s = "INFO";
                break;
            case Severity::kVERBOSE:
                s = "VERBOSE";
                break;
        }
        std::cerr << s << ": " << msg << std::endl;
    }
};
//======================================================================================================================

/// Using unique_ptr with Destroy is optional, but beats calling destroy() for everything
/// Borrowed from the NVidia tutorial, nice C++ skills !
template<typename T>
struct Destroy {
    void operator()(T *t) const {
        t->destroy();
    }
};

struct ObjectPose
{
	uint32_t ID;	/**< Object ID in the image frame, starting with 0 */

	float Left;	/**< Bounding box left, as determined by the left-most keypoint in the pose */
	float Right;	/**< Bounding box right, as determined by the right-most keypoint in the pose */
	float Top;	/**< Bounding box top, as determined by the top-most keypoint in the pose */
	float Bottom;	/**< Bounding box bottom, as determined by the bottom-most keypoint in the pose */

	/**
	 * A keypoint or joint in the topology. A link is formed between two keypoints.
	 */
	struct Keypoint
	{
		uint32_t ID;	/**< Type ID of the keypoint - the name can be retrieved with poseNet::GetKeypointName() */
		float x;		/**< The x coordinate of the keypoint */
		float y;		/**< The y coordinate of the keypoint */
	};

	std::vector<Keypoint> Keypoints;			/**< List of keypoints in the object, which contain the keypoint ID and x/y coordinates */
	std::vector<std::array<uint32_t, 2>> Links;	/**< List of links in the object.  Each link has two keypoint indexes into the Keypoints list */

	/**< Find a keypoint index by it's ID, or return -1 if not found.  This returns an index into the Keypoints list */
	inline int FindKeypoint(uint32_t id) const;

	/**< Find a link index by two keypoint ID's, or return -1 if not found.  This returns an index into the Links list */
	inline int FindLink(uint32_t a, uint32_t b) const;
};

// FindKeypoint
inline int ObjectPose::FindKeypoint(uint32_t id) const
{
	const uint32_t numKeypoints = Keypoints.size();

	for (uint32_t n = 0; n < numKeypoints; n++)
	{
		if (id == Keypoints[n].ID)
			return n;
	}

	return -1;
}

// FindLink
inline int ObjectPose::FindLink(uint32_t a, uint32_t b) const
{
	const uint32_t numLinks = Links.size();

	for (uint32_t n = 0; n < numLinks; n++)
	{
		if (a == Keypoints[Links[n][0]].ID && b == Keypoints[Links[n][1]].ID)
			return n;
	}

	return -1;

}



size_t getSizeByDim(const nvinfer1::Dims& dims)
{
	size_t size = 1;
	for (size_t i = 0; i < dims.nbDims; ++i)
	{
		size *= dims.d[i];
	}
	return size;
}

void App::cudaInitBuffer(GLuint buffer_input_ID, size_t in_bufferSize, GLuint buffer_output_cmap_ID, size_t out_cmap_bufferSize, GLuint buffer_output_paf_ID, size_t out_paf_bufferSize) {


	cudaGraphicsGLRegisterBuffer(&cuda_buffer_resource_input,
		buffer_input_ID, cudaGraphicsRegisterFlagsNone);
	cudaGraphicsMapResources(1, &cuda_buffer_resource_input, 0);
	cudaGraphicsResourceGetMappedPointer(&cuda_buffer_pointer_input, &in_bufferSize, cuda_buffer_resource_input);

	cudaGraphicsGLRegisterBuffer(&cuda_buffer_resource_output_cmap,
		buffer_output_cmap_ID, cudaGraphicsRegisterFlagsNone);
	cudaGraphicsMapResources(1, &cuda_buffer_resource_output_cmap, 0);
	cudaGraphicsResourceGetMappedPointer(&cuda_buffer_pointer_output_cmap, &out_cmap_bufferSize, cuda_buffer_resource_output_cmap);

	cudaGraphicsGLRegisterBuffer(&cuda_buffer_resource_output_paf,
		buffer_output_paf_ID, cudaGraphicsRegisterFlagsNone);
	cudaGraphicsMapResources(1, &cuda_buffer_resource_output_paf, 0);
	cudaGraphicsResourceGetMappedPointer(&cuda_buffer_pointer_output_paf, &out_paf_bufferSize, cuda_buffer_resource_output_paf);

	cudaGraphicsUnmapResources(1, &cuda_buffer_resource_input);
	cudaGraphicsUnmapResources(1, &cuda_buffer_resource_output_cmap);
	cudaGraphicsUnmapResources(1, &cuda_buffer_resource_output_paf);

	//cudaGraphicsUnregisterResource(cuda_buffer_resource_input);

}

App::App() {

};

App::~App() {

};

/// Parse onnx file and create a TRT engine
nvinfer1::ICudaEngine* App::createCudaEngine(const std::string& onnxFileName, nvinfer1::ILogger& logger) {
    using namespace std;
    using namespace nvinfer1;

    std::unique_ptr<IBuilder, Destroy<IBuilder>> builder{ createInferBuilder(logger) };

    std::unique_ptr<INetworkDefinition, Destroy<INetworkDefinition>> network{
            builder->createNetworkV2(1U << (unsigned)NetworkDefinitionCreationFlag::kEXPLICIT_BATCH) };
    std::unique_ptr<nvonnxparser::IParser, Destroy<nvonnxparser::IParser>> parser{
            nvonnxparser::createParser(*network, logger) };

    if (!parser->parseFromFile(onnxFileName.c_str(), static_cast<int>(ILogger::Severity::kINFO)))
        throw runtime_error("ERROR: could not parse ONNX model " + onnxFileName + " !");

    // Modern version with config
    std::unique_ptr<IBuilderConfig, Destroy<IBuilderConfig>> config(builder->createBuilderConfig());
    // This is needed for TensorRT 6, not needed by 7 !
    //config->setMaxWorkspaceSize(64*1024*1024);
    config->setMaxWorkspaceSize(1ULL << 32);
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    //config->setFlag(nvinfer1::BuilderFlag::kINT8);
    return builder->buildEngineWithConfig(*network, *config);
}

void App::copyPointersToBuffers(void* inputPtr, void* cmap_outputPtr, void* paf_outputPtr) {

	// get sizes of input and output and allocate memory required for input data and for output data
	//std::vector<nvinfer1::Dims> input_dims; // we expect only one input
	// //std::vector<nvinfer1::Dims> output_dims; // and one output
	buffers_.resize(mEngine->getNbBindings()); // buffers for input and output data
	// //cudaError_t error = cudaGetLastError();
	//std::cout << "buffers size " << buffers_.size();
	buffers_[0] = inputPtr;
	buffers_[1] = cmap_outputPtr;
	buffers_[2] = paf_outputPtr;




	// // INPUT

}

void App::mainLoop()
{

	auto retu = glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	//glfwWindowHint(GLFW_REFRESH_RATE, 30);
	glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
	window_ = glfwCreateWindow(display_width_, display_height_, "trt pose", NULL, NULL);
	if (!window_)
	{
		glfwTerminate();
		std::cout << "Failed to create window context" << std::endl;

	}

	glfwMakeContextCurrent(window_);
	glfwSwapInterval(0);
	glfwSetWindowSize(window_, display_width_, display_height_);

	if (glewInit() != GLEW_OK)
	{
		glfwTerminate();
		throw "Failed to initialize GLEW...";
	}

	glEnable(GL_DEPTH_TEST);

	// Parse args
	CommandLineParser commandLineParser;


	commandLineParser.parse(args);
	if (commandLineParser.isSet("help")) {
		commandLineParser.printHelp();
		std::cin.get();
		exit(0);
	}
	if (commandLineParser.isSet("bag")) {
		ros_bag_filename_ = commandLineParser.getValueAsString("bag", "./data/20190423_161107.bag");
	}
	if (commandLineParser.isSet("output")) {
		size_t pos = ros_bag_filename_.find(".bag");
		std::string outputdir;
		if (pos != std::string::npos) {
			outputdir = ros_bag_filename_.substr(0, pos);
		}
		jsonFileName = commandLineParser.getValueAsString("output", outputdir + "_poses.json");

	}

	quad = new gl::Quad;
	quad->updateVerts(1.0f, 1.0f);

	glGenVertexArrays(1, &vao_);
	glBindVertexArray(vao_);

	// Load rosbag

	rs2::config config_rs;
	rs2::context context_rs;

	const rs2::playback playback = context_rs.load_device(ros_bag_filename_);
	const std::vector<rs2::sensor> sensors = playback.query_sensors();
	for (const rs2::sensor& sensor : sensors) {
		const std::vector<rs2::stream_profile> stream_profiles = sensor.get_stream_profiles();
		for (const rs2::stream_profile& stream_profile : stream_profiles) {
			config_rs.enable_stream(stream_profile.stream_type(), stream_profile.stream_index());
		}
	}

	// Start Pipeline
	config_rs.enable_device_from_file(playback.file_name());
	pipeline_profile_ = pipeline_.start(config_rs);

	// Set Non Real Time Playback
	pipeline_profile_.get_device().as<rs2::playback>().set_real_time(false);

	// Show Enable Streams
	const std::vector<rs2::stream_profile> stream_profiles = pipeline_profile_.get_streams();
	for (const rs2::stream_profile stream_profile : stream_profiles) {
		std::cout << stream_profile.stream_name() << std::endl;
	}

	rs2_stream align_to = RS2_STREAM_COLOR;
	rs2::align align(align_to);

	// Retrieve Last Position
	uint64_t last_position = pipeline_profile_.get_device().as<rs2::playback>().get_position();

	// REALSENSE
	frameset_ = pipeline_.wait_for_frames();

	//Get processed aligned frame
	auto processed = align.process(frameset_);

	// Trying to get both other and aligned depth frames
	//rs2::video_frame other_frame = processed.first(align_to);
	rs2::depth_frame aligned_depth_frame = processed.get_depth_frame();
	auto depth_profile = aligned_depth_frame.get_profile().as<rs2::video_stream_profile>();
	auto z_intrin = depth_profile.get_intrinsics();

	auto color_profile = frameset_.get_color_frame().get_profile().as<rs2::video_stream_profile>();
	auto color_intrin = color_profile.get_intrinsics();

	std::cout << z_intrin.fx << " " << z_intrin.fy << " " << z_intrin.ppx << " " << z_intrin.ppy << std::endl;
	std::cout << color_intrin.fx << " " << color_intrin.fy << " " << color_intrin.ppx << " " << color_intrin.ppy << std::endl;

	// Retrieve Color Flame
	color_frame_rs_ = frameset_.get_color_frame();

	// Retrieve Depth Flame
	depth_frame_rs_ = frameset_.get_depth_frame();

	// Retrive Frame Size
	color_width_rs_ = color_frame_rs_.as<rs2::video_frame>().get_width();
	color_height_rs_ = color_frame_rs_.as<rs2::video_frame>().get_height();

	// Retrive Frame Size
	depth_width_rs_ = depth_frame_rs_.as<rs2::video_frame>().get_width();
	depth_height_rs_ = depth_frame_rs_.as<rs2::video_frame>().get_height();



	color_frame_ = std::make_shared<gl::Texture>();
	color_frame_->createStorage(1, image_width_, image_height_, GL_RGBA_INTEGER, GL_RGBA8UI, GL_UNSIGNED_BYTE, false);
	color_frame_->setFiltering(GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);
	color_frame_->setWarp(GL_CLAMP_TO_EDGE);

	depth_frame_ = std::make_shared<gl::Texture>();
	depth_frame_->createStorage(1, depth_width_rs_, depth_height_rs_, GL_RED_INTEGER, GL_R16UI, GL_UNSIGNED_SHORT, false);
	depth_frame_->setFiltering(GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);
	depth_frame_->setWarp(GL_CLAMP_TO_EDGE);

	cmap_frame_ = std::make_shared<gl::Texture3D>();
	cmap_frame_->createStorage(1, 64, 64, 18, GL_TEXTURE_2D_ARRAY, GL_RED, GL_R32F, GL_FLOAT, false);
	cmap_frame_->setFiltering(GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);
	cmap_frame_->setWarp(GL_CLAMP_TO_EDGE);

	paf_frame_ = std::make_shared<gl::Texture3D>();
	paf_frame_->createStorage(1, 64, 64, 42, GL_TEXTURE_2D_ARRAY, GL_RED, GL_R32F, GL_FLOAT, false);
	paf_frame_->setFiltering(GL_LINEAR, GL_LINEAR_MIPMAP_LINEAR);
	paf_frame_->setWarp(GL_CLAMP_TO_EDGE);



	Logger logger;
	logger.log(nvinfer1::ILogger::Severity::kINFO, "C++ TensorRT DenseNet121 ");
	logger.log(nvinfer1::ILogger::Severity::kINFO, "Creating engine ...");

	std::vector<char> trtModelStream_;
	size_t size{ 0 };

	std::ifstream srr_engine_file("./models/pose_densenet121_body_f16.engine", std::ios::binary);

	if (srr_engine_file.good())
	{
		srr_engine_file.seekg(0, srr_engine_file.end);
		size = srr_engine_file.tellg();
		srr_engine_file.seekg(0, srr_engine_file.beg);
		trtModelStream_.resize(size);
		std::cout << "size of engine file : " << trtModelStream_.size() << std::endl;
		srr_engine_file.read(trtModelStream_.data(), size);
		srr_engine_file.close();

		std::unique_ptr<nvinfer1::IRuntime, Destroy<nvinfer1::IRuntime>> runtime(nvinfer1::createInferRuntime(logger));
		// https://github.com/onnx/onnx-tensorrt/issues/597
		bool didInitPlugins = initLibNvInferPlugins(nullptr, "");
		//std::unique_ptr<nvinfer1::ICudaEngine, Destroy<nvinfer1::ICudaEngine>> engine(runtime->deserializeCudaEngine(trtModelStream_.data(), trtModelStream_.size()));shared_ptr
		mEngine = runtime->deserializeCudaEngine(trtModelStream_.data(), size);

		if (!mEngine) {
			throw std::runtime_error("Deserialize error !");
		}


		context = mEngine->createExecutionContext();


		//context.reset(mEngine->createExecutionContext());


	}
	else {
		std::unique_ptr<nvinfer1::ICudaEngine, Destroy<nvinfer1::ICudaEngine>> engine(createCudaEngine("./models/pose_densenet121_body.onnx", logger));

		// Write engine to disk
		std::unique_ptr<nvinfer1::IHostMemory, Destroy<nvinfer1::IHostMemory>> serializedEngine(engine->serialize());
		std::cout << "\nSerialized engine : size = " << serializedEngine->size() << ", dtype = " << (int)serializedEngine->type()
			<< std::endl;

		std::ofstream out("./models/pose_densenet121_body_f16.engine", std::ios::binary);
		out.write((char*)serializedEngine->data(), serializedEngine->size());
	}


	color_buffer_input = new gl::ShaderStorageBuffer<float>;
	cmap_buffer_output = new gl::ShaderStorageBuffer<float>;
	paf_buffer_output = new gl::ShaderStorageBuffer<float>;
	skeleton_keypoints = new gl::ShaderStorageBuffer<float>;
	skeleton_links = new gl::ShaderStorageBuffer<float>;

	int in_buffsize = getSizeByDim(mEngine->getBindingDimensions(0)); // 1 x 3 x 256 x 256
	int out_cmap_buffsize = getSizeByDim(mEngine->getBindingDimensions(1)); // 1 x 18 x 64 x 64
	int out_paf_buffsize = getSizeByDim(mEngine->getBindingDimensions(2)); // 1 x 42 x 64 x 64


	std::vector<float> tmpInputData(in_buffsize, 0);

	color_buffer_input->bind();
	color_buffer_input->create(tmpInputData.data(), tmpInputData.size(), GL_DYNAMIC_DRAW);
	color_buffer_input->bindBase(0);
	color_buffer_input->unbind();

	std::vector<float> tmpOutputDataCmap(out_cmap_buffsize);

	cmap_buffer_output->bind();
	cmap_buffer_output->create(tmpOutputDataCmap.data(), tmpOutputDataCmap.size(), GL_DYNAMIC_DRAW);
	cmap_buffer_output->bindBase(1);
	cmap_buffer_output->unbind();

	std::vector<float> tmpOutputDataPaf(out_paf_buffsize);

	paf_buffer_output->bind();
	paf_buffer_output->create(tmpOutputDataPaf.data(), tmpOutputDataPaf.size(), GL_DYNAMIC_DRAW);
	paf_buffer_output->bindBase(2);
	paf_buffer_output->unbind();

	skeleton_keypoints->bind();
	skeleton_keypoints->create(nullptr, 18 * 3, GL_DYNAMIC_DRAW);
	skeleton_keypoints->bindBase(3);
	skeleton_keypoints->unbind();

	skeleton_links->bind();
	skeleton_links->create(nullptr, 21 * 2, GL_DYNAMIC_DRAW);
	skeleton_links->bindBase(4);
	skeleton_links->unbind();

	
	cudaInitBuffer(color_buffer_input->get_id(), in_buffsize, cmap_buffer_output->get_id(), out_cmap_buffsize, paf_buffer_output->get_id(), out_paf_buffsize);

	copyPointersToBuffers(cuda_buffer_pointer_input, cuda_buffer_pointer_output_cmap, cuda_buffer_pointer_output_paf);

	memset(topology.links, 0, sizeof(topology.links));

    topology.numLinks = 0;

	// load the json
	nlohmann::json topology_json;
	std::string json_path = "./data/human_pose.json";
	try
	{
		std::ifstream topology_file(json_path);
		topology_file >> topology_json;
	}
	catch (...)
	{
		std::cout << "poseNet -- failed to load topology json from " << json_path << std::endl;
		//return false;
	}

	// https://nlohmann.github.io/json/features/arbitrary_types/
	topology.category = topology_json["supercategory"].get<std::string>();
	topology.keypoints = topology_json["keypoints"].get<std::vector<std::string>>();

	for (size_t n = 0; n < topology.keypoints.size(); n++)
		std::cout << "topology -- keypoint " << n << " " << topology.keypoints[n].c_str() << std::endl;

	// load skeleton links
	const auto skeleton = topology_json["skeleton"].get<std::vector<std::vector<int>>>();

	if (skeleton.size() >= 100)
	{
		std::cout << "topology from '%s' has more than the maximum number of skeleton links " << std::endl;
		//return false;
	}

	for (size_t n = 0; n < skeleton.size(); n++)
	{
		if (skeleton[n].size() != 2)
		{
			std::cout << "invalid skeleton link from topology " << std::endl;
			//return false;
		}


		topology.links[n * 4 + 0] = n * 2;
		topology.links[n * 4 + 1] = n * 2 + 1;
		topology.links[n * 4 + 2] = skeleton[n][0] - 1;
		topology.links[n * 4 + 3] = skeleton[n][1] - 1;

		topology.numLinks++;
	}


	//
	// Test with camera source
	// 
	//cv::VideoCapture cap(0);

	//if (!cap.isOpened()) {
	//	std::cout << "cannot open camera";
	//}
	//cv::Mat camMat(480, 640, CV_8UC3);
	//cv::Mat camMat_small(256, 256, CV_8UC3);
	//cv::Mat rgba;




	std::string pathToShaders("./shaders/");
	progs.insert(std::make_pair("imageToBuffer", std::make_shared<gl::Shader>(pathToShaders + "imageToBuffer.comp")));
	progs.insert(std::make_pair("bufferToImages", std::make_shared<gl::Shader>(pathToShaders + "bufferToImages.comp")));

	progs.insert(std::make_pair("screenQuad", std::make_shared<gl::Shader>(pathToShaders + "screenQuad.vert", pathToShaders + "screenQuad.frag")));
	progs.insert(std::make_pair("screenSkeleton", std::make_shared<gl::Shader>(pathToShaders + "screenSkeleton.vert", pathToShaders + "screenSkeleton.frag")));


	// alloc post-processing buffers
	mPeaks = (int*)malloc(C * M * 2 * sizeof(int));
	mPeakCounts = (int*)malloc(C * sizeof(int));
	mRefinedPeaks = (float*)malloc(C * M * 2 * sizeof(float));

	mScoreGraph = (float*)malloc(K * M * M * sizeof(float));
	mConnections = (int*)malloc(K * M * 2 * sizeof(int));
	mObjects = (int*)malloc(MAX_OBJECTS * C * sizeof(int));



	mAssignmentWorkspace = malloc(trt_pose::parse::assignment_out_workspace(M));
	mConnectionWorkspace = malloc(trt_pose::parse::connect_parts_out_workspace(C, M));




	// run pose estimation
	std::vector<ObjectPose> poses;

	GLuint error = glGetError();

	outputJsonFile.open(jsonFileName, std::ios::out | std::ios::app | std::ios::ate);


	while (!glfwWindowShouldClose(window_)) {

		glfwPollEvents();
		//glfwGetFramebufferSize(window_, &m_displayWidth, &m_displayHeight);
		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);



		// REALSENSE
		frameset_ = pipeline_.wait_for_frames();

		//Get processed aligned frame
		auto processed = align.process(frameset_);

		// Trying to get both other and aligned depth frames
		// rs2::video_frame other_frame = processed.first(align_to);
		rs2::depth_frame aligned_depth_frame = processed.get_depth_frame();


		// Retrieve Color Flame
		color_frame_rs_ = frameset_.get_color_frame();
		if (!color_frame_rs_) {
			continue;
		}
		// Retrieve Depth Flame
		depth_frame_rs_ = frameset_.get_depth_frame();
		if (!depth_frame_rs_) {
			continue;
		}

		// Retrive Frame Size
		color_width_rs_ = color_frame_rs_.as<rs2::video_frame>().get_width();
		color_height_rs_ = color_frame_rs_.as<rs2::video_frame>().get_height();

		// Retrive Frame Size
		depth_width_rs_ = depth_frame_rs_.as<rs2::video_frame>().get_width();
		depth_height_rs_ = depth_frame_rs_.as<rs2::video_frame>().get_height();

		// Create cv::Mat form Color Frame
		const rs2_format color_format = color_frame_rs_.get_profile().format();
		switch (color_format) {
			// RGB8
		case rs2_format::RS2_FORMAT_RGB8:
		{
			color_mat_rs_ = cv::Mat(color_height_rs_, color_width_rs_, CV_8UC3, const_cast<void*>(color_frame_rs_.get_data())).clone();
			cv::cvtColor(color_mat_rs_, color_mat_rs_, cv::COLOR_RGB2BGRA);
			cv::resize(color_mat_rs_, color_mat_rs_, cv::Size(256, 256));
			
			break;
		}
		// RGBA8
		case rs2_format::RS2_FORMAT_RGBA8:
		{
			color_mat_rs_ = cv::Mat(color_height_rs_, color_width_rs_, CV_8UC4, const_cast<void*>(color_frame_rs_.get_data())).clone();
			cv::cvtColor(color_mat_rs_, color_mat_rs_, cv::COLOR_RGBA2BGRA);
			cv::resize(color_mat_rs_, color_mat_rs_, cv::Size(256, 256));

			break;
		}
		// BGR8
		case rs2_format::RS2_FORMAT_BGR8:
		{
			color_mat_rs_ = cv::Mat(color_height_rs_, color_width_rs_, CV_8UC3, const_cast<void*>(color_frame_rs_.get_data())).clone();
			cv::cvtColor(color_mat_rs_, color_mat_rs_, cv::COLOR_BGR2BGRA);
			cv::resize(color_mat_rs_, color_mat_rs_, cv::Size(256, 256));

			break;
		}
		// BGRA8
		case rs2_format::RS2_FORMAT_BGRA8:
		{
			color_mat_rs_ = cv::Mat(color_height_rs_, color_width_rs_, CV_8UC4, const_cast<void*>(color_frame_rs_.get_data())).clone();
			cv::resize(color_mat_rs_, color_mat_rs_, cv::Size(256, 256));

			break;
		}
		// Y16 (GrayScale)
		case rs2_format::RS2_FORMAT_Y16:
		{
			color_mat_rs_ = cv::Mat(color_height_rs_, color_width_rs_, CV_16UC1, const_cast<void*>(color_frame_rs_.get_data())).clone();
			constexpr double scaling = static_cast<double>(std::numeric_limits<uint8_t>::max()) / static_cast<double>(std::numeric_limits<uint16_t>::max());
			color_mat_rs_.convertTo(color_mat_rs_, CV_8U, scaling);
			break;
		}
		// YUYV
		case rs2_format::RS2_FORMAT_YUYV:
		{
			color_mat_rs_ = cv::Mat(color_height_rs_, color_width_rs_, CV_8UC2, const_cast<void*>(color_frame_rs_.get_data())).clone();
			cv::cvtColor(color_mat_rs_, color_mat_rs_, cv::COLOR_YUV2BGR_YUYV);
			break;
		}
		default:
			throw std::runtime_error("unknown color format");
			break;
		}

		// depth_mat_rs_ = cv::Mat(depth_height_rs_, depth_width_rs_, CV_16UC1, const_cast<void*>(depth_frame_rs_.get_data())).clone();


		// WEBCAM
		// cap >> camMat;
		// cv::resize(camMat, camMat_small, cv::Size(256, 256));
		// cv::cvtColor(camMat_small, rgba, cv::COLOR_RGB2RGBA);



		color_frame_->update(color_mat_rs_.data);
		depth_frame_->update(aligned_depth_frame.get_data());

		error = glGetError();

		cudaInitBuffer(color_buffer_input->get_id(), in_buffsize, cmap_buffer_output->get_id(), out_cmap_buffsize, paf_buffer_output->get_id(), out_paf_buffsize);


		progs["imageToBuffer"]->use();
		color_frame_->bindImage(0, 0, GL_READ_ONLY);

		color_buffer_input->bind();
		color_buffer_input->bindBase(0);
		glDispatchCompute(GLHelper::divup(image_width_, 32), GLHelper::divup(image_height_, 32), 1);
		glMemoryBarrier(GL_ALL_BARRIER_BITS);
		progs["imageToBuffer"]->disuse();

		color_buffer_input->unbind();

		error = glGetError();



		color_buffer_input->read(tmpInputData.data(), 0, tmpInputData.size());
		error = glGetError();



		//progs["bufferToImages"]->use();
		//cmap_frame_->bindLayeredImage(0, 0, GL_WRITE_ONLY);

		//cmap_buffer_output->bindBase(0);

		//glDispatchCompute(GLHelper::divup(64, 32), GLHelper::divup(64, 32), 18);
		//glMemoryBarrier(GL_ALL_BARRIER_BITS);
		//progs["bufferToImages"]->disuse();
		//cmap_buffer_output->unbind();

		//progs["bufferToImages"]->use();
		//paf_frame_->bindLayeredImage(0, 0, GL_WRITE_ONLY);

		//paf_buffer_output->bindBase(0);

		//glDispatchCompute(GLHelper::divup(64, 32), GLHelper::divup(64, 32), 42);
		//glMemoryBarrier(GL_ALL_BARRIER_BITS);
		//progs["bufferToImages"]->disuse();
		//paf_buffer_output->unbind();



		//std::vector<float> ch0;
		//std::vector<float> ch1;
		//std::vector<float> ch2;

		//ch0.resize(256 * 256);
		//ch1.resize(256 * 256);
		//ch2.resize(256 * 256);

		//for (int i = 0; i < 256 * 256; i++) {
		//	ch0[i] = tmpInputData[i];
		//	ch1[i] = tmpInputData[i + (256 * 256)];
		//	ch2[i] = tmpInputData[i + (2 * 256 * 256)];
		//}

		//cv::imshow("ch0", cv::Mat(256, 256, CV_32FC1, ch0.data()));
		//cv::imshow("ch1", cv::Mat(256, 256, CV_32FC1, ch1.data()));
		//cv::imshow("ch2", cv::Mat(256, 256, CV_32FC1, ch2.data()));



		
		context->executeV2(buffers_.data());

		error = glGetError();

		cmap_buffer_output->read(tmpOutputDataCmap.data(), 0, tmpOutputDataCmap.size());

		//std::vector<std::vector<float>> chCmap(18, std::vector<float>(64 * 64));
		//for (int i = 0; i < 64 * 64; i++) {
		//	for (int j = 0; j < 18; j++) {
		//		//std::cout << i + (j * 64 * 64) << std::endl;
		//		chCmap[j][i] = tmpOutputDataCmap[i + (j * 64 * 64)] * 1000.0f;
		//	}
		//}

		////for (int j = 0; j < 18; j++) {
		//std::string titleMap = "ch" + std::to_string(4);
		//cv::imshow(titleMap, cv::Mat(64, 64, CV_32FC1, chCmap[1].data()));
		//cv::waitKey(1);

		//}


		paf_buffer_output->read(tmpOutputDataPaf.data(), 0, tmpOutputDataPaf.size());

		//cv::imshow("small", rgba);
		//cv::waitKey(1);



		trt_pose::parse::find_peaks_out_nchw(mPeakCounts, mPeaks, tmpOutputDataCmap.data(), 1, C, H, W, M, 0.25f, 5);  // 0.5f threshold
		trt_pose::parse::refine_peaks_out_nchw(mRefinedPeaks, mPeakCounts, mPeaks, tmpOutputDataCmap.data(), 1, C, H, W, M, 5);
		// compute score graph
		trt_pose::parse::paf_score_graph_out_nkhw(mScoreGraph, topology.links, tmpOutputDataPaf.data(), mPeakCounts, mRefinedPeaks, 1, K, C, H, W, M, 7);

		// generate connections
		memset(mConnections, -1, K * M * 2 * sizeof(int));
		memset(mObjects, -1, MAX_OBJECTS * C * sizeof(int));

		trt_pose::parse::assignment_out_nk(mConnections, mScoreGraph, topology.links, mPeakCounts, 1, C, K, M, 0.25f, mAssignmentWorkspace);  // 0.5f threshold

		trt_pose::parse::connect_parts_out_batch(&mNumObjects, mObjects, mConnections, topology.links, mPeakCounts, 1, K, C, M, MAX_OBJECTS, mConnectionWorkspace);

		glDisable(GL_DEPTH_TEST);
		glEnable(GL_PROGRAM_POINT_SIZE);


		progs["screenQuad"]->use();
		glViewport(0, 0, display_width_, display_height_);
		color_frame_->use(0);
		depth_frame_->use(1);

		quad->render();
		progs["screenQuad"]->disuse();

		nlohmann::ordered_json outputPoseJson;

		// collate results
		for (int i = 0; i < mNumObjects; i++)
		{
			ObjectPose obj_pose;

			obj_pose.ID = i;
			obj_pose.Left = 9999999;
			obj_pose.Top = 9999999;
			obj_pose.Right = 0;
			obj_pose.Bottom = 0;

			// add valid keypoints
			for (int j = 0; j < C; j++)
			{
				const int k = mObjects[i * C + j];

				if (k >= 0)
				{
					const int peak_idx = j * M * 2 + k * 2;

					ObjectPose::Keypoint keypoint;

					keypoint.ID = j;
					keypoint.x = mRefinedPeaks[peak_idx + 1] * 256;
					keypoint.y = mRefinedPeaks[peak_idx + 0] * 256;

					obj_pose.Keypoints.push_back(keypoint);
				}
			}

			// add valid links
			for (int k = 0; k < K; k++)
			{
				const int c_a = topology.links[k * 4 + 2];
				const int c_b = topology.links[k * 4 + 3];

				const int obj_a = mObjects[i * C + c_a];
				const int obj_b = mObjects[i * C + c_b];

				if (obj_a >= 0 && obj_b >= 0)
				{
					int a = obj_pose.FindKeypoint(c_a);
					int b = obj_pose.FindKeypoint(c_b);

					if (a < 0 || b < 0)
					{
						//LogError(LOG_TRT "poseNet::postProcess() -- missing keypoint in output object pose, skipping...\n");
						std::cout << "poseNet::postProcess() -- missing keypoint in output object pose, skipping..." << std::endl;
						continue;
					}

					const int link_idx = obj_pose.FindLink(a, b);

					if (link_idx >= 0)
					{
						std::cout << "poseNet::postProcess() --duplicate link detected, skipping..." << std::endl;

						//LogWarning(LOG_TRT "poseNet::postProcess() -- duplicate link detected, skipping...\n");
						continue;
					}

					if (a > b)
					{
						const int c = a;
						a = b;
						b = c;
					}

					obj_pose.Links.push_back({ (uint32_t)a, (uint32_t)b });
				}
			}

			// get bounding box
			const uint32_t numKeypoints = obj_pose.Keypoints.size();

			if (numKeypoints < 2)
				continue;

			for (uint32_t n = 0; n < numKeypoints; n++)
			{
				obj_pose.Left = MIN(obj_pose.Keypoints[n].x, obj_pose.Left);
				obj_pose.Top = MIN(obj_pose.Keypoints[n].y, obj_pose.Top);
				obj_pose.Right = MAX(obj_pose.Keypoints[n].x, obj_pose.Right);
				obj_pose.Bottom = MAX(obj_pose.Keypoints[n].y, obj_pose.Bottom);
			}

			poses.push_back(obj_pose);

			std::vector<float> detected_keypoints;
			for (int i = 0; i < obj_pose.Keypoints.size(); i++) {
				float depth = aligned_depth_frame.get_distance(obj_pose.Keypoints[i].x / 255.0f * 848.0f, obj_pose.Keypoints[i].y / 255.0f * 480.0f);
				
				float point[3] = { 0 };
				float pixel[2] = { obj_pose.Keypoints[i].x / 255.0f * 848.0f , obj_pose.Keypoints[i].y / 255.0f * 480.0f };

				rs2_deproject_pixel_to_point(point, &z_intrin, (float*)&pixel, depth);
				detected_keypoints.push_back(point[0]);
				detected_keypoints.push_back(point[1]);

				//detected_keypoints.push_back(obj_pose.Keypoints[i].x);
				//detected_keypoints.push_back(obj_pose.Keypoints[i].y);

				detected_keypoints.push_back(point[2]);
			}

			std::vector<float> detected_links;
			for (int i = 0; i < obj_pose.Links.size(); i++) {
				detected_links.push_back(obj_pose.Links[i][0]);
				detected_links.push_back(obj_pose.Links[i][1]);
			}

			nlohmann::ordered_json tempJ = {
				{"ID", obj_pose.ID},
				{"Left", obj_pose.Left},
				{"Top", obj_pose.Top},
				{"Right", obj_pose.Right},
				{"Bottom", obj_pose.Bottom},
				{"Keypoints", detected_keypoints},
				{"Links", detected_links}
			};

			outputPoseJson.push_back(tempJ);

			//std::cout << tempJ.dump() << std::endl;




			skeleton_keypoints->update(detected_keypoints.data(), 0, detected_keypoints.size());


			skeleton_links->update(detected_links.data(), 0, detected_links.size());

			glBindVertexArray(vao_);


			progs["screenSkeleton"]->use();
			glViewport(0, 0, display_width_, display_height_);
			depth_frame_->use(0);

			skeleton_keypoints->bindBase(0);
			skeleton_links->bindBase(1);

			glDrawArrays(GL_POINTS, 0, detected_keypoints.size() / 3);
			progs["screenSkeleton"]->disuse();

			glBindVertexArray(0);

		}

		outputJsonFile << outputPoseJson.dump() << std::endl;



		glfwSwapBuffers(window_);

	}






}