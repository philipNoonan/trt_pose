# Mocat trt_pose
App to get and parse poses from kinect or realsense cameras

## Dependencies
Cuda
Glfw3
GLM
OpenCV
Nvidia TensorRT
Librealsense2
Kinect Azure SDK

## Installation
```
$ mkdir build && cd build
$ cmake ../
$ make -j4
```

## Running
Ensure that posenet onnx file exists in the /models directory. On first run, tensorRT will generate an engine file for this model. This will take some time.

Command line options are shown by running

```
$ ./bin/app --help
```

