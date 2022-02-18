# HVS Viewer OpenGL
Software for hyperspectral imaging, running on Linux/Windows

## Dependencies
Cuda
Glew
Glfw3
GLM
Ximea SDK
OpenCV
Nvidia TensorRT
Nvidia Video SDK

## Installation
```
$ mkdir build && cd build
$ cmake ../
$ make -j4
```

## Running
Ensure that super_resolution.onnx exists in the /models directory. On first run, tensorRT will generate an engine file for this model. This will take some time.

```
$ ./bin/app -p ./shaders -o ./data
```

