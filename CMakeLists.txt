cmake_minimum_required(VERSION 3.18)

# # https://github.com/pytorch/pytorch/issues/54174#issuecomment-801860434
# function(CUDA_CONVERT_FLAGS EXISTING_TARGET)
#     get_property(old_flags TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS)
#     if(NOT "${old_flags}" STREQUAL "")
#         string(REPLACE ";" "," CUDA_flags "${old_flags}")
#         set_property(TARGET ${EXISTING_TARGET} PROPERTY INTERFACE_COMPILE_OPTIONS
#             "$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CXX>>:${old_flags}>$<$<BUILD_INTERFACE:$<COMPILE_LANGUAGE:CUDA>>:-Xcompiler=${CUDA_flags}>"
#             )
#     endif()
# endfunction()

# Toolchain file needs to be set before project() 
### FIND VCPKG
if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "")
else()
	message(STATUS "VCPKG_ROOT not set in environment variables")	  
endif()

project(mocat_trt LANGUAGES C CXX CUDA)

set(CMAKE_VERBOSE_MAKEFILE ON)
if (WIN32)

# if ("${CMAKE_VERSION}" VERSION_GREATER 3.6.3)
#     set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
# endif ("${CMAKE_VERSION}" VERSION_GREATER 3.6.3)

endif()

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Save the command line compile commands in the build output
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
#list(APPEND CMAKE_PREFIX_PATH "C:\\code\\libtorchRel")	
#include(BasicConfig)

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(glm REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(k4a REQUIRED)
find_package(k4arecord REQUIRED)


if (UNIX)

# enabling C++ 11 support for gcc
if(CMAKE_COMPILER_IS_GNUCXX)
  list( APPEND CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS} -pthread")
endif()

endif()

find_package(OpenCV REQUIRED)
find_package(TensorRT REQUIRED)
find_package(VideoSDK REQUIRED)

find_package(realsense2 REQUIRED)
find_package(k4a REQUIRED)
include_directories(external/json/single_include/nlohmann)


#file(GLOB_RECURSE SRC "src/*")
file(GLOB_RECURSE GFX "graphics/*")
file(GLOB_RECURSE SHD "shaders/*")
file(GLOB_RECURSE HDR "include/*")
file(GLOB_RECURSE MOD "models/*")
file(GLOB_RECURSE RES "resources/*")
file(GLOB_RECURSE DAT "data/*")
file(GLOB_RECURSE PAR "src/parse/*")

list(APPEND SRC ${SHD})
list(APPEND SRC ${GFX})
list(APPEND SRC ${HDR})
list(APPEND SRC ${PAR})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

### COPY SHADER FILES TO BUILD AND BINARY LOCATIONS IF THEY HAVE BEEN CHANGED ###
add_custom_target(
  shaders ALL
  DEPENDS ${SHD}
  COMMENT "Copying shader files"
  VERBATIM
)
foreach(ShaderFile ${SHD})
get_filename_component(ShaderName ${ShaderFile} NAME)
	add_custom_command(
		TARGET shaders
		COMMAND ${CMAKE_COMMAND}
		ARGS -E copy_if_different
			${ShaderFile}
			${PROJECT_BINARY_DIR}/bin/$(Configuration)/shaders/${ShaderName}
	)
		add_custom_command(
		TARGET shaders
		COMMAND ${CMAKE_COMMAND}
		ARGS -E copy_if_different
			${ShaderFile}
			${PROJECT_BINARY_DIR}/shaders/${ShaderName}
	)
endforeach()


### COPY MODEL FILES TO BUILD AND BINARY LOCATIONS IF THEY HAVE BEEN CHANGED ###
add_custom_target(
  models ALL
  DEPENDS ${MOD}
  COMMENT "Copying models files"
  VERBATIM
)
foreach(ModelFile ${MOD})
get_filename_component(ModelName ${ModelFile} NAME)
	add_custom_command(
		TARGET models
		COMMAND ${CMAKE_COMMAND}
		ARGS -E copy_if_different
			${ModelFile}
			${PROJECT_BINARY_DIR}/bin/$(Configuration)/models/${ModelName}
	)
		add_custom_command(
		TARGET models
		COMMAND ${CMAKE_COMMAND}
		ARGS -E copy_if_different
			${ModelFile}
			${PROJECT_BINARY_DIR}/models/${ModelName}
	)
endforeach()

### COPY Resources FILES TO BUILD AND BINARY LOCATIONS IF THEY HAVE BEEN CHANGED ###
add_custom_target(
  resources ALL
  DEPENDS ${RES}
  COMMENT "Copying resource files"
  VERBATIM
)
foreach(ResourceFile ${RES})
get_filename_component(ResourceName ${ResourceFile} NAME)
	add_custom_command(
		TARGET resources
		COMMAND ${CMAKE_COMMAND}
		ARGS -E copy_if_different
			${ResourceFile}
			${PROJECT_BINARY_DIR}/bin/$(Configuration)/resources/${ResourceName}
	)
	message(STATUS "resources... " ${ResourceName})
		add_custom_command(
		TARGET resources
		COMMAND ${CMAKE_COMMAND}
		ARGS -E copy_if_different
			${ResourceFile}
			${PROJECT_BINARY_DIR}/resources/${ResourceName}
	)
endforeach()

### COPY Data FILES TO BUILD AND BINARY LOCATIONS IF THEY HAVE BEEN CHANGED ###
add_custom_target(
  data ALL
  DEPENDS ${DAT}
  COMMENT "Copying data files"
  VERBATIM
)
foreach(DataFile ${DAT})
get_filename_component(DataName ${DataFile} NAME)
	add_custom_command(
		TARGET data
		COMMAND ${CMAKE_COMMAND}
		ARGS -E copy_if_different
			${DataFile}
			${PROJECT_BINARY_DIR}/bin/$(Configuration)/data/${DataName}
	)
		add_custom_command(
		TARGET data
		COMMAND ${CMAKE_COMMAND}
		ARGS -E copy_if_different
			${DataFile}
			${PROJECT_BINARY_DIR}/data/${DataName}
	)
endforeach()

set(K4A_PROJECT_NAME ${PROJECT_NAME}_k4a)
set(RS_PROJECT_NAME ${PROJECT_NAME}_rs)

add_executable(${K4A_PROJECT_NAME} ${SRC} src/app_k4a.cpp include/app_k4a.h)
add_dependencies(${K4A_PROJECT_NAME} shaders models resources)

add_executable(${RS_PROJECT_NAME} ${SRC} src/app_rs.cpp include/app_rs.h)
add_dependencies(${RS_PROJECT_NAME} shaders models resources)


set_target_properties(${K4A_PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)

target_link_libraries(${K4A_PROJECT_NAME} k4a::k4a k4a::k4arecord)
target_link_libraries(${K4A_PROJECT_NAME} ${CUDA_cudart_static_LIBRARY} ${CUDA_cuda_driver_LIBRARY} GLEW::GLEW glfw ${VIDEO_SDK_LIBRARIES} ${TensorRT_LIBRARIES} ${OpenCV_LIBS} ${CUDA_cudart_static_LIBRARY} ${CUDA_cuda_driver_LIBRARY})
target_include_directories(${K4A_PROJECT_NAME} PUBLIC "include" "graphics" "src/parse" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${OpenCV_INCLUDE_DIRS} ${VIDEO_SDK_INCLUDE_PATHS} ${TensorRT_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} )

target_link_libraries(${RS_PROJECT_NAME} realsense2::realsense-file realsense2::realsense2)
target_link_libraries(${RS_PROJECT_NAME} ${CUDA_cudart_static_LIBRARY} ${CUDA_cuda_driver_LIBRARY} GLEW::GLEW glfw ${VIDEO_SDK_LIBRARIES} ${TensorRT_LIBRARIES} ${OpenCV_LIBS} ${CUDA_cudart_static_LIBRARY} ${CUDA_cuda_driver_LIBRARY})
target_include_directories(${RS_PROJECT_NAME} PUBLIC "include" "graphics" "src/parse" ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${OpenCV_INCLUDE_DIRS} ${VIDEO_SDK_INCLUDE_PATHS} ${TensorRT_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} )

if (UNIX)
	target_include_directories(${K4A_PROJECT_NAME} PUBLIC ${GLM_INCLUDE_DIRS} /usr/src/tensorrt/samples/common)
	target_link_libraries(${K4A_PROJECT_NAME} ${OPENGL_LIBRARIES} ) # may case a DSO error on linux, use OPENGL_LIBRAIES instead
endif()

if (WIN32)
	target_link_libraries(${K4A_PROJECT_NAME} ${OPENGL_gl_LIBRARY}) # may case a DSO error on linux, use OPENGL_LIBRAIES instead
	target_link_libraries(${K4A_PROJECT_NAME} glm::glm)
endif()

message(STATUS "nvenc static_lib " ${VIDEO_SDK_LIBRARIES})	
message(STATUS "nvenc dirs " ${VIDEO_SDK_INCLUDE_PATHS})	

message(STATUS "CUDA_rt static_lib " ${CUDA_cudart_static_LIBRARY})	  
message(STATUS "CUDA_cuda_driver_lib " ${CUDA_cuda_driver_LIBRARY})	  
message(STATUS "include dirs for cuda " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})	  

message(STATUS "tensorrt libs" ${TensorRT_LIBRARIES})	  
message(STATUS "tensorrt dirs " ${TensorRT_INCLUDE_DIRS})	

message(STATUS "CUDA_rt static_lib " ${CUDA_cudart_static_LIBRARY})	  
message(STATUS "CUDA_cuda_driver_lib " ${CUDA_cuda_driver_LIBRARY})	  
message(STATUS "include dirs for cuda " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})	 




