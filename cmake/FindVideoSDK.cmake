# - Find NVidia Video SDK
# Find the native xiAPI includes and libraries
#
#  VIDEO_SDK_INCLUDE_DIRS - where to find NvEncoder.h, etc
#  VIDEO_SDK_LIBRARIES   - List of libraries when using the VIDEO_SDK.
#  VIDEO_SDK_FOUND       - True if nvcuvid and nvencodeapi found.


if (UNIX)
set(VIDEO_SDK_ROOT_DIR "~/code/Video_Codec_SDK_11.0.10")

set(LIBRARY_PATHS
	~/usr/lib
	~/usr/local/lib
	/usr/lib
	/usr/local/lib
	${VIDEO_SDK_ROOT_DIR}/Lib/linux/stubs/aarch64
	)
endif()

if (WIN32)
set(VIDEO_SDK_ROOT_DIR "C:/code/Video_Codec_SDK_11.0.10")

set(LIBRARY_PATHS
	~/usr/lib
	~/usr/local/lib
	/usr/lib
	/usr/local/lib
	${VIDEO_SDK_ROOT_DIR}/Lib/x64
	)

endif()


find_library(NVCUVID_LIBRARY 
	NAMES nvcuvid
	PATHS ${LIBRARY_PATHS}
	)

if (UNIX)
find_library(NVENCODEAPI_LIBRARY 
	NAMES nvidia-encode
	PATHS ${LIBRARY_PATHS}
	)
endif()

if (WIN32)

find_library(NVENCODEAPI_LIBRARY 
	NAMES nvencodeapi
	PATHS ${LIBRARY_PATHS}
	)
endif()

find_path(VIDEO_SDK_INTERFACE_INCLUDE_PATH nvcuvid.h
 	~/usr/include
	~/usr/local/include
	/usr/include
	/usr/local/include
	${VIDEO_SDK_ROOT_DIR}/Interface
	)

find_path(NVENCODER_INCLUDE_PATH NvEncoder.h
 	~/usr/include
	~/usr/local/include
	/usr/include
	/usr/local/include
	${VIDEO_SDK_ROOT_DIR}/Samples/NvCodec/NvEncoder
	)

set (NVCODEC_PATH ${VIDEO_SDK_ROOT_DIR}/Samples/NvCodec)



find_path(NVUTILS_INCLUDE_PATH NvCodecUtils.h
	~/usr/include
   ~/usr/local/include
   /usr/include
   /usr/local/include
   ${VIDEO_SDK_ROOT_DIR}/Samples/Utils
   )

		
if(NVCUVID_LIBRARY AND NVENCODEAPI_LIBRARY AND VIDEO_SDK_INTERFACE_INCLUDE_PATH AND NVENCODER_INCLUDE_PATH AND NVUTILS_INCLUDE_PATH)
	set(VIDEO_SDK_FOUND TRUE)
	set(VIDEO_SDK_INCLUDE_PATHS ${NVCODEC_PATH} ${VIDEO_SDK_INTERFACE_INCLUDE_PATH} ${NVENCODER_INCLUDE_PATH} ${NVUTILS_INCLUDE_PATH} CACHE STRING "The include paths needed to use the nv video sdk api")
    set(VIDEO_SDK_LIBRARIES ${NVCUVID_LIBRARY} ${NVENCODEAPI_LIBRARY} CACHE STRING "The libraries needed to use the nv video sdk api")
endif()



	
mark_as_advanced(
    VIDEO_SDK_INCLUDE_PATHS
    VIDEO_SDK_LIBRARIES
	)
