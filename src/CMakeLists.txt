cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(TensorRTInference LANGUAGES CXX CUDA)  # Specify languages here

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# Find packages
find_package(CUDA REQUIRED)
#find_package(TensorRT REQUIRED)  # If available, use find_package to locate TensorRT
find_package(OpenCV REQUIRED)

list(APPEND CMAKE_PREFIX_PATH "$ENV{HOME}/.local/lib/python3.8/site-packages/torch")
#list(APPEND CMAKE_PREFIX_PATH "/home/albert/Installs/libtorch/share/cmake")
find_package(Torch REQUIRED)



# Add include directories for CUDA and TensorRT
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TensorRT_INCLUDE_DIRS})  # Assuming find_package finds TensorRT include dirs
include_directories(${Torch_INCLUDE_DIRS})

# Print out the CUDA and TensorRT Include Directories and Libraries
message(STATUS "CUDA Include Directories: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
message(STATUS "TensorRT Include Directories: ${TensorRT_INCLUDE_DIRS}")
message(STATUS "Torch Libraries: ${TORCH_LIBRARIES}")
# Create the executable target
add_executable(tensorrt_inference main.cpp)  # Replace with your source files

# Link libraries to your target
target_link_libraries(tensorrt_inference
        ${CUDA_LIBRARIES}
        ${CUDA_CUDART_LIBRARY}
#        ${TORCH_LIBRARIES}
        ${OpenCV_LIBS}
        nvinfer
        nvinfer_plugin
        )


