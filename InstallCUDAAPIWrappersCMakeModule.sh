#!/bin/bash

export CUDA_API_WRAPPERS_LIBRARY_DIR=/usr/local/lib
export CUDA_API_WRAPPERS_INCLUDE_DIR=/usr/local/include

# Replace this with your version number or a more robust way to determine path to CMake Modules
export CMAKE_VERSION=3.10

# I made up this CMake module for the cuda-api-wrappers to be 'found' in my CMakeLists.txt
sudo cp FindCUDAAPIWrappers.cmake /usr/share/cmake-${CMAKE_VERSION}/Modules
