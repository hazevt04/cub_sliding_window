cmake_minimum_required( VERSION 2.8.0 )


project( test_cuda_api_wrapper LANGUAGES CXX CUDA )

message( "CMAKE_MODULE_PATH is ${CMAKE_MODULE_PATH}" )

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA QUIET REQUIRED)
find_package(cuda-api-wrappers QUIET REQUIRED)
#include(CompileWithWarnings)

# Selects GPU arch flags for nvcc based on target_CUDA_architectures
# Jetson Xavier GPUs are compute capability 7.2, Tegra X2, and Geforce GTX 1050 are 6.1
#cuda_select_nvcc_arch_flags(ARCH_FLAGS 7.2)
cuda_select_nvcc_arch_flags(ARCH_FLAGS 6.1)

list(APPEND CUDA_NVCC_FLAGS ${ARCH_FLAGS})
list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=-v")
list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")

set(CUDA_VERBOSE_BUILD ON)

set(EXEC_NAME "${CMAKE_PROJECT_NAME}")

set(SRCS
   "src/utils.cpp"
   "src/my_file_io_funcs.cpp"
   "src/sliding_window_kernel.cu"
   "src/SlidingWindow.cu"
   "src/main.cpp"
   )


set(INC_DIRS
   "include"
   "${CUDA_API_WRAPPERS_INCLUDE_DIR}"
   )

include_directories( "${INC_DIRS}" )

cuda_add_executable( ${EXEC_NAME} "${SRCS}" )

target_link_libraries( ${EXEC_NAME} "${CUDA_API_WRAPPERS_LIB}" m )

set_property( TARGET ${EXEC_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON )


