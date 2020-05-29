# Test CUDA API Wrapper

Example that uses the cuda-api-wrapper by Eyal Rosenburg on Github:<br>
 
https://github.com/eyalroz/cuda-api-wrappers

Computes the sliding window average for a given a number of values and a window size.

This was used mainly for trying out newer C++11 features such as smart pointers and move semantics with CUDA. 
The CUDA memory allocation and deallocation is done more robustly 
using the RAII method, rather than manually, thanks to the cuda-api-wrapper functions. 
This is could actually be used as a CUDA template.

It was also used to experiment with CUDA kernel optimization techniques

## Dependencies

Development environment:
- Ubuntu 18.04.1 (Linux Kernel Release 5.3.0-53-generic, Kernel Version: #47~18.04.1-Ubuntu SMP)
- Intel Core i5 8300H, 2.3GHz, 16 GB RAM
- nVidia Geforce GTX 1050 (Pascal Architecture, Compute Capability 6.1)
- nVidia Driver Version 440.59
- CUDA 10.2
- g++/gcc 7.5
- Cmake 3.10.2

## Building

```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Usage

```bash
./test_cuda_api_wrapper [options]
Calculate num_vals sliding window averages, for a given window_size.
Outputs the num_vals,window_size, and GPU execution time in milliseconds
 Options:
--num_vals <n>:     Number of Values
--window_size <w>:  Window Size
--debug <d>:        Increased verbosity for debug
--help <h>:         Show help
```

Example command line:

```bash
./test_cuda_api_wrapper -n 1000000 -w 4000
 1000000,4000,41.9598;
```

The last value output is the time in milliseconds for the overall GPU execution including memory transfer to and from the GPU

## FindCUDAAPIWrapper CMake Module
I also included my version of the FindCUDAAPIWrapper Cmake module and even a BASH script to install it. This will help my CMakeLists.txt find
Cmake module for cuda-api-wrapper
