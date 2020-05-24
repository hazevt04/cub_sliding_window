#ifndef __SLIDING_WINDOW_CUH__
#define __SLIDING_WINDOW_CUH__

#include <cuda/api_wrappers.hpp>
#include <algorithm>
#include <random>

#include "cuda_utils.h"
#include "sliding_window_kernel.cuh"

#define RAND_MAX_RECIP (1.0f/(float)RAND_MAX)

float2 get_rand_float2( void );

float2 get_next_float2( void );

int run_kernel( const int window_size, const int num_vals, const bool debug );

#endif
