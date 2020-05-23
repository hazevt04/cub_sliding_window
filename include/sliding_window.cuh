#ifndef __SLIDING_WINDOW_CUH__
#define __SLIDING_WINDOW_CUH__

#include <cuda/api_wrappers.hpp>
#include <algorithm>
#include <random>

#include "cuda_utils.h"
#include "my_float2.h"
#include "sliding_window_kernel.cuh"

#define RAND_MAX_RECIP (1.0f/(float)RAND_MAX)

my_float2 get_rand_my_float2( void );
float2 get_rand_float2( void );
int my_test( const int window_size, const int num_vals, const bool debug );
#endif
