#ifndef _CUDA_UTILS_H_
#define _CUDA_UTILS_H_

// Utility Macros for CUDA

#include <cuda_runtime.h>
#include "utils.h"

#define check_cuda_error(cerror,loc) { \
  if ( cerror != cudaSuccess ) { \
    printf( "%s(): "#loc " ERROR: %s (%d)\n", __func__, \
      cudaGetErrorString( cerror ), cerror ); \
    exit(EXIT_FAILURE); \
  } \
}

#define check_cuda_error_flag(cerror,loc) { \
  if ( cerror != cudaSuccess ) { \
    printf( "%s(): "#loc " ERROR: %s (%d)\n", __func__, \
      cudaGetErrorString( cerror ), cerror ); \
    error_flag = true; \
    return FAILURE; \
  } \
}

#define check_cuda_error_return(cerror,loc) { \
  if ( cerror != cudaSuccess ) { \
    printf( "%s(): "#loc " ERROR: %s (%d)\n", __func__, \
      cudaGetErrorString( cerror ), cerror ); \
    return; \
  } \
}

#define check_cuda_error_return_failure(cerror,loc) { \
  if ( cerror != cudaSuccess ) { \
    printf( "%s(): "#loc " ERROR: %s (%d)\n", __func__, \
      cudaGetErrorString( cerror ), cerror ); \
    return FAILURE; \
  } \
}

#define check_cuda_error_throw(cerror,loc) { \
  if ( cerror != cudaSuccess ) { \
    throw std::runtime_error(cudaGetErrorString( cerror )); \
  } \
}


#define try_cuda_func(cerror, func) { \
  cerror = func; \
  check_cuda_error( cerror, func ); \
}

#define try_cuda_func_error_flag(cerror, func) { \
  cerror = func; \
  check_cuda_error_flag( cerror, func ); \
}

#define try_cuda_func_return(cerror, func) { \
  cerror = func; \
  check_cuda_error_return( cerror, func ); \
}

#define try_cuda_func_return_failure(cerror, func) { \
  cerror = func; \
  check_cuda_error_return_failure( cerror, func ); \
}

#define try_cuda_func_throw(cerror, func) { \
  cerror = func; \
  check_cuda_error_throw( cerror, func ); \
}


#define try_cuda_free( cerror, ptr ) { \
  if ((ptr)) { \
    try_cuda_func( (cerror), cudaFree((ptr))); \
    (ptr) = nullptr; \
  } \
}

#define try_cuda_free_host( cerror, ptr ) { \
  if ((ptr)) { \
    try_cuda_func( (cerror), cudaFreeHost((ptr))); \
    (ptr) = nullptr; \
  } \
}

#define try_cuda_free_return( cerror, ptr ) { \
  if (ptr) { \
    try_cuda_func_return( (cerror), cudaFree((ptr)) ); \
  } \
}

#define try_cuda_free_throw( cerror, ptr ) { \
  if (ptr) { \
    try_cuda_func_throw( (cerror), cudaFree((ptr)) ); \
    (ptr) = nullptr; \
  } \
}


#ifdef TRY_FAST_MATH

__device__ __host__ inline float2 float2_add(float2 left, float2 right) {
    float2 result;
    result.x = __fadd_rn( left.x, right.x );
    result.y = __fadd_rn( left.y, right.y );
    return result;
}

// Complex subtraction
__device__ __host__ inline float2 float2_subtract(float2 left, float2 right) {
    float2 result;
    result.x = __fsub_rn( left.x, right.x );
    result.y = __fsub_rn( left.y, right.y );
    return result;
}

// Complex division
__device__ __host__ inline float2 float2_division(float2 left, float2 right) {
    float2 result;
    result.x = __fdividef( left.x, right.x );
    result.y = __fdividef( left.y, right.y );
    return result;
}

// Complex division by scalar
__device__ __host__ inline float2 float2_division_scalar(float2 left, float scalar_right) {
    float2 result;
    result.x = __fdividef( left.x, scalar_right );
    result.y = __fdividef( left.y, scalar_right );
    return result;
}

#else

// Complex addition
__device__ __host__ inline float2 float2_add(float2 left, float2 right) {
    float2 result;
    result.x = left.x + right.x;
    result.y = left.y + right.y;
    return result;
}

// Complex subtraction
__device__ __host__ inline float2 float2_subtract(float2 left, float2 right) {
    float2 result;
    result.x = left.x - right.x;
    result.y = left.y - right.y;
    return result;
}

// Complex division
__device__ __host__ inline float2 float2_division(float2 left, float2 right) {
    float2 result;
    result.x = left.x / right.x;
    result.y = left.y / right.y;
    return result;
}

// Complex division by scalar
__device__ __host__ inline float2 float2_division_scalar(float2 left, float scalar_right) {
    float2 result;
    result.x = left.x / scalar_right;
    result.y = left.y / scalar_right;
    return result;
}

#endif

#endif // ifndef _CUDA_UTILS_H_
