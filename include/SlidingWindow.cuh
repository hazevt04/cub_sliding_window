#ifndef __SLIDINGWINDOW_CUH__

#include <cuda/api_wrappers.hpp>
#include <algorithm>
#include <random>
#include <memory>
#include "cuda_utils.h"
#include "sliding_window_kernel.cuh"

class SlidingWindow {
   public:
      SlidingWindow() {
         debug = false;
      }
      SlidingWindow( int new_num_vals, int new_window_size, bool new_debug );
      // Move constructor
      SlidingWindow( SlidingWindow&& other ) noexcept;
      // Move assignment constructor
      SlidingWindow& operator=( SlidingWindow&& other ) noexcept;
      void run();
      ~SlidingWindow();
   protected:
      void gen_vals();
      void gen_expected();
   private:
      // No copying allowed!
      SlidingWindow( const SlidingWindow& other );
      SlidingWindow& operator=( const SlidingWindow& other );

      std::unique_ptr<float2> h_vals;
      std::unique_ptr<float2> h_results;
      std::unique_ptr<float2> expected_results;
      
      cuda::memory::device::unique_ptr<float2 []> d_vals;
      cuda::memory::device::unique_ptr<float2 []> d_results;

      int num_vals;
      int window_size;
      int num_results;
      bool debug;

};

#endif
