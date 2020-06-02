#ifndef __SLIDINGWINDOW_CUH__

#include <cuda/api_wrappers.hpp>
#include <algorithm>
#include <random>
#include <memory>
#include <iomanip>

#include "cuda_utils.h"
#include "KernelSel.h"
#include "sliding_window_kernel.cuh"
#include "SlidingWindowConfig.cuh"

class SlidingWindow {
   public:
      SlidingWindow() {
         debug = false;
      }
      SlidingWindow( int new_num_vals, int new_window_size, 
         KernelSel new_kernel_sel, bool new_debug );
      SlidingWindow( const SlidingWindowConfig& config );
      // Move constructor
      SlidingWindow( SlidingWindow&& other ) noexcept;
      // Move assignment constructor
      SlidingWindow& operator=( SlidingWindow&& other ) noexcept;
      void run();
      ~SlidingWindow();
   protected:
      void gen_vals();
      void gen_expected();
      void check_results();
   private:
      // No copying allowed!
      SlidingWindow( const SlidingWindow& other );
      SlidingWindow& operator=( const SlidingWindow& other );
      inline void check_cache_pref( cuda::device_t<cuda::detail::assume_device_is_current> device ) {
         auto cache_pref = device.cache_preference();
         enum cudaFuncCache t_cache_pref = static_cast<enum cudaFuncCache>(cache_pref);
         // decode_cache_pref() is in cuda_utils.h and does not depend on cuda-api-wrappers
         // It directly uses the cudaFuncCache enum.
         std::cout << "Cache Preference is " << 
            decode_cache_pref(t_cache_pref) << "\n"; 
      }

      cuda::memory::managed::unique_ptr<float2 []> vals;
      cuda::memory::managed::unique_ptr<float2 []> results;
      cuda::memory::managed::unique_ptr<float2 []> expected_results;
      
      int num_vals;
      int window_size;
      int num_results;
      KernelSel kernel_sel;
      bool debug;

};

#endif
