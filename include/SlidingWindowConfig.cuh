#ifndef __SLIDINGWINDOWCONFIG_CUH__
#define __SLIDINGWINDOWCONFIG_CUH__

#include "KernelSel.h"

class SlidingWindowConfig {
   public:
      int num_vals;
      int window_size;
      int num_results;
      KernelSel kernel_sel;
      bool debug;

      SlidingWindowConfig() {};

      SlidingWindowConfig( int new_num_vals, int new_window_size, 
            KernelSel new_kernel_sel = KernelSel::rolled_sel, 
            bool new_debug = false ):
         
         num_vals( new_num_vals ),
         window_size( new_window_size ),
         kernel_sel( new_kernel_sel ),
         debug( new_debug ) {

         num_results = new_num_vals - new_window_size;
      }
      
      // Copy Constructor
      SlidingWindowConfig( const SlidingWindowConfig& other ) {
         num_vals = other.num_vals;
         window_size = other.window_size;
         num_results = other.num_results;
         kernel_sel = other.kernel_sel;
         debug = other.debug;
      }

      // Move Constructor
      SlidingWindowConfig( SlidingWindowConfig&& other ) noexcept {
         num_vals = other.num_vals;
         window_size = other.window_size;
         num_results = other.num_results;
         kernel_sel = other.kernel_sel;
         debug = other.debug;
         
         other.num_vals = 0;
         other.window_size = 0;
         other.num_results = 0;
         other.kernel_sel = KernelSel::rolled_sel;
         other.debug = false;
      }

      // Copy Assignment Constructor
      SlidingWindowConfig& operator=( const SlidingWindowConfig& other ) {
         if ( this != &other ) {
            num_vals = other.num_vals;
            window_size = other.window_size;
            num_results = other.num_results;
            kernel_sel = other.kernel_sel;
            debug = other.debug;
         }

         return *this;
      }

      // Move Assignment Constructor
      SlidingWindowConfig& operator=( SlidingWindowConfig&& other ) noexcept {
         if ( this != &other ) {
            num_vals = other.num_vals;
            window_size = other.window_size;
            num_results = other.num_results;
            kernel_sel = other.kernel_sel;
            debug = other.debug;

            other.num_vals = 0;
            other.window_size = 0;
            other.num_results = 0;
            other.kernel_sel = KernelSel::rolled_sel;
            other.debug = false;
         }

         return *this;
      }
};

#endif
