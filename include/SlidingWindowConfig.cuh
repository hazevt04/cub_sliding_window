#ifndef __SLIDINGWINDOWCONFIG_CUH__
#define __SLIDINGWINDOWCONFIG_CUH__

class SlidingWindowConfig {
   public:
      int num_vals;
      int window_size;
      int num_results;
      bool debug;

      SlidingWindowConfig() {};

      SlidingWindowConfig( int new_num_vals, int new_window_size, bool new_debug = false ):
         num_vals( new_num_vals ),
         window_size( new_window_size ),
         debug( new_debug ) {

         num_results = new_num_vals - new_window_size;
      }
      
      // Copy Constructor
      SlidingWindowConfig( const SlidingWindowConfig& other ) {
         num_vals = other.num_vals;
         window_size = other.window_size;
         num_results = other.num_results;
         debug = other.debug;
      }

      // Move Constructor
      SlidingWindowConfig( SlidingWindowConfig&& other ) noexcept {
         num_vals = other.num_vals;
         window_size = other.window_size;
         num_results = other.num_results;
         debug = other.debug;
         
         other.num_vals = 0;
         other.window_size = 0;
         other.num_results = 0;
         other.debug = false;
      }

      // Copy Assignment Constructor
      SlidingWindowConfig& operator=( const SlidingWindowConfig& other ) {
         if ( this != &other ) {
            num_vals = other.num_vals;
            window_size = other.window_size;
            num_results = other.num_results;
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
            debug = other.debug;

            other.num_vals = 0;
            other.window_size = 0;
            other.num_results = 0;
            other.debug = false;
         }

         return *this;
      }
};

#endif
