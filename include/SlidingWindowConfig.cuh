#ifndef __SLIDING_WINDOW_CONFIG__
#define __SLIDING_WINDOW_CONFIG__

class SlidingWindowConfig {
   public:
      SlidingWindowConfig():
         num_vals( 0 ),
         num_results( 0 ),
         window_size( 0 ),
         debug( false ) {}

      SlidingWindowConfig( int new_num_vals, int new_window_size, bool new_debug ) :
         num_vals( new_num_vals ),
         window_size( new_window_size ),
         debug( new_debug ) {
         
         num_results = new_num_vals - new_window_size;
      }
      
      int num_vals;
      int num_results;
      int window_size;
      bool debug;
};

#endif // __SLIDING_WINDOW_CONFIG
