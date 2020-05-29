// C++ File for main
#include <getopt.h>
#include "SlidingWindow.cuh"

void print_usage( const char* prog_name ) {
   std::cout << prog_name << " [options]\n"
   "Calculate num_vals sliding window averages, for a given window_size.\n"
   "Outputs the num_vals,window_size, and GPU execution time in milliseconds\n"
   " Options:\n"
   "--num_vals <n>:     Number of Values\n"
   "--window_size <w>:  Window Size\n"
   "--debug <d>:        Increased verbosity for debug\n"
   "--help <h>:         Show help\n";
    exit(1);
}

typedef struct args_s {
   int num_vals;
   int window_size;
   int num_results;
   bool debug;
} args_t;


void get_args( args_t& args, int argc, char** argv ) {

   try {
      const char* const short_opts = "n:w:dh";
      const option long_opts[] = {
         {"num_vals", required_argument, nullptr, 'n'},
         {"window_size", required_argument, nullptr, 'w'},
         {"debug", no_argument, nullptr, 'd'},
         {"help", no_argument, nullptr, 'h'},
         {nullptr, no_argument, nullptr, 0}
      };
      while (true) {
         const auto opt = getopt_long(argc, argv, short_opts, long_opts, nullptr);

         if (-1 == opt)
            break;

         switch (opt) {
         case 'n':
            args.num_vals = std::strtol(optarg, nullptr, 10);
            break;
         case 'w':
            args.window_size = std::strtol(optarg, nullptr, 10);
            break;
         case 'h': // -h or --help
            print_usage( argv[0] );
            break;
         case 'd': // -d or --help
            args.debug = true;
            break;
         case '?': // Unrecognized option
         default:
            break;
         }
      } // end of while (true)
      args.num_results = args.num_vals - args.window_size;
      if ( args.debug ) {
         std::cout << "Num Vals set to: " << args.num_vals << std::endl;
         std::cout << "Window Size Vals set to: " << args.window_size << std::endl;
         std::cout << "Num Results set to " << args.num_results << std::endl;
      }
   } catch( std::exception& ex ) {
      std::cout << __func__ << "(): ERROR: " << ex.what() << std::endl;
   }
   
}

void run_kernel( const args_t& args ) {
   
   try {
      if ( args.debug ) {
         printf( "%s Before instantiation of sliding_window\n", __func__  );
      } 
      SlidingWindowConfig config;
      config.num_vals = args.num_vals;
      config.window_size = args.window_size;
      config.num_results = args.num_results;
      config.debug = args.debug;

      //SlidingWindow sliding_window( num_vals, window_size, debug );
      SlidingWindow sliding_window( config );
      sliding_window.run();
   } catch( std::exception& ex ) {
      std::cout << __func__ << "(): ERROR: " << ex.what() << std::endl;
   }
}


int main(int argc, char **argv) {
   try {
      args_t args;
      args.window_size = 4000;
      args.num_vals = 1000000;
      args.debug = false;
       
      get_args( args, argc, argv );

      run_kernel( args );
      exit(EXIT_SUCCESS);

   } catch( std::exception& ex ) {
      std::cout << __func__ << "(): ERROR: " << ex.what() << std::endl;
      exit(EXIT_FAILURE);
   }
}
