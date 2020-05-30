// C++ File for main
#include <getopt.h>
#include "KernelSel.h"
#include "SlidingWindow.cuh"

void print_usage( const char* prog_name ) {
   std::cout << prog_name << " [options]\n"
   "Calculate num_vals sliding window averages, for a given window_size.\n"
   "Outputs the num_vals,window_size, and GPU execution time in milliseconds\n"
   " Options:\n"
   "--num_vals <n>:     Number of Values\n"
   "--window_size <w>:  Window Size\n"
   "--kernel_sel <k>    Kernel Selector. One of:\n"
   "                      1 - Rolled\n"
   "                      2 - Unrolled by 2\n"
   "                      4 - Unrolled by 4\n"
   "                      8 - Unrolled by 8\n"
   "--debug <d>:        Increased verbosity for debug\n"
   "--help <h>:         Show help\n";
    exit(EXIT_SUCCESS);
}

typedef struct args_s {
   int num_vals;
   int window_size;
   int num_results;
   KernelSel kernel_sel;
   bool debug;
} args_t;


void get_args( args_t& args, int argc, char** argv ) {

   try {
      const char* const short_opts = "n:w:k:dh";
      const option long_opts[] = {
         {"num_vals", required_argument, nullptr, 'n'},
         {"window_size", required_argument, nullptr, 'w'},
         {"kernel_sel", optional_argument, nullptr, 'k'},
         {"debug", no_argument, nullptr, 'd'},
         {"help", no_argument, nullptr, 'h'},
         {nullptr, no_argument, nullptr, 0}
      };
      int temp_kernel_sel = 1;
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
         case 'k':
            temp_kernel_sel = std::strtol(optarg, nullptr, 10);
            args.kernel_sel = static_cast<KernelSel>(temp_kernel_sel);
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
         std::cout << "Num Vals set to: " << args.num_vals << "\n";
         std::cout << "Window Size Vals set to: " << args.window_size << "\n";
         std::cout << "Kernel Select set to " << args.kernel_sel << "\n";
         std::cout << "Num Results set to " << args.num_results << "\n";
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
      config.kernel_sel = args.kernel_sel;
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
      args.kernel_sel = KernelSel::rolled_sel;
      args.debug = false;
       
      get_args( args, argc, argv );

      run_kernel( args );
      exit(EXIT_SUCCESS);

   } catch( std::exception& ex ) {
      std::cout << __func__ << "(): ERROR: " << ex.what() << std::endl;
      exit(EXIT_FAILURE);
   }
}
