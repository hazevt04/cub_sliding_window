// C++ File for main

#include "sliding_window.cuh"


int main(int argc, char **argv) {
   int window_size = 50;
   int num_vals = 1024;
   bool debug = true;
   int status = SUCCESS;

   try_func( status, "my_test failed", my_test( window_size, num_vals, debug ) );
}
