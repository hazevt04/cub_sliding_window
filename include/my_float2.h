#ifndef __MY_FLOAT2_H__
#define __MY_FLOAT2_H__

#include "cuda_utils.h"

// WHY DOESN'T CUDA HAVE OPERATORS FOR THEIR VECTOR TYPES BUILT-IN?!!!!!
class my_float2 {
   public:
      my_float2() {}

      my_float2( const float new_x, const float new_y );
      
      my_float2( const float2& lame_float2 );

      // Copy constructor
      my_float2( const my_float2& other );
      
      // Copy Assignment Operator
      my_float2& operator=( const my_float2& other );
      
      // Moveconstructor
      my_float2( my_float2&& other ) noexcept;

      // Move Assignment Operator
      my_float2& operator=( my_float2&& other ) noexcept;
      
      my_float2 operator+( const my_float2& other );
      
      my_float2 operator/( const my_float2& other );
      
      my_float2 operator/( const float& other );

      friend std::ostream& operator<<(std::ostream& my_ostream, const my_float2& f2);

   private: 
      float2 val;
};
#endif // end of #ifndef __MY_FLOAT2_H__
