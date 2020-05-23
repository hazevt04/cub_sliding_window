#include <iostream>
#include "my_float2.h"

my_float2::my_float2( const float new_x, const float new_y ) {
   val.x = new_x;
   val.y = new_y;
}

my_float2::my_float2( const float2& lame_float2 ) {
   val.x = lame_float2.x;
   val.y = lame_float2.y;
}

// Copy constructor
my_float2::my_float2( const my_float2& other ) {
   val.x = other.val.x;
   val.y = other.val.y;
}
  

// Copy Assignment Operator
my_float2& my_float2::operator=( const my_float2& other ) {
   if ( this != &other ) {
      val.x = other.val.x;
      val.y = other.val.y;
   }

   return *this;
}

my_float2 my_float2::operator+( const my_float2& other ) {
   my_float2 result;
   result.val.x = val.x + other.val.x;
   result.val.y = val.y + other.val.y;
   return result;
}


my_float2 my_float2::operator/( const my_float2& other ) {
   my_float2 result;
   result.val.x = val.x / other.val.x;
   result.val.y = val.y / other.val.y;
   return result;
}

my_float2 my_float2::operator/( const float& other ) {
   my_float2 result;
   result.val.x = val.x / other;
   result.val.y = val.y / other;
   return result;
}

// Move constructor
my_float2::my_float2( my_float2&& other ) noexcept {
   val.x = other.val.x;
   val.y = other.val.y;

   other.val.x = 0.f;
   other.val.y = 0.f;
}   

// Move Assignment Operator
my_float2& my_float2::operator=( my_float2&& other ) noexcept {
   if ( this != &other ) {
      val.x = other.val.x;
      val.y = other.val.y;
   
      other.val.x = 0.f;
      other.val.y = 0.f;
   } 
   return *this;
}

std::ostream& operator<<(std::ostream& my_ostream, const my_float2& f2) {
   my_ostream << "{" << f2.val.x << ", " << f2.val.y << "}";
   return my_ostream; 
}

