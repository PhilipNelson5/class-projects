#pragma once

#include <algorithm>
#include <cuda.h>
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief stream insertion operator for a vector
 * 
 * @tparam T type of element in the vector
 * @param o ostream to insert into
 * @param v vector to insert into the ostream
 * @return std::ostream& the ostream passed in
 */
template<typename T>
std::ostream& operator<<(std::ostream& o, const std::vector<T>& v)
{
  if (v.size() == 0) { o << "[ ]"; return o; }
  o << "[ ";
  std::for_each(begin(v), end(v)-1, [&o](T elem){ o << elem << ", "; });
  o << v.back() << " ]";
  return o;
}

/**
 * @brief stream insertion operator for a dim3
 * 
 * @param o ostream to insert into
 * @param d dim3 to insert into the ostream
 * @return std::ostream& the ostream passed in
 */
std::ostream& operator<<(std::ostream& o, const dim3& d)
{
  o << "[ " << d.x << " " << d.y << " " << d.z << " ]";
  return o;
}

/**
 * @brief recursive helper for the print function
 * 
 * @tparam T type of argument to be printed
 * @param t argument to be printed
 */
void print() { std::cout << '\n'; };

/**
 * @brief recursive helper for the print function
 * 
 * @tparam T type of argument to be printed
 * @param t argument to be printed
 */
template<typename T>
void print(T t) { std::cout << t; print(); };

/**
 * @brief simple print function which allows printing of multiple space separated values
 * 
 * @note print is designed to work like the python print function.
 *       Multiple values can be passed and they will be printed to the console
 *       with spaces in between and a newline at the end.
 *       Supports any type with a stream insertion operator defined for it.
 * 
 * @example print("hello", "world") // hello world\n
 * 
 * @tparam T the type of the first argument to be printed
 * @tparam Args the rest of the arguments
 * @param t the first argument to be printed
 * @param args the rest of the arguments
 */
template<typename T, typename... Args>
void print(T t, Args... args) { std::cout << t << ' '; print(args...); };
