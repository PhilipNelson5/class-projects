#pragma once

#include <type_traits>
#include <cassert>
#include <complex>
#include <cstdint>
#include <mpi.h>

/**
 * @brief get the MPI Type from a C++ type
 * 
 * This is a constexpr function used to support generic C++
 * functions that send or receive data with templated types
 * 
 * @tparam T C++ type
 * @return constexpr MPI_Datatype The MPI Type
 */
template <typename T>
[[nodiscard]] constexpr MPI_Datatype mpi_get_type() noexcept
{
  MPI_Datatype mpi_type = MPI_DATATYPE_NULL;
  
  if constexpr (std::is_same<T, char>::value)
  {
    mpi_type = MPI_CHAR;
  }
  else if constexpr (std::is_same<T, signed char>::value)
  {
    mpi_type = MPI_SIGNED_CHAR;
  }
  else if constexpr (std::is_same<T, unsigned char>::value)
  {
    mpi_type = MPI_UNSIGNED_CHAR;
  }
  else if constexpr (std::is_same<T, wchar_t>::value)
  {
    mpi_type = MPI_WCHAR;
  }
  else if constexpr (std::is_same<T, signed short>::value)
  {
    mpi_type = MPI_SHORT;
  }
  else if constexpr (std::is_same<T, unsigned short>::value)
  {
    mpi_type = MPI_UNSIGNED_SHORT;
  }
  else if constexpr (std::is_same<T, signed int>::value)
  {
    mpi_type = MPI_INT;
  }
  else if constexpr (std::is_same<T, unsigned int>::value)
  {
    mpi_type = MPI_UNSIGNED;
  }
  else if constexpr (std::is_same<T, signed long int>::value)
  {
    mpi_type = MPI_LONG;
  }
  else if constexpr (std::is_same<T, unsigned long int>::value)
  {
    mpi_type = MPI_UNSIGNED_LONG;
  }
  else if constexpr (std::is_same<T, signed long long int>::value)
  {
    mpi_type = MPI_LONG_LONG;
  }
  else if constexpr (std::is_same<T, unsigned long long int>::value)
  {
    mpi_type = MPI_UNSIGNED_LONG_LONG;
  }
  else if constexpr (std::is_same<T, float>::value)
  {
    mpi_type = MPI_FLOAT;
  }
  else if constexpr (std::is_same<T, double>::value)
  {
    mpi_type = MPI_DOUBLE;
  }
  else if constexpr (std::is_same<T, long double>::value)
  {
    mpi_type = MPI_LONG_DOUBLE;
  }
  else if constexpr (std::is_same<T, std::int8_t>::value)
  {
    mpi_type = MPI_INT8_T;
  }
  else if constexpr (std::is_same<T, std::int16_t>::value)
  {
    mpi_type = MPI_INT16_T;
  }
  else if constexpr (std::is_same<T, std::int32_t>::value)
  {
    mpi_type = MPI_INT32_T;
  }
  else if constexpr (std::is_same<T, std::int64_t>::value)
  {
    mpi_type = MPI_INT64_T;
  }
  else if constexpr (std::is_same<T, std::uint8_t>::value)
  {
    mpi_type = MPI_UINT8_T;
  }
  else if constexpr (std::is_same<T, std::uint16_t>::value)
  {
    mpi_type = MPI_UINT16_T;
  }
  else if constexpr (std::is_same<T, std::uint32_t>::value)
  {
    mpi_type = MPI_UINT32_T;
  }
  else if constexpr (std::is_same<T, std::uint64_t>::value)
  {
    mpi_type = MPI_UINT64_T;
  }
  else if constexpr (std::is_same<T, bool>::value)
  {
    mpi_type = MPI_C_BOOL;
  }
  else if constexpr (std::is_same<T, std::complex<float>>::value)
  {
    mpi_type = MPI_C_COMPLEX;
  }
  else if constexpr (std::is_same<T, std::complex<double>>::value)
  {
    mpi_type = MPI_C_DOUBLE_COMPLEX;
  }
  else if constexpr (std::is_same<T, std::complex<long double>>::value)
  {
    mpi_type = MPI_C_LONG_DOUBLE_COMPLEX;
  }
  
  assert(mpi_type != MPI_DATATYPE_NULL);
  return mpi_type;
}

