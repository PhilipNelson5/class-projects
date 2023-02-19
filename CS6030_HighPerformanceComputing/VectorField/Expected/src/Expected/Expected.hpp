#pragma once

#include <functional>
#include <type_traits>
#include <assert.h>
#include <iostream>

#ifdef __CUDACC__
#define CUDA_HD __host__ __device__
#else
#define CUDA_HD
#endif

/**
 * @brief Represents a value that is expected to be a T but might have failed
 * 
 * Expected<T> can be operated on in both _has_value and !_has_value states.
 * When it does not have a value, all computations are skipped and the error
 * state is propagated through. To use the underlying value, check 
 * if (expected.has_value()) and handle the error case accordingly. 
 * 
 * @tparam T underlying type held by the Expected value
 */
template <typename T>
class Expected
{
    T _value;
    bool _has_value;
    
public:
    using value_type = T;
    CUDA_HD Expected(): _has_value(false) {}
    CUDA_HD Expected(T value): _value(value), _has_value(true) {}
    CUDA_HD Expected(const Expected & other): _value(other._value), _has_value(other._has_value) {}
    CUDA_HD Expected(const Expected && other): _value(std::move(other._value)), _has_value(std::move(other._has_value)) {}

    /**
     * @brief Move assignment operator
     * 
     * @param other the assigned value
     * @return Expected<T>& 
     */
    CUDA_HD Expected<T>& operator=(const Expected<T> && other)
    {
        _value = std::move(other._value);
        _has_value = std::move(other._has_value);
        return *this;
    }

    /**
     * @brief Copy assignment operator
     * 
     * @param other the assigned value
     * @return Expected<T>& 
     */
    CUDA_HD Expected<T>& operator=(const Expected<T> & other)
    {
        _value = other._value;
        _has_value = other._has_value;
        return *this;
    }

    /**
     * @brief Cast to the underlying value.
     * 
     * If the underlying value is invalid, this will throw or assert(0).
     * Make sure to check if(expected.has_value()) first.
     * 
     * @return T underlying value
     */
    CUDA_HD operator T() const {
        if (std::is_same<Expected<typename T::value_type>, T>::value)
            return _value;
        return value(); }

    /**
     * @brief check if the expected has a value or is invalid
     * 
     * @return bool true if in a valid state
     */
    CUDA_HD bool has_value() const {
        return _has_value;
    }
    
    /**
     * @brief Get the underlying value
     * 
     * If the underlying value is invalid, this will throw or assert(0).
     * Make sure to check if(expected.has_value()) first.
     *
     * @return T the underlying value 
     */
    CUDA_HD T value() const {
        if (_has_value) return _value;
#ifdef __CUDACC__
        assert(0);
        return _value;
#else
        throw std::runtime_error("expected value was invalid");
#endif
    }

    // with newer compiler use:
    // template <typename F, typename U = std::invoke_result_t<F&, T>>
    /**
     * @brief apply the value to a function which takes a T.
     * If the expected value is invalid, the function is not applied
     * and the invalid state is returned.
     * 
     * @tparam F A function 
     * @tparam U The result of F(T)
     * @param f The function to be applied
     * @return Expected<U> the expected result of f(t)
     */
    template <typename F, typename U = typename std::result_of<F(T)>::type>
    CUDA_HD Expected<U> apply(F f) const
    {
        if (_has_value)
            return f(_value);
        return Expected<U>();
    }
    
    /**
     * @brief The bind operator
     * 
     * @tparam F A function 
     * @tparam U The result of F(T)
     * @param f The function to be applied
     * @return Expected<U> the expected result of f(t)
     */
    template <typename F, typename U = typename std::result_of<F(T)>::type>
    CUDA_HD Expected<U> operator>>=(F f) const
    {
        return apply(f);
    }

};

/**
 * @brief definitions of the operator to be used in mixed modes
 * with expected types:
 * Expected<T> op Expected<V>
 * Expected<T> op V
 * V op Expected<T>
 */
#define MixedMode(op)\
template <typename T, typename V>\
CUDA_HD auto operator op (Expected<T> t, Expected<V> v)\
{\
    using U = decltype(std::declval<T>() op std::declval<V>());\
    if (!t.has_value() || !v.has_value() ) return Expected<U>{};\
    return Expected<U>(t.value() op v.value());\
}\
template <typename T, typename V>\
CUDA_HD auto operator op (Expected<T> t, V v)\
{\
    using U = decltype(std::declval<T>() op std::declval<V>());\
    if (!t.has_value()) return Expected<U>{};\
    return Expected<U>(t.value() op v);\
}\
template <typename T, typename V>\
CUDA_HD auto operator op (V v, Expected<T> t)\
{\
    using U = decltype(std::declval<V>() op std::declval<T>());\
    if (!t.has_value()) return Expected<U>{};\
    return Expected<U>(v op t.value());\
}\

MixedMode(+)
MixedMode(-)
MixedMode(*)
MixedMode(/)
MixedMode(%)
MixedMode(>)
MixedMode(>=)
MixedMode(<)
MixedMode(<=)
MixedMode(==)

/**
 * @brief stream insertion operator for an Expected<T>
 * 
 * @tparam T underlying expected type
 * @param o an ostream
 * @param t an expected value
 * @return std::ostream& 
 */
template <typename T>
std::ostream& operator<<(std::ostream& o, Expected<T> t)
{
    if (t.has_value()) o << t.value();
    else o << "ERROR";
    return o;
}