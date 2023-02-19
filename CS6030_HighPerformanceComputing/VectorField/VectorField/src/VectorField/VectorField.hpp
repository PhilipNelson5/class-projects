#pragma once

#ifdef __CUDACC__
#define CUDA_HD __host__ __device__
#else
#define CUDA_HD
#endif

#include <Point/Pnt2.hpp>
#include <Vector/Vec2.hpp>
#include <Misc/lerp.hpp>
#include <Solvers/RungeKutta.hpp>
#include <Solvers/Euler.hpp>
#include <Expected/Expected.hpp>

#include <cstddef>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>

using std::size_t;

namespace {

/**
 * @brief read a 2D block of a file into a 1D array
 * 
 * @tparam T 
 * @param f ifstream to the file to read
 * @param file_width width of the file
 * @param file_height height of the file
 * @param block_width width of the block
 * @param block_height height of the block
 * @param x x coordinate of the block to read
 * @param y y coordinate of the block to read
 * @return T* 1D array of block data
 */
template <typename T>
T* read_block(std::ifstream& f, const int file_width, const int file_height, const int block_width, const int block_height, const int x, const int y)
{
    const int block_size = block_width * block_height;
    const int block_width_byte = block_width * sizeof(T);
    T* block = new T[block_size];
    
    const int first_pos_byte = (y * block_height * file_width + x * block_width) * sizeof(T);
    const int last_pos_byte = first_pos_byte + (file_width * (block_height - 1) + block_width) * sizeof(T);
    for(int g_pos = first_pos_byte, b_pos = 0; g_pos < last_pos_byte; g_pos += file_width * sizeof(T), b_pos += block_width_byte)
    {
        f.seekg(g_pos);
        f.read((char*)block+b_pos, block_width_byte);
    }
    
    return block;
}

} // namespace


/**
 * @brief Represents a vector field and operations on it
 * 
 * @tparam T type of data in field
 */
template <typename T>
class VectorField
{
    size_t _width, _height;
    Math::Vec2<T>* _data;
    bool _own_data;
    
public:
    using value_type = Math::Vec2<T>;
    CUDA_HD VectorField(const size_t w, const size_t h): _width(w), _height(h), _data(new Math::Vec2<T>[_width * _height]) {}
    
    CUDA_HD VectorField(const size_t w, const size_t h, Math::Vec2<T>* data, bool own_data = true): _width(w), _height(h), _data(data), _own_data(own_data) {}
    CUDA_HD ~VectorField() { if (_own_data) delete[] _data; }
    
    /**
     * @brief get vector field width
     * 
     * @return size_t width 
     */
    CUDA_HD size_t width() const { return _width; }
    /**
     * @brief get vector field height
     * 
     * @return size_t height
     */
    CUDA_HD size_t height() const { return _height; }
    /**
     * @brief get vector field size
     * 
     * @return size_t number of elements
     */
    CUDA_HD size_t size() const { return _width * _height; }
    
    /**
     * @brief immutable pointer to vector field data
     * 
     * @return const Math::Vec2<T>* immutable pointer to vector field data
     */
    CUDA_HD const Math::Vec2<T>* data() const { return _data; }

    /**
     * @brief mutable pointer to vector field data
     * 
     * @return Math::Vec2<T>* mutable pointer to vector field data
     */
    CUDA_HD Math::Vec2<T>* data() { return _data; }
    
    /**
     * @brief retrieve a const reference to a value at a grid point p
     * 
     * @param p point
     * @return const Math::Vec2<T>& value at point
     */
    CUDA_HD const Math::Vec2<T>& at(const Math::Pnt2<size_t> p) const
    {
        return _data[p.y * _width + p.x];
    }
    
    /**
     * @brief retrieve a reference to a value at a grid point p
     * 
     * @param p point
     * @return const Math::Vec2<T>& value at point
     */
    CUDA_HD Math::Vec2<T>& at(const Math::Pnt2<size_t> p)
    {
        return _data[p.y * _width + p.x];
    }
    
    /**
     * @brief get a copy of a value at a grid point p
     * 
     * @param p point
     * @return Math::Vec2<T> value at point
     */
    CUDA_HD Math::Vec2<T> get(const Math::Pnt2<size_t> p) const
    {
        return _data[p.y * _width + p.x];
    }
    
    /**
     * @brief perform bilinear interpolation to estimate values at points in between grid point
     * This operation may fail and so it returns an Expected result with will be an invalid state
     * in the case of attempting to access a value off the edge of the vector field
     * 
     * @param p point
     * @return Expected<Math::Vec2<T>> Expected result value
     */
    CUDA_HD Expected<Math::Vec2<T>> operator()(const Math::Pnt2<T> p) const
    {
        const T x = p.x;
        const T y = p.y;

        // 4 adjacent grid point components  
        int x1 = std::floor(p.x);
        int x2 = std::ceil(p.x);
        int y1 = std::floor(p.y);
        int y2 = std::ceil(p.y);

        // bounds check
        if(x1 < 0 || y1 < 0 || x2 < 0 || y2 < 0
            || static_cast<size_t>(x1) >= _width
            || static_cast<size_t>(x2) >= _width
            || static_cast<size_t>(y1) >= _height
            || static_cast<size_t>(y2) >= _height)
        {
            return {};
        }

        // TODO: try profiling with a branch predictor hint
        // https://stackoverflow.com/a/30131034/7480820
        // #define likely(x)    __builtin_expect (!!(x), 1)
        // #define unlikely(x)  __builtin_expect (!!(x), 0)
        // check that x or y values are not equal and adjust accordingly
        if (x1 == x2)
        {
            if (static_cast<size_t>(x2) == _width) x2 += -1;
            else x2 += 1;
        }
        if (y1 == y2)
        {
            if (static_cast<size_t>(y2) == _height) y2 += -1;
            else y2 += 1;
        }
        

        // 4 adjacent grid points
        const auto Q11 = Math::Pnt2<size_t>{static_cast<size_t>(x1), static_cast<size_t>(y1)};
        const auto Q12 = Math::Pnt2<size_t>{static_cast<size_t>(x1), static_cast<size_t>(y2)};
        const auto Q21 = Math::Pnt2<size_t>{static_cast<size_t>(x2), static_cast<size_t>(y1)};
        const auto Q22 = Math::Pnt2<size_t>{static_cast<size_t>(x2), static_cast<size_t>(y2)};
        
        // bilinear interpolate
        return (
            (x2-x)*(y2-y)*get(Q11) +
            (x-x1)*(y2-y)*get(Q21) +
            (x2-x)*(y-y1)*get(Q12) + 
            (x-x1)*(y-y1)*get(Q22)
        ) / T((x2-x1)*(y2-y1));
    }
    
    /**
     * @brief read a vector field from a raw file
     * 
     * @param filename name of the file to read from
     * @param width width of the vector field
     * @param height height of the vector field
     * @return VectorField constructed vector field
     */
    static VectorField read(const std::string& filename, const size_t width, const size_t height)
    {
        const auto size = width * height;
        Math::Vec2<T>* data = new Math::Vec2<T>[size];

        std::ifstream fin(filename, std::ios::binary);
        if (!fin) throw std::runtime_error("Failed to open vector field file");
        fin.read((char *)data, sizeof(Math::Vec2<T>) * size);

        return VectorField(width, height, data);
    }
    
    /**
     * @brief read a block of a vector field from a raw file
     * 
     * @param filename name of the file to read from
     * @param width width of the vector field
     * @param height height of the vector field
     * @param block_width with of the block
     * @param block_height height of the block
     * @param x x coordinate of the block to read
     * @param y y coordinate of the block to read
     * @return VectorField constructed vector field
     */
    static VectorField read_block(const std::string& filename, const size_t width, const size_t height, const int block_width, const int block_height, const int coord_x, const int coord_y)
    {
        std::ifstream fin(filename);
        Math::Vec2<T>* block = ::read_block<Math::Vec2<float>>(fin, width, height, block_width, block_height, coord_x, coord_y);
        return VectorField(block_width, block_height, block);
    }
    
    /**
     * @brief generate a vector field from a function, applying it at each grid point
     * 
     * @tparam F 
     * @param f function
     * @param xMin min x value
     * @param xMax max x value
     * @param yMin min y value
     * @param yMax max y value
     * @param nX number of grid points in x
     * @param nY number of grid points in y
     * @return generated vector field
     */
    template <typename F>
    CUDA_HD static VectorField generate(F f, const T xMin, const T xMax, const T yMin, const T yMax, const size_t nX, const size_t nY)
    {
        VectorField vf(nX, nY);
        for(size_t y = 0; y < nY; ++y)
        {
            for(size_t x = 0; x < nX; ++x)
            {
                vf.at({x,y}) = f({
                    Math::lerp(xMin, xMax, (float)x/nX),
                    Math::lerp(yMin, yMax, (float)y/nY)
                });
            }
        }
        return vf;
    }
    
    std::vector<std::pair<Math::Pnt2<T>, Math::Vec2<T>>> calculateArrowField(const int nX, const int nY) const
    {
        std::vector<std::pair<Math::Pnt2<T>, Math::Vec2<T>>> arrows;
        arrows.reserve(nX*nY*2);
        
        for (float y = 0; y < _height-1; y += (float)_height / nY)
        {
            for (float x = 0; x < _width-1; x += (float)_width / nX)
            {
                const Math::Pnt2<T> p {
                    x + _width / nX / 2.0f,
                    y + _height / nY / 2.0f
                };
                const auto v = (*this)(p);
                const auto uV = v >>= Math::normalize<T>;
                if (uV.has_value())
                    arrows.emplace_back(p, uV.value());
            }
        }
        return arrows;
    }
    CUDA_HD std::vector<std::pair<Math::Pnt2<T>, Math::Vec2<T>>> calculateArrowField() const
    { return calculateArrowField(_width, _height); }

    /**
     * @brief Calculate a streamline from a starting point using Euler method
     * CPU implementation
     * 
     * @param start 
     * @param max_len 
     * @return std::vector<Math::Pnt2<T>> 
     */
    std::vector<Math::Pnt2<T>> calculateStreamLineEuler(const Math::Pnt2<T> start, const int max_len) const
    {
        const T dt = .15;
        std::vector<Math::Pnt2<T>> points {start};
        Expected<Math::Pnt2<T>> curr = start;
        int len = 0;
        while (curr.value().x >= 0 && curr.value().x < _width-5 && curr.value().y >= 0 && curr.value().y < _height-5 && len < max_len)
        {
            curr = Math::euler(*this, curr.value(), dt);
            if (curr.has_value()) points.push_back(curr.value());
            else break;
            ++len;
        }
        return points;
    }
    
    /**
     * @brief Calculate a streamline from a starting point using Runge Kutta method
     * CPU implementation
     * 
     * @param start starting point
     * @param max_len maximum streamline length
     * @return std::vector<Math::Pnt2<T>> 
     */
    std::vector<Math::Pnt2<T>> calculateStreamLineRK(const Math::Pnt2<T> start, const int max_len) const
    {
        using namespace std::placeholders;
        const T dt = .15;
        std::vector<Math::Pnt2<T>> points {start};
        Expected<Math::Pnt2<T>> curr = start;
        int len = 0;
        while (curr.value().x >= 0 && curr.value().x < _width-5 && curr.value().y >= 0 && curr.value().y < _height-5 && len < max_len)
        {
            curr = Math::runge_kutta(*this, curr.value(), dt);
            if (curr.has_value()) points.push_back(curr.value());
            else break;
            ++len;
        }
        return points;
    }

    /**
     * @brief Calculate a streamline from a starting point using Runge Kutta method
     * CUDA implementation
     * 
     * @param start starting point
     * @param points output array of points 
     * @param max_len maximum streamline length
     */
    CUDA_HD void calculateStreamLineRK(Math::Pnt2<T> start, Math::Pnt2<T>* points, const int max_len) const
    {
        const T dt = .15;
        // reserve first element to store final length of streamline
        int len = 1;
        auto curr = Expected<Math::Pnt2<T>>(start);
        points[len] = start;

        while (curr.value().x >= 0 && curr.value().x < _width-5 && curr.value().y >= 0 && curr.value().y < _height-5 && len < max_len)
        {
            curr = Math::runge_kutta(*this, curr.value(), dt);
            if (curr.has_value()) points[++len] = curr.value();
            else break;
        }

        // store the length of the curve as the first "point" in the list
        points[0] = {(T)len, (T)len};
    }
};
