#pragma once

#include "Pixel.hpp"
#include <Point/Pnt2.hpp>

#include <cstddef>
#include <fstream>
#include <string>

namespace Graphics {

/**
 * @brief an image stored as a one dimensional array of pixels with height and width
 * 
 */
class Image
{
    int _width, _height;
    Pixel* _data;
    
public:
    Image(const int w, const int h): _width(w), _height(h), _data(new Pixel[_width*_height]) {}
    Image(const int w, const int h, Pixel* data): _width(w), _height(h), _data(data) {}
    ~Image() { delete[] _data; }

    /**
     * @brief get image width
     * 
     * @return int image width
     */
    int width() const { return _width; }
    
    /**
     * @brief get image height
     * 
     * @return int image height 
     */
    int height() const { return _height; }
    
    /**
     * @brief get a reference to the pixel at given point (x,y) or (col, row)
     * 
     * @param p point in the image of desired pixel
     * @return const Pixel& the pixel
     */
    inline const Pixel& at(const Math::Pnt2<int> p) const
    {
        return _data[p.y * _width + p.x];
    }
    
    /**
     * @brief get a const reference to the pixel at given point (x,y) or (col, row)
     * 
     * @param p point in the image of desired pixel
     * @return Pixel& the pixel
     */
    inline Pixel& at(const Math::Pnt2<int> p)
    {
        return _data[p.y * _width + p.x];
    }
    
    /**
     * @brief fill the image with a single color
     * 
     * @param color color
     */
    inline void fill_image(const Pixel color)
    {
        for (int i = 0; i < _height * _width; ++i)
        {
            _data[i] = color;
        }
    }
    
    /**
     * @brief write the image to a file in ascii ppm format P3
     * 
     * @param file_name name of file, should use file extension .ppm
     */
    inline void write_ascii_ppm(const std::string& file_name) const 
    {
        std::ofstream ppm(file_name);
        ppm << "P3\n" << _width << " " << _height << " 255\n";
        for (int row = 0; row < _height; ++row)
        {
            for (int col = 0; col < _width; ++col)
            {
                const auto p = at({col, row});
                ppm << (int)p.r << " " << (int)p.g << " " << (int)p.b << "  ";
            }
            ppm << "\n";
        }
        ppm.close();
    }

    /**
     * @brief write the image to a file in binary ppm format P6
     * 
     * @param file_name name of file, should use file extension .ppm
     */
    inline void write_ppm(const std::string& file_name) const 
    {
        std::ofstream ppm(file_name);
        ppm << "P6 " << _width << " " << _height << " 255\n";
        ppm.write((char*)_data, _width * _height * sizeof(Pixel));
        ppm.close();
    }
};
        
} // namespace Graphics
