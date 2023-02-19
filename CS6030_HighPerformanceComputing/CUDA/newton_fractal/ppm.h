#pragma once
#include <fstream>
#include <iostream>

void write_ppm(const std::string file_name, uint8_t* image, const int width, const int height)
{
  std::ofstream fout(file_name);
  fout << "P3\n" << width << " " << height <<" 255\n";
  for(int i = 0; i < height; ++i)
  {
    for(int j = 0; j < width; ++j)
    {
      uint8_t* p= &image[(i*width+j)*4];
      fout << (int)p[0] << " " << (int)p[1] << " " << (int)p[2] << " " ;
    }
    fout << "\n";
  }
  fout.close();
}
