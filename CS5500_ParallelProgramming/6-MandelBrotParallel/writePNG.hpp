#ifndef WRITE_PNG_HPP
#define WRITE_PNG_HPP

#include <png.h>
#include <string>

bool save_png_libpng(const std::string filename, uint8_t* pixels, int w, int h)
{
  png_structp png =
    png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (!png)
  {
    return false;
  }

  png_infop info = png_create_info_struct(png);
  if (!info)
  {
    png_destroy_write_struct(&png, &info);
    return false;
  }

  FILE* fp = fopen(filename.c_str(), "wb");
  if (!fp)
  {
    png_destroy_write_struct(&png, &info);
    return false;
  }

  png_init_io(png, fp);
  png_set_IHDR(png,
               info,
               w,
               h,
               8 /* depth */,
               PNG_COLOR_TYPE_RGB,
               PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_BASE,
               PNG_FILTER_TYPE_BASE);
  png_colorp palette =
    (png_colorp)png_malloc(png, PNG_MAX_PALETTE_LENGTH * sizeof(png_color));
  if (!palette)
  {
    fclose(fp);
    png_destroy_write_struct(&png, &info);
    return false;
  }
  png_set_PLTE(png, info, palette, PNG_MAX_PALETTE_LENGTH);
  png_write_info(png, info);
  png_set_packing(png);

  png_bytepp rows = (png_bytepp)png_malloc(png, h * sizeof(png_bytep));
  for (int i = 0; i < h; ++i)
  {
    rows[i] = (png_bytep)(pixels + (h - i) * w * 3);
  }

  png_write_image(png, rows);
  png_write_end(png, info);
  png_free(png, palette);
  png_destroy_write_struct(&png, &info);

  fclose(fp);
  delete[] rows;
  return true;
}

#endif
