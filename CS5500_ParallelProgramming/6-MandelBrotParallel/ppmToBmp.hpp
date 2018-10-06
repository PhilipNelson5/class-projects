#ifndef PPM_TO_BMP_HPP
#define PPM_TO_BMP_HPP

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/**
 * @author Bryan Hansen
 * @author Erik Falor
 * @author Philip Nelson
 * @date 10/9/2017
 *
 * @history
 * 10/16/17  Fixed padding in generated BMP for non-word aligned sizes
 * 10/31/17  Fixed BMP files store rows of pixels from bottom-to-top
 * 09/26/18  Now converts a vector of iteration data
 *             instead of reading from a file. A color_scheme function is
 *             passed in to turn the raw iterations into [r,g,b] colors
 */

namespace stayOffMyLawn
{

  /**
   * Writes the standard BMP header to the provided file
   * @param bmpFile File stream to which the header will be written
   * @param width   The width of the PPM file in pixels
   * @param height  The height of the PPM file in pixels
   */
  void writeBmpHeader(std::ofstream& bmpFile, int width, int height)
  {
    // BMP header (14 bytes)
    // A two character signature to indicate the file is a bitmap file
    // (typically “BM”). A 32bit unsigned little-endian integer representing the
    // size of the file itself. A pair of 16bit unsigned little-endian integers
    // reserved for application specific uses. A 32bit unsigned little-endian
    // integer representing the offset to where the pixel array starts in the
    // file.

    const uint32_t HEADER_SIZE_BYTES = 54;
    const uint32_t BYTES_PER_PIXEL = 3;
    uint32_t padBytes = 0;
    if ((width * BYTES_PER_PIXEL) % sizeof(uint32_t) != 0)
    {
      padBytes =
        sizeof(uint32_t) - ((width * BYTES_PER_PIXEL) % sizeof(uint32_t));
    }

    const uint32_t paddedWidthBytes = (width * BYTES_PER_PIXEL) + padBytes;
    const uint32_t totalSize = HEADER_SIZE_BYTES + (height * paddedWidthBytes);
    const char sigOne = 'B';
    const char sigTwo = 'M';
    const uint16_t reserved = 0;
    const uint32_t pixelOffset = HEADER_SIZE_BYTES;

    /* clang-format off */
    bmpFile.write(&sigOne, sizeof(uint8_t));
    bmpFile.write(&sigTwo, sizeof(uint8_t));
    bmpFile.write(
        reinterpret_cast<const char*>(&totalSize), sizeof(uint32_t));
    bmpFile.write(
        reinterpret_cast<const char*>(&reserved), sizeof(uint16_t));
    bmpFile.write(
        reinterpret_cast<const char*>(&reserved), sizeof(uint16_t));
    bmpFile.write(
      reinterpret_cast<const char*>(&pixelOffset), sizeof(uint32_t));
    /* clang-format on */
  }

  /**
   * Writes the BMP image header to the provided file
   * @param bmpFile File stream to which image header will be written
   * @param width   The width of the PPM file in pixels
   * @param height  The height of the PPM file in pixels
   */
  void writeBmpImageHeader(std::ofstream& bmpFile, int width, int height)
  {
    // Image header (40 bytes)
    // biSize           4 Header Size - Must be at least 40
    // biWidth          4 Image width in pixels
    // biHeight         4 Image height in pixels
    // biPlanes         2 Must be 1
    // biBitCount       2 Bits per pixel - 1, 4, 8, 16, 24, or 32
    // biCompression    4 Compression type (0 = uncompressed)
    // biSizeImage      4 Image Size - may be zero for uncompressed images
    // biXPelsPerMeter  4 Preferred resolution in pixels per meter
    // biYPelsPerMeter  4 Preferred resolution in pixels per meter
    // biClrUsed        4 Number Color Map entries that are actually used
    // biClrImportant   4 Number of significant colors

    const uint32_t headerSizeBytes = 40;
    const uint16_t planes = 1;
    const uint16_t bitsPerPixel = 24;
    const uint32_t compression = 0;
    const uint32_t imageSize = 0;
    const uint32_t preferredResolution = 0;
    const uint32_t colorMapEntries = 0;
    const uint32_t significantColors = 0;

    /* clang-format off */
    bmpFile.write(
      reinterpret_cast<const char*>(&headerSizeBytes), sizeof(uint32_t));
    bmpFile.write(
        reinterpret_cast<const char*>(&width), sizeof(uint32_t));
    bmpFile.write(
        reinterpret_cast<const char*>(&height), sizeof(uint32_t));
    bmpFile.write(
        reinterpret_cast<const char*>(&planes), sizeof(uint16_t));
    bmpFile.write(
      reinterpret_cast<const char*>(&bitsPerPixel), sizeof(uint16_t));
    bmpFile.write(
      reinterpret_cast<const char*>(&compression), sizeof(uint32_t));
    bmpFile.write(
        reinterpret_cast<const char*>(&imageSize), sizeof(uint32_t));
    bmpFile.write(
      reinterpret_cast<const char*>(&preferredResolution), sizeof(uint32_t));
    bmpFile.write(
      reinterpret_cast<const char*>(&preferredResolution), sizeof(uint32_t));
    bmpFile.write(
      reinterpret_cast<const char*>(&colorMapEntries), sizeof(uint32_t));
    bmpFile.write(
      reinterpret_cast<const char*>(&significantColors), sizeof(uint32_t));
    /* clang-format on */
  }

  /**
   * Writes all pixels from the PPM file (ascii) into the BMP file (binary)
   * @param ppmBuffer File stream from which ascii pixels will be read
   * @param bmpFile   File stream to which binary pixels will be written
   * @param width     The width of the PPM file in pixels
   * @param height    The height of the PPM file in pixels
   */
  template <typename F>
  bool writePixels(std::vector<int>& ppmBuffer,
                   std::ofstream& bmpFile,
                   int width,
                   int height,
                   F color_scheme)
  {
    // Write pixels to BMP file (24 bits per pixel), padding each row to be
    // 4-byte divisible The BMP image is stored bottom-to-top, so we have to
    // wrote the rows backwards relative to the PPM image

    char** map = new char*[height];
    const uint32_t BYTES_PER_PIXEL = 3;
    uint32_t padBytes = 0;
    if ((width * BYTES_PER_PIXEL) % sizeof(uint32_t) != 0)
      padBytes =
        sizeof(uint32_t) - ((width * BYTES_PER_PIXEL) % sizeof(uint32_t));

    // Copy the top of the PPM into the bottom of the bitmap
    auto ppmIt = begin(ppmBuffer);
    for (int row = height - 1; row >= 0; --row)
    {
      map[row] = new char[width * BYTES_PER_PIXEL + padBytes];
      auto col = 0u;
      for (; col < width * BYTES_PER_PIXEL; col += 3)
      {
        auto [red, green, blue] = color_scheme(*ppmIt++);

        map[row][col + 0] = (char)blue;
        map[row][col + 1] = (char)green;
        map[row][col + 2] = (char)red;
      }

      // Pad if needed
      const uint8_t padData = 0x00;
      for (auto pad = 0u; pad < padBytes; ++pad)
        map[row][col++] = (char)padData;
    }

    // Write the bitmap out to the bmpFile
    for (int row = 0; row < height; ++row)
    {
      bmpFile.write(map[row], width * BYTES_PER_PIXEL + padBytes);
      delete[] map[row];
    }

    delete[] map;

    return true;
  }
} // namespace stayOffMyLawn

/**
 * Program converts an vector of iteration data into a 24-bit BMP file
 * @param ppmBuffer    buffer of pixel information
 * @param ppmWidth     width of the image in pixels
 * @param ppmHeight    height of the image in pixels
 * @param color_scheme std::tuple<int, int, int>(int) function
 * @param bmpFileName  name of the bmp image to write
 * @return true on success, false on failure
 */
template <typename F>
bool ppmToBmp(std::vector<int> ppmBuffer,
              uint32_t ppmWidth,
              uint32_t ppmHeight,
              F color_scheme,
              std::string bmpFileName)
{
  std::cout << "Writing " << bmpFileName << "...\n";

  // Read out PPM header to get size information
  std::ofstream bmpFile(bmpFileName.c_str(), std::ios::binary);

  if (ppmBuffer.size() == ppmWidth * ppmHeight)
  {
    stayOffMyLawn::writeBmpHeader(bmpFile, ppmWidth, ppmHeight);
    stayOffMyLawn::writeBmpImageHeader(bmpFile, ppmWidth, ppmHeight);

    if (stayOffMyLawn::writePixels(
          ppmBuffer, bmpFile, ppmWidth, ppmHeight, color_scheme))
    {
      std::cout << "Success!" << std::endl;
      return true;
    }
  }

  bmpFile.close();

  return false;
}

#endif
