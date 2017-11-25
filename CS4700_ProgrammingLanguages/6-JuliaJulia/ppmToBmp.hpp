#include <cstdint>

/**
 * @author Bryan Hansen
 * @author Erik Falor
 * @date 10/9/2017
 * 
 * @history
 * 10/16/17  Fixed padding in generated BMP for non-word aligned sizes
 * 10/31/17  Fixed BMP files store rows of pixels from bottom-to-top
 */

namespace stayOffMyLawn {

    /**
     * Writes the standard BMP header to the provided file
     * @param bmpFile File stream to which the header will be written
     * @param width The width of the PPM file in pixels
     * @param height The height of the PPM file in pixels
     */
    void writeBmpHeader(std::ofstream& bmpFile, int width, int height)
    {
        // BMP header (14 bytes)
        // A two character signature to indicate the file is a bitmap file (typically “BM”).
        // A 32bit unsigned little-endian integer representing the size of the file itself.
        // A pair of 16bit unsigned little-endian integers reserved for application specific uses.
        // A 32bit unsigned little-endian integer representing the offset to where the pixel array starts in the file.

        const uint32_t HEADER_SIZE_BYTES = 54;
        const uint32_t BYTES_PER_PIXEL = 3;
        uint32_t padBytes = 0;
        if ((width * BYTES_PER_PIXEL) % sizeof(uint32_t) != 0)
        {
            padBytes = sizeof(uint32_t) - ((width * BYTES_PER_PIXEL) % sizeof(uint32_t));
        }

        uint32_t paddedWidthBytes = (width * BYTES_PER_PIXEL) + padBytes;
        uint32_t totalSize = HEADER_SIZE_BYTES + (height * paddedWidthBytes);
        const char sigOne = 'B';
        const char sigTwo = 'M';
        uint16_t reserved = 0;
        uint32_t pixelOffset = HEADER_SIZE_BYTES;

        bmpFile.write(&sigOne, sizeof(uint8_t));
        bmpFile.write(&sigTwo, sizeof(uint8_t));
        bmpFile.write(reinterpret_cast<const char*>(&totalSize), sizeof(uint32_t));
        bmpFile.write(reinterpret_cast<const char*>(&reserved), sizeof(uint16_t));
        bmpFile.write(reinterpret_cast<const char*>(&reserved), sizeof(uint16_t));
        bmpFile.write(reinterpret_cast<const char*>(&pixelOffset), sizeof(uint32_t));
    }

    /**
     * Writes the BMP image header to the provided file
     * @param bmpFile File stream to which image header will be written
     * @param width The width of the PPM file in pixels
     * @param height The height of the PPM file in pixels
     */
    void writeBmpImageHeader(std::ofstream& bmpFile, int width, int height)
    {
        // Image header (40 bytes)
        // biSize	        4	Header Size - Must be at least 40
        // biWidth	        4	Image width in pixels
        // biHeight	        4	Image height in pixels
        // biPlanes	        2	Must be 1
        // biBitCount	    2	Bits per pixel - 1, 4, 8, 16, 24, or 32
        // biCompression	4	Compression type (0 = uncompressed)
        // biSizeImage	    4	Image Size - may be zero for uncompressed images
        // biXPelsPerMeter	4	Preferred resolution in pixels per meter
        // biYPelsPerMeter	4	Preferred resolution in pixels per meter
        // biClrUsed	    4	Number Color Map entries that are actually used
        // biClrImportant	4	Number of significant colors

        uint32_t headerSizeBytes = 40;
        uint16_t planes = 1;
        uint16_t bitsPerPixel = 24;
        uint32_t compression = 0;
        uint32_t imageSize = 0;
        uint32_t preferredResolution = 0;
        uint32_t colorMapEntries = 0;
        uint32_t significantColors = 0;

        bmpFile.write(reinterpret_cast<const char*>(&headerSizeBytes), sizeof(uint32_t));
        bmpFile.write(reinterpret_cast<const char*>(&width), sizeof(uint32_t));
        bmpFile.write(reinterpret_cast<const char*>(&height), sizeof(uint32_t));
        bmpFile.write(reinterpret_cast<const char*>(&planes), sizeof(uint16_t));
        bmpFile.write(reinterpret_cast<const char*>(&bitsPerPixel), sizeof(uint16_t));
        bmpFile.write(reinterpret_cast<const char*>(&compression), sizeof(uint32_t));
        bmpFile.write(reinterpret_cast<const char*>(&imageSize), sizeof(uint32_t));
        bmpFile.write(reinterpret_cast<const char*>(&preferredResolution), sizeof(uint32_t));
        bmpFile.write(reinterpret_cast<const char*>(&preferredResolution), sizeof(uint32_t));
        bmpFile.write(reinterpret_cast<const char*>(&colorMapEntries), sizeof(uint32_t));
        bmpFile.write(reinterpret_cast<const char*>(&significantColors), sizeof(uint32_t));
    }

    /**
     * Writes all pixels from the PPM file (ascii) into the BMP file (binary)
     * @param ppmFile File stream from which ascii pixels will be read
     * @param bmpFile File stream to which binary pixels will be written
     * @param width The width of the PPM file in pixels
     * @param height The height of the PPM file in pixels
     */
    bool writePixels(std::ifstream& ppmFile, std::ofstream& bmpFile, int width, int height)
    {
        // Write pixels to BMP file (24 bits per pixel), padding each row to be 4-byte divisible
        // The BMP image is stored bottom-to-top, so we have to wrote the rows backwards relative
        // to the PPM image

        char** map = new char*[height];
        const uint32_t BYTES_PER_PIXEL = 3;
        uint32_t padBytes = 0;
        if ((width * BYTES_PER_PIXEL) % sizeof(uint32_t) != 0)
            padBytes = sizeof(uint32_t) - ((width * BYTES_PER_PIXEL) % sizeof(uint32_t));

        // Copy the top of the PPM into the bottom of the bitmap
        for (int row = height - 1; row >= 0; row--)
        {
            map[row] = new char[width * BYTES_PER_PIXEL + padBytes];
            int col = 0;
            for (; col < width * BYTES_PER_PIXEL; col += 3)
            {
                uint32_t red;
                uint32_t green;
                uint32_t blue;

                ppmFile >> red >> green >> blue;

                if (!ppmFile)
                {
                    std::cout << "Error, not enough pixels in PPM file!" << std::endl;
                    return false;
                }

                map[row][col + 0] = (char)blue;
                map[row][col + 1] = (char)green;
                map[row][col + 2] = (char)red;
            }

            // Pad if needed
            const uint8_t padData = 0x00;
            for (int pad = 0; pad < padBytes; pad++)
                map[row][col++] = (char)padData;
        }

        // Write the bitmap out to the bmpFile
        for (int row = 0; row < height; ++row) {
            bmpFile.write(map[row], width * BYTES_PER_PIXEL + padBytes);
            delete[] map[row];
        }

        delete[] map;

        return true;
    }
}

/**
 * Program converts a PPM file into a 24-bit BMP file
 * @return 0 on success, -1 otherwise
 */
int ppmToBmp(std::string ppmFileName) {
    int success = -1;

    // Make a filename ending in '.bmp'
    std::string bmpFileName(ppmFileName);
    std::string::iterator i = bmpFileName.end();
    *(--i) = 'p';
    *(--i) = 'm';
    *(--i) = 'b';
    *(--i) = '.';

    std::cout << "Writing " << bmpFileName << "...\n";

    // Read out PPM header to get size information
    std::string ppmFileType;
    uint32_t ppmWidth;
    uint32_t ppmHeight;
    uint32_t ppmMaxVal;
    
    std::ifstream ppmFile(ppmFileName.c_str());
    std::ofstream bmpFile(bmpFileName.c_str(), std::ios::binary);
    
    ppmFile >> ppmFileType >> ppmWidth >> ppmHeight >> ppmMaxVal;
    
    // Validate PPM header
    if (ppmFile && (ppmFileType == "P3") && (ppmMaxVal == 255))
    {
        stayOffMyLawn::writeBmpHeader(bmpFile, ppmWidth, ppmHeight);
        stayOffMyLawn::writeBmpImageHeader(bmpFile, ppmWidth, ppmHeight);
        
        if (stayOffMyLawn::writePixels(ppmFile, bmpFile, ppmWidth, ppmHeight))
        {
            std::cout << "Success!" << std::endl;
            success = 0;
        }
    }
    
    ppmFile.close();
    bmpFile.close();

    return success;
}

