#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "ppmToBmp.hpp" // provides the function int ppmToBmp(std::string ppmFileName);

using namespace std;

struct Color {
    int r;
    int g;
    int b;
};

struct MandelbrotConfig {
    //
    // These values come from the configuration file
    //
    int pixels;            // dimensions of square image, in pixels

    double midX, midY;     // center of image in imaginary coords
    double axisLen;        // width and height of image in imaginary units

    int maxIterations;     // max number of iterations to test for escape

    Color colorOne;        // color of pixels outside of Mandelbrot set
    Color colorTwo;        // color of pixels within Mandelbrot set

    string outputPPMfile;  // name of PPM file to create


    //
    // These values can be calculated from the above, and will
    // make your algorithm more convenient and efficient
    //
    double minX;
    double minY;                          // X coords in imaginary units of left & right sides of image
    double maxX;
    double maxY;                          // Y coords in imaginary units of top & bottom sides of image
    double pixelSize;                     // how many imaginary units of length/height each pixel takes
    double step_r;                        // how far the red channel changes between iterations
    double step_g;                        // how far the green channel changes between iterations
    double step_b;                        // how far the blue channel changes between iterations
    
};



// Read fractal config file and return a MandelbrotConfig struct
MandelbrotConfig readConfig(string fileName){
    MandelbrotConfig cfg;
        
    //Open File and read info from it
    ifstream fin(fileName);
    if(!fin) {
        cout << "File Not Found!" << endl;
        return cfg;
    }
    
    //read in given values
    fin >> cfg.pixels >> cfg.midX >> cfg.midY >> cfg.axisLen >> cfg.maxIterations >> cfg.colorOne.r >> cfg.colorOne.g >> cfg.colorOne.b 
    >> cfg.colorTwo.r >> cfg.colorTwo.g >> cfg.colorTwo.b >> cfg.outputPPMfile;
    
    //calculate values
    cfg.minX = cfg.midX - (cfg.axisLen/2);
    cfg.minY = cfg.midY - (cfg.axisLen/2);  
    cfg.maxX = cfg.midX + (cfg.axisLen/2);
    cfg.maxY = cfg.midY + (cfg.axisLen/2); 
    cfg.pixelSize = cfg.axisLen/cfg.pixels;
    cfg.step_r = static_cast <double>((cfg.colorTwo.r - cfg.colorOne.r))/cfg.maxIterations; 
    cfg.step_g = static_cast <double>((cfg.colorTwo.g - cfg.colorOne.g))/cfg.maxIterations;
    cfg.step_b = static_cast <double>((cfg.colorTwo.b - cfg.colorOne.b))/cfg.maxIterations;
    
    
    return cfg;
}

// Helper function: given a coordinate corresponding to an imaginary number,
// run the Escape Time algorithm and count how many iterations this number takes
int countIterations(MandelbrotConfig cfg, int i, int j){
    double xtemp = 0.0;
    double x = 0.0;
    double y = 0.0;
    int iteration = 0;
    // Scale current pixel coordinates to lie inside the complex Mandelbrot X/Y scale
    double x0 = cfg.minX + j * cfg.pixelSize;
    double y0 = cfg.maxY - i * cfg.pixelSize;
    
    //find out number of iterations
    while (((x*x + y*y) < 4) && (iteration < cfg.maxIterations)){
              xtemp = x*x - y*y + x0;
              y = 2*x*y + y0;
              x = xtemp;
              iteration = iteration + 1;
    }
    return iteration;
}

// Helper function: decide what color corresponds to a given number of iterations
Color getPixelColor(MandelbrotConfig cfg, int iterations){
    Color pixelColor;
    
    pixelColor.r = cfg.colorOne.r + (cfg.step_r*iterations);
    pixelColor.b = cfg.colorOne.b + (cfg.step_b*iterations);
    pixelColor.g = cfg.colorOne.g + (cfg.step_g*iterations);
    
    return pixelColor;
}

// Create the PPM file, including header, and build the image pixel by pixel
// Loop over the grid of pixels and call the above helper functions as needed
bool drawMandelbrot(MandelbrotConfig cfg){
    double x0;
    double y0;
    double xtemp;
    int y= 0;
    int iteration;
    Color pixelColor;
    
    //Open PPM file 
    ofstream fout (cfg.outputPPMfile);
    
    //Create Header
    fout << "P3" << endl << cfg.pixels << " " << cfg.pixels << endl << "255" << endl;
    
    //For each row in image, do:
    for (int i=0; i< cfg.pixels; i++){
        
        //For each col in row, do:
        for (int j=0; j<cfg.pixels; j++){
           
           iteration = countIterations(cfg, i, j);
    
           pixelColor = getPixelColor(cfg, iteration);
           
           //add pixel to file
           fout << setw(4) << left << static_cast<int> (pixelColor.r) << setw(4) << left << static_cast<int> (pixelColor.g) 
                << setw(6)<< left << static_cast<int> (pixelColor.b);
        }
    
    }
    fout.close();
    return true;
}

int main(int argc, char* argv[]) {
    string filename;

    //Read in config file location from use
    cout << "Please enter your file name:";                                                    
    cin >> filename;                                                                           

    // Create a configuration struct from the file
    MandelbrotConfig cfg = readConfig(filename);
    
    // Compute and write specified mandelbrot image to PPM file
    if (drawMandelbrot(cfg)) {
        // use the provided function to create a BMP image
        ppmToBmp(cfg.outputPPMfile);
    }
    else {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}