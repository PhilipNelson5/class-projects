include("color.jl")
include("config.jl")

function readConfig(fileName)
  f = open(fileName)
  pixles = split(readline(f)," ")
  mid = split(readline(f)," ")
  axisLen = split(readline(f)," ")
  c1 = split(readline(f), " ")
  c2 = split(readline(f), " ")

  pixlesX = parse(Int, pixles[1])
  pixlesY = parse(Int, pixles[2])
  midX = parse(Float64, mid[1])
  midY = parse(Float64, mid[2])
  axisLenX = parse(Float64, axisLen[1])
  axisLenY = parse(Float64, axisLen[2])
  maxIters = parse(Int, readline(f))

  color1 = Color(parse(Int, c1[1]), parse(Int, c1[2]), parse(Int, c1[3]))
  color2 = Color(parse(Int, c2[1]), parse(Int, c2[2]), parse(Int, c2[3]))

  outputFile = readline(f)

  minX = midX - axisLenX / 2
  maxX = midX + axisLenX / 2
  minY = midY - axisLenY / 2
  maxY = midY + axisLenY / 2
  pxlSizeX = axisLenX / pixlesX
  pxlSizeY = axisLenY / pixlesY
  step_r = (color2.r - color1.r) / maxIters
  step_g = (color2.g - color1.g) / maxIters
  step_b = (color2.b - color1.b) / maxIters

  Config(pixlesX, pixlesY, midX, midY, axisLenX, axisLenY, maxIters, color1, color2, outputFile, minX, maxX, minY, maxY, pxlSizeX, pxlSizeY, step_r, step_g, step_b)

end

function getIters(cfg)
  //calculate the initial real and imaginary part of z, based on the pixel location and zoom and position values
    newRe = 1.5 * (x - w / 2) / (0.5 * zoom * w) + moveX;
    newIm = (y - h / 2) / (0.5 * zoom * h) + moveY;
    int i;
    //start the iteration process
    for(i = 0; i < maxIterations; i++)
    {
      //remember value of previous iteration
      oldRe = newRe;
      oldIm = newIm;
      //the actual iteration, the real and imaginary part are calculated
      newRe = oldRe * oldRe - oldIm * oldIm + cRe;
      newIm = 2 * oldRe * oldIm + cIm;
      //if the point is outside the circle with radius 2: stop
      if((newRe * newRe + newIm * newIm) > 4) break;
    }
    //use color model conversion to get rainbow palette, make brightness black if maxIterations reached
    color = HSVtoRGB(ColorHSV(i % 256, 255, 255 * (i < maxIterations)));
    //draw the pixel
    pset(x, y, color);

int countIterations(MandelbrotConfig cfg, int i, int j){
    double xtemp = 0.0;
    double x = 0.0;
    double y = 0.0;
    int iteration = 0;
    double x0 = cfg.minX + j * cfg.pixelSize;
    double y0 = cfg.maxY - i * cfg.pixelSize;
    
    while (((x*x + y*y) < 4) && (iteration < cfg.maxIterations)){
              xtemp = x*x - y*y + x0;
              y = 2*x*y + y0;
              x = xtemp;
              iteration = iteration + 1;
    }
    return iteration;
}

  //each iteration, it calculates: new = old*old + c, where c is a constant and old starts at current pixel
  double cRe, cIm;           //real and imaginary part of the constant c, determinate shape of the Julia Set
  double newRe, newIm, oldRe, oldIm;   //real and imaginary parts of new and old

  //pick some values for the constant c, this determines the shape of the Julia Set
  cRe = -0.7;
  cIm = 0.27015;

  //loop through every pixel
  for(int y = 0; y < h; y++)
  for(int x = 0; x < w; x++)
  {
    //calculate the initial real and imaginary part of z, based on the pixel location and zoom and position values
    newRe = 1.5 * (x - w / 2) / (0.5 * zoom * w) + moveX;
    newIm = (y - h / 2) / (0.5 * zoom * h) + moveY;
    //i will represent the number of iterations
    int i;
    //start the iteration process
    for(i = 0; i < maxIterations; i++)
    {
      //remember value of previous iteration
      oldRe = newRe;
      oldIm = newIm;
      //the actual iteration, the real and imaginary part are calculated
      newRe = oldRe * oldRe - oldIm * oldIm + cRe;
      newIm = 2 * oldRe * oldIm + cIm;
      //if the point is outside the circle with radius 2: stop
      if((newRe * newRe + newIm * newIm) > 4) break;
    }
    //use color model conversion to get rainbow palette, make brightness black if maxIterations reached
    color = HSVtoRGB(ColorHSV(i % 256, 255, 255 * (i < maxIterations)));
    //draw the pixel
    pset(x, y, color);
  }
  //make the Julia Set visible and wait to exit
  redraw();
  sleep();
  return 0;
