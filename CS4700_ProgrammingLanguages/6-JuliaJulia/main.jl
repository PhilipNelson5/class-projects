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
