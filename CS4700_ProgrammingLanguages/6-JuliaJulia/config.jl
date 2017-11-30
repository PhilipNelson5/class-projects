struct Config
  pixlesX::Int64
  pixlesY::Int64
  c::Complex64
  midX::Float64
  midY::Float64
  axisLenX::Float64
  axisLenY::Float64
  maxIters::Int

  color1::RGB
  color2::RGB

  outputFile::String

  minX::Float64
  maxX::Float64
  minY::Float64
  maxY::Float64
  pxlSizeX::Float64
  pxlSizeY::Float64
  step_r::Float64
  step_g::Float64
  step_b::Float64
end
