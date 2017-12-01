include("rgb.jl") # includeing files is very similar to C++
include("hsv.jl")
include("config.jl")

function readConfig(fileName)
  f = open(fileName)
  pixles = split(readline(f)," ")
  cval = split(readline(f)," ")
  mid = split(readline(f)," ")
  axisLen = split(readline(f)," ")
  c1 = split(readline(f), " ")
  c2 = split(readline(f), " ")

  pixlesX = parse(Int, pixles[1])
  pixlesY = parse(Int, pixles[2])
  cvalR = parse(Float64, cval[1])
  cvalI = parse(Float64, cval[2])
  c = cvalR + (cvalI)im
  midX = parse(Float64, mid[1])
  midY = parse(Float64, mid[2])
  axisLenX = parse(Float64, axisLen[1])
  axisLenY = parse(Float64, axisLen[2])
  maxIters = parse(Int, split(readline(f), " ")[1])

  color1 = RGB(parse(Int, c1[1]), parse(Int, c1[2]), parse(Int, c1[3]))
  color2 = RGB(parse(Int, c2[1]), parse(Int, c2[2]), parse(Int, c2[3]))

  outputFile = split(readline(f), " ")[1]

  minX = midX - axisLenX / 2
  maxX = midX + axisLenX / 2
  minY = midY - axisLenY / 2
  maxY = midY + axisLenY / 2
  pxlSizeX = axisLenX / pixlesX
  pxlSizeY = axisLenY / pixlesY
  step_r = (color2.r - color1.r) / maxIters
  step_g = (color2.g - color1.g) / maxIters
  step_b = (color2.b - color1.b) / maxIters

  Config(pixlesX, pixlesY, c, midX, midY, axisLenX, axisLenY, maxIters, color1, color2, outputFile, minX, maxX, minY, maxY, pxlSizeX, pxlSizeY, step_r, step_g, step_b)

end

function hsvTOrgb(hsv::HSV)
  H = hsv.h
  S = hsv.s
  V = hsv.v

  (H == 360.0)?(H = 0.0):(H /= 60.0)
  fract = H - floor(H)

  P = floor(V*(1.0 - S)*255)
  Q = floor(V*(1.0 - S*fract)*255)
  T = floor(V*(1.0 - S*(1.0 - fract))*255)

  if (0 <= H && H < 1)
    return RGB(V, T, P)
  elseif (1 <= H && H < 2)
    return RGB(Q,V,P)
  elseif (2 <= H && H < 3)
    return RGB(P,V,T)
  elseif (3 <= H && H < 4)
    return RGB(P,Q,V)
  elseif (4 <= H && H < 5)
    return RGB(T,P,V)
  elseif (5 <= H && H < 6)
    return RGB(V,P,Q)
  else
    return RGB(0,0,0)
  end
end

function gradient1(cfg, iters)

    r = floor(cfg.color1.r + (cfg.step_r*iters))
    b = floor(cfg.color1.b + (cfg.step_b*iters))
    g = floor(cfg.color1.g + (cfg.step_g*iters))

    return RGB(r,g,b)
end

function getColor(cfg, iters, style)
  if style == 0
    return hsvTOrgb(HSV((iters % 256), 1, (1 * (iters < cfg.maxIters))));
  elseif style == 1
    return gradient1(cfg, iters)
  end
end

function getIters(cfg, row, col)
  zr = cfg.minX + (row - 1) * cfg.pxlSizeX
  zi = cfg.minY + (col - 1) * cfg.pxlSizeY
  z = zr + (zi)im # native support for complex numbers

  for n = 1:cfg.maxIters
    if abs2(z) > 4
      return n-1
    end
    z = z^2 + cfg.c # ^ operator does powers
  end
  return cfg.maxIters

end

function main()
  cfg = readConfig("config.txt")

  println(cfg.c)

  fout = open(cfg.outputFile,"w")
  write(fout, "P3\n$(cfg.pixlesX) $(cfg.pixlesY)\n255\n") # string interpolation is very convenient

  for col = 1:cfg.pixlesY
    for row = 1:cfg.pixlesX

    iters = getIters(cfg, row, col)
    color = getColor(cfg, iters, 0)
    write(fout, "$(color.r) $(color.g) $(color.b)\t")

    end
  end

  close(fout)
end

main()

# Julia does not do implicit type casting
# Having a REPL is amazing for learning a language

