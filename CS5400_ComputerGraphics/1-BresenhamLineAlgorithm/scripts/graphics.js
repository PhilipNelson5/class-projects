// ------------------------------------------------------------------
//
// This is the graphics object.  It provides a pseudo pixel rendering
// space for use in demonstrating some basic rendering techniques.
//
// ------------------------------------------------------------------
MySample.graphics = (function(pixelsX, pixelsY)
{
  'use strict';

  let canvas = document.getElementById('canvas-main');
  let context = canvas.getContext('2d');

  let deltaX = canvas.width / pixelsX;
  let deltaY = canvas.height / pixelsY;

  //------------------------------------------------------------------
  //
  // Public function that allows the client code to clear the canvas.
  //
  //------------------------------------------------------------------

  /**
   * Draw a very light background to show the "pixels" for the framebuffer.
   */
  function drawGrid()
  {
    context.save();
    context.lineWidth = .1;
    context.strokeStyle = 'rgb(0, 0, 0)';
    context.beginPath();

    for (let y = 0; y <= pixelsY; y++)
    {
      context.moveTo(1, y * deltaY);
      context.lineTo(canvas.width, y * deltaY);
    }

    for (let x = 0; x <= pixelsX; x++)
    {
      context.moveTo(x * deltaX, 1);
      context.lineTo(x * deltaX, canvas.width);
    }

    context.stroke();
    context.restore();
  }

  /**
   * Clear the canvas
   */
  function clear()
  {
    context.save();
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.restore();

    //drawGrid();
  }

  //------------------------------------------------------------------
  //
  // Public function that renders a "pixel" on the framebuffer.
  //
  //------------------------------------------------------------------

  /**
   * draw one pixel
   * @param x the x coordinate
   * @param y the y coordinate
   * @param color the pixel color
   */
  function drawPixel(x, y, color)
  {
    x = Math.trunc(x);
    y = Math.trunc(y);

    context.fillStyle = color;
    context.fillRect(x * deltaX, y * deltaY, deltaX, deltaY);
  }

  //------------------------------------------------------------------
  //
  // Bresenham line drawing implementation.
  //
  //------------------------------------------------------------------

  /**
   * Draw a line one X pixel at a time
   * @param x0 the starting x coordinate
   * @param y0 the starting y coordinate
   * @param x1 the ending x coordinate
   * @param y1 the ending y coordinate
   * @param color the line color
   */
  function drawLineX(x0, y0, x1, y1, color)
  {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let yi = 1;
    if (dy < 0)
    {
      yi = -1;
      dy = -dy;
    }
    let eps = 2 * dy - dx;
    let mid = (x0 + x1) / 2;

    for (let x = x0, y = y0, xe = x1, ye = y1; x <= mid; ++x, --xe)
    {
      drawPixel(x, y, color);
      drawPixel(xe, ye, color);
      if (eps >= 0)
      {
        y += yi;
        ye -= yi;
        eps -= 2 * dx;
      }
      eps += 2 * dy;
    }
  }

  /**
   * Draw a line one Y pixel at a time
   * @param x0 the starting x coordinate
   * @param y0 the starting y coordinate
   * @param x1 the ending x coordinate
   * @param y1 the ending y coordinate
   * @param color the line color
   */
  function drawLineY(x0, y0, x1, y1, color)
  {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let xi = 1;
    if (dx < 0)
    {
      xi = -1;
      dx = -dx;
    }
    let eps = 2 * dx - dy;
    let mid = (y0 + y1) / 2;

    for (let y = y0, x = x0, ye = y1, xe = x1; y <= mid; ++y, --ye)
    {
      drawPixel(x, y, color);
      drawPixel(xe, ye, color);
      if (eps > 0)
      {
        x += xi;
        xe -= xi;
        eps -= 2 * dy;
      }
      eps += 2 * dx;
    }
  }

  /**
   * Draw a line using the Bresenham Line Algorithm
   * @param x0 the starting x coordinate
   * @param y0 the starting y coordinate
   * @param x1 the ending x coordinate
   * @param y1 the ending y coordinate
   * @param color the line color
   */
  function drawLine(x0, y0, x1, y1, color)
  {
    if (x0 == x1 && y0 == y1)
    {
      drawPixel(x0, y0);
      return;
    }
    if (Math.abs(y1 - y0) < Math.abs(x1 - x0))
    {
      if (x0 > x1)
      {
        drawLineX(x1, y1, x0, y0, color);
      }
      else
      {
        drawLineX(x0, y0, x1, y1, color);
      }
    }
    else
    {
      if (y0 > y1)
      {
        drawLineY(x1, y1, x0, y0, color);
      }
      else
      {
        drawLineY(x0, y0, x1, y1, color);
      }
    }
  }

  /**
   * Calculate the X coordinate for an Epicycloid
   * @param R the diameter of the inner circle
   * @param r the diameter of the outer circle
   * @param th the angle theta in radians
   * @return the x coordinate
   */
  function cal_x(R, r, th)
  {
    return (R + r) * Math.cos(th) - r * Math.cos((R / r + 1) * th);
  }

  /**
   * Calculate the Y coordinate for an Epicycloid
   * @param R the diameter of the inner circle
   * @param r the diameter of the outer circle
   * @param th the angle theta in radians
   * @return the y coordinate
   */
  function cal_y(R, r, th)
  {
    return (R + r) * Math.sin(th) - r * Math.sin((R / r + 1) * th);
  }

  /**
   * Draw an Epicycloid
   * @param args {
   *   R: Radius of the inner circle
   *   r: radius of the outer circle
   *   d_th: delta theta
   *   th_max: maximum theta
   *   scale: the scale of the Epicycloid
   *   rotation: rotation of the Epicycloid
   *   color: color of the Epicycloid
   * }
   */
  function drawEpicycloid(args)
  {
    context.save();
    context.translate(args.center.x, args.center.y);
    context.rotate(args.rotation);
    let x0 = (cal_x(args.R, args.r, 0)*args.scale);
    let y0 = (cal_y(args.R, args.r, 0)*args.scale);
    for (let th = args.d_th; th < args.th_max; th+=args.d_th)
    {
      let x1 = (cal_x(args.R, args.r, th)*args.scale);
      let y1 = (cal_y(args.R, args.r, th)*args.scale);
      drawLine(x0, y0, x1, y1, args.color);
      x0 = x1;
      y0 = y1;
    }
    context.restore();
  }

  return {
    clear,
    drawPixel,
    drawLine,
    drawEpicycloid,
  };
}(1000, 1000));
