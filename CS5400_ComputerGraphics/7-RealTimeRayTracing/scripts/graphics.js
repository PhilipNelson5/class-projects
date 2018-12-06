// ------------------------------------------------------------------
//
// This is the graphics object.  It provides a pseudo pixel rendering
// space for use in demonstrating some basic rendering techniques.
//
// ------------------------------------------------------------------
MySample.graphics = (function (pixelsX, pixelsY) {
  'use strict';

  let canvas = document.getElementById('canvas-curve');
  let context = canvas.getContext('2d');

  let deltaX = canvas.width / pixelsX;
  let deltaY = canvas.height / pixelsY;
  let curveSegments = 10;
  let du = 1 / curveSegments;
  let BEZ = [[[]]];
  let C = [];
  let U = [[]];

  function fallingFactorial(n, k) {
    let ffact = n;
    for (++k; k < n; ++k) {
      ffact *= k;
    }
    return ffact;
  }

  function factorial(n) {
    let fact = 1;
    for (let i = 1; i <= n; ++i) {
      fact *= i;
    }

    return fact;
  }

  //------------------------------------------------------------------
  //
  // Public function that allows the client code to clear the canvas.
  //
  //------------------------------------------------------------------
  function clear(renderGrid) {
    context.save();
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.restore();

    //
    // Draw a very light background to show the "pixels" for the framebuffer.
    if (renderGrid) {
      context.save();
      context.lineWidth = .1;
      context.strokeStyle = 'rgb(0, 0, 0)';
      context.beginPath();
      for (let y = 0; y <= pixelsY; y++) {
        context.moveTo(1, y * deltaY);
        context.lineTo(canvas.width, y * deltaY);
      }
      for (let x = 0; x <= pixelsX; x++) {
        context.moveTo(x * deltaX, 1);
        context.lineTo(x * deltaX, canvas.width);
      }
      context.stroke();
      context.restore();
    }
  }

  //------------------------------------------------------------------
  //
  // Public function that renders a "pixel" on the framebuffer.
  //
  //------------------------------------------------------------------
  function drawPixel(x, y, color) {
    x = Math.trunc(x);
    y = Math.trunc(y);

    context.fillStyle = color;
    context.fillRect(x * deltaX, y * deltaY, deltaX, deltaY);
  }

  //------------------------------------------------------------------
  //
  // Helper function used to draw an X centered at a point.
  //
  //------------------------------------------------------------------
  function drawPoint(x, y, ptColor) {
    drawPixel(x - 2, y - 2, ptColor);
    drawPixel(x - 1, y - 1, ptColor);
    drawPixel(x + 2, y - 2, ptColor);
    drawPixel(x + 1, y - 1, ptColor);
    drawPixel(x, y, ptColor);
    drawPixel(x + 1, y + 1, ptColor);
    drawPixel(x + 2, y + 2, ptColor);
    drawPixel(x - 1, y + 1, ptColor);
    drawPixel(x - 2, y + 2, ptColor);
  }

  //------------------------------------------------------------------
  //
  // Bresenham line drawing algorithm.
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
  function drawLineX(x0, y0, x1, y1, color) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let yi = 1;
    if (dy < 0) {
      yi = -1;
      dy = -dy;
    }
    let eps = 2 * dy - dx;
    let mid = (x0 + x1) / 2;

    for (let x = x0, y = y0, xe = x1, ye = y1; x <= mid; ++x, --xe) {
      drawPixel(x, y, color);
      drawPixel(xe, ye, color);
      if (eps >= 0) {
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
  function drawLineY(x0, y0, x1, y1, color) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let xi = 1;
    if (dx < 0) {
      xi = -1;
      dx = -dx;
    }
    let eps = 2 * dx - dy;
    let mid = (y0 + y1) / 2;

    for (let y = y0, x = x0, ye = y1, xe = x1; y <= mid; ++y, --ye) {
      drawPixel(x, y, color);
      drawPixel(xe, ye, color);
      if (eps > 0) {
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
  function drawLine(x0, y0, x1, y1, color) {
    if (x0 == x1 && y0 == y1) {
      drawPixel(x0, y0);
      return;
    }
    if (Math.abs(y1 - y0) < Math.abs(x1 - x0)) {
      if (x0 > x1) {
        drawLineX(x1, y1, x0, y0, color);
      }
      else {
        drawLineX(x0, y0, x1, y1, color);
      }
    }
    else {
      if (y0 > y1) {
        drawLineY(x1, y1, x0, y0, color);
      }
      else {
        drawLineY(x0, y0, x1, y1, color);
      }
    }
  }


  //------------------------------------------------------------------
  //
  // Renders an Hermite curve based on the input parameters.
  //
  //------------------------------------------------------------------
  /**
   * @param controls {
   *   p0: {x: , y: } The initial point
   *   p1: {x: , y: } The final point
   *   s0: {x: , y: } The slope at the initial point
   *   s1: {x: , y: } The slope at the final point
   * }
   * @param showPoints  Boolean flag to show the intersection of line segments
   * @param showLine    Boolean flag to show the drawn curve
   * @param showControl Boolean flag to show the control points
   * @param color       Color of the line
   */
  function drawCurveHermite(controls, showPoints, showLine, showControl, color) {
    let px, py;
    let px_o = controls.p0.x;
    let py_o = controls.p0.y;

    if (showPoints) drawPoint(px_o, py_o, 'rgb(255, 0, 0)');

    if (showControl) {
      drawLine(
        controls.p0.x,
        controls.p0.y,
        controls.p0.x + controls.s0.x,
        controls.p0.y + controls.s0.y,
        color
      );

      drawLine(
        controls.p1.x,
        controls.p1.y,
        controls.p1.x + controls.s1.x,
        controls.p1.y + controls.s1.y,
        color
      );
    }

    let u = du;
    for (let i = 0; i <= curveSegments; ++i, u += du) {

      const a = (2 * U[i][3] - 3 * U[i][2] + 1);
      const b = (- 2 * U[i][3] + 3 * U[i][2]);
      const c = (U[i][3] - 2 * U[i][2] + U[i][1]);
      const d = (U[i][3] - U[i][1]);

      px = controls.p0.x * a
        + controls.p1.x * b
        + controls.s0.x * c
        + controls.s1.x * d;

      py = controls.p0.y * a
        + controls.p1.y * b
        + controls.s0.y * c
        + controls.s1.y * d;

      if (showPoints) drawPoint(px, py, 'rgb(255, 0, 0)');

      if (showLine) drawLine(px_o, py_o, px, py, color);

      px_o = px;
      py_o = py;
    }
  }

  //------------------------------------------------------------------
  //
  // Renders a Cardinal curve based on the input parameters.
  //
  //------------------------------------------------------------------
  /**
   * @param controls {
   *   p0: {x: , y: } The initial point
   *   p1: {x: , y: } The final point
   *   s2: {x: , y: } The slope at the initial point
   *   s3: {x: , y: } The slope at the final point
   *   t :            The tension parameter
   * }
   * @param showPoints  Boolean flag to show the intersection of line segments
   * @param showLine    Boolean flag to show the drawn curve
   * @param showControl Boolean flag to show the control points
   * @param color       Color of the line
   */
  function drawCurveCardinal(controls, showPoints, showLine, showControl, color) {
    let px, py;
    let px0 = controls.p1.x;
    let py0 = controls.p1.y;
    let s = (1 - controls.t) / 2;

    if (showPoints) drawPoint(px0, py0, 'rgb(255, 0, 0)');

    if (showControl) {
      drawLine(
        controls.p0.x,
        controls.p0.y,
        controls.p1.x,
        controls.p1.y,
        color
      );

      drawLine(
        controls.p2.x,
        controls.p2.y,
        controls.p3.x,
        controls.p3.y,
        color
      );
    }

    let u = du;
    for (let i = 0; i <= curveSegments; ++i, u += du) {

      const a = (- s * U[i][3] + 2 * s * U[i][2] - s * U[i][1]);
      const b = ((2 - s) * U[i][3] + (s - 3) * U[i][2] + 1);
      const c = ((s - 2) * U[i][3] + (3 - 2 * s) * U[i][2] + s * U[i][1]);
      const d = (s * U[i][3] - s * U[i][2]);

      px = controls.p0.x * a
        + controls.p1.x * b
        + controls.p2.x * c
        + controls.p3.x * d;

      py = controls.p0.y * a
        + controls.p1.y * b
        + controls.p2.y * c
        + controls.p3.y * d;

      if (showPoints) drawPoint(px, py, 'rgb(255, 0, 0)');

      if (showLine) drawLine(px0, py0, px, py, color);

      px0 = px;
      py0 = py;
    }
  }



  //------------------------------------------------------------------
  //
  // Renders a Bezier curve based on the input parameters.
  //
  //------------------------------------------------------------------
  /**
   * @param controls {
   *   p0: {x: , y: } The first control point
   *   p1: {x: , y: } The second control point
   *   s2: {x: , y: } The third control point
   *   s3: {x: , y: } The fourth control point
   * }
   * @param showPoints  Boolean flag to show the intersection of line segments
   * @param showLine    Boolean flag to show the drawn curve
   * @param showControl Boolean flag to show the control points
   * @param color       Color of the line
   */
  function drawCurveBezier(controls, showPoints, showLine, showControl, color) {
    let px, py;
    let px0 = controls.p0.x;
    let py0 = controls.p0.y;

    if (showPoints) drawPoint(px0, py0, 'rgb(255, 0, 0)');

    if (showControl) {
      drawPoint(controls.p0.x, controls.p0.y, 'rgb(255, 0, 0)');
      drawPoint(controls.p1.x, controls.p1.y, 'rgb(255, 0, 0)');
      drawPoint(controls.p2.x, controls.p2.y, 'rgb(255, 0, 0)');
      drawPoint(controls.p3.x, controls.p3.y, 'rgb(255, 0, 0)');
      // drawLine(controls.p0.x, controls.p0.y, controls.p1.x, controls.p1.y, 'rgb(255, 0, 0)');
      // drawLine(controls.p1.x, controls.p1.y, controls.p2.x, controls.p2.y, 'rgb(255, 0, 0)');
      // drawLine(controls.p2.x, controls.p2.y, controls.p3.x, controls.p3.y, 'rgb(255, 0, 0)');
    }
    let u = 0;
    for (let i = 0; i <= curveSegments; ++i, u += du) {
      px = controls.p0.x * BEZ[0][u]
        + controls.p1.x * BEZ[1][u]
        + controls.p2.x * BEZ[2][u]
        + controls.p3.x * BEZ[3][u];

      py = controls.p0.y * BEZ[0][u]
        + controls.p1.y * BEZ[1][u]
        + controls.p2.y * BEZ[2][u]
        + controls.p3.y * BEZ[3][u];

      if (showPoints) drawPoint(px, py, 'rgb(255, 0, 0)');

      if (showLine) drawLine(px0, py0, px, py, color);

      px0 = px;
      py0 = py;
    }
  }

  //------------------------------------------------------------------
  //
  // Renders a Bezier curve based on the input parameters; using the matrix form.
  // This follows the Mathematics for Game Programmers form.
  //
  //------------------------------------------------------------------
  /**
   * @param controls {
   *   p0: {x: , y: } The first control point
   *   p1: {x: , y: } The second control point
   *   s2: {x: , y: } The third control point
   *   s3: {x: , y: } The fourth control point
   * }
   * @param showPoints  Boolean flag to show the intersection of line segments
   * @param showLine    Boolean flag to show the drawn curve
   * @param showControl Boolean flag to show the control points
   * @param color       Color of the line
   */
  function drawCurveBezierMatrix(controls, showPoints, showLine, showControl, color) {
    let px, py;
    let px0 = controls.p3.x;
    let py0 = controls.p3.y;

    if (showPoints) drawPoint(px0, py0, 'rgb(255, 0, 0)');

    if (showControl) {
      drawPoint(controls.p0.x, controls.p0.y, 'rgb(255, 0, 0)');
      drawPoint(controls.p1.x, controls.p1.y, 'rgb(255, 0, 0)');
      drawPoint(controls.p2.x, controls.p2.y, 'rgb(255, 0, 0)');
      drawPoint(controls.p3.x, controls.p3.y, 'rgb(255, 0, 0)');
      drawLine(controls.p0.x, controls.p0.y, controls.p1.x, controls.p1.y, 'rgb(255, 0, 0)');
      drawLine(controls.p1.x, controls.p1.y, controls.p2.x, controls.p2.y, 'rgb(255, 0, 0)');
      drawLine(controls.p2.x, controls.p2.y, controls.p3.x, controls.p3.y, 'rgb(255, 0, 0)');
    }

    for (let i = 1; i <= curveSegments; ++i) {

      const a = U[i][3];
      const b = - 3 * U[i][3] + 3 * U[i][2];
      const c = 3 * U[i][3] - 6 * U[i][2] + 3 * U[i][1];
      const d = - U[i][3] + 3 * U[i][2] - 3 * U[i][1] + 1;

      px = controls.p0.x * a
        + controls.p1.x * b
        + controls.p2.x * c
        + controls.p3.x * d;

      py = controls.p0.y * a
        + controls.p1.y * b
        + controls.p2.y * c
        + controls.p3.y * d;

      if (showPoints) drawPoint(px, py, 'rgb(255, 0, 0)');

      if (showLine) drawLine(px0, py0, px, py, color);

      px0 = px;
      py0 = py;
    }
  }

  //------------------------------------------------------------------
  //
  // Entry point for rendering the different types of curves.
  //
  //------------------------------------------------------------------
  function drawCurve(type, controls, showPoints, showLine, showControl, lineColor) {
    switch (type) {
      case api.Curve.Hermite:
        drawCurveHermite(controls, showPoints, showLine, showControl, lineColor);
        break;
      case api.Curve.Cardinal:
        drawCurveCardinal(controls, showPoints, showLine, showControl, lineColor);
        break;
      case api.Curve.Bezier:
        drawCurveBezier(controls, showPoints, showLine, showControl, lineColor);
        break;
      case api.Curve.BezierMatrix:
        drawCurveBezierMatrix(controls, showPoints, showLine, showControl, lineColor);
        break;
    }
  }

  //------------------------------------------------------------------
  //
  // Specifies the number of line segments to use when rendering curves.
  // Any pre-compute optimization can be initiated from this function.
  //
  //------------------------------------------------------------------
  function setCurveSegments(segments) {
    curveSegments = segments;
    du = 1 / curveSegments;
    for (let k = 0; k <= 3; ++k) {
      let u = 0;
      BEZ[k] = [];
      // C[k] = fallingFactorial(3, k) / factorial(3 - k);
      C[k] = factorial(3) / factorial(k) / factorial(3 - k);
      for (let i = 0; i <= curveSegments; ++i, u += du) {
        BEZ[k][u] = C[k] * Math.pow(u, k) * Math.pow((1 - u), (3 - k));
        U[i] = [1];
        for (let p = 1; p <= 3; ++p) {
          U[i][p] = U[i][p - 1] * u;
        }
      }
    }
    // console.log(U);
    // console.log(C);
    // console.log(BEZ);
  }

  //
  // This is what we'll export as the rendering API
  const api = {
    clear: clear,
    drawPixel: drawPixel,
    drawLine: drawLine,
    drawCurve: drawCurve
  };

  Object.defineProperty(api, 'pixelsX', {
    value: pixelsX,
    writable: false
  });
  Object.defineProperty(api, 'pixelsY', {
    value: pixelsY,
    writable: false
  });
  Object.defineProperty(api, 'curveSegments', {
    get: () => { return curveSegments; },
    set: value => setCurveSegments(value)
  });
  Object.defineProperty(api, 'Curve', {
    value: Object.freeze({
      Hermite: 0,
      Cardinal: 1,
      Bezier: 2,
      BezierMatrix: 3
    }),
    writable: false
  });

  return api;
}(900, 900));
