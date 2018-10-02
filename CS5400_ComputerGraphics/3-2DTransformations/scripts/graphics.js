// ------------------------------------------------------------------
//
// This is the graphics object.  It provides a pseudo pixel rendering
// space for use in demonstrating some basic rendering techniques.
//
// ------------------------------------------------------------------
MySample.graphics = (function (pixelsX, pixelsY) {
  'use strict';

  let canvas = document.getElementById('canvas-main');
  let context = canvas.getContext('2d');

  let deltaX = canvas.width / pixelsX;
  let deltaY = canvas.height / pixelsY;
  let curveSegments = 10;     // Initial setting for how many line segments to use in curve rendering
  let du = 1 / curveSegments;
  let BEZ = [[[]]];
  let C = [];
  let U = [[]];

  // function fallingFactorial(n, k) {
  //   let ffact = n;
  //   for (++k; k < n; ++k) {
  //     ffact *= k;
  //   }
  //   return ffact;
  // }

  /**
   * A factorial function
   * 
   * @param {Integer} n - The n in n!
   * @return {Integer} n!
   */
  function factorial(n) {
    let fact = 1;
    for (let i = 1; i <= n; ++i) {
      fact *= i;
    }

    return fact;
  }

  /**
   * Public function that allows the client code to clear the canvas.
   * 
   * @param {Boolean} renderGrid - Turns on drawing the grid
   */
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

  /**
   * Public function that renders a "pixel" on the framebuffer.
   * 
   * @param {Number} x - x coordinate of the pixel
   * @param {Number} y - y coordinate of the pixel
   * @param {String} color - Color of the pixel
   */
  function drawPixel(x, y, color) {
    x = Math.trunc(x);
    y = Math.trunc(y);

    context.fillStyle = color;
    context.fillRect(x * deltaX, y * deltaY, deltaX, deltaY);
  }

  /**
   * Helper function used to draw an X centered at a point.
   * 
   * @param {Number} x - x coordinate of the point
   * @param {Number} y - y coordinate of the point
   * @param {String} ptColor - Color of the point
   */
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
   * 
   * @param {Number} x0 - The starting x coordinate
   * @param {Number} y0 - The starting y coordinate
   * @param {Number} x1 - The ending x coordinate
   * @param {Number} y1 - The ending y coordinate
   * @param {String} color - The line color
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
   * 
   * @param {Number} x0 - The starting x coordinate
   * @param {Number} y0 - The starting y coordinate
   * @param {Number} x1 - The ending x coordinate
   * @param {Number} y1 - The ending y coordinate
   * @param {String} color - The line color
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
   * 
   * @param {Number} x0 - The starting x coordinate
   * @param {Number} y0 - The starting y coordinate
   * @param {Number} x1 - The ending x coordinate
   * @param {Number} y1 - The ending y coordinate
   * @param {String} color - the line color
   */
  function drawLine(x0, y0, x1, y1, color) {
    x0 *= pixelsX;
    y0 *= pixelsY;
    x1 *= pixelsX;
    y1 *= pixelsY;
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

  /**
   * Renders an Hermite curve based on the input parameters.
   * 
   * @param {Object} controls {
   *   @member {Point} p0 - The initial point
   *   @member {Point} p1 - The final point
   *   @member {Point} s0 - The slope at the initial point
   *   @member {Point} s1 - The slope at the final point
   * }
   * @param {Boolean} showPoints  - Boolean flag to show the intersection of line segments
   * @param {Boolean} showLine    - Boolean flag to show the drawn curve
   * @param {Boolean} showControl - Boolean flag to show the control points
   * @param {String}  color       - Color of the line
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

  /**
   * Renders a Cardinal curve based on the input parameters.
   * 
   * @param {Object} controls {
   *   @member {Point} p0 - The initial point
   *   @member {Point} p1 - The final point
   *   @member {Point} s2 - The slope at the initial point
   *   @member {Point} s3 - The slope at the final point
   *   @member {Number} t - The tension parameter
   * }
   * @param {Boolean} showPoints  - Boolean flag to show the intersection of line segments
   * @param {Boolean} showLine    - Boolean flag to show the drawn curve
   * @param {Boolean} showControl - Boolean flag to show the control points
   * @param {String} color        - Color of the line
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

  /**
   * Renders a Bezier curve based on the input parameters.
   * 
   * @param {Object} controls {
   *   @member {Point} p0 - The first control point
   *   @member {Point} p1 - The second control point
   *   @member {Point} s2 - The third control point
   *   @member {Point} s3 - The fourth control point
   * }
   * @param {Boolean} showPoints  - Boolean flag to show the intersection of line segments
   * @param {Boolean} showLine    - Boolean flag to show the drawn curve
   * @param {Boolean} showControl - Boolean flag to show the control points
   * @param {String} color        - Color of the line
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

  /**
   * Renders a Bezier curve based on the input parameters; using the matrix form.
   * This follows the Mathematics for Game Programmers form.
   * 
   * @param {Object} controls{
   *   @member {Point} p0 - The first control point
   *   @member {Point} p1 - The second control point
   *   @member {Point} s2 - The third control point
   *   @member {Point} s3 - The fourth control point
   * }
   * @param {Boolean} showPoints  - Boolean flag to show the intersection of line segments
   * @param {Boolean} showLine    - Boolean flag to show the drawn curve
   * @param {Boolean} showControl - Boolean flag to show the control points
   * @param {String}  color       - Color of the line
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

  /**
   * Entry point for rendering the different types of curves.
   * 
   * @param {Enum}    type        - The type of curve to draw
   * @param {Object}  controls    - The controls for the specified curve
   * @param {Boolean} showPoints  - Show the line segment points
   * @param {Boolean} showLine    - Show the line
   * @param {Boolean} showControl - Show the control points or lines
   * @param {String}  lineColor   - The line color
   */
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

  /**
   * Specifies the number of line segments to use when rendering curves.
   * Any pre-compute optimization can be initiated from this function.
   * 
   * @param {Number} segments - The number of line segments for curves
   */
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
  }

  /**
   * Renders a primitive polygon
   * 
   * @param {Object} primitive {
   *   @member {Point} center - Center of the polygon
   *   @member {Point[]} verts - Array of verticies (must have 2+)
   * }
   * @param {Boolean} connect - If true, draw a line from the last vertex to the first
   * @param {String} color    - The color of the lines
   */
  function drawPrimitive(primitive, connect, color) {
    for (let i = 1; i < primitive.verts.length; ++i) {
      drawLine(primitive.verts[i - 1].x,
        primitive.verts[i - 1].y,
        primitive.verts[i].x,
        primitive.verts[i].y,
        color
      );
    }
    if (connect) {
      drawLine(primitive.verts[0].x,
        primitive.verts[0].y,
        primitive.verts[primitive.verts.length - 1].x,
        primitive.verts[primitive.verts.length - 1].y,
        color
      );
    }
  }

  /**
   * Translate a point by a distance
   * 
   * @param {Point} point - The point to translate
   * @param {Point} distance - A vector representing the distance to translate the point by
   * @return {Object} New point translated by distance
   */
  function translatePoint(point, distance) {
    return { x: point.x + distance.x, y: point.y + distance.y };
  }

  /**
   * translate a primitive by a distance
   * 
   * @param {Object} primitive {
   *   @member {Point} center - Center of the polygon
   *   @member {Point[]} verts - Array of verticies (must have 2+)
   * }
   * @param {Point} distance - A vector representing the distance to translate the point by
   * @return {Object} New primitive translated by distance
   */
  function translatePrimitive(primitive, distance) {
    let newPrim = { verts: [] };

    newPrim.center = translatePoint(primitive.center, distance);

    for (let i = 0; i < primitive.verts.length; ++i) {
      newPrim.verts[i] = translatePoint(primitive.verts[i], distance);
    }

    return newPrim;
  }

  /**
   * Scales a primitive by an ammount in the x and y
   * 
   * @param {Object} primitive {
   *   @member {Point} center - Center of the polygon
   *   @member {Point[]} verts - Array of verticies (must have 2+)
   * }
   * @param {Point} scale - Scale in the x and y directions
   */
  function scalePrimitive(primitive, scale) {
    let newPrim;
    if (primitive.center.x !== 0 && primitive.center.y !== 0) {
      newPrim = translatePrimitive(primitive, { x: -primitive.center.x, y: -primitive.center.y });
      console.log('nonzero center [scale]');
    } else {
      newPrim = JSON.parse(JSON.stringify(primitive));
    }

    for (let i = 0; i < newPrim.verts.length; ++i) {
      newPrim.verts[i].x *= scale.x;
      newPrim.verts[i].y *= scale.y;
    }

    if (primitive.center.x !== 0 && primitive.center.y !== 0) {
      return newPrim;
    } else {
      return translatePrimitive(newPrim, { x: primitive.center.x, y: primitive.center.y });
    }
  }

  /**
   * Rotates a primitive by a number of radians
   * 
   * @param {Object} primitive {
   *   @member {Point} center - Center of the polygon
   *   @member {Point[]} verts - Array of verticies (must have 2+)
   * }
   * @param {Number} angle - The angle in Radians to rotate the primitive
   */
  function rotatePrimitive(primitive, angle) {
    let newPrim;
    if (primitive.center.x !== 0 && primitive.center.y !== 0) {
      newPrim = translatePrimitive(primitive, { x: -primitive.center.x, y: -primitive.center.y });
      console.log('nonzero center [rotate]');
    } else {
      newPrim = JSON.parse(JSON.stringify(primitive));
    }

    const sina = Math.sin(angle);
    const cosa = Math.cos(angle);
    for (let i = 0; i < newPrim.verts.length; ++i) {
      let newx = newPrim.verts[i].x * cosa - newPrim.verts[i].y * sina;
      let newy = newPrim.verts[i].x * sina + newPrim.verts[i].y * cosa;
      newPrim.verts[i].x = newx;
      newPrim.verts[i].y = newy;
    }

    if (primitive.center.x !== 0 && primitive.center.y !== 0) {
      return newPrim;
    } else {
      return translatePrimitive(newPrim, { x: primitive.center.x, y: primitive.center.y });
    }
  }

  /**
   * Translate a curve by a distance
   * 
   * @param {Enum} type - The type of curve to draw (Cardinal, Bezier)
   * @param {Object} controls - The controls for the specified curve
   * @param {Point} distance - A vector representing the distance to translate the curve by
   */
  function translateCurve(type, controls, distance) {
    let newControls = JSON.parse(JSON.stringify(controls));

    switch (type) {
      case api.Curve.Hermite:
        newControls.p0 = translatePoint(controls.p0, distance);
        newControls.p1 = translatePoint(controls.p1, distance);
        newControls.s0 = translatePoint(controls.s0, distance);
        newControls.s1 = translatePoint(controls.s1, distance);
        break;
      case api.Curve.Cardinal:
      case api.Curve.Bezier:
        newControls.p0 = translatePoint(controls.p0, distance);
        newControls.p1 = translatePoint(controls.p1, distance);
        newControls.p2 = translatePoint(controls.p2, distance);
        newControls.p3 = translatePoint(controls.p3, distance);
        break;
    }

    return newControls;
  }

  /**
   * Scales a curve relative to its center.
   * 
   * @param {Enum} type - The type of curve to draw (Cardinal, Bezier)
   * @param {Object} controls - The controls for the specified curve
   * @param {Point} scale - Scale in the x and y directions
   */
  function scaleCurve(type, controls, scale) {
    let center = {};
    let newCurve = {};
    switch (type) {
      case api.Curve.Hermite:
        center.x = (controls.p0.x + controls.p1.x) / 2;
        center.y = (controls.p0.y + controls.p1.y) / 2;

        newCurve = translateCurve(api.Curve.Hermite, controls, { x: -center.x, y: -center.y });

        newCurve.p0.x *= scale.x;
        newCurve.p0.y *= scale.y;
        newCurve.p1.x *= scale.x;
        newCurve.p1.y *= scale.y;
        newCurve.s0.x *= scale.x;
        newCurve.s0.y *= scale.y;
        newCurve.s1.x *= scale.x;
        newCurve.s1.y *= scale.y;

        return translateCurve(api.Curve.Hermite, newCurve, { x: center.x, y: center.y });

      case api.Curve.Cardinal:

        center.x = (controls.p1.x + controls.p2.x) / 2;
        center.y = (controls.p1.y + controls.p2.y) / 2;

        newCurve = translateCurve(api.Curve.Cardinal, controls, { x: -center.x, y: -center.y });

        newCurve.p0.x *= scale.x;
        newCurve.p0.y *= scale.y;
        newCurve.p1.x *= scale.x;
        newCurve.p1.y *= scale.y;
        newCurve.p2.x *= scale.x;
        newCurve.p2.y *= scale.y;
        newCurve.p3.x *= scale.x;
        newCurve.p3.y *= scale.y;

        return translateCurve(api.Curve.Cardinal, newCurve, { x: center.x, y: center.y });

      case api.Curve.Bezier:
        center.x = (controls.p0.x + controls.p3.x) / 2;
        center.y = (controls.p0.y + controls.p3.y) / 2;

        newCurve = translateCurve(api.Curve.Bezier, controls, { x: -center.x, y: -center.y });

        newCurve.p0.x *= scale.x;
        newCurve.p0.y *= scale.y;
        newCurve.p1.x *= scale.x;
        newCurve.p1.y *= scale.y;
        newCurve.p2.x *= scale.x;
        newCurve.p2.y *= scale.y;
        newCurve.p3.x *= scale.x;
        newCurve.p3.y *= scale.y;

        return translateCurve(api.Curve.Bezier, newCurve, { x: center.x, y: center.y });
    }
  }

  /**
   * Rotates a curve about its center.
   * 
   * @param {Enum} type - The type of curve to draw (Cardinal, Bezier)
   * @param {Object} controls - The controls for the specified curve
   * @param {Number} angle - The angle in Radians to rotate the primitive
   */
  function rotateCurve(type, controls, angle) {
    let center = {};
    let newCurve = {};
    let newx, newy;
    const sina = Math.sin(angle);
    const cosa = Math.cos(angle);

    switch (type) {
      case api.Curve.Hermite:
        center.x = (controls.p0.x + controls.p1.x) / 2;
        center.y = (controls.p0.y + controls.p1.y) / 2;

        newCurve = translateCurve(api.Curve.Hermite, controls, { x: -center.x, y: -center.y });

        newx = newCurve.p0.x * cosa - newCurve.p0.y * sina;
        newy = newCurve.p0.x * sina + newCurve.p0.y * cosa;
        newCurve.p0.x = newx;
        newCurve.p0.y = newy;

        newx = newCurve.p1.x * cosa - newCurve.p1.y * sina;
        newy = newCurve.p1.x * sina + newCurve.p1.y * cosa;
        newCurve.p1.x = newx;
        newCurve.p1.y = newy;

        newx = newCurve.s0.x * cosa - newCurve.s0.y * sina;
        newy = newCurve.s0.x * sina + newCurve.s0.y * cosa;
        newCurve.s0.x = newx;
        newCurve.s0.y = newy;

        newx = newCurve.s1.x * cosa - newCurve.s1.y * sina;
        newy = newCurve.s1.x * sina + newCurve.s1.y * cosa;
        newCurve.s1.x = newx;
        newCurve.s1.y = newy;

        return translateCurve(api.Curve.Hermite, newCurve, { x: center.x, y: center.y });

      case api.Curve.Cardinal:

        center.x = (controls.p1.x + controls.p2.x) / 2;
        center.y = (controls.p1.y + controls.p2.y) / 2;

        newCurve = translateCurve(api.Curve.Cardinal, controls, { x: -center.x, y: -center.y });

        newx = newCurve.p0.x * cosa - newCurve.p0.y * sina;
        newy = newCurve.p0.x * sina + newCurve.p0.y * cosa;
        newCurve.p0.x = newx;
        newCurve.p0.y = newy;

        newx = newCurve.p1.x * cosa - newCurve.p1.y * sina;
        newy = newCurve.p1.x * sina + newCurve.p1.y * cosa;
        newCurve.p1.x = newx;
        newCurve.p1.y = newy;

        newx = newCurve.p2.x * cosa - newCurve.p2.y * sina;
        newy = newCurve.p2.x * sina + newCurve.p2.y * cosa;
        newCurve.p2.x = newx;
        newCurve.p2.y = newy;

        newx = newCurve.p3.x * cosa - newCurve.p3.y * sina;
        newy = newCurve.p3.x * sina + newCurve.p3.y * cosa;
        newCurve.p3.x = newx;
        newCurve.p3.y = newy;

        return translateCurve(api.Curve.Cardinal, newCurve, { x: center.x, y: center.y });

      case api.Curve.Bezier:
        center.x = (controls.p0.x + controls.p3.x) / 2;
        center.y = (controls.p0.y + controls.p3.y) / 2;

        newCurve = translateCurve(api.Curve.Bezier, controls, { x: -center.x, y: -center.y });

        newx = newCurve.p0.x * cosa - newCurve.p0.y * sina;
        newy = newCurve.p0.x * sina + newCurve.p0.y * cosa;
        newCurve.p0.x = newx;
        newCurve.p0.y = newy;

        newx = newCurve.p1.x * cosa - newCurve.p1.y * sina;
        newy = newCurve.p1.x * sina + newCurve.p1.y * cosa;
        newCurve.p1.x = newx;
        newCurve.p1.y = newy;

        newx = newCurve.p2.x * cosa - newCurve.p2.y * sina;
        newy = newCurve.p2.x * sina + newCurve.p2.y * cosa;
        newCurve.p2.x = newx;
        newCurve.p2.y = newy;

        newx = newCurve.p3.x * cosa - newCurve.p3.y * sina;
        newy = newCurve.p3.x * sina + newCurve.p3.y * cosa;
        newCurve.p3.x = newx;
        newCurve.p3.y = newy;

        return translateCurve(api.Curve.Bezier, newCurve, { x: center.x, y: center.y });
    }
  }

  //
  // This is what we'll export as the rendering API
  const api = {
    clear,
    drawPixel,
    drawLine,
    drawCurve,
    drawPrimitive,
    translatePrimitive,
    scalePrimitive,
    rotatePrimitive,
    translateCurve,
    scaleCurve,
    rotateCurve,
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
  /**
   * Enum for curve types
   * @enum {Number}
   */
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
}(1000, 1000));
