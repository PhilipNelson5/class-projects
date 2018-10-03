
MySample.main = (function (graphics) {
  'use strict';

  /**
   * Make a regular polygon centered around (0, 0) with verticies 
   * distance one from the origin
   * 
   * @param {Integer} n The number of verticies of the regular polygon
   */
  function makeRegularPrimitive(n) {
    let verts = [];
    let dth = 2 * Math.PI / n;

    for (let i = 0; i < n; ++i) {
      verts.push({ x: .1 * Math.cos(i * dth), y: .1 * Math.sin(i * dth) });
    }

    return {
      center: { x: 0, y: 0 },
      verts: verts
    };
  }

  /**
   * The representation of a point in 2D space
   * @typedef {Object} Point
   * @property {Number} x - x coordinate of the point
   * @property {Number} y - y coordinate of the point
   */

  /**
   * Make a primitive with the specified dimensions
   * @constructor
   * 
   * @param {Point} center The center of the primitive
   * @param {Point[]} verts List of verticies (must have 2+)
   * @return The new primitive
   */
  function makePrimitive(center, verts) {
    let primitive = {
      center: { x: center.x, y: center.y },
      verts: []
    };

    for (let i = 0; i < verts.length; ++i) {
      primitive.verts[i] = { x: verts[i].x, y: verts[i].y };
    }

    return primitive;
  }


  let cHermite = {
    p0: {
      x: 0.1,
      y: 0.125
    },
    p1: {
      x: 0.9,
      y: 0.125
    },
    s0: {
      x: 0.1,
      y: -0.75
    },
    s1: {
      x: -0.1,
      y: 0.15
    },
    t: 0,
    update: dt => {
      dt /= 1000;
      cHermite.t += dt * 2;
      cHermite.s0.x = (1 / 4 * Math.cos(cHermite.t / 2));
      cHermite.s0.y = (1 / 4 * Math.sin(cHermite.t));
      cHermite.s1.x = (1 / 4 * Math.cos(cHermite.t / 2));
      cHermite.s1.y = (1 / 4 * Math.sin(cHermite.t / 5));
    }
  };

  let cCardinal = {
    p0: {
      x: 0.05,
      y: 0.25,
      x0: 0.05,
      y0: 0.25
    },
    p1: {
      x: 0.1,
      y: 0.375
    },
    p2: {
      x: 0.9,
      y: 0.375
    },
    p3: {
      x: 0.95,
      y: 0.5,
      x0: 0.95,
      y0: 0.5
    },
    t: 1,
    th: 0,
    dir: 1,
    update: dt => {
      dt /= 1000;
      cCardinal.t += 5 * dt * cCardinal.dir;
      if (cCardinal.t > 5 || cCardinal.t < -10) {
        cCardinal.dir *= -1;
        cCardinal.t += 5 * dt * cCardinal.dir;
      }
      cCardinal.th += dt * 2;
      cCardinal.th %= 360;
      cCardinal.p0.x = (cCardinal.p0.x0 + 1 / 4 * Math.cos(cCardinal.th / 2));
      cCardinal.p0.y = (cCardinal.p0.y0 + 1 / 4 * Math.sin(cCardinal.th));
      cCardinal.p3.x = (cCardinal.p3.x0 + 1 / 4 * Math.cos(cCardinal.th / 3));
      cCardinal.p3.y = (cCardinal.p3.y0 + 1 / 4 * Math.sin(cCardinal.th / 9));

    }
  };

  let cBezier = {
    p0: {
      x: 0.1,
      y: 0.625
    },
    p1: {
      x: 0.25,
      y: 0.725,
      x0: 0.25,
      y0: 0.725
    },
    p2: {
      x: 0.75,
      y: 0.525,
      x0: 0.75,
      y0: 0.525
    },
    p3: {
      x: 0.9,
      y: 0.625
    },
    t: 0,
    update: dt => {
      dt /= 1000;
      cBezier.t += dt * 7;
      cBezier.p1.x = cBezier.p1.x0 + 1 / 5 * Math.cos(cBezier.t / 2);
      cBezier.p1.y = cBezier.p1.y0 + 1 / 5 * Math.sin(cBezier.t / 3);
      cBezier.p2.x = cBezier.p2.x0 + 1 / 7 * Math.cos(cBezier.t / 4);
      cBezier.p2.y = cBezier.p2.y0 + 1 / 7 * Math.sin(cBezier.t / 5);
    }
  };

  let cBezier2 = {
    p0: {
      x: 0.1,
      y: 0.875
    },
    p1: {
      x: 0.05,
      y: 0.70,
      x0: 0.05,
      y0: 0.70
    },
    p2: {
      x: 0.75,
      y: 0.90,
      x0: 0.75,
      y0: 0.90
    },
    p3: {
      x: 0.9,
      y: 0.875
    },
    t: 0,
    update: dt => {
      dt /= 1000;
      cBezier2.t += dt * 5;
      cBezier2.p1.x = cBezier2.p1.x0 + 1 / 5 * Math.cos(cBezier2.t / 2);
      cBezier2.p1.y = cBezier2.p1.y0 + 1 / 5 * Math.sin(cBezier2.t / 2.5);
      cBezier2.p2.x = cBezier2.p2.x0 + 1 / 7 * Math.cos(cBezier2.t / 3);
      cBezier2.p2.y = cBezier2.p2.y0 + 1 / 7 * Math.sin(cBezier2.t / 3);
    }
  };

  let circle = {
    primitive: makeRegularPrimitive(3),
    nextVert: .075,
    verts: 3,
    curTime: 0,
    vertDir: -1,
    update: dt => {
      dt /= 1000;
      circle.curTime += dt;
      if (circle.curTime > circle.nextVert) {
        circle.primitive = makeRegularPrimitive(circle.verts += circle.vertDir);
        circle.curTime = 0;
      }
      if (circle.verts > 20 || circle.verts <= 3) {
        circle.vertDir *= -1;
        circle.verts += circle.vertDir;
      }
    }
  };

  let shapes = [];
  for (let i = 3; i < 11; ++i) {
    shapes.push(makeRegularPrimitive(i));
  }

  let heptagon = makeRegularPrimitive(3);

  //------------------------------------------------------------------
  //
  // Scene updates go here.
  //
  //------------------------------------------------------------------
  let theta = 0;
  let tx = .8;
  let ty = .25;
  let dtx = 1.05;
  let dty = 1.3;
  let r = 0, g = 0, b = 0, dr = 1, dg = 1, db = 1;
  function update(dt) {
    cHermite.update(dt);
    cCardinal.update(dt);
    cBezier.update(dt);
    cBezier2.update(dt);
    theta += Math.PI / 3500 * dt;

    circle.update(dt);

    dt /= 1000;
    tx += dt * .1 * dtx;
    ty += dt * .1 * dty;
    if (tx > .9 || tx < .75) {
      dtx *= -1;
      tx += dt * .1 * dtx;
    }
    if (ty > .4 || ty < .1) {
      dty *= -1;
      ty += dt * .1 * dty;
    }

    r += dt * 2 * dr;
    g += dt * 10 * dg;
    b += dt * 35 * db;
    if (r > 255 || r < 0) {
      dr *= -1;
      r += dt * 2 * dr;
    }
    if (g > 100 || g < 0) {
      dg *= -1;
      g += dt * 10 * dg;
    }
    if (b > 255 || b < 0) {
      db *= -1;
      b += dt * 35 * db;
    }

  }

  //------------------------------------------------------------------
  //
  // Rendering code goes here
  //
  //------------------------------------------------------------------
  let points = false;
  let line = true;
  let controls = false;
  function render() {
    graphics.clear(false);

    for (let i = 0, s = .05; i < .25; i += .01, s += .03) {
      graphics.drawCurve(graphics.Curve.Bezier,
        graphics.translateCurve(graphics.Curve.Bezier,
          graphics.scaleCurve(graphics.Curve.Bezier,
            graphics.rotateCurve(graphics.Curve.Bezier, cBezier, theta / 3),
            { x: s, y: s }),
          { x: 0, y: i - .35 }),
        points, line, controls, `rgb(${g}, ${r}, ${b})`);
    }
    // graphics.drawCurve(graphics.Curve.Cardinal, cCardinal, points, line, controls, 'rgb(0, 0, 255)');
    // graphics.drawCurve(graphics.Curve.Bezier, cBezier, points, line, controls, 'rgb(0, 155, 44)');
    // graphics.drawCurve(graphics.Curve.BezierMatrix, cBezier2, points, line, controls, 'rgb(255, 0, 0)');
    let i = .0275;
    let s = .1;
    let dir = 1;
    shapes.forEach((shape) => {
      graphics.drawPrimitive(
        graphics.translatePrimitive(
          graphics.scalePrimitive(
            graphics.rotatePrimitive(shape, theta*dir),
            { x: s * 1.5, y: s * 1.5 }),
          { x: .1, y: i * 5 - .1 }),
        true, `rgb(0, ${255*s}, ${255*s}`);
      i *= 1.3;
      s *= 1.3;
      dir*=-1;
    });

    let s2 = .05;
    for (let i = 0; i < 35; ++i) {
      graphics.drawPrimitive(
        graphics.translatePrimitive(
          graphics.scalePrimitive(
            graphics.rotatePrimitive(heptagon, theta / 20 * (i + 1)),
            { x: s2, y: s2 }),
          { x: tx, y: ty }),
        true, `rgb(${r}, ${g}, ${b})`);
      s2 += .025;
    }

    graphics.drawPrimitive(
      graphics.translatePrimitive(
        graphics.rotatePrimitive(circle.primitive,
          theta * 2),
        { x: .85, y: .85 }),
      true, 'rgb(204, 63, 63)');
  }

  //------------------------------------------------------------------
  //
  // This is the animation loop.
  //
  //------------------------------------------------------------------
  let prevTime = performance.now();
  function animationLoop(time) {

    let dt = time - prevTime;
    prevTime = time;

    update(dt);
    render();

    requestAnimationFrame(animationLoop);
  }

  console.log('initializing...');
  graphics.curveSegments = 50;
  requestAnimationFrame(animationLoop);

}(MySample.graphics));