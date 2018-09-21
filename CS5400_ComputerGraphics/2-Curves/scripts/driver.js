
MySample.main = (function (graphics) {
  'use strict';

  let cHermite = {
    p0: {
      x: Math.trunc(graphics.pixelsX * 0.2),
      y: Math.trunc(graphics.pixelsY * 0.125)
    },
    p1: {
      x: Math.trunc(graphics.pixelsX * 0.8),
      y: Math.trunc(graphics.pixelsY * 0.125)
    },
    s0: {
      x: Math.trunc(graphics.pixelsX * 0.1),
      y: -Math.trunc(graphics.pixelsY * 0.75)
    },
    s1: {
      x: Math.trunc(graphics.pixelsX * -0.1),
      y: Math.trunc(graphics.pixelsY * 0.15)
    },
    t: 0,
    update: dt => {
      dt /= 1000;
      cHermite.t += dt * 2;
      cHermite.s0.x = (graphics.pixelsX / 4 * Math.cos(cHermite.t / 2));
      cHermite.s0.y = (graphics.pixelsY / 4 * Math.sin(cHermite.t));
      cHermite.s1.x = (graphics.pixelsX / 4 * Math.cos(cHermite.t / 2));
      cHermite.s1.y = (graphics.pixelsY / 4 * Math.sin(cHermite.t / 5));
    }
  };

  let cCardinal = {
    p0: {
      x: Math.trunc(graphics.pixelsX * 0.05),
      y: Math.trunc(graphics.pixelsY * 0.25),
      x0: Math.trunc(graphics.pixelsX * 0.05),
      y0: Math.trunc(graphics.pixelsY * 0.25)
    },
    p1: {
      x: Math.trunc(graphics.pixelsX * 0.1),
      y: Math.trunc(graphics.pixelsY * 0.375)
    },
    p2: {
      x: Math.trunc(graphics.pixelsX * 0.9),
      y: Math.trunc(graphics.pixelsY * 0.375)
    },
    p3: {
      x: Math.trunc(graphics.pixelsX * 0.95),
      y: Math.trunc(graphics.pixelsY * 0.5),
      x0: Math.trunc(graphics.pixelsX * 0.95),
      y0: Math.trunc(graphics.pixelsY * 0.5)
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
      cCardinal.p0.x = (cCardinal.p0.x0 + graphics.pixelsX / 4 * Math.cos(cCardinal.th / 2));
      cCardinal.p0.y = (cCardinal.p0.y0 + graphics.pixelsY / 4 * Math.sin(cCardinal.th));
      cCardinal.p3.x = (cCardinal.p3.x0 + graphics.pixelsX / 4 * Math.cos(cCardinal.th / 3));
      cCardinal.p3.y = (cCardinal.p3.y0 + graphics.pixelsY / 4 * Math.sin(cCardinal.th / 9));

    }
  };

  let cBezier = {
    p0: {
      x: Math.trunc(graphics.pixelsX * 0.1),
      y: Math.trunc(graphics.pixelsY * 0.625)
    },
    p1: {
      x: Math.trunc(graphics.pixelsX * 0.25),
      y: Math.trunc(graphics.pixelsY * 0.725),
      x0: Math.trunc(graphics.pixelsX * 0.25),
      y0: Math.trunc(graphics.pixelsY * 0.725)
    },
    p2: {
      x: Math.trunc(graphics.pixelsX * 0.75),
      y: Math.trunc(graphics.pixelsY * 0.525),
      x0: Math.trunc(graphics.pixelsX * 0.75),
      y0: Math.trunc(graphics.pixelsY * 0.525)
    },
    p3: {
      x: Math.trunc(graphics.pixelsX * 0.9),
      y: Math.trunc(graphics.pixelsY * 0.625)
    },
    t: 0,
    update: dt => {
      dt /= 1000;
      cBezier.t += dt * 5;
      cBezier.p1.x = cBezier.p1.x0 + graphics.pixelsX / 5 * Math.cos(cBezier.t / 2);
      cBezier.p1.y = cBezier.p1.y0 + graphics.pixelsX / 5 * Math.sin(cBezier.t / 2.5);
      cBezier.p2.x = cBezier.p2.x0 + graphics.pixelsX / 7 * Math.cos(cBezier.t / 3);
      cBezier.p2.y = cBezier.p2.y0 + graphics.pixelsX / 7 * Math.sin(cBezier.t / 7);
    }
  };

  let cBezier2 = {
    p0: {
      x: Math.trunc(graphics.pixelsX * 0.1),
      y: Math.trunc(graphics.pixelsY * 0.875)
    },
    p1: {
      x: Math.trunc(graphics.pixelsX * 0.05),
      y: Math.trunc(graphics.pixelsY * 0.70),
      x0: Math.trunc(graphics.pixelsX * 0.05),
      y0: Math.trunc(graphics.pixelsY * 0.70)
    },
    p2: {
      x: Math.trunc(graphics.pixelsX * 0.75),
      y: Math.trunc(graphics.pixelsY * 0.90),
      x0: Math.trunc(graphics.pixelsX * 0.75),
      y0: Math.trunc(graphics.pixelsY * 0.90)
    },
    p3: {
      x: Math.trunc(graphics.pixelsX * 0.9),
      y: Math.trunc(graphics.pixelsY * 0.875)
    },
    t: 0,
    update: dt => {
      dt /= 1000;
      cBezier2.t += dt * 5;
      cBezier2.p1.x = cBezier2.p1.x0 + graphics.pixelsX / 5 * Math.cos(cBezier2.t / 2);
      cBezier2.p1.y = cBezier2.p1.y0 + graphics.pixelsX / 5 * Math.sin(cBezier2.t / 2.5);
      cBezier2.p2.x = cBezier2.p2.x0 + graphics.pixelsX / 7 * Math.cos(cBezier2.t / 3);
      cBezier2.p2.y = cBezier2.p2.y0 + graphics.pixelsX / 7 * Math.sin(cBezier2.t / 3);
    }
  };

  //------------------------------------------------------------------
  //
  // Scene updates go here.
  //
  //------------------------------------------------------------------
  function update(dt) {
    cHermite.update(dt);
    cCardinal.update(dt);
    cBezier.update(dt);
    cBezier2.update(dt);
  }

  //------------------------------------------------------------------
  //
  // Rendering code goes here
  //
  //------------------------------------------------------------------
  let points = false;
  let line = true;
  let controls = true;
  function render() {
    graphics.clear(false);

    graphics.drawCurve(graphics.Curve.Hermite, cHermite, points, line, controls, 'rgb(0, 0, 0)');
    graphics.drawCurve(graphics.Curve.Cardinal, cCardinal, points, line, controls, 'rgb(0, 0, 255)');
    graphics.drawCurve(graphics.Curve.Bezier, cBezier, points, line, controls, 'rgb(0, 155, 44)');
    graphics.drawCurve(graphics.Curve.BezierMatrix, cBezier2, points, line, controls, 'rgb(255, 0, 0)');
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
