
MySample.main = (function (graphics, input) {
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
      y: Math.trunc(graphics.pixelsY * 0.5)
    },
    p1: {
      x: Math.trunc(graphics.pixelsX * 0.05),
      y: Math.trunc(graphics.pixelsY * 0.70),
      x0: Math.trunc(graphics.pixelsX * 0.05),
      y0: Math.trunc(graphics.pixelsY * 0.20)
    },
    p2: {
      x: Math.trunc(graphics.pixelsX * 0.75),
      y: Math.trunc(graphics.pixelsY * 0.90),
      x0: Math.trunc(graphics.pixelsX * 0.75),
      y0: Math.trunc(graphics.pixelsY * 0.40)
    },
    p3: {
      x: Math.trunc(graphics.pixelsX * 0.9),
      y: Math.trunc(graphics.pixelsY * 0.5)
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
    // cHermite.update(dt);
    // cCardinal.update(dt);
    // cBezier.update(dt);
    // cBezier2.update(dt);
    mouse.update(dt);
    keyboard.update(dt);
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

    // graphics.drawCurve(graphics.Curve.Hermite, cHermite, points, line, controls, 'rgb(0, 0, 0)');
    // graphics.drawCurve(graphics.Curve.Cardinal, cCardinal, points, line, controls, 'rgb(0, 0, 255)');
    // graphics.drawCurve(graphics.Curve.Bezier, cBezier, points, line, controls, 'rgb(0, 155, 44)');
    for (let i = 0; i < curves.length; ++i)
    {
      graphics.drawCurve(graphics.Curve.BezierMatrix, curves[i], points, line, controls, 'rgb(255, 0, 0)');
    }
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

  function newBezier()
  {
    return cBezier = {
      p0: {
        x: NaN,
        y: NaN
      },
      p1: {
        x: NaN,
        y: NaN
      },
      p2: {
        x: NaN,
        y: NaN
      },
      p3: {
        x: NaN,
        y: NaN
      }
    };
  }

  function near(obj, x, y)
  {
    const dx = graphics.pixelsX*.025;
    const dy = graphics.pixelsY*.025;
    const inX = obj.x + dx > x && obj.x - dx < x;
    const inY = obj.y + dy > y && obj.y - dy < y;
    return inX && inY;
  }

  console.log('initializing...');
  graphics.curveSegments = 50;
  const mouse = input.Mouse();
  const keyboard = input.Keyboard();
  let mouseCapture = false;
  let target = null;
  let m_x0, m_y0;
  let curves = [];
  let newpoint = 4;
  curves.push(cBezier2);

  mouse.registerCommand('mousedown', function(e) {
    mouseCapture = true;
    m_x0 = e.clientX;
    m_y0 = e.clientY;

    if (newpoint < 4)
    {
      if (newpoint === 0)
      {
        curves[curves.length - 1].p0.x = m_x0;
        curves[curves.length - 1].p0.y = m_y0;
      }
      if (newpoint === 1)
      {
        curves[curves.length - 1].p1.x = m_x0;
        curves[curves.length - 1].p1.y = m_y0;
      }
      if (newpoint === 2)
      {
        curves[curves.length - 1].p2.x = m_x0;
        curves[curves.length - 1].p2.y = m_y0;
      }
      if (newpoint === 3)
      {
        curves[curves.length - 1].p3.x = m_x0;
        curves[curves.length - 1].p3.y = m_y0;
      }
      ++newpoint;
    }
    else
    {
      for (let i = 0; i < curves.length; ++i)
      {
        if (near(curves[i].p0, m_x0, m_y0))
        {
          target = curves[i].p0;
          break;
        }
        else if (near(curves[i].p1, m_x0, m_y0))
        {
          target = curves[i].p1;
          break;
        }
        else if (near(curves[i].p2, m_x0, m_y0))
        {
          target = curves[i].p2;
          break;
        }
        else if (near(curves[i].p3, m_x0, m_y0))
        {
          target = curves[i].p3;
          break;
        }
      }
    }
  });

  window.addEventListener('keydown', (e) =>{
    if(e.key === 'b' || e.key === 'n')
    {
      curves.push(newBezier());
      newpoint = 0;
    }
  });

  mouse.registerCommand('mouseup', function(e) {
    void e;
    mouseCapture = false;
    target = null;
  });

  mouse.registerCommand('mousemove', function(e) {
    if (mouseCapture) {
      let dx = e.clientX - m_x0;
      let dy = e.clientY - m_y0;
      if (target)
      {
        target.x += dx;
        target.y += dy;
        m_x0 = e.clientX;
        m_y0 = e.clientY;
      }
    }
  });
  requestAnimationFrame(animationLoop);

}(MySample.graphics, MySample.input));
