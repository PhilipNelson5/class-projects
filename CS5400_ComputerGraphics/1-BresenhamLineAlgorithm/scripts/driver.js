MySample.main = (function(graphics) {
  'use strict';

  let ptCenter = {
    x: 500,
    y: 500,
  };
  let secEnd = {
    x: 950,
    y: 85,
  };
  let minEnd = {
    x: 950,
    y: 85,
  };
  let hrsEnd = {
    x: 950,
    y: 85,
  };
  let epicycloid = {
    x: 500,
    y: 500,
    rotation: 0,
  };

  //------------------------------------------------------------------
  //
  // Scene updates go here.
  //
  //------------------------------------------------------------------
  let i = 0, di = 25;
  let r = 3, R = 5;
  let dr = 0.0025, dR = 0;
  function update(dt)
  {
    console.log(dt);

    let d = new Date();
    let sec = d.getSeconds();
    let min = d.getMinutes();
    let hrs = d.getHours();

    if (hrs > 12) hrs -= 12;

    let secRot = (sec/60*2*Math.PI)-Math.PI/2;
    let minRot = (min/60*2*Math.PI)-Math.PI/2;
    let hrsRot = (hrs/12*2*Math.PI)-Math.PI/2;

    secEnd = {
      x:ptCenter.x + Math.cos(secRot) * 50,
      y:ptCenter.y + Math.sin(secRot) * 50
    };

    minEnd = {
      x:ptCenter.x + Math.cos(minRot) * 50,
      y:ptCenter.y + Math.sin(minRot) * 50
    };

    hrsEnd = {
      x:ptCenter.x + Math.cos(hrsRot) * 35,
      y:ptCenter.y + Math.sin(hrsRot) * 35
    };

    dt/=1000;
    i += dt*di;
    if (i > 200 || i < 0)
    {
      di *= -1;
      i += di;
    }

    R += dt*dR;
    if (dR > 10 || dR < 3)
    {
      dR *= -1;
    }

    r += dt*dr;

    epicycloid.rotation += .005;
    epicycloid.rotation %= 360;
  }

  //------------------------------------------------------------------
  //
  // Rendering code goes here
  //
  //------------------------------------------------------------------
  function render()
  {
    graphics.clear();

    graphics.drawLine(ptCenter.x,
      ptCenter.y,
      Math.trunc(minEnd.x),
      Math.trunc(minEnd.y),
      'rgb(0, 0, 0)');

    graphics.drawLine(ptCenter.x,
      ptCenter.y,
      Math.trunc(hrsEnd.x),
      Math.trunc(hrsEnd.y),
      'rgb(0, 0, 0)');

    graphics.drawLine(ptCenter.x,
      ptCenter.y,
      Math.trunc(secEnd.x),
      Math.trunc(secEnd.y),
      'rgb(255, 0, 0)');

    graphics.drawEpicycloid({
      center:{x:epicycloid.x, y:epicycloid.y},
      R: R,
      r: r,
      d_th: .1,
      th_max: 360,
      scale: 20,
      rotation: epicycloid.rotation,
      color:'rgb(0, 45, ' + i + ')'
    });
  }

  //------------------------------------------------------------------
  //
  // This is the animation loop.
  //
  //------------------------------------------------------------------
  let prevTime = performance.now();
  function animationLoop(time)
  {
    let dt = time - prevTime;
    prevTime = time;

    update(dt);
    render();

    requestAnimationFrame(animationLoop);
  }

  console.log('initializing...');
  requestAnimationFrame(animationLoop);

}(MySample.graphics));
