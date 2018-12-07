
MySample.main = (function(graphics, input) {
  'use strict';

  //let cBezier = {
  //p0: {x: 5, y: -15},
  //p1: {x: -4, y: 0, x0: -4, y0: 0},
  //p2: {x: 4, y: -15, x0: 4, y0: -15},
  //p3: {x: -5, y: 0},
  //t: 0,
  //update: dt => {
  //dt /= 1000;
  //cBezier.t += dt * 5;
  //cBezier.p1.x = cBezier.p1.x0 + graphics.pixelsX / 5 * Math.cos(cBezier.t / 2);
  //cBezier.p1.y = cBezier.p1.y0 + graphics.pixelsX / 5 * Math.sin(cBezier.t / 2.5);
  //cBezier.p2.x = cBezier.p2.x0 + graphics.pixelsX / 7 * Math.cos(cBezier.t / 3);
  //cBezier.p2.y = cBezier.p2.y0 + graphics.pixelsX / 7 * Math.sin(cBezier.t / 7);
  //}
  //};

  let cBezier = {
    p0: {
      x: Math.trunc(graphics.pixelsX * 0.8),
      y: Math.trunc(graphics.pixelsY * 0.2)
    },
    p1: {
      x: Math.trunc(graphics.pixelsX * 0.01),
      y: Math.trunc(graphics.pixelsY * 0.01),
    },
    p2: {
      x: Math.trunc(graphics.pixelsX * 0.99),
      y: Math.trunc(graphics.pixelsY * 0.99),
    },
    p3: {
      x: Math.trunc(graphics.pixelsX * 0.8),
      y: Math.trunc(graphics.pixelsY * 0.2)
    }
  };
  let cBezier2 = {
    p0: cBezier.p3,
    p1: {
      x: Math.trunc(graphics.pixelsX * 0.5),
      y: Math.trunc(graphics.pixelsY * 0.01),
    },
    p2: {
      x: Math.trunc(graphics.pixelsX * 0.99),
      y: Math.trunc(graphics.pixelsY * 0.5),
    },
    p3: cBezier.p0
  };

  const mouse = input.Mouse();
  const keyboard = input.Keyboard();
  let mouseCapture = false;
  let target = null;
  let m_x0, m_y0;

  let maxSegments = 50;

  const MATERIAL_DIFFUSE =    0;
  const MATERIAL_SPECULAR =   1;
  const MATERIAL_REFLECTIVE = 2;
  const MATERIAL_MIXTURE =    3;

  let canvas = document.getElementById('canvas-main');
  let gl = canvas.getContext('webgl');

  let multiRayElem = document.getElementById('MultiRay');
  let multiRay = multiRayElem.checked;

  let sliderNormalXElem = document.getElementById('sliderNormalX');
  let sliderNormalYElem = document.getElementById('sliderNormalY');
  let sliderNormalZElem = document.getElementById('sliderNormalZ');

  let sliderNormalXLableElem = document.getElementById('sliderNormalXLable');
  let sliderNormalYLableElem = document.getElementById('sliderNormalYLable');
  let sliderNormalZLableElem = document.getElementById('sliderNormalZLable');

  let sliderSphereDiffuseRadiusElem = document.getElementById('sliderSphereDiffuseRadius');
  let sliderSphereReflectiveRadiusElem = document.getElementById('sliderSphereReflectiveRadius');
  let sliderSphereMixedRadiusElem = document.getElementById('sliderSphereMixedRadius');

  let sliderDiffuseRadiusLabelElem = document.getElementById('sliderDiffuseRadiusLabel');
  let sliderReflectiveRadiusLabelElem = document.getElementById('sliderReflectiveRadiusLabel');
  let sliderMixedRadiusLabelElem = document.getElementById('sliderMixedRadiusLabel');

  let previousTime = performance.now();
  let scene = {};
  let data = [];
  let buffers = {};
  let shaders = {};
  let offsetX = 1.0/canvas.width;
  let offsetY = 1.0/canvas.height;
  let resolution = canvas.width;

  let sphereDiffuse = {
    c : new Float32Array([0.0, 0.0, -10.0]),
    r : 0.25,
    color : new Float32Array([1.0, 1.0, 0.0]),
    material : MATERIAL_SPECULAR,
  };
  let sphereReflective = {
    c : new Float32Array([0.0, 1.0, -10.0]),
    r : 0.5,
    color : new Float32Array([0.0, 0.0, 0.0]),
    material : MATERIAL_REFLECTIVE,
  };
  let sphereMixture = {
    c : new Float32Array([0.0, 0.0, -10.0]),
    r : 0.5,
    color : new Float32Array([0.0, 1.0, 0.0]),
    material : MATERIAL_MIXTURE,
  };
  let plane = {
    a : new Float32Array([0.0, -1.0, 0.0]),
    n : normalize(new Float32Array([0.0, 1.0, 0.0])),
    color : new Float32Array([1.0, 0.0, 0.0]),
    material : MATERIAL_DIFFUSE,
  };

  //------------------------------------------------------------------
  //
  // Prepare the data to be rendered
  //
  //------------------------------------------------------------------
  function initializeData() {
    data.vertices = new Float32Array([
      // -0.75, -0.75, 0.0,
      // -0.75, 0.75, 0.0,
      // 0.75, 0.75, 0.0,
      // 0.75, -0.75, 0.0
      -1.0, -1.0, 0.0,
      -1.0,  1.0, 0.0,
      1.0,   1.0, 0.0,
      1.0,  -1.0, 0.0
    ]);

    data.indices = new Uint16Array([ 0, 1, 2, 3, 0, 2 ]);

    data.eye = new Float32Array([0.0, 0.0, 5.0]);

    data.light = new Float32Array([0.0, 5.0, 1.0]);
  }

  //------------------------------------------------------------------
  //
  // Prepare and set the Vertex Buffer Object to render.
  //
  //------------------------------------------------------------------
  function initializeBufferObjects() {
    buffers.vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, data.vertices, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    buffers.indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, data.indices, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
  }

  //------------------------------------------------------------------
  //
  // Associate the vertex and pixel shaders, and the expected vertex
  // format with the VBO.
  //
  //------------------------------------------------------------------
  function associateShadersWithBuffers() {
    gl.useProgram(shaders.shaderProgram);

    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.vertexBuffer);
    let position = gl.getUniformLocation(shaders.shaderProgram, 'aPosition');
    gl.vertexAttribPointer(position, 3, gl.FLOAT, false, data.vertices.BYTES_PER_ELEMENT * 3, 0);
    gl.enableVertexAttribArray(position);

  }

  //------------------------------------------------------------------
  //
  // Prepare and set the shaders to be used.
  //
  //------------------------------------------------------------------
  function initializeShaders() {
    return new Promise((resolve, reject) => {
      loadFileFromServer('shaders/ray-trace.vs')
        .then(source => {
          shaders.vertexShader = gl.createShader(gl.VERTEX_SHADER);
          gl.shaderSource(shaders.vertexShader, source);
          gl.compileShader(shaders.vertexShader);
          return loadFileFromServer('shaders/ray-trace.frag');
        })
        .then(source => {
          shaders.fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
          gl.shaderSource(shaders.fragmentShader, source);
          gl.compileShader(shaders.fragmentShader);
        })
        .then(() => {
          shaders.shaderProgram = gl.createProgram();
          gl.attachShader(shaders.shaderProgram, shaders.vertexShader);
          gl.attachShader(shaders.shaderProgram, shaders.fragmentShader);
          gl.linkProgram(shaders.shaderProgram);

          let errVertex = gl.getShaderInfoLog(shaders.vertexShader);
          if (errVertex.length > 0) {
            console.log('Vertex errors: ', errVertex);
          }
          let errFragment = gl.getShaderInfoLog(shaders.fragmentShader);
          if (errFragment.length > 0) {
            console.log('Frag errors: ', errFragment);
          }

          shaders.locOffsetX = gl.getUniformLocation(shaders.shaderProgram, 'uOffsetX');
          shaders.locOffsetY = gl.getUniformLocation(shaders.shaderProgram, 'uOffsetY');
          shaders.locEye = gl.getUniformLocation(shaders.shaderProgram, 'uEye');

          shaders.locSphereDiffuseCenter   = gl.getUniformLocation(shaders.shaderProgram, 'uSphereDiffuse.c');
          shaders.locSphereDiffuseRadius   = gl.getUniformLocation(shaders.shaderProgram, 'uSphereDiffuse.r');
          shaders.locSphereDiffuseColor    = gl.getUniformLocation(shaders.shaderProgram, 'uSphereDiffuse.color');
          shaders.locSphereDiffuseMaterial = gl.getUniformLocation(shaders.shaderProgram, 'uSphereDiffuse.material');

          shaders.locSphereReflectiveCenter   = gl.getUniformLocation(shaders.shaderProgram, 'uSphereReflective.c');
          shaders.locSphereReflectiveRadius   = gl.getUniformLocation(shaders.shaderProgram, 'uSphereReflective.r');
          shaders.locSphereReflectiveColor    = gl.getUniformLocation(shaders.shaderProgram, 'uSphereReflective.color');
          shaders.locSphereReflectiveMaterial = gl.getUniformLocation(shaders.shaderProgram, 'uSphereReflective.material');

          shaders.locSphereMixtureCenter   = gl.getUniformLocation(shaders.shaderProgram, 'uSphereMixture.c');
          shaders.locSphereMixtureRadius   = gl.getUniformLocation(shaders.shaderProgram, 'uSphereMixture.r');
          shaders.locSphereMixtureColor    = gl.getUniformLocation(shaders.shaderProgram, 'uSphereMixture.color');
          shaders.locSphereMixtureMaterial = gl.getUniformLocation(shaders.shaderProgram, 'uSphereMixture.material');

          shaders.locPlanePoint    = gl.getUniformLocation(shaders.shaderProgram, 'uPlane.a');
          shaders.locPlaneNormal   = gl.getUniformLocation(shaders.shaderProgram, 'uPlane.n');
          shaders.locPlaneColor    = gl.getUniformLocation(shaders.shaderProgram, 'uPlane.color');
          shaders.locPlaneMaterial = gl.getUniformLocation(shaders.shaderProgram, 'uPlane.material');

          shaders.locLight = gl.getUniformLocation(shaders.shaderProgram, 'uLightPos');

          shaders.locSeed = gl.getUniformLocation(shaders.shaderProgram, 'uSeed');
          shaders.locResolution = gl.getUniformLocation(shaders.shaderProgram, 'uResolution');
          shaders.locMultiRay = gl.getUniformLocation(shaders.shaderProgram, 'uMultiRay');

          resolve();
        })
        .catch(error => {
          console.log('(initializeShaders) something bad happened: ', error);
          reject();
        });
    });
  }

  function interpolate(x, x0, x1, y0, y1)
  {
    return y0 + (x-x0)*(y1-y0)/(x1-x0);
  }

  //------------------------------------------------------------------
  //
  // Scene updates go here.
  //
  //------------------------------------------------------------------
  let th = 0;
  let segment = 0;
  let activeCurve = true;
  function update(dt) {
    mouse.update(dt);
    keyboard.update(dt);

    gl.uniform1f(shaders.locOffsetX, offsetX);
    gl.uniform1f(shaders.locOffsetY, offsetY);
    gl.uniform3fv(shaders.locEye, data.eye);

    gl.uniform3fv(shaders.locSphereDiffuseCenter,   sphereDiffuse.c);
    gl.uniform1f( shaders.locSphereDiffuseRadius,   sphereDiffuse.r);
    gl.uniform3fv(shaders.locSphereDiffuseColor,    sphereDiffuse.color);
    gl.uniform1i( shaders.locSphereDiffuseMaterial, sphereDiffuse.material);

    gl.uniform3fv(shaders.locSphereReflectiveCenter,   sphereReflective.c);
    gl.uniform1f( shaders.locSphereReflectiveRadius,   sphereReflective.r);
    gl.uniform3fv(shaders.locSphereReflectiveColor,    sphereReflective.color);
    gl.uniform1i( shaders.locSphereReflectiveMaterial, sphereReflective.material);

    gl.uniform3fv(shaders.locSphereMixtureCenter,   sphereMixture.c);
    gl.uniform1f( shaders.locSphereMixtureRadius,   sphereMixture.r);
    gl.uniform3fv(shaders.locSphereMixtureColor,    sphereMixture.color);
    gl.uniform1i( shaders.locSphereMixtureMaterial, sphereMixture.material);

    gl.uniform3fv(shaders.locPlanePoint,    plane.a);
    gl.uniform3fv(shaders.locPlaneNormal,   plane.n);
    gl.uniform3fv(shaders.locPlaneColor,    plane.color);
    gl.uniform1i( shaders.locPlaneMaterial, plane.material);

    gl.uniform3fv(shaders.locLight, data.light);

    let now = performance.now();
    gl.uniform1f(shaders.locSeed, Math.random()*1000000);
    //gl.uniform1f(shaders.locSeed, performance.now());

    gl.uniform1f(shaders.locResolution, resolution);
    gl.uniform1i(shaders.locMultiRay, multiRay);

    //data.light[0] = 5 * Math.cos(th);
    //data.light[1] = 5 * Math.sin(th);
    sphereDiffuse.c[0] = Math.cos(th);
    sphereDiffuse.c[1] = Math.cos(th);
    sphereDiffuse.c[2] = -10.0 + 2 * Math.sin(th);

    let curve = drawCurveBezier(activeCurve?cBezier:cBezier2, segment);
    let px = curve.px;
    px = interpolate(px, 0, graphics.pixelsX, -5, 5);
    let py = curve.py; 
    py = -interpolate(py, 0, graphics.pixelsY, -5, 5);
    sphereReflective.c[0] = px;
    sphereReflective.c[1] = py;
    th += dt / 1000;
    ++segment;
    if (segment > maxSegments)
    {
      activeCurve = !activeCurve;
      segment = 0;
    }

    multiRay = multiRayElem.checked;
    plane.n[0] = sliderNormalXElem.value;
    plane.n[1] = sliderNormalYElem.value;
    plane.n[2] = sliderNormalZElem.value;
    plane.n = normalize(plane.n);

    sliderNormalXLableElem.innerText = sliderNormalXElem.value;
    sliderNormalYLableElem.innerText = sliderNormalYElem.value;
    sliderNormalZLableElem.innerText = sliderNormalZElem.value;

    sliderDiffuseRadiusLabelElem.innerText = sliderSphereDiffuseRadiusElem.value;
    sliderReflectiveRadiusLabelElem.innerText = sliderSphereReflectiveRadiusElem.value;
    sliderMixedRadiusLabelElem.innerText = sliderSphereMixedRadiusElem.value;

    sphereDiffuse.r = sliderSphereDiffuseRadiusElem.value;
    sphereReflective.r = sliderSphereReflectiveRadiusElem.value;
    sphereMixture.r = sliderSphereMixedRadiusElem.value;
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
    gl.clearColor(
      0.3921568627450980392156862745098,
      0.58431372549019607843137254901961,
      0.92941176470588235294117647058824,
      1.0);
    gl.clearDepth(1.0);

    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    //
    // This sets which buffer to use for the draw call in the render function.
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.indexBuffer);
    gl.drawElements(gl.TRIANGLES, data.indices.length, gl.UNSIGNED_SHORT, 0);

    graphics.clear(false);
    graphics.drawCurve(graphics.Curve.BezierMatrix, cBezier, points, line, controls, 'rgb(255, 0, 0)');
    graphics.drawCurve(graphics.Curve.BezierMatrix, cBezier2, points, line, controls, 'rgb(255, 0, 0)');
  }

  function near(obj, x, y)
  {
    const dx = graphics.pixelsX*.025;
    const dy = graphics.pixelsY*.025;
    const inX = obj.x + dx > x && obj.x - dx < x;
    const inY = obj.y + dy > y && obj.y - dy < y;
    return inX && inY;
  }

  //------------------------------------------------------------------
  //
  // This is the animation loop.
  //
  //------------------------------------------------------------------
  function animationLoop(time) {
    let elapsedTime = time - previousTime;
    previousTime = time;

    update(elapsedTime);
    render();

    requestAnimationFrame(animationLoop);
  }

  console.log('initializing...');
  console.log('    raw data')
  initializeData();
  console.log('    vertex buffer objects');
  initializeBufferObjects();
  console.log('    shaders');
  initializeShaders()
    .then(() => {
      console.log('    binding shaders to VBOs');
      associateShadersWithBuffers();
      console.log('initialization complete!');
      previousTime = performance.now();
      console.log('initializing...');

      setCurveSegments(maxSegments);
      graphics.curveSegments = maxSegments;

      mouse.registerCommand('mousedown', function(e) {
        mouseCapture = true;
        m_x0 = e.clientX;
        m_y0 = e.clientY;

        if (near(cBezier.p0, m_x0, m_y0))
        {
          target = cBezier.p0;
        }
        else if (near(cBezier.p1, m_x0, m_y0))
        {
          target = cBezier.p1;
        }
        else if (near(cBezier.p2, m_x0, m_y0))
        {
          target = cBezier.p2;
        }
        else if (near(cBezier.p3, m_x0, m_y0))
        {
          target = cBezier.p3;
        }
        else if (near(cBezier2.p1, m_x0, m_y0))
        {
          target = cBezier2.p1;
        }
        else if (near(cBezier2.p2, m_x0, m_y0))
        {
          target = cBezier2.p2;
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
    });

}(MySample.graphics, MySample.input));
