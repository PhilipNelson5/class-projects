
MySample.main = (function() {
  'use strict';

  let canvas = document.getElementById('canvas-main');
  let gl = canvas.getContext('webgl');

  let previousTime = performance.now();
  let scene = {};
  let data = [];
  let buffers = {};
  let shaders = {};
  let offsetX = 1.0/canvas.width;
  let offsetY = 1.0/canvas.height;
  let circleDiffuse = {
    c : new Float32Array([0.0, 0.0, -10.0]),
    r : 1.0,
    color : new Float32Array([0.0, 0.25, 0.0]),
    material : 0,
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

    data.light = new Float32Array([5.0, 5.0, 25.0]);
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
          shaders.locSphereDiffuseCenter = gl.getUniformLocation(shaders.shaderProgram, 'uSphereDiffuse.c');
          shaders.locSphereDiffuseRadius = gl.getUniformLocation(shaders.shaderProgram, 'uSphereDiffuse.r');
          shaders.locSphereDiffuseColor = gl.getUniformLocation(shaders.shaderProgram, 'uSphereDiffuse.color');
          shaders.locSphereDiffuseMaterial = gl.getUniformLocation(shaders.shaderProgram, 'uSphereDiffuse.material');
          shaders.locLight = gl.getUniformLocation(shaders.shaderProgram, 'uLightPos');

          resolve();
        })
        .catch(error => {
          console.log('(initializeShaders) something bad happened: ', error);
          reject();
        });
    });
  }

  //------------------------------------------------------------------
  //
  // Scene updates go here.
  //
  //------------------------------------------------------------------
  let th = 0;
  function update(dt) {
    gl.uniform1f(shaders.locOffsetX, offsetX);
    gl.uniform1f(shaders.locOffsetY, offsetY);
    gl.uniform3fv(shaders.locEye, data.eye);
    gl.uniform3fv(shaders.locSphereDiffuseCenter, circleDiffuse.c);
    gl.uniform1f(shaders.locSphereDiffuseRadius, circleDiffuse.r);
    gl.uniform3fv(shaders.locSphereDiffuseColor, circleDiffuse.color);
    gl.uniform1i(shaders.locSphereDiffuseMaterial, circleDiffuse.material);
    gl.uniform3fv(shaders.locLight, data.light);

    //data.light[0] = 5 * Math.cos(th);
    //data.light[1] = 5 * Math.sin(th);
    //th += dt / 1000 * 2;
  }

  //------------------------------------------------------------------
  //
  // Rendering code goes here
  //
  //------------------------------------------------------------------
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
  }

  //------------------------------------------------------------------
  //
  // This is the animation loop.
  //
  //------------------------------------------------------------------
  function animationLoop(time) {
    let elapsedTime = previousTime - time;
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
      requestAnimationFrame(animationLoop);
    });

}());
