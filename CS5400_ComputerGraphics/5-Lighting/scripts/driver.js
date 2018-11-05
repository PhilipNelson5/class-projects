Engine.main = (function() {
  'use strict';

  let environment = {};
  let model = {};
  let buffers = {};
  let shaders = { vecLightPos:[], vecLightColor:[] };
  let previousTime = performance.now();
  let lightPos = [
    [1, 1, 1, 1],
    [-1, -1, -1, 1],
    [1, -1, 1, 1],
  ];
  let lightColor = [
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
  ];

  //------------------------------------------------------------------
  //
  // Helper function used to set the rotation state and parameters for
  // the model.
  //
  //------------------------------------------------------------------
  function initializeModelRotation(model) {
    //
    // Current rotation status
    model.rotation = {  // Radians
      x: Math.PI/4,
      y: 0,
      z: 0
    };
    //
    // Rotation update rate
    model.rotationRate = {   // Radians per second (divide by 1000 to go from ms to seconds)
      //x: Math.PI / 4 / 1000,
      x: 0,
      y: Math.PI / 4 / 1000,
      //y: 0,
      //z: Math.PI / 4 / 1000,
      z: 0
    };
  }

  //------------------------------------------------------------------
  //
  // Prepare the rendering environment.
  //
  //------------------------------------------------------------------
  function initializeData() {

    environment.matAspect = [
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0
    ];
    if (canvas.width > canvas.height) {
      environment.matAspect[0] = canvas.height / canvas.width;
    } else {
      environment.matAspect[5] = canvas.width / canvas.height;
    }

    //
    // Obtain the projection matrix
    environment.matProjection = projectionPerspectiveFOV(Math.PI / 2, 1.0, 10.0);

    environment.vEye = new Float32Array([0.0, 0.0, 1.0]);
    environment.matView = transposeMatrix4x4([
      1,  0,  0,  -environment.vEye[0],
      0,  1,  0,  -environment.vEye[1],
      0,  0,  1,  -environment.vEye[2],
      0,  0,  0,  1
    ]);
  }

  //------------------------------------------------------------------
  //
  // Creates a Perspective Projection matrix based on a requested FOV.
  // The matrix results in the vertices in Normalized Device Coordinates...
  //
  //------------------------------------------------------------------
  function projectionPerspectiveFOV(fov, near, far) {
    let scale = Math.tan(Math.PI * 0.5 - 0.5 * fov);
    let m = [
      scale,  0.0,  0.0, 0.0,
      0.0,   scale, 0.0, 0.0,
      0.0,    0.0, -(far + near) / (far - near), -(2 * far * near) / (far - near),
      0.0,    0.0, -1,   0
    ];
    return transposeMatrix4x4(m);
  }

  //------------------------------------------------------------------
  //
  // Prepare and set the Vertex Buffer Object to render.
  //
  //------------------------------------------------------------------
  function initializeBufferObjects() {
    buffers.vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, model.vertices, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    buffers.vertexNormalBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.vertexNormalBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, model.normals, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    buffers.vertexColorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.vertexColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, model.vertexColors, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    buffers.indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, model.indices, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
  }

  //------------------------------------------------------------------
  //
  // Prepare and set the shaders to be used.
  //
  //------------------------------------------------------------------
  function initializeShaders() {
    return new Promise((resolve, reject) => {
      loadFileFromServer('shaders/simple.vs')
        .then(source => {
          shaders.vertexShader = gl.createShader(gl.VERTEX_SHADER);
          gl.shaderSource(shaders.vertexShader, source);
          gl.compileShader(shaders.vertexShader);

          if (!gl.getShaderParameter(shaders.vertexShader, gl.COMPILE_STATUS)){
            console.log("ERROR - createShader: ", gl.getShaderInfoLog(shaders.vertexShader));
          }

          return loadFileFromServer('shaders/simple.frag');
        })
        .then(source => {
          shaders.fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
          gl.shaderSource(shaders.fragmentShader, source);
          gl.compileShader(shaders.fragmentShader);
          if (!gl.getShaderParameter(shaders.fragmentShader, gl.COMPILE_STATUS)){
            console.log("ERROR - createShader: ", gl.getShaderInfoLog(shaders.fragmentShader));
          }
        })
        .then(() => {
          shaders.shaderProgram = gl.createProgram();
          gl.attachShader(shaders.shaderProgram, shaders.vertexShader);
          gl.attachShader(shaders.shaderProgram, shaders.fragmentShader);
          gl.linkProgram(shaders.shaderProgram);

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
  // Associate the vertex and pixel shaders, and the expected vertex
  // format with the VBO.
  //
  //------------------------------------------------------------------
  function associateShadersWithBuffers() {
    gl.useProgram(shaders.shaderProgram);

    shaders.matAspect          = gl.getUniformLocation(shaders.shaderProgram, 'uAspect');
    shaders.matProjection      = gl.getUniformLocation(shaders.shaderProgram, 'uProjection');
    shaders.matView            = gl.getUniformLocation(shaders.shaderProgram, 'uView');
    shaders.matModel           = gl.getUniformLocation(shaders.shaderProgram, 'uModel');
    shaders.matNormal          = gl.getUniformLocation(shaders.shaderProgram, 'uNormal');
    for (let i = 0; i < lightPos.length; ++i) {
      shaders.vecLightPos[i]   = gl.getUniformLocation(shaders.shaderProgram, `uLightPos${i}`);
      shaders.vecLightColor[i] = gl.getUniformLocation(shaders.shaderProgram, `uLightColor${i}`);
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.vertexBuffer);
    let position = gl.getAttribLocation(shaders.shaderProgram, 'aPosition');
    gl.vertexAttribPointer(position, 3, gl.FLOAT, false, model.vertices.BYTES_PER_ELEMENT * 3, 0);
    gl.enableVertexAttribArray(position);

    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.vertexNormalBuffer);
    let normal = gl.getAttribLocation(shaders.shaderProgram, 'aNormal');
    gl.vertexAttribPointer(normal, 3, gl.FLOAT, false, model.normals.BYTES_PER_ELEMENT * 3, 0);
    gl.enableVertexAttribArray(normal);

    gl.bindBuffer(gl.ARRAY_BUFFER, buffers.vertexColorBuffer);
    let color = gl.getAttribLocation(shaders.shaderProgram, 'aColor');
    gl.vertexAttribPointer(color, 3, gl.FLOAT, false, model.vertexColors.BYTES_PER_ELEMENT * 3, 0);
    gl.enableVertexAttribArray(color);
  }

  //------------------------------------------------------------------
  //
  // Prepare some WegGL settings, things like the clear color, depth buffer, etc.
  //
  //------------------------------------------------------------------
  function initializeWebGLSettings() {
    let ext = gl.getExtension('OES_element_index_uint');
    gl.clearColor(
      0.3921568627450980392156862745098,
      0.58431372549019607843137254901961,
      0.92941176470588235294117647058824, 1.0);
    gl.clearDepth(1.0);
    gl.depthFunc(gl.LEQUAL);
    gl.enable(gl.DEPTH_TEST);
  }

  //------------------------------------------------------------------
  //
  // Scene updates go here.
  //
  //------------------------------------------------------------------
  function update(elapsedTime) {
    //
    // Update the rotation matrices
    model.rotation.x += (model.rotationRate.x * elapsedTime);
    let sin = Math.sin(model.rotation.x);
    let cos = Math.cos(model.rotation.x);
    let matRotateX = [
      1,   0,    0,   0,
      0,  cos, -sin,  0,
      0,  sin,  cos,  0,
      0,   0,    0,   1
    ];
    matRotateX = transposeMatrix4x4(matRotateX);

    model.rotation.y += (model.rotationRate.y * elapsedTime);
    sin = Math.sin(model.rotation.y);
    cos = Math.cos(model.rotation.y);
    let matRotateY = transposeMatrix4x4([
      cos,  0,  sin, 0,
      0,    1,   0,  0,
      -sin, 0,  cos, 0,
      0,    0,   0,  1
    ]);

    model.rotation.z += (model.rotationRate.z * elapsedTime);
    sin = Math.sin(model.rotation.z);
    cos = Math.cos(model.rotation.z);
    let matRotateZ = transposeMatrix4x4([
      cos, -sin, 0, 0,
      sin,  cos, 0, 0,
      0,     0,  1, 0,
      0,     0,  0, 1
    ]);

    let matScale = transposeMatrix4x4([
      model.scale.x,       0,             0,             0,
      0,             model.scale.y,       0,             0,
      0,                   0,       model.scale.z,       0,
      0,                   0,             0,             1,
    ]);

    let matTranslate = transposeMatrix4x4([
      1,  0,  0, model.center.x,
      0,  1,  0, model.center.y,
      0,  0,  1, model.center.z,
      0,  0,  0, 1
    ]);

    model.matModel = multiplyMatrix4x4(
      matTranslate,
      matScale,
      matRotateX,
      matRotateY, 
      matRotateZ
    );

    model.matNormal = invert(multiplyMatrix4x4(model.matModel, environment.matView));
  }

  //------------------------------------------------------------------
  //
  // Rendering code goes here
  //
  //------------------------------------------------------------------
  function render() {
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    //
    // This sets which buffers/shaders to use for the draw call in the render function.
    associateShadersWithBuffers();
    gl.uniformMatrix4fv(shaders.matAspect, false, environment.matAspect);
    gl.uniformMatrix4fv(shaders.matProjection, false, environment.matProjection);
    gl.uniformMatrix4fv(shaders.matView, false, environment.matView);
    gl.uniformMatrix4fv(shaders.matModel, false, model.matModel);
    gl.uniformMatrix4fv(shaders.matNormal, false, model.matNormal);
    for(let i = 0; i < lightPos.length; ++i){
      gl.uniform4fv(shaders.vecLightPos[i], lightPos[i]);
      gl.uniform4fv(shaders.vecLightColor[i], lightColor[i]);
    }
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.indexBuffer);
    gl.drawElements(gl.TRIANGLES, model.indices.length, model.indices_type, 0);
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
  console.log('    Loading model');
  //ModelLoaderPLY.load('models/cube.ply')
  ModelLoaderPLY.load('models/dodecahedron.ply')
  //ModelLoaderPLY.load('models/bunny.ply')
  //ModelLoaderPLY.load('models/galleon.ply')
  //ModelLoaderPLY.load('models/happy.ply')

    .then(modelSource => {
      model = modelSource;
      initializeModelRotation(model);
      console.log('    WebGL settings');
      initializeWebGLSettings();
      console.log('    raw data');
      initializeData();
      console.log('    vertex buffer objects');
      initializeBufferObjects();
      console.log('    shaders');
      return initializeShaders();
    })
    .then(() => {
      console.log('initialization complete!');
      requestAnimationFrame(animationLoop);
    })
    .catch(error => console.error('[ERROR] ' + error));

}());
