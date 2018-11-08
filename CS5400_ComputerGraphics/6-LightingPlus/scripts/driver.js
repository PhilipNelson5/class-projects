Engine.main = (function() {
  'use strict';

  let environment = {};
  let models = [];
  let buffers = {};
  let shaders = { vecLightPos:[], vecLightColor:[] };
  let previousTime = performance.now();
  let lightPos = [
    [0, 10, 0, 1],
    [0, 0, 10, 1],
    [10, 0, 0, 1],
  ];
  let lightColor = [
    [0, 0, 1, 1],
    [1, 0, 0, 1],
    [1, 1, 0, 1],
  ];
  let lightOn = [
    true,
    true,
    true,
  ];
  let lightCheckBox = [
    document.getElementById('color0'),
    document.getElementById('color1'),
    document.getElementById('color2'),
  ];
  let lightColorPicker = [
    document.getElementById('colorpicker0'),
    document.getElementById('colorpicker1'),
    document.getElementById('colorpicker2'),
  ];

  //------------------------------------------------------------------
  //
  // Helper function used to set the rotation state and parameters for
  // the model.
  //
  //------------------------------------------------------------------
  function initializeModelRotation() {
    for (let i = 0; i < models.length; ++i) {
      //
      // Current rotation status
      models[i].rotation = {  // Radians
        x: Math.PI/10,
        //x: i,
        y: 0,
        z: 0
      };
      //
      // Rotation update rate
      models[i].rotationRate = {   // Radians per second (divide by 1000 to go from ms to seconds)
        //x: Math.PI / 4 / 1000,
        x: 0,
        y: Math.PI / 4 / 1000,
        //y: 0,
        //z: Math.PI / 4 / 1000,
        z: 0
      };
    }
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
    environment.matView = [
      1,  0,  0,  -environment.vEye[0],
      0,  1,  0,  -environment.vEye[1],
      0,  0,  1,  -environment.vEye[2],
      0,  0,  0,  1
    ];
  }

  //------------------------------------------------------------------
  //
  // Creates a Perspective Projection matrix based on a requested FOV.
  // The matrix results in the vertices in Normalized Device Coordinates...
  //
  //------------------------------------------------------------------
  function projectionPerspectiveFOV(fov, near, far) {
    let scale = Math.tan(Math.PI * 0.5 - 0.5 * fov);
    return [
      scale,  0.0,  0.0, 0.0,
      0.0,   scale, 0.0, 0.0,
      0.0,    0.0, -(far + near) / (far - near), -(2 * far * near) / (far - near),
      0.0,    0.0,  -1,   0
    ];
  }

  //------------------------------------------------------------------
  //
  // Prepare and set the Vertex Buffer Object to render.
  //
  //------------------------------------------------------------------
  function initializeBufferObjects() {
    for (let i = 0; i < models.length; ++i) {
      models[i].vertexBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, models[i].vertexBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, models[i].vertices, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);

      models[i].vertexNormalBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, models[i].vertexNormalBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, models[i].normals, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);

      models[i].vertexColorBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, models[i].vertexColorBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, models[i].vertexColors, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);

      models[i].indexBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, models[i].indexBuffer);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, models[i].indices, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    }
  }

  function initializeBufferObject(model) {
    model.vertexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, model.vertexBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, model.vertices, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    model.vertexNormalBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, model.vertexNormalBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, model.normals, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    model.vertexColorBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, model.vertexColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, model.vertexColors, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);

    model.indexBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, model.indexBuffer);
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
  function associateShadersWithBuffers(i) {
    gl.useProgram(shaders.shaderProgram);

    shaders.matAspect              = gl.getUniformLocation(shaders.shaderProgram, 'uAspect');
    shaders.matProjection          = gl.getUniformLocation(shaders.shaderProgram, 'uProjection');
    shaders.matView                = gl.getUniformLocation(shaders.shaderProgram, 'uView');
    shaders.matModel               = gl.getUniformLocation(shaders.shaderProgram, 'uModel');
    shaders.matNormal              = gl.getUniformLocation(shaders.shaderProgram, 'uNormal');

    for (let light = 0; light < lightPos.length; ++light) {
      shaders.vecLightPos[light]   = gl.getUniformLocation(shaders.shaderProgram, `uLightPos${light}`);
      shaders.vecLightColor[light] = gl.getUniformLocation(shaders.shaderProgram, `uLightColor${light}`);
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, models[i].vertexBuffer);
    let position = gl.getAttribLocation(shaders.shaderProgram, 'aPosition');
    gl.vertexAttribPointer(position, 3, gl.FLOAT, false, models[i].vertices.BYTES_PER_ELEMENT * 3, 0);
    gl.enableVertexAttribArray(position);

    gl.bindBuffer(gl.ARRAY_BUFFER, models[i].vertexNormalBuffer);
    let normal = gl.getAttribLocation(shaders.shaderProgram, 'aNormal');
    gl.vertexAttribPointer(normal, 3, gl.FLOAT, false, models[i].normals.BYTES_PER_ELEMENT * 3, 0);
    gl.enableVertexAttribArray(normal);

    gl.bindBuffer(gl.ARRAY_BUFFER, models[i].vertexColorBuffer);
    let color = gl.getAttribLocation(shaders.shaderProgram, 'aColor');
    gl.vertexAttribPointer(color, 3, gl.FLOAT, false, models[i].vertexColors.BYTES_PER_ELEMENT * 3, 0);
    gl.enableVertexAttribArray(color);
  }

  //------------------------------------------------------------------
  //
  // Prepare some WegGL settings, things like the clear color, depth buffer, etc.
  //
  //------------------------------------------------------------------
  function initializeWebGLSettings() {
    gl.getExtension('OES_element_index_uint');
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
  let th = 0;
  function update(dt) {
    for (let i = 0; i < models.length; ++i) {
      //
      // Update the rotation matrices
      //dt = 0;
      models[i].rotation.x += (models[i].rotationRate.x * dt);
      let sin = Math.sin(models[i].rotation.x);
      let cos = Math.cos(models[i].rotation.x);
      let matRotateX = [
        1,   0,    0,   0,
        0,  cos, -sin,  0,
        0,  sin,  cos,  0,
        0,   0,    0,   1
      ];

      models[i].rotation.y += (models[i].rotationRate.y * dt);
      sin = Math.sin(models[i].rotation.y);
      cos = Math.cos(models[i].rotation.y);
      let matRotateY = [
        cos,  0,  sin, 0,
        0,    1,   0,  0,
        -sin, 0,  cos, 0,
        0,    0,   0,  1
      ];

      models[i].rotation.z += (models[i].rotationRate.z * dt);
      sin = Math.sin(models[i].rotation.z);
      cos = Math.cos(models[i].rotation.z);
      let matRotateZ = [
        cos, -sin, 0, 0,
        sin,  cos, 0, 0,
        0,     0,  1, 0,
        0,     0,  0, 1
      ];

      let matScale = [
        models[i].scale.x,       0,             0,             0,
        0,             models[i].scale.y,       0,             0,
        0,                   0,       models[i].scale.z,       0,
        0,                   0,             0,             1,
      ];

      let matTranslate = [
        1,  0,  0, models[i].center.x,
        0,  1,  0, models[i].center.y,
        0,  0,  1, models[i].center.z,
        0,  0,  0, 1
      ];

      models[i].matModel = multiplyMatrix4x4(
        matRotateX,
        matRotateY, 
        matRotateZ,
        matScale,
        matTranslate
      );

      models[i].matNormal = invert(
        multiplyMatrix4x4(
          models[i].matModel,
          environment.matView
        )
      );
      lightPos[1][0] = 5 * Math.cos(th);
      lightPos[1][2] = 5 * Math.sin(th);

      lightPos[2][0] = 5 * Math.cos(th/10);
      lightPos[2][1] = 5 * Math.sin(th/10);
      th += dt / 1000 / 2;

      lightOn[0] = lightCheckBox[0].checked;
      lightOn[1] = lightCheckBox[1].checked;
      lightOn[2] = lightCheckBox[2].checked;

      lightColor[0] = hexToRgba(lightColorPicker[0].value)
      lightColor[1] = hexToRgba(lightColorPicker[1].value)
      lightColor[2] = hexToRgba(lightColorPicker[2].value)
    }
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
    gl.uniformMatrix4fv(shaders.matAspect, false, transposeMatrix4x4(environment.matAspect));
    gl.uniformMatrix4fv(shaders.matProjection, false, transposeMatrix4x4(environment.matProjection));
    gl.uniformMatrix4fv(shaders.matView, false, transposeMatrix4x4(environment.matView));

    for (let i = 0; i < lightPos.length; ++i){
      gl.uniform4fv(shaders.vecLightPos[i], lightPos[i]);
      gl.uniform4fv(shaders.vecLightColor[i], lightOn[i] ? lightColor[i] : [0,0,0, 1]);
    }

    for (let i = 0; i < models.length; ++i){
      associateShadersWithBuffers(i);
      gl.uniformMatrix4fv(shaders.matModel, false, transposeMatrix4x4(models[i].matModel));
      gl.uniformMatrix4fv(shaders.matNormal, false, models[i].matNormal);

      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, models[i].indexBuffer);
      gl.drawElements(gl.TRIANGLES, models[i].indices.length, models[i].indices_type, 0);
    }
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
  //ModelLoaderPLY.load('models/dodecahedron.ply')
  //ModelLoaderPLY.load('models/bunny.ply')
  ModelLoaderPLY.load('models/cube.ply')
    .then(model => {
      model.rotation = {
        x: Math.PI/100,
        y: 0,
        z: 0
      };
      model.rotationRate = {
        x: 0,
        y: 0,
        z: 0,
      };
      model.center = {
        x:0,
        y:-3,
        z:-3,
      }
      model.scale = {
        x:5,
        y:1,
        z:5,
      }
      models.push(model);

      console.log('    WebGL settings');
      initializeWebGLSettings();
      console.log('    raw data');
      initializeData();
      console.log('    vertex buffer objects');
      initializeBufferObject(model);
      console.log('    shaders');
      return initializeShaders();
    })
    .then(() => {
      console.log('initialization complete!');
      requestAnimationFrame(animationLoop);
    })
    .catch(error => console.error('[ERROR] ' + error));

  //ModelLoaderPLY.load('models/bunny.ply')
  //.then(model => {
  //model.rotation = {
  //x: Math.PI/16,
  //y: 0,
  //z: 0
  //};
  //model.rotationRate = {
  //x: 0,
  //y: 0,
  //z: 0,
  //};
  //model.center = {
  //x:1,
  //y:-1.75,
  //z:-3,
  //}
  //models.push(model)
  //initializeBufferObject(model);
  //})
  //.catch(error => console.error('[ERROR] ' + error));

  ModelLoaderPLY.load('models/galleon.ply')
    .then(model => {
      model.rotation = {
        x: -Math.PI/2,
        y: Math.PI/2.5,
        z: 0
      };
      model.rotationRate = {
        x: 0,
        y: Math.PI / 4 / 1000,
        z: 0
      };
      initializeBufferObject(model);
      models.push(model)
    })
    .catch(error => console.error('[ERROR] ' + error));


  function dcopy(src) {
    let copy = {};
    for (let prop in src) {
      if (src.hasOwnProperty(prop)) {
        copy[prop] = src[prop];
      }
    }
    return copy;
  }

  function cloneObject(obj) {
    var clone = {};
    for(let prop in obj) {
      if(obj[prop] != null &&  typeof(obj[prop])=="object")
        clone[prop] = cloneObject(obj[prop]);
      else
        clone[prop] = obj[prop];
    }
    return clone;
  }

  function hexToRgba(hex) {
    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? [
      parseInt(result[1], 16)/255,
      parseInt(result[2], 16)/255,
      parseInt(result[3], 16)/255, 1
    ] : null;
  }
}());
