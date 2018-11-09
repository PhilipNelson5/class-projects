Engine.main = (function() {
  'use strict';

  let environment = {};
  let models = [];
  let skybox = {};
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

    environment.matProjection = projectionPerspectiveFOV(Math.PI/2 , 1.0, 50.0);

    environment.vEye = new Float32Array([0.0, 0.0, 1.5]);
    environment.matView = translate(
      -environment.vEye[0],
      -environment.vEye[1],
      -environment.vEye[2]);

  }

  //------------------------------------------------------------------
  //
  // Prepare and set the Vertex Buffer Object to render.
  //
  //------------------------------------------------------------------
  function initializeBufferObjects() {
    for (let i = 0; i < models.length; ++i) {
      models[i].vertexBuffer = createBuffer(
        gl, models[i].vertices, gl.ARRAY_BUFFER, gl.STATIC_DRAW);

      models[i].vertexNormalBuffer = createBuffer(
        gl, models[i].normals, gl.ARRAY_BUFFER, gl.STATIC_DRAW);

      models[i].vertexColorBuffer = createBuffer(
        gl, models[i].vertexColors, gl.ARRAY_BUFFER, gl.STATIC_DRAW);

      models[i].indexBuffer = createBuffer(
        gl, models[i].indices, gl.ELEMENT_ARRAY_BUFFER, gl.STATIC_DRAW);
    }
  }

  function initializeCubeMap(model, texCube){
    model.cube = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, model.cube);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
    gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_X, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, texCube.posx);
    gl.texImage2D(gl.TEXTURE_CUBE_MAP_NEGATIVE_X, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, texCube.negx);
    gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_Y, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, texCube.posy);
    gl.texImage2D(gl.TEXTURE_CUBE_MAP_NEGATIVE_Y, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, texCube.negy);
    gl.texImage2D(gl.TEXTURE_CUBE_MAP_POSITIVE_Z, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, texCube.posz);
    gl.texImage2D(gl.TEXTURE_CUBE_MAP_NEGATIVE_Z, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, texCube.negz);
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_CUBE_MAP, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  function initializeBufferObject(model) {
    model.vertexBuffer = createBuffer(
      gl, model.vertices, gl.ARRAY_BUFFER, gl.STATIC_DRAW);

    model.vertexNormalBuffer = createBuffer(
      gl, model.normals, gl.ARRAY_BUFFER, gl.STATIC_DRAW);

    model.vertexColorBuffer = createBuffer(
      gl, model.vertexColors, gl.ARRAY_BUFFER, gl.STATIC_DRAW);

    model.indexBuffer = createBuffer(
      gl, model.indices, gl.ELEMENT_ARRAY_BUFFER, gl.STATIC_DRAW);
  }

  //------------------------------------------------------------------
  //
  // Prepare and set the shaders to be used.
  //
  //------------------------------------------------------------------
  function initializeShaders() {
    return new Promise((resolve, reject) => {
      loadFileFromServer('shaders/diffuse.vs')
        .then(source => {
          shaders.diffuse = {};
          shaders.diffuse.vShader = createShader(gl, gl.VERTEX_SHADER, source);

          return loadFileFromServer('shaders/diffuse.frag');
        })
        .then(source => {
          shaders.diffuse.fShader = createShader(gl, gl.FRAGMENT_SHADER, source);

          shaders.diffuse.program = createProgram(
            gl, shaders.diffuse.vShader, shaders.diffuse.fShader);

          return loadFileFromServer('shaders/skybox.vs');
        })
        .then(source => {
          shaders.skybox = {};
          shaders.skybox.vShader = createShader(gl, gl.VERTEX_SHADER, source);

          return loadFileFromServer('shaders/skybox.frag');
        })
        .then(source => {
          shaders.skybox.fShader = createShader(gl, gl.FRAGMENT_SHADER, source);

          shaders.skybox.program = createProgram(
            gl, shaders.diffuse.vShader, shaders.diffuse.fShader);

          resolve();
        })
        .catch(error => {
          console.error('(initializeShaders) ERROR : ', error);
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
  function associateShadersWithBuffers(model, program) {
    gl.useProgram(program);

    shaders.matAspect              = gl.getUniformLocation(program, 'uAspect');
    shaders.matProjection          = gl.getUniformLocation(program, 'uProjection');
    shaders.matView                = gl.getUniformLocation(program, 'uView');
    shaders.matModel               = gl.getUniformLocation(program, 'uModel');
    shaders.matNormal              = gl.getUniformLocation(program, 'uNormal');

    for (let light = 0; light < lightPos.length; ++light) {
      shaders.vecLightPos[light]   = gl.getUniformLocation(program, `uLightPos${light}`);
      shaders.vecLightColor[light] = gl.getUniformLocation(program, `uLightColor${light}`);
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, model.vertexBuffer);
    let position = gl.getAttribLocation(program, 'aPosition');
    gl.vertexAttribPointer(position, 3, gl.FLOAT, false, model.vertices.BYTES_PER_ELEMENT * 3, 0);
    gl.enableVertexAttribArray(position);

    gl.bindBuffer(gl.ARRAY_BUFFER, model.vertexNormalBuffer);
    let normal = gl.getAttribLocation(program, 'aNormal');
    gl.vertexAttribPointer(normal, 3, gl.FLOAT, false, model.normals.BYTES_PER_ELEMENT * 3, 0);
    gl.enableVertexAttribArray(normal);

    gl.bindBuffer(gl.ARRAY_BUFFER, model.vertexColorBuffer);
    let color = gl.getAttribLocation(program, 'aColor');
    gl.vertexAttribPointer(color, 3, gl.FLOAT, false, model.vertexColors.BYTES_PER_ELEMENT * 3, 0);
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

  function renderSkybox(sb){
    gl.useProgram(shaders.skybox.program);

    //
    // setup uniforms
    let uAspect = gl.getUniformLocation(shaders.skybox.program, 'uAspect');
    gl.uniformMatrix4fv(uAspect, false, transposeMatrix4x4(environment.matAspect));

    let uProjection = gl.getUniformLocation(shaders.skybox.program, 'uProjection');
    gl.uniformMatrix4fv(uProjection, false, transposeMatrix4x4(environment.matProjection));

    let uModel = gl.getUniformLocation(shaders.skybox.program, 'uModel');
    gl.uniformMatrix4fv(uModel, false, transposeMatrix4x4(scale(sb.model.scale.x, sb.model.scale.y, sb.model.scale.z)));

    let uView = gl.getUniformLocation(shaders.skybox.program, 'uView');
    gl.uniformMatrix4fv(uView, false, transposeMatrix4x4(environment.matView));

    //
    // setup position attribute
    gl.bindBuffer(gl.ARRAY_BUFFER, sb.model.vertexBuffer);
    let position = gl.getAttribLocation(shaders.skybox.program, 'aPosition');
    gl.vertexAttribPointer(position, 3, gl.FLOAT, false, sb.model.vertices.BYTES_PER_ELEMENT * 3, 0);
    gl.enableVertexAttribArray(position);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, sb.model.indexBuffer);

    //
    // draw the box
    gl.drawElements(gl.TRIANGLES, sb.model.indices.length, sb.model.indices_type, 0);
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
      models[i].rotation.y += (models[i].rotationRate.y * dt);
      models[i].rotation.z += (models[i].rotationRate.z * dt);

      let matRotateX = x_axis_rotate(models[i].rotation.x);
      let matRotateY = y_axis_rotate(models[i].rotation.y);
      let matRotateZ = z_axis_rotate(models[i].rotation.z);

      let matScale = scale(
        models[i].scale.x,
        models[i].scale.y,
        models[i].scale.z); 

      let matTranslate = translate(
        models[i].center.x,
        models[i].center.y,
        models[i].center.z);

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

    renderSkybox(skybox);

    gl.useProgram(shaders.diffuse.program);

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
      associateShadersWithBuffers(models[i], shaders.diffuse.program);
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
  loadTexCube('NightPark', 'jpg')
    .then(texCube => {
      skybox.texCube = texCube;
      initializeCubeMap(skybox, skybox.texCube);
      return ModelLoaderPLY.load('models/cube.ply');
    })
    .then(model => {
      skybox.model = model;
      initializeBufferObject(skybox.model);
      model.center = { x:0, y:0, z:0 };
      model.scale  = { x:10, y:10, z:10 };
      return ModelLoaderPLY.load('models/cube.ply');
    })
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

}());
