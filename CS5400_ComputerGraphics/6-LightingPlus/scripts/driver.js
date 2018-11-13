Engine.main = (function() {
  'use strict';

  let environment = {};
  let models = [];
  let skybox = {};
  let shaders = { vecLightPos:[], vecLightColor:[] };
  let previousTime = performance.now();
  let refractiveIndex = 1.0;
  let shiny = 1;
  let phongVal = 80;
  let reflectVal = 20;
  let lightPos = [
    [10, 10, 10, 1],
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

  let reflectChecked = document.getElementById('radioReflect');

  let refractionSlider = document.getElementById('sliderRefraction');
  let refractionSliderLabel = document.getElementById('sliderRefractionLabel');

  let shinySlider = document.getElementById('sliderShiny');
  let shinySliderLabel = document.getElementById('sliderShinyLabel');

  let diffuseReflectSlider = document.getElementById('sliderDiffuseReflect');
  let diffuseReflectSliderLabel = document.getElementById('sliderDiffuseReflectLabel');

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

    environment.vEye = new Float32Array([0.0, 0.0, 1.5, 1.0]);
    environment.matView = translate(
      -environment.vEye[0],
      -environment.vEye[1],
      -environment.vEye[2]);

  }

  //------------------------------------------------------------------
  //
  // Initialize a CubeMap for a skybox
  //
  //------------------------------------------------------------------
  function initializeCubeMap(sb, texCube){
    sb.cubeMap = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, sb.cubeMap);
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

  //------------------------------------------------------------------
  //
  // Initialize the buffers for a model:
  //   vertex, normal, color, index
  //
  //------------------------------------------------------------------
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
      // --------------------
      // Diffuse Shader
      // --------------------
      loadFileFromServer('shaders/diffuse.vs')
        .then(source => {
          shaders.diffuse = {};
          shaders.specular = {};
          shaders.diffuse.vShader = createShader(gl, gl.VERTEX_SHADER, source);
          shaders.specular.vShader = shaders.diffuse.vShader;

          return loadFileFromServer('shaders/diffuse.frag');
        })
        .then(source => {
          shaders.diffuse.fShader = createShader(gl, gl.FRAGMENT_SHADER, source);

          shaders.diffuse.program = createProgram(
            gl, shaders.diffuse.vShader, shaders.diffuse.fShader);

          // Get Uniform Locations
          shaders.diffuse.uAspect     = gl.getUniformLocation(shaders.diffuse.program, 'uAspect');
          shaders.diffuse.uProjection = gl.getUniformLocation(shaders.diffuse.program, 'uProjection');
          shaders.diffuse.uView       = gl.getUniformLocation(shaders.diffuse.program, 'uView');
          shaders.diffuse.uModel      = gl.getUniformLocation(shaders.diffuse.program, 'uModel');
          shaders.diffuse.uNormal     = gl.getUniformLocation(shaders.diffuse.program, 'uNormal');

          shaders.diffuse.uLightPos   = [];
          shaders.diffuse.uLightColor = [];
          for (let i = 0; i < lightPos.length; ++i) {
            shaders.diffuse.uLightPos[i]   = gl.getUniformLocation(shaders.diffuse.program, `uLightPos${i}`);
            shaders.diffuse.uLightColor[i] = gl.getUniformLocation(shaders.diffuse.program, `uLightColor${i}`);
          }

          // Get Attribute Locations
          shaders.diffuse.aPosition = gl.getAttribLocation(shaders.diffuse.program, 'aPosition');
          shaders.diffuse.aNormal   = gl.getAttribLocation(shaders.diffuse.program, 'aNormal');
          shaders.diffuse.aColor    = gl.getAttribLocation(shaders.diffuse.program, 'aColor');

          // --------------------
          // Specular Shader
          // --------------------
          return loadFileFromServer('shaders/specular.frag');
        })
        .then(source => {
          shaders.specular.fShader = createShader(gl, gl.FRAGMENT_SHADER, source);

          shaders.specular.program = createProgram(
            gl, shaders.specular.vShader, shaders.specular.fShader);

          // Get Uniform Locations
          shaders.specular.uAspect     = gl.getUniformLocation(shaders.specular.program, 'uAspect');
          shaders.specular.uProjection = gl.getUniformLocation(shaders.specular.program, 'uProjection');
          shaders.specular.uView       = gl.getUniformLocation(shaders.specular.program, 'uView');
          shaders.specular.uModel      = gl.getUniformLocation(shaders.specular.program, 'uModel');
          shaders.specular.uNormal     = gl.getUniformLocation(shaders.specular.program, 'uNormal');
          shaders.specular.uEye        = gl.getUniformLocation(shaders.specular.program, 'uEye');
          shaders.specular.uShine      = gl.getUniformLocation(shaders.specular.program, 'uShine');

          shaders.specular.uLightPos   = [];
          shaders.specular.uLightColor = [];
          for (let i = 0; i < lightPos.length; ++i) {
            shaders.specular.uLightPos[i]   = gl.getUniformLocation(shaders.specular.program, `uLightPos${i}`);
            shaders.specular.uLightColor[i] = gl.getUniformLocation(shaders.specular.program, `uLightColor${i}`);
          }

          // Get Attribute Locations
          shaders.specular.aPosition = gl.getAttribLocation(shaders.specular.program, 'aPosition');
          shaders.specular.aNormal   = gl.getAttribLocation(shaders.specular.program, 'aNormal');
          shaders.specular.aColor    = gl.getAttribLocation(shaders.specular.program, 'aColor');

          // --------------------
          // Phong Reflect Shader
          // --------------------
          return loadFileFromServer('shaders/phongReflection.vs')
        })
        .then(source => {
          shaders.phongReflect = {};
          shaders.phongReflect.vShader = createShader(gl, gl.VERTEX_SHADER, source);

          return loadFileFromServer('shaders/phongReflection.frag')
        })
        .then(source => {
          shaders.phongReflect.fShader = createShader(gl, gl.FRAGMENT_SHADER, source);

          shaders.phongReflect.program = createProgram(
            gl, shaders.phongReflect.vShader, shaders.phongReflect.fShader);

          // Get Uniform Locations
          shaders.phongReflect.uAspect          = gl.getUniformLocation(shaders.phongReflect.program, 'uAspect');
          shaders.phongReflect.uProjection      = gl.getUniformLocation(shaders.phongReflect.program, 'uProjection');
          shaders.phongReflect.uView            = gl.getUniformLocation(shaders.phongReflect.program, 'uView');
          shaders.phongReflect.uModel           = gl.getUniformLocation(shaders.phongReflect.program, 'uModel');
          shaders.phongReflect.uNormal          = gl.getUniformLocation(shaders.phongReflect.program, 'uNormal');
          shaders.phongReflect.uEye             = gl.getUniformLocation(shaders.phongReflect.program, 'uEye');
          shaders.phongReflect.uShine           = gl.getUniformLocation(shaders.phongReflect.program, 'uShine');
          shaders.phongReflect.uSampler         = gl.getUniformLocation(shaders.phongReflect.program, 'uSampler');
          shaders.phongReflect.uReflection      = gl.getUniformLocation(shaders.phongReflect.program, 'uReflection');
          shaders.phongReflect.uRefractiveIndex = gl.getUniformLocation(shaders.phongReflect.program, 'uRefractiveIndex');
          shaders.phongReflect.uPercentPhong    = gl.getUniformLocation(shaders.phongReflect.program, 'uPercentPhong');

          shaders.phongReflect.uLightPos   = [];
          shaders.phongReflect.uLightColor = [];
          for (let i = 0; i < lightPos.length; ++i) {
            shaders.phongReflect.uLightPos[i]   = gl.getUniformLocation(shaders.phongReflect.program, `uLightPos${i}`);
            shaders.phongReflect.uLightColor[i] = gl.getUniformLocation(shaders.phongReflect.program, `uLightColor${i}`);
          }

          // Get Attribute Locations
          shaders.phongReflect.aPosition = gl.getAttribLocation(shaders.phongReflect.program, 'aPosition');
          shaders.phongReflect.aNormal   = gl.getAttribLocation(shaders.phongReflect.program, 'aNormal');
          shaders.phongReflect.aColor    = gl.getAttribLocation(shaders.phongReflect.program, 'aColor');

          // --------------------
          // Environment Mapping Shader
          // --------------------
          return loadFileFromServer('shaders/environmentMapping.vs')
        })
        .then(source => {
          shaders.envMap = {};
          shaders.envMap.vShader = createShader(gl, gl.VERTEX_SHADER, source);

          return loadFileFromServer('shaders/environmentMapping.frag')
        })
        .then(source => {
          shaders.envMap.fShader = createShader(gl, gl.FRAGMENT_SHADER, source);

          shaders.envMap.program = createProgram(
            gl, shaders.envMap.vShader, shaders.envMap.fShader);

          // Get Uniform Locations
          shaders.envMap.uAspect     = gl.getUniformLocation(shaders.envMap.program, 'uAspect');
          shaders.envMap.uProjection = gl.getUniformLocation(shaders.envMap.program, 'uProjection');
          shaders.envMap.uView       = gl.getUniformLocation(shaders.envMap.program, 'uView');
          shaders.envMap.uModel      = gl.getUniformLocation(shaders.envMap.program, 'uModel');
          shaders.envMap.uNormal     = gl.getUniformLocation(shaders.envMap.program, 'uNormal');
          shaders.envMap.uEye        = gl.getUniformLocation(shaders.envMap.program, 'uEye');
          shaders.envMap.uSampler    = gl.getUniformLocation(shaders.envMap.program, 'uSampler');
          shaders.envMap.uReflection = gl.getUniformLocation(shaders.envMap.program, 'uReflection');
          shaders.envMap.uRefractiveIndex = gl.getUniformLocation(shaders.envMap.program, 'uRefractiveIndex');

          // Get Attribute Locations
          shaders.envMap.aPosition = gl.getAttribLocation(shaders.envMap.program, 'aPosition');
          shaders.envMap.aNormal   = gl.getAttribLocation(shaders.envMap.program, 'aNormal');

          // --------------------
          // Skybox Shader
          // --------------------
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
            gl, shaders.skybox.vShader, shaders.skybox.fShader);

          // Get Uniform Locations
          shaders.skybox.uAspect = gl.getUniformLocation(shaders.skybox.program, 'uAspect');
          shaders.skybox.uModel = gl.getUniformLocation(shaders.skybox.program, 'uModel');
          shaders.skybox.uProjection = gl.getUniformLocation(shaders.skybox.program, 'uProjection');
          shaders.skybox.uSampler = gl.getUniformLocation(shaders.skybox.program, 'uSampler');
          shaders.skybox.uView = gl.getUniformLocation(shaders.skybox.program, 'uView');

          // Get Attribute Locations
          shaders.skybox.aPosition = gl.getAttribLocation(shaders.skybox.program, 'aPosition');

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
  // Prepare WegGL settings like the clear color, depth buffer, etc.
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
  // Render a skybox
  //
  //------------------------------------------------------------------
  function renderSkybox(sb){

    // Use the skybox shader program
    gl.useProgram(shaders.skybox.program);

    // Activate the cubeMap texture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, sb.cubeMap);

    // Setup the cubeSampler
    gl.uniform1i(shaders.skybox.uSampler, 0);

    // setup uniforms
    gl.uniformMatrix4fv(shaders.skybox.uAspect, false, transposeMatrix4x4(environment.matAspect));
    gl.uniformMatrix4fv(shaders.skybox.uProjection, false, transposeMatrix4x4(environment.matProjection));
    gl.uniformMatrix4fv(shaders.skybox.uModel, false, transposeMatrix4x4(scale(sb.model.scale.x, sb.model.scale.y, sb.model.scale.z)));
    gl.uniformMatrix4fv(shaders.skybox.uView, false, transposeMatrix4x4(environment.matView));

    // setup position attribute
    gl.bindBuffer(gl.ARRAY_BUFFER, sb.model.vertexBuffer);
    gl.vertexAttribPointer(shaders.skybox.aPosition, 3, gl.FLOAT, false, sb.model.vertices.BYTES_PER_ELEMENT * 3, 0);
    gl.enableVertexAttribArray(shaders.skybox.aPosition);

    // Index buffer
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, sb.model.indexBuffer);

    // draw the box
    gl.drawElements(gl.TRIANGLES, sb.model.indices.length, sb.model.indices_type, 0);
  }

  //------------------------------------------------------------------
  //
  // Render a model with diffuse lighting
  //
  //------------------------------------------------------------------
  function renderDiffuse(model)
  {
    gl.useProgram(shaders.diffuse.program);

    // Set uniforms
    gl.uniformMatrix4fv(shaders.diffuse.uAspect, false, transposeMatrix4x4(environment.matAspect));
    gl.uniformMatrix4fv(shaders.diffuse.uModel, false, transposeMatrix4x4(model.matModel));
    gl.uniformMatrix4fv(shaders.diffuse.uNormal, false, model.matNormal);
    gl.uniformMatrix4fv(shaders.diffuse.uProjection, false, transposeMatrix4x4(environment.matProjection));
    gl.uniformMatrix4fv(shaders.diffuse.uView, false, transposeMatrix4x4(environment.matView));

    for (let i = 0; i < lightPos.length; ++i){
      gl.uniform4fv(shaders.diffuse.uLightPos[i], lightPos[i]);
      gl.uniform4fv(shaders.diffuse.uLightColor[i], lightOn[i] ? lightColor[i] : [0, 0, 0, 1]);
    }

    // aPosition
    gl.bindBuffer(gl.ARRAY_BUFFER, model.vertexBuffer);
    gl.vertexAttribPointer(shaders.diffuse.aPosition, 3, gl.FLOAT, false, model.vertices.BYTES_PER_ELEMENT * 3, 0);
    gl.enableVertexAttribArray(shaders.diffuse.aPosition);

    // aNormal
    gl.bindBuffer(gl.ARRAY_BUFFER, model.vertexNormalBuffer);
    gl.vertexAttribPointer(shaders.diffuse.aNormal, 3, gl.FLOAT, false, model.normals.BYTES_PER_ELEMENT * 3, 0);
    gl.enableVertexAttribArray(shaders.diffuse.aNormal);

    // aColor
    gl.bindBuffer(gl.ARRAY_BUFFER, model.vertexColorBuffer);
    gl.vertexAttribPointer(shaders.diffuse.aColor, 3, gl.FLOAT, false, model.vertexColors.BYTES_PER_ELEMENT * 3, 0);
    gl.enableVertexAttribArray(shaders.diffuse.aColor);

    // index buffer
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, model.indexBuffer);

    // draw model
    gl.drawElements(gl.TRIANGLES, model.indices.length, model.indices_type, 0);
  }

  //------------------------------------------------------------------
  //
  // Render a model with diffuse lighting
  //
  //------------------------------------------------------------------
  function renderSpecular(model)
  {
    gl.useProgram(shaders.specular.program);

    // Set uniforms
    gl.uniformMatrix4fv(shaders.specular.uAspect, false, transposeMatrix4x4(environment.matAspect));
    gl.uniformMatrix4fv(shaders.specular.uModel, false, transposeMatrix4x4(model.matModel));
    gl.uniformMatrix4fv(shaders.specular.uNormal, false, model.matNormal);
    gl.uniformMatrix4fv(shaders.specular.uProjection, false, transposeMatrix4x4(environment.matProjection));
    gl.uniformMatrix4fv(shaders.specular.uView, false, transposeMatrix4x4(environment.matView));
    gl.uniform4fv(shaders.specular.uShine, model.specularMaterial);
    gl.uniform4fv(shaders.specular.uEye, environment.vEye);

    for (let i = 0; i < lightPos.length; ++i){
      gl.uniform4fv(shaders.specular.uLightPos[i], lightPos[i]);
      gl.uniform4fv(shaders.specular.uLightColor[i], lightOn[i] ? lightColor[i] : [0, 0, 0, 1]);
    }

    // aPosition
    gl.bindBuffer(gl.ARRAY_BUFFER, model.vertexBuffer);
    gl.vertexAttribPointer(shaders.specular.aPosition, 3, gl.FLOAT, false, model.vertices.BYTES_PER_ELEMENT * 3, 0);
    gl.enableVertexAttribArray(shaders.specular.aPosition);

    // aNormal
    gl.bindBuffer(gl.ARRAY_BUFFER, model.vertexNormalBuffer);
    gl.vertexAttribPointer(shaders.specular.aNormal, 3, gl.FLOAT, false, model.normals.BYTES_PER_ELEMENT * 3, 0);
    gl.enableVertexAttribArray(shaders.specular.aNormal);

    // aColor
    gl.bindBuffer(gl.ARRAY_BUFFER, model.vertexColorBuffer);
    gl.vertexAttribPointer(shaders.specular.aColor, 3, gl.FLOAT, false, model.vertexColors.BYTES_PER_ELEMENT * 3, 0);
    gl.enableVertexAttribArray(shaders.specular.aColor);

    // index buffer
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, model.indexBuffer);

    // draw model
    gl.drawElements(gl.TRIANGLES, model.indices.length, model.indices_type, 0);
  }

  //------------------------------------------------------------------
  //
  // Render a model with environment mapping
  //
  //------------------------------------------------------------------
  let vEye = new Float32Array([0.0, 0.0, 1.5]);
  function renderEnvMap(model, sb)
  {
    gl.useProgram(shaders.envMap.program);

    // Activate the cubeMap texture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, sb.cubeMap);

    // Setup the cubeSampler
    gl.uniform1i(shaders.envMap.uSampler, 0);

    // Set uniforms
    gl.uniformMatrix4fv(shaders.envMap.uAspect, false, transposeMatrix4x4(environment.matAspect));
    gl.uniformMatrix4fv(shaders.envMap.uModel, false, transposeMatrix4x4(model.matModel));
    gl.uniformMatrix4fv(shaders.envMap.uNormal, false, model.matNormal);
    gl.uniformMatrix4fv(shaders.envMap.uProjection, false, transposeMatrix4x4(environment.matProjection));
    gl.uniformMatrix4fv(shaders.envMap.uView, false, transposeMatrix4x4(environment.matView));
    gl.uniform3fv(shaders.envMap.uEye, vEye);

    // true - reflection, false - refraction
    gl.uniform1i(shaders.envMap.uReflection, reflectChecked.checked);
    gl.uniform1f(shaders.envMap.uRefractiveIndex, refractiveIndex);

    // aPosition
    gl.bindBuffer(gl.ARRAY_BUFFER, model.vertexBuffer);
    gl.vertexAttribPointer(shaders.specular.aPosition, 3, gl.FLOAT, false, model.vertices.BYTES_PER_ELEMENT * 3, 0);
    gl.enableVertexAttribArray(shaders.specular.aPosition);

    // aNormal
    gl.bindBuffer(gl.ARRAY_BUFFER, model.vertexNormalBuffer);
    gl.vertexAttribPointer(shaders.specular.aNormal, 3, gl.FLOAT, false, model.normals.BYTES_PER_ELEMENT * 3, 0);
    gl.enableVertexAttribArray(shaders.specular.aNormal);

    // index buffer
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, model.indexBuffer);

    // draw model
    gl.drawElements(gl.TRIANGLES, model.indices.length, model.indices_type, 0);
  }

  function renderPhongReflection(model, sb, phongPercent){
    gl.useProgram(shaders.phongReflect.program);

    // Activate the cubeMap texture
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_CUBE_MAP, sb.cubeMap);

    // Setup the cubeSampler
    gl.uniform1i(shaders.phongReflect.uSampler, 0);

    // Set uniforms
    gl.uniformMatrix4fv(shaders.phongReflect.uAspect,     false, transposeMatrix4x4(environment.matAspect));
    gl.uniformMatrix4fv(shaders.phongReflect.uModel,      false, transposeMatrix4x4(model.matModel));
    gl.uniformMatrix4fv(shaders.phongReflect.uNormal,     false, model.matNormal);
    gl.uniformMatrix4fv(shaders.phongReflect.uProjection, false, transposeMatrix4x4(environment.matProjection));
    gl.uniformMatrix4fv(shaders.phongReflect.uView,       false, transposeMatrix4x4(environment.matView));
    gl.uniform3fv(shaders.phongReflect.uEye, vEye);
    gl.uniform4fv(shaders.phongReflect.uShine, model.specularMaterial);

    for (let i = 0; i < lightPos.length; ++i){
      gl.uniform4fv(shaders.phongReflect.uLightPos[i], lightPos[i]);
      gl.uniform4fv(shaders.phongReflect.uLightColor[i], lightOn[i] ? lightColor[i] : [0, 0, 0, 1]);
    }

    // true - reflection, false - refraction
    gl.uniform1i(shaders.phongReflect.uReflection, reflectChecked.checked);
    gl.uniform1f(shaders.phongReflect.uRefractiveIndex, refractiveIndex);
    gl.uniform1f(shaders.phongReflect.uPercentPhong, phongPercent);

    // aPosition
    gl.bindBuffer(gl.ARRAY_BUFFER, model.vertexBuffer);
    gl.vertexAttribPointer(shaders.specular.aPosition, 3, gl.FLOAT, false, model.vertices.BYTES_PER_ELEMENT * 3, 0);
    gl.enableVertexAttribArray(shaders.specular.aPosition);

    // aNormal
    gl.bindBuffer(gl.ARRAY_BUFFER, model.vertexNormalBuffer);
    gl.vertexAttribPointer(shaders.specular.aNormal, 3, gl.FLOAT, false, model.normals.BYTES_PER_ELEMENT * 3, 0);
    gl.enableVertexAttribArray(shaders.specular.aNormal);

    // index buffer
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, model.indexBuffer);

    // draw model
    gl.drawElements(gl.TRIANGLES, model.indices.length, model.indices_type, 0);
  }

  //------------------------------------------------------------------
  //
  // The scene updates
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

      //lightPos[1][0] = 5 * Math.cos(th);
      //lightPos[1][2] = 5 * Math.sin(th);

      //lightPos[2][0] = 5 * Math.cos(th/10);
      //lightPos[2][1] = 5 * Math.sin(th/10);
      //th += dt / 1000 / 2;

      lightOn[0] = lightCheckBox[0].checked;
      lightOn[1] = lightCheckBox[1].checked;
      lightOn[2] = lightCheckBox[2].checked;

      lightColor[0] = hexToRgba(lightColorPicker[0].value);
      lightColor[1] = hexToRgba(lightColorPicker[1].value);
      lightColor[2] = hexToRgba(lightColorPicker[2].value);
    }

    refractionSliderLabel.innerText = refractionSlider.value;
    refractiveIndex = parseFloat(refractionSlider.value);

    shinySliderLabel.innerText = shinySlider.value;
    models[1].specularMaterial[3] = parseFloat(shinySlider.value);

    phongVal = parseFloat(diffuseReflectSlider.value);
    reflectVal = 100 - phongVal;
    diffuseReflectSliderLabel.innerText = `${phongVal} / ${reflectVal}`;

  }

  //------------------------------------------------------------------
  //
  // Rendering the models
  //
  //------------------------------------------------------------------
  function render() {
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    renderSkybox(skybox);


    renderDiffuse(models[0]);
    renderSpecular(models[1]);
    renderEnvMap(models[2], skybox);
    //renderDiffuse(models[3]);
    renderPhongReflection(models[3], skybox, phongVal/100);
  }

  //------------------------------------------------------------------
  //
  // Animation loop.
  //
  //------------------------------------------------------------------
  function animationLoop(time) {
    let elapsedTime = previousTime - time;
    previousTime = time;

    update(elapsedTime);
    render();

    requestAnimationFrame(animationLoop);
  }

  //------------------------------------------------------------------
  //
  // Load and initialize everything
  //
  //------------------------------------------------------------------
  console.log('initializing...');
  console.log('    Loading model');
  //ModelLoaderPLY.load('models/cube.ply')
  //ModelLoaderPLY.load('models/dodecahedron.ply')
  //ModelLoaderPLY.load('models/bunny.ply')
  //loadTexCube('NightPark', 'jpg')
  loadTexCube('alps', 'jpg')
    .then(texCube => {
      skybox.texCube = texCube;
      initializeCubeMap(skybox, skybox.texCube);
      return ModelLoaderPLY.load('models/cube.ply');
    })
    .then(model => {
      model.center = { x:0, y:0, z:0 };
      model.scale  = { x:10, y:10, z:10 };
      skybox.model = model;
      initializeBufferObject(skybox.model);
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
        z:30,
      };
      model.scale = {
        x:5,
        y:1,
        z:5,
      };
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

  // diffuse galleon
  let file = 'models/galleon.ply'
  ModelLoaderPLY.load(file)
    .then(model => {
      model.center.x = 1.5;
      model.rotation = {
        x: -Math.PI/2,
        y: Math.PI/2,
        z: 0
      };
      model.rotationRate = {
        x: 0,
        y: Math.PI / 4 / 1000,
        z: 0
      };
      initializeBufferObject(model);
      models.push(model);
    })
    .catch(error => console.error('[ERROR] ' + error));

  // specular galleon
  ModelLoaderPLY.load(file)
    .then(model => {
      model.specularMaterial = new Float32Array([0.5, 0.5, 0.5, 2.0]);
      model.center.x = -1.5;

      model.rotation = {
        x: -Math.PI/2,
        y: Math.PI/2,
        z: 0
      };
      model.rotationRate = {
        x: 0,
        y: Math.PI / 4 / 1000,
        z: 0
      };
      initializeBufferObject(model);
      models.push(model);
    })
    .catch(error => console.error('[ERROR] ' + error));

  // environment mapping galleon
  ModelLoaderPLY.load(file)
    .then(model => {
      model.center.x = 0.0;

      model.rotation = {
        x: -Math.PI/2,
        y: Math.PI/2,
        z: 0
      };
      model.rotationRate = {
        x: 0,
        y: Math.PI / 4 / 1000,
        z: 0
      };
      initializeBufferObject(model);
      models.push(model);
    })
    .catch(error => console.error('[ERROR] ' + error));

  // diffuse and environment mapping galleon
  ModelLoaderPLY.load(file)
    .then(model => {
      model.specularMaterial = new Float32Array([0.5, 0.5, 0.5, 2.0]);
      model.center.y = -1.5;

      model.rotation = {
        x: -Math.PI/2,
        y: Math.PI/2,
        z: 0
      };
      model.rotationRate = {
        x: 0,
        y: Math.PI / 4 / 1000,
        z: 0
      };
      initializeBufferObject(model);
      models.push(model);
    })
    .catch(error => console.error('[ERROR] ' + error));

}());
