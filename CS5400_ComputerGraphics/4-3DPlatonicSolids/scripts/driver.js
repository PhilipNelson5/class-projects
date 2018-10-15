Engine.main = (function(graphics, objs, glUtils) {
  'use strict';

  let canvas = document.getElementById('canvas-main');
  let gl = canvas.getContext('webgl');

  let solids = [];
  solids.push(objs.make_tetrahedron());
  solids[0].center = {x:-.5, y:-.5, z:-.5};
  solids.push(objs.make_octahedron());
  solids.push(objs.make_hexahedron());
  solids[2].center = {x:.5, y:.5, z:.5};


  let buffers = {};
  buffers.vertexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.vertexBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, objs.vertices, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);

  buffers.vertexColorBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.vertexColorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, objs.vertexColors, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);

  for(let i = 0; i < solids.length; ++i){
    solids[i].indexBuffer= gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, solids[i].indexBuffer);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, solids[i].indices, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
  }

  let vertexShaderSource = `
  uniform mat4 matRotateX;
  uniform mat4 matRotateY;
  uniform mat4 matRotateZ;
  uniform mat4 matScale;
  uniform mat4 matTranslate;
  uniform mat4 matProject;
  attribute vec4 aPosition;
  attribute vec4 aColor;
  varying vec4 vColor;
  void main()
  {
    gl_Position = 
      matProject
      * matTranslate
      * matRotateX
      * matRotateY
      * matRotateZ
      * matScale
      * aPosition;

    vColor = aColor;
  }`;

  let vertexShader = glUtils.createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  //let vertexShader = gl.createShader(gl.VERTEX_SHADER);
  //gl.shaderSource(vertexShader, vertexShaderSource);
  //gl.compileShader(vertexShader);
  //console.log(gl.getShaderInfoLog(vertexShader));// for debugging

  let fragmentShaderSource = `
  precision lowp float;
  varying vec4 vColor;
  void main()
  {
    gl_FragColor = vColor;
  }`;

  let fragmentShader = glUtils.createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);
  //let fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
  //gl.shaderSource(fragmentShader, fragmentShaderSource);
  //gl.compileShader(fragmentShader);

  let shaderProgram = glUtils.createProgram(gl, vertexShader, fragmentShader);
  //let shaderProgram = gl.createProgram();
  //gl.attachShader(shaderProgram, vertexShader);
  //gl.attachShader(shaderProgram, fragmentShader);
  //gl.linkProgram(shaderProgram);
  gl.useProgram(shaderProgram);

  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.vertexBuffer);
  let position = gl.getAttribLocation(shaderProgram, 'aPosition');
  gl.vertexAttribPointer(position, 3, gl.FLOAT, false, objs.vertices.BYTES_PER_ELEMENT * 3, 0);
  gl.enableVertexAttribArray(position);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.vertexColorBuffer);

  let color = gl.getAttribLocation(shaderProgram, 'aColor');
  gl.vertexAttribPointer(color, 3, gl.FLOAT, false, objs.vertexColors.BYTES_PER_ELEMENT * 3, 0);
  gl.enableVertexAttribArray(color);

  //------------------------------------------------------------------
  //
  // Scene updates go here.
  //
  //------------------------------------------------------------------
  let th = 0;
  let matRotateXLoc = gl.getUniformLocation(shaderProgram, 'matRotateX');
  let matRotateYLoc = gl.getUniformLocation(shaderProgram, 'matRotateY');
  let matRotateZLoc = gl.getUniformLocation(shaderProgram, 'matRotateZ');
  let matScaleLoc = gl.getUniformLocation(shaderProgram, 'matScale');
  let matTranslateLoc = gl.getUniformLocation(shaderProgram, 'matTranslate');
  let matProjectLoc = gl.getUniformLocation(shaderProgram, 'matProject');
  function update(dt) {
    dt/=1000;
    th += .5*dt;
    gl.clearColor(
      0.3921568627450980392156862745098,
      0.58431372549019607843137254901961,
      0.92941176470588235294117647058824, 1.0);

    gl.clearDepth(1.0);
    gl.depthFunc(gl.LEQUAL);
    gl.enable(gl.DEPTH_TEST);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  }

  //------------------------------------------------------------------
  //
  // Rendering code goes here
  //
  //------------------------------------------------------------------
  function render() {
    gl.uniformMatrix4fv(matProjectLoc, false,
      ///transposeMatrix4x4(graphics.project_parallel(2, 2, 2)));
      transposeMatrix4x4(graphics.project_parallel(1, -1, 1, -1, 1, -1)));
    //transposeMatrix4x4(graphics.project_perspective(1, -1, 1, -1, 1, -1)));
    //transposeMatrix4x4(graphics.project_perspective(1, 1, 1, 1)));

    for(let i = 0; i < solids.length; ++i){
      gl.uniformMatrix4fv(matRotateXLoc, false,
        transposeMatrix4x4(graphics.x_axis_rotate(th)));

      gl.uniformMatrix4fv(matRotateYLoc, false,
        transposeMatrix4x4(graphics.y_axis_rotate(th)));

      gl.uniformMatrix4fv(matRotateZLoc, false,
        transposeMatrix4x4(graphics.z_axis_rotate(0)));

      gl.uniformMatrix4fv(matScaleLoc, false,
        transposeMatrix4x4(graphics.scale(.5, .5, .5)));

      gl.uniformMatrix4fv(matTranslateLoc, false,
        transposeMatrix4x4(graphics.translate(
          solids[i].center.x,
          solids[i].center.y,
          solids[i].center.z)));

      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, solids[i].indexBuffer);
      gl.drawElements(gl.TRIANGLES, solids[i].indices.length, gl.UNSIGNED_SHORT, 0);
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

  /**
   * Transpose a 4x4 matrix
   *
   * @param {array} m - a 4x4 matrix represented as a single array
   * @return a new transposed matrix represented as an array
   */
  function transposeMatrix4x4(m) {
    return [
      m[0], m[4], m[8], m[12],
      m[1], m[5], m[9], m[13],
      m[2], m[6], m[10], m[14],
      m[3], m[7], m[11], m[15]
    ];
  }

  console.log('initializing...');
  requestAnimationFrame(animationLoop);

}(Engine.graphics, Engine.objects, Engine.glUtils));
