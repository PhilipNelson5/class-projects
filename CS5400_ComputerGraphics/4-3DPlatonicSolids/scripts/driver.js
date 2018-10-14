MySample.main = (function() {
  'use strict';

  let canvas = document.getElementById('canvas-main');
  let gl = canvas.getContext('webgl');

  let vertices = new Float32Array([
    //0.0, 0.75, 0.0,
    //0.75, -0.75, 0.0,
    //-0.75, -0.75, 0.0,
    0              , 1   , 0              , // 0
    0              , -1/3, -Math.sqrt(8/9), // 1
    Math.sqrt(2/3) , -1/3, Math.sqrt(2/9) , // 2

    0              , -1/3, -Math.sqrt(8/9), // 3
    Math.sqrt(2/3) , -1/3, Math.sqrt(2/9) , // 4
    -Math.sqrt(2/3), -1/3, Math.sqrt(2/9) , // 5

    Math.sqrt(2/3) , -1/3, Math.sqrt(2/9) , // 6
    -Math.sqrt(2/3), -1/3, Math.sqrt(2/9) , // 7
    0              , 1   , 0              , // 8

    -Math.sqrt(2/3), -1/3, Math.sqrt(2/9) , // 9
    0              , 1   , 0              , // 10
    0              , -1/3, -Math.sqrt(8/9), // 11
  ]);

  let indices = new Uint16Array([
    0, 1, 2,
    3, 4, 5,
    6, 7, 8,
    9, 10, 11,
  ]);

  let vertexColors = new Float32Array([
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,

    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,

    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,

    1.0, 1.0, 0.0,
    1.0, 1.0, 0.0,
    1.0, 1.0, 0.0,
  ]);

  let buffers = {};
  buffers.vertexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.vertexBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  
  buffers.indexBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.indexBuffer);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, indices, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

  buffers.vertexColorBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.vertexColorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, vertexColors, gl.STATIC_DRAW);
  gl.bindBuffer(gl.ARRAY_BUFFER, null);

  let vertexShaderSource = `
    attribute vec4 aPosition;
    attribute vec4 aColor;
    varying vec4 vColor;
    void main()
      {
        gl_Position = aPosition;
        vColor = aColor;
      }`;
  let vertexShader = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vertexShader, vertexShaderSource);
  gl.compileShader(vertexShader);
  console.log(gl.getShaderInfoLog(vertexShader));// for debugging

  let fragmentShaderSource = `
    precision lowp float;
    varying vec4 vColor;
    void main()
    {
      gl_FragColor = vColor;
    }`;
  let fragmentShader = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fragmentShader, fragmentShaderSource);
  gl.compileShader(fragmentShader);

  let shaderProgram = gl.createProgram();
  gl.attachShader(shaderProgram, vertexShader);
  gl.attachShader(shaderProgram, fragmentShader);
  gl.linkProgram(shaderProgram);
  gl.useProgram(shaderProgram);

  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.vertexBuffer);
  let position = gl.getAttribLocation(shaderProgram, 'aPosition');
  gl.vertexAttribPointer(position, 3, gl.FLOAT, false, vertices.BYTES_PER_ELEMENT * 3, 0);
  gl.enableVertexAttribArray(position);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.vertexColorBuffer);
  let color = gl.getAttribLocation(shaderProgram, 'aColor');
  gl.vertexAttribPointer(color, 3, gl.FLOAT, false,vertexColors.BYTES_PER_ELEMENT * 3, 0);
  gl.enableVertexAttribArray(color);

  // This sets which buffer to use for the draw call in the render function.
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.indexBuffer);
  //------------------------------------------------------------------
  //
  // Scene updates go here.
  //
  //------------------------------------------------------------------
  function update() {
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
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, buffers.indexBuffer);
    gl.drawElements(gl.TRIANGLES, indices.length, gl.UNSIGNED_SHORT, 0);
  }

  //------------------------------------------------------------------
  //
  // This is the animation loop.
  //
  //------------------------------------------------------------------
  function animationLoop(time) {

    update();
    render();

    //requestAnimationFrame(animationLoop);
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

}());
