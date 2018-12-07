'use strict';

//------------------------------------------------------------------
//
// Helper function used to create a shader
//
//------------------------------------------------------------------
function createShader(gl, type, source){
  let shader = gl.createShader(type);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (gl.getShaderParameter(shader, gl.COMPILE_STATUS)){
    return shader;
  }

  console.error("ERROR - createShader: ", gl.getShaderInfoLog(shader));
  gl.deleteShader(shader);
}

//------------------------------------------------------------------
//
// Helper function used to create a program
//
//------------------------------------------------------------------
function createProgram(gl, vertexShader, fragmentShader) {
  let program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (gl.getProgramParameter(program, gl.LINK_STATUS)) {
    return program;
  }

  console.error("ERROR - createProgram: ", gl.getProgramInfoLog(program));
  gl.deleteProgram(program);
}

//------------------------------------------------------------------
//
// Helper function used to create a buffer
//
//------------------------------------------------------------------
function createBuffer(gl, data, type, usage){
  let buffer = gl.createBuffer();
  gl.bindBuffer(type, buffer);
  gl.bufferData(type, data, usage);
  gl.bindBuffer(type, null);
  return buffer;
}
