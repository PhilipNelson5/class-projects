// ------------------------------------------------------------------
//
// This is the glUtils object.
// It provides wrappers for webGL operations
//
// ------------------------------------------------------------------
Engine.glUtils = (function() {
  'use strict';

  function createShader(gl, type, source){
    let shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (gl.getShaderParameter(shader, gl.COMPILE_STATUS)){
      return shader;
    }

    console.log("ERROR - createShader: ", gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
  }

  function createProgram(gl, vertexShader, fragmentShader) {
    let program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    if (gl.getProgramParameter(program, gl.LINK_STATUS)) {
      return program;
    }

    console.log("ERROR - createProgram: ", gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
  }

  function createBuffer(gl, data, type, usage){
    let buffer = gl.createBuffer();
    gl.bindBuffer(type, buffer);
    gl.bufferData(type, data, usage);
    gl.bindBuffer(type, null);
    return buffer;
  }

  const api = {
    createShader,
    createProgram,
    createBuffer,
  };

  console.log('utils...');
  return api;
}());
