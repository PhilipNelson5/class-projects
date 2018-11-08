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

/**
 * create a matrix to rotate about the x axis
 *
 * @param {number} theta - the angle in radials to rotate
 * @return {array} The x axis rotation matrix
 */
function x_axis_rotate(theta){
  let cos = Math.cos(theta);
  let sin = Math.sin(theta);
  return [
    1,  0,   0,    0,
    0, cos, -sin,  0,
    0, sin,  cos,  0,
    0,  0,   0,    1
  ];
}

/**
 * create a matrix to rotate about the y axis
 *
 * @param {number} theta - the angle in radials to rotate
 * @return {array} The y axis rotation matrix
 */
function y_axis_rotate(theta){
  let cos = Math.cos(theta);
  let sin = Math.sin(theta);
  return [
    cos,  0,  sin,  0,
    0,    1,   0,   0,
    -sin, 0,  cos,  0,
    0,    0,   0,   1
  ];
}

/**
 * create a matrix to rotate about the z axis
 *
 * @param {number} theta - the angle in radials to rotate
 * @return {array} The z axis rotation matrix
 */
function z_axis_rotate(theta){
  let cos = Math.cos(theta);
  let sin = Math.sin(theta);
  return [
    cos, -sin, 0,  0,
    sin,  cos, 0,  0,
    0,     0,  1,  0,
    0,     0,  0,  1
  ];
}

/**
 * create a matrix to scale an object about the origin
 *
 * @param {number} sx - scale in x direction
 * @param {number} sy - scale in y direction
 * @param {number} sz - scale in z direction
 * @return {array} The scaling matrix
 */
function scale(sx, sy, sz){
  return [
    sx, 0,  0,  0,
    0, sy,  0,  0,
    0,  0, sz,  0,
    0,  0,  0,  1,
  ];
}

/**
 * create a matrix to translate an object
 *
 * @param {number} dx - translation in the x direction
 * @param {number} dy - translation in the y direction
 * @param {number} dz - translation in the z direction
 * @return {array} The translation matrix
 */
function translate(dx, dy, dz){
  return [
    1,  0,  0,  dx,
    0,  1,  0,  dy,
    0,  0,  1,  dz,
    0,  0,  0,  1,
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
    scale,  0.0,   0.0,  0.0,
    0.0,   scale,  0.0,  0.0,
    0.0,    0.0, -(far + near) / (far - near), -(2 * far * near) / (far - near),
    0.0,    0.0,   -1,    0
  ];
}
