// ------------------------------------------------------------------
//
// This is the graphics object.
// It provides functions to create 3D transformations
//
// ------------------------------------------------------------------
Engine.graphics = (function() {
  'use strict';

  /**
   * multiplies an arbitrary number of matrices together
   *
   * @param {array} matrices - 2 or more matrices
   * @return {array} a composite matrix
   */
  function mat4Multiply(...matricies){
    return matricies.reduce((a, b) => {
      return [
        b[0]  * a[0] + b[1]  * a[4] + b[2]  * a[8]  + b[3]  * a[12],
        b[0]  * a[1] + b[1]  * a[5] + b[2]  * a[9]  + b[3]  * a[13],
        b[0]  * a[2] + b[1]  * a[6] + b[2]  * a[10] + b[3]  * a[14],
        b[0]  * a[3] + b[1]  * a[7] + b[2]  * a[11] + b[3]  * a[15],
        b[4]  * a[0] + b[5]  * a[4] + b[6]  * a[8]  + b[7]  * a[12],
        b[4]  * a[1] + b[5]  * a[5] + b[6]  * a[9]  + b[7]  * a[13],
        b[4]  * a[2] + b[5]  * a[6] + b[6]  * a[10] + b[7]  * a[14],
        b[4]  * a[3] + b[5]  * a[7] + b[6]  * a[11] + b[7]  * a[15],
        b[8]  * a[0] + b[9]  * a[4] + b[10] * a[8]  + b[11] * a[12],
        b[8]  * a[1] + b[9]  * a[5] + b[10] * a[9]  + b[11] * a[13],
        b[8]  * a[2] + b[9]  * a[6] + b[10] * a[10] + b[11] * a[14],
        b[8]  * a[3] + b[9]  * a[7] + b[10] * a[11] + b[11] * a[15],
        b[12] * a[0] + b[13] * a[4] + b[14] * a[8]  + b[15] * a[12],
        b[12] * a[1] + b[13] * a[5] + b[14] * a[9]  + b[15] * a[13],
        b[12] * a[2] + b[13] * a[6] + b[14] * a[10] + b[15] * a[14],
        b[12] * a[3] + b[13] * a[7] + b[14] * a[11] + b[15] * a[15],
      ];
    });
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

  /**
   * create a matrix to perform parallel projection
   *
   * @return {array} The parallel projection matrix
   */
  function project_parallel(r, t, n, f){
    return [
      1/r,     0,       0,          0,
      0,      1/t,      0,          0,
      0,       0,    -2/(f-n), -(f+n)/(f-n),
      0,       0,       0,          1
    ];
  }

  /**
   * create a matrix to perform perspective projection
   *
   * @param {number} r - the offset to the right
   * @param {number} t - the offset to the top
   * @param {number} n - distance to the near clipping plane
   * @param {number} f - distance to the far clipping plane
   * @return {array} The perspective projection matrix
   */
  function project_perspective(r, t, n, f){
    return [
      n/r,       0,           0,           0,
      0,        n/t,          0,           0,
      0,         0,     -(f+n)/(f-n), -2*f*n/(f-n),
      0,         0,          -1,           0
    ];
  }

  const api = {
    x_axis_rotate,
    y_axis_rotate,
    z_axis_rotate,
    scale,
    translate,
    project_parallel,
    project_perspective,
    mat4Multiply,
  };

  console.log('graphics...');
  return api;
}());
