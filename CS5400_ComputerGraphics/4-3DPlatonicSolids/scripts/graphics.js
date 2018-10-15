// ------------------------------------------------------------------
//
// This is the graphics object.
// It provides functions to create 3D transformations
//
// ------------------------------------------------------------------
Engine.graphics = (function() {
  'use strict';

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

  function scale(sx, sy, sz){
    return [
      sx, 0,  0,  0,
      0, sy,  0,  0,
      0,  0, sz,  0,
      0,  0,  0,  1,
    ];
  }

  function translate(dx, dy, dz){
    return [
      1,  0,  0,  dx,
      0,  1,  0,  dy,
      0,  0,  1,  dz,
      0,  0,  0,  1,
    ];
  }

  //function project_parallel(w, h, d){
    //return [
      //2/w,    0,      0,    0,
      //0,    -2/h,     0,    0,
      //0,      0,    -2/d,   0,
      //0,      0,      0,    1
    //];
  //}

  function project_parallel(r, l, t, b, n, f){
    return [
      2/(r-l),    0,        0,    -(l+r)/(r-l),
      0,       2/(t-b),     0,    -(t+b)/(t-b),
      0,          0,    -2/(f-n), -(f+n)/(f-n),
      0,          0,        0,          1
    ];
  }

  function project_perspective(r, l, t, b, n, f){
    return [
      2*n/(r-l),    0,      (r+l)/(r-l),      0,
      0,        2*n/(t-b),  (t+b)/(t-b),      0,
      0,            0,     -(f+n)/(f-n), -(2*f*n)/(f-n),
      0,            0,          -1,           0
    ];
  }

  //function project_perspective(r, t, n, f){
    //return [
      //n/r,       0,          0,            0,
      //0,        n/t,         0,            0,
      //0,         0,    -(f+n)/(f-n), -2*f*n/(f-n),
      //0,         0,         -1,            0
    //];
  //}

  //function project_perspective(r, t, n, f){
    //return [
      //1/r,       0,          0,            0,
      //0,        1/t,         0,            0,
      //0,         0,       -2/(f-n), -(f+n)/(f-n),
      //0,         0,          0,            0
    //];
  //}

  const api = {
    x_axis_rotate,
    y_axis_rotate,
    z_axis_rotate,
    scale,
    translate,
    project_parallel,
    project_perspective
  };

  console.log('graphics...');
  return api;
}());
