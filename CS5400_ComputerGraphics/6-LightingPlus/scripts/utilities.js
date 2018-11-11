'use strict';

//------------------------------------------------------------------
//
// Helper function used to load a file from the server
//
//------------------------------------------------------------------
function loadFileFromServer(filename) {
  return new Promise((resolve, reject) => {
    let xmlHttp = new XMLHttpRequest();

    xmlHttp.onreadystatechange = function() {
      if (xmlHttp.readyState === XMLHttpRequest.DONE) {
        resolve(xmlHttp.responseText);
      }
    };

    xmlHttp.open("GET", filename, true);
    xmlHttp.send();
  });
}

//------------------------------------------------------------------//
//
// Helper function used to load a texture file from the server.//
//
//------------------------------------------------------------------
function loadTextureFromServer(filename) {
  return new Promise((resolve, reject) => {
    let xmlHttp = new XMLHttpRequest();
    xmlHttp.onload = function() {
      if (xmlHttp.status === 200) {
        let asset = new Image();
        asset.onload = function() {
          window.URL.revokeObjectURL(asset.src);
          resolve(asset);
        }
        asset.src = window.URL.createObjectURL(xmlHttp.response);
      } else {
        reject();
      }
    };
    xmlHttp.open("GET", filename, true);
    xmlHttp.responseType = 'blob';
    xmlHttp.send();
  });
}

//------------------------------------------------------------------
//
// Given the base name of a texture cube, load all the images
//
//------------------------------------------------------------------
function loadTexCube(name, ext){
  return new Promise((resolve, reject) => {
    let texCube = {};
    loadTextureFromServer(`assets/${name}/negx.${ext}`)
      .then(texture => {
        texCube.negx = texture;
        return loadTextureFromServer(`assets/${name}/negy.${ext}`);
      })
      .then(texture => {
        texCube.negy = texture;
        return loadTextureFromServer(`assets/${name}/negz.${ext}`);
      })
      .then(texture => {
        texCube.negz = texture;
        return loadTextureFromServer(`assets/${name}/posx.${ext}`);
      })
      .then(texture => {
        texCube.posx = texture;
        return loadTextureFromServer(`assets/${name}/posy.${ext}`);
      })
      .then(texture => {
        texCube.posy = texture;
        return loadTextureFromServer(`assets/${name}/posz.${ext}`);
      })
      .then(texture => {
        texCube.posz = texture;
        resolve(texCube);
      })
      .catch(error => {
        console.error('(loadTexCube) ERROR : ', error);
        reject();
      });
  });
}

//------------------------------------------------------------------
//
// Multiply an arbitrary number of 4x4 matrices.
//
//------------------------------------------------------------------
function multiplyMatrix4x4(...matricies) {
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

//------------------------------------------------------------------
//
// Invert a matrix
//
//------------------------------------------------------------------
function invert(m) {

  let r = [
    m[5]*m[10]*m[15]  - m[5]*m[14]*m[11] - m[6]*m[9]*m[15] + m[6]*m[13]*m[11] + m[7]*m[9]*m[14] - m[7]*m[13]*m[10],
    -m[1]*m[10]*m[15] + m[1]*m[14]*m[11] + m[2]*m[9]*m[15] - m[2]*m[13]*m[11] - m[3]*m[9]*m[14] + m[3]*m[13]*m[10],
    m[1]*m[6]*m[15]   - m[1]*m[14]*m[7]  - m[2]*m[5]*m[15] + m[2]*m[13]*m[7]  + m[3]*m[5]*m[14] - m[3]*m[13]*m[6] ,
    -m[1]*m[6]*m[11]  + m[1]*m[10]*m[7]  + m[2]*m[5]*m[11] - m[2]*m[9]*m[7]   - m[3]*m[5]*m[10] + m[3]*m[9]*m[6]  ,
    -m[4]*m[10]*m[15] + m[4]*m[14]*m[11] + m[6]*m[8]*m[15] - m[6]*m[12]*m[11] - m[7]*m[8]*m[14] + m[7]*m[12]*m[10],
    m[0]*m[10]*m[15]  - m[0]*m[14]*m[11] - m[2]*m[8]*m[15] + m[2]*m[12]*m[11] + m[3]*m[8]*m[14] - m[3]*m[12]*m[10],
    -m[0]*m[6]*m[15]  + m[0]*m[14]*m[7]  + m[2]*m[4]*m[15] - m[2]*m[12]*m[7]  - m[3]*m[4]*m[14] + m[3]*m[12]*m[6] ,
    m[0]*m[6]*m[11]   - m[0]*m[10]*m[7]  - m[2]*m[4]*m[11] + m[2]*m[8]*m[7]   + m[3]*m[4]*m[10] - m[3]*m[8]*m[6]  ,
    m[4]*m[9]*m[15]   - m[4]*m[13]*m[11] - m[5]*m[8]*m[15] + m[5]*m[12]*m[11] + m[7]*m[8]*m[13] - m[7]*m[12]*m[9] ,
    -m[0]*m[9]*m[15]  + m[0]*m[13]*m[11] + m[1]*m[8]*m[15] - m[1]*m[12]*m[11] - m[3]*m[8]*m[13] + m[3]*m[12]*m[9] ,
    m[0]*m[5]*m[15]   - m[0]*m[13]*m[7]  - m[1]*m[4]*m[15] + m[1]*m[12]*m[7]  + m[3]*m[4]*m[13] - m[3]*m[12]*m[5] ,
    -m[0]*m[5]*m[11]  + m[0]*m[9]*m[7]   + m[1]*m[4]*m[11] - m[1]*m[8]*m[7]   - m[3]*m[4]*m[9]  + m[3]*m[8]*m[5]  ,
    -m[4]*m[9]*m[14]  + m[4]*m[13]*m[10] + m[5]*m[8]*m[14] - m[5]*m[12]*m[10] - m[6]*m[8]*m[13] + m[6]*m[12]*m[9] ,
    m[0]*m[9]*m[14]   - m[0]*m[13]*m[10] - m[1]*m[8]*m[14] + m[1]*m[12]*m[10] + m[2]*m[8]*m[13] - m[2]*m[12]*m[9] ,
    -m[0]*m[5]*m[14]  + m[0]*m[13]*m[6]  + m[1]*m[4]*m[14] - m[1]*m[12]*m[6]  - m[2]*m[4]*m[13] + m[2]*m[12]*m[5] ,
    m[0]*m[5]*m[10]   - m[0]*m[9]*m[6]   - m[1]*m[4]*m[10] + m[1]*m[8]*m[6]   + m[2]*m[4]*m[9]  - m[2]*m[8]*m[5]  ,
  ];
  var det = m[0]*r[0] + m[1]*r[4] + m[2]*r[8] + m[3]*r[12];
  for (var i = 0; i < 16; i++) r[i] /= det;
  return r;
};

//------------------------------------------------------------------
//
// Transpose a matrix
//
//------------------------------------------------------------------
function transposeMatrix4x4(m) {
  return [
    m[0], m[4], m[8],  m[12],
    m[1], m[5], m[9],  m[13],
    m[2], m[6], m[10], m[14],
    m[3], m[7], m[11], m[15]
  ];
}

//------------------------------------------------------------------
//
// Two deep copy functions, not sure which one works
//
//------------------------------------------------------------------
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

//------------------------------------------------------------------
//
// Helper function to turn hex into rgba
//
//------------------------------------------------------------------
function hexToRgba(hex) {
  var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? [
    parseInt(result[1], 16)/255,
    parseInt(result[2], 16)/255,
    parseInt(result[3], 16)/255, 1
  ] : null;
}

//------------------------------------------------------------------
//
// Debugging function to print out the uniforms of a shader program
//
//------------------------------------------------------------------
function printUniforms(program){
  const numUniforms = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < numUniforms; ++i) {
    const info = gl.getActiveUniform(program, i);
    console.log(info);
  }
}

