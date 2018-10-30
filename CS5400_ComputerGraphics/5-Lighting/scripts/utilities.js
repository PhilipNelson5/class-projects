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

//------------------------------------------------------------------
//
// Helper function to multiply two 4x4 matrices.
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
// Transpose a matrix.
//
//------------------------------------------------------------------
function transposeMatrix4x4(m) {
  let transpose = [
    m[0], m[4], m[8], m[12],
    m[1], m[5], m[9], m[13],
    m[2], m[6], m[10], m[14],
    m[3], m[7], m[11], m[15]
  ];

  return transpose;
}
