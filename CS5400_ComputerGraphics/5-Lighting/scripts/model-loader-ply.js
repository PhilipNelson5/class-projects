ModelLoaderPLY = (function() {
  'use strict';

  function randDouble(min, max) {
    return Math.random() * (max - min) + min;
  }

  /*
   *function getRandomInt(min, max) {
   *  return Math.floor(Math.random() * (max - min + 1)) + min;
   *}
   */

  function norm(array){
    let max = array.reduce((acc, val)=>{return Math.max(acc, val);}, array[0]);
    let min = array.reduce((acc, val)=>{return Math.min(acc, val);}, array[0]);
    let nMin = -1;
    let nMax = 1;

    for(let i = 0; i < array.length; ++i){
      let x = array[i];
      array[i] = nMin + (nMax - nMin)/(max - min)*(x - min);
    }
  }


  //------------------------------------------------------------------
  //
  // Placeholder function that returns a hard-coded cube.
  //
  //------------------------------------------------------------------
  function defineModel(fileText) {
    let model = {status:'ok'};

    let row = 0;
    let lines = fileText.split(/\r?\n/);

    // check for "magic" ply
    let words = lines[row++].split('/\s*/');
    if (words[0] !== 'ply') {
      return {
        status:'error',
        error:'File does not begin with ply'
      };
    }

    // initialize vertex and face count
    let vct = 0;
    let fct = 0;
    let verts, faces;

    // get the count of vertices and faces
    while (lines[row] !== 'end_header'){
      words = lines[row].split(/\s+/);
      if (words[0] == 'element'){
        if (words[1] === 'vertex'){
          vct = parseInt(words[2])
          words = lines[++row].split(' ');
          if (words[1] === 'float' || words[1] === 'float32'){
            verts = new Float32Array(vct*3);
            console.log('verts: Float32Array');
          } else if (words[1] === 'double') {
            verts = new Float64Array(vct*3);
            console.log('verts: Float64Array');
          } else {
            return {
              status:'error',
              error:'vertices has an invalid type'
            };
          }
        }
        if (words[1] === 'face'){
          fct = parseInt(words[2]);
          words = lines[++row].split(' ');
          if (words[3] === 'int8' || words[3] === 'uint8' || words[3] == 'uchar'){
            faces = new Uint8Array(fct*3);
            console.log('indices: Uint8Array');
          } else if (words[3] === 'int16' || words[3] === 'uint16' || words[3] == 'int'){
            faces = new Uint16Array(fct*3);
            console.log('indices: Uint16Array');
          } else if (words[3] === 'int32' || words[3] === 'uint32'){
            faces = new Uint32Array(fct*3);
            console.log('indices: Uint32Array');
          } else {
            return {
              status:'error',
              error:'vertex indices has an invalid type'
            };
          }
        }
      }
      row++;
    }
    row++;
    console.log({vct, fct});

    // parse vertices
    let i, j, n;
    for (i = 0; i < vct; ++i, ++row){
      words = lines[row].split(/\s+/);
      if (words[0] === ""){
        words = words.slice(1);
      }
      for (j = 0; j < 3; ++j){
        verts[i*3+j] = parseFloat(words[j]);
      }
    }

    // parse face indices
    for (i = 0; i < fct; ++i, ++row){
      words = lines[row].split(/\s+/);
      if (words[0] === ""){
        words = words.slice(1);
      }
      n = parseInt(words[0]);
      for (j = 1; j <= n; ++j){
        faces[i*3+j-1] = parseFloat(words[j]);
      }
    }
    console.log({vct:verts.length/3, fct:faces.length/3});

    // random vertex colors
    let colors = new Float32Array(verts.length);
    for(i = 0; i < verts.length; ++i)
    {
      colors[i*3+0]=(randDouble(.5, 1));
      colors[i*3+1]=(randDouble(.5, 1));
      colors[i*3+2]=(randDouble(.5, 1));
    }

    norm(verts);

    model.vertices = verts;
    model.indices = faces;
    model.vertexColors = colors;

    /*
     *    model.vertices = new Float32Array([
     *      -0.5, -0.5, 0.5,   // 0 - 3 (Front face)
     *      0.5, -0.5, 0.5,
     *      0.5,  0.5, 0.5,
     *      -0.5,  0.5, 0.5,
     *
     *      -0.5, -0.5, -0.5,   // 4 - 7 (Back face)
     *      0.5, -0.5, -0.5,
     *      0.5,  0.5, -0.5,
     *      -0.5,  0.5, -0.5,
     *    ]);
     *
     *    model.vertexColors = new Float32Array([
     *      0.0, 0.0, 1.0,  // Front face
     *      0.0, 0.0, 1.0,
     *      0.0, 0.0, 1.0,
     *      0.0, 0.0, 1.0,
     *
     *      1.0, 0.0, 0.0,  // Back face
     *      1.0, 0.0, 0.0,
     *      1.0, 0.0, 0.0,
     *      1.0, 0.0, 0.0,
     *    ]);
     *
     *
     *    // CCW winding order
     *    model.indices = new Uint16Array([
     *      0, 1, 2, 0, 2, 3,   // Front face
     *      5, 4, 7, 5, 7, 6,   // Back face
     *      1, 5, 6, 1, 6, 2,   // Right face
     *      7, 4, 0, 3, 7, 0,   // Left face
     *      3, 2, 6, 3, 6, 7,   // Top face
     *      5, 1, 0, 5, 0, 4    // Bottom face
     *    ]);
     */


    model.center = {
      x: 0.0,
      y: 0.0,
      z: -2.0
    };

    return model;
  }

  //------------------------------------------------------------------
  //
  // Loads and parses a PLY formatted file into an object ready for
  // rendering.
  //
  //------------------------------------------------------------------
  function load(filename) {
    return new Promise((resolve, reject) => {
      loadFileFromServer(filename)
        .then(fileText => {
          let model = defineModel(fileText);
          if(model.status === 'ok'){
            resolve(model);
          }
          else{
            reject(model.error);
          }
        })
        .catch(error => {
          reject(error);
        });
    });
  }

  return {
    load
  };

}());
