ModelLoaderPLY = (function() {
  'use strict';

  //------------------------------------------------------------------
  //
  // Placeholder function that returns a hard-coded cube.
  //
  //------------------------------------------------------------------
  function defineModel(fileLines) {
    let model = {};
    let row = 0;

    // check for "magic" ply
    if (fileLines[row++] !== 'ply') {
      return {
        status:'error',
        error:'File does not begin with ply'
      };
    }

    // initialize vertex and face count
    let vct = 0;
    let fct = 0;
    let verts, faces, words;

    // get the count of vertices and faces
    while (fileLines[row++] !== 'end_header'){
      words = fileLines[row].split(' ');
      if (words[0] == 'element'){
        if (words[1] === 'vertex'){
          vct = parseInt(words[2])
          words = fileLines[++row].split(' ');
          if (words[1] === 'float'){
            verts = new Float32Array();
          } else if (words[1] === 'double') {
            verts = new Float64Array();
          } else {
            return {
              status:'error',
              error:'vertices has an invalid type'
            };
          }
        }
        if (words[1] === 'face'){
          fct = parseInt(words[2]);
          words = fileLines[++row].split(' ');
          if (words[3] === 'int8' || words[3] === 'uint8'){
            faces = new Uint8Array();
          } else if (words[3] === 'int16' || words[3] === 'uint16'){
            faces = new Uint16Array();
          } else if (words[3] === 'int32' || words[3] === 'uint32'){
            faces = new Uint32Array();
          } else {
            return {
              status:'error',
              error:'vertex indices has an invalid type'
            };
          }
        }
      }
    }

    // parse vertices
    let i, j, n;
    for (i = 0; i < vct; ++i, ++row){
      words = fileLines[row].split(/\s+/);
      if (words[0] === "") words.slice(1);
      for (j = 0; j < 3; ++i){
        verts.push(parseFloat(words[j]));
      }
    }

    // parse face indicies
    for (i = 0; i < fct; ++i, ++row){
      words = fileLines[row].split(/\s+/);
      if (words[0] === "") words.slice(1);
      n = parseInt(words[0]);
      for (j = 1; j < n; ++i){
        faces.push(parseFloat(words[j]));
      }
    }

    model.vertices = new Float32Array([
      -0.5, -0.5, 0.5,   // 0 - 3 (Front face)
      0.5, -0.5, 0.5,
      0.5,  0.5, 0.5,
      -0.5,  0.5, 0.5,

      -0.5, -0.5, -0.5,   // 4 - 7 (Back face)
      0.5, -0.5, -0.5,
      0.5,  0.5, -0.5,
      -0.5,  0.5, -0.5,
    ]);

    model.vertexColors = new Float32Array([
      0.0, 0.0, 1.0,  // Front face
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,
      0.0, 0.0, 1.0,

      1.0, 0.0, 0.0,  // Back face
      1.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
      1.0, 0.0, 0.0,
    ]);

    //
    // CCW winding order
    model.indices = new Uint16Array([
      0, 1, 2, 0, 2, 3,   // Front face
      5, 4, 7, 5, 7, 6,   // Back face
      1, 5, 6, 1, 6, 2,   // Right face
      7, 4, 0, 3, 7, 0,   // Left face
      3, 2, 6, 3, 6, 7,   // Top face
      5, 1, 0, 5, 0, 4    // Bottom face
    ]);

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
          let fileLines = fileText.split('\n');
          let model = defineModel(fileLines);
          resolve(model);
        })
        .catch(error => {
          reject(error);
        });
    });
  }

  return {
    load : load
  };

}());
