ModelLoaderPLY = (function() {
  'use strict';

  function randDouble(min, max) {
    return Math.random() * (max - min) + min;
  }

  function getRandomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

  //normalize the normal vectors
  function normalize(normals){
    for(let i = 0; i < normals.length; i += 3){
      let vx = normals[i];
      let vy = normals[i+1];
      let vz = normals[i+2];

      let norm = Math.sqrt(vx*vx + vy*vy + vz*vz)

      normals[i]   /= norm;
      normals[i+1] /= norm;
      normals[i+2] /= norm;
    }
  }

  /**
   * figure out which vertices are actually used and
   * normalize only based on those vertices.
   */
  function normUsed(verts, indices){
    let usedVerts = [];

    for(let i = 0; i < verts.length; ++i)
      usedVerts[i] = false;

    for(let i = 0; i < indices.length; ++i){
      usedVerts[indices[i]*3+0] = true;
      usedVerts[indices[i]*3+1] = true;
      usedVerts[indices[i]*3+2] = true;
    }

    let min = verts[0];
    let max = verts[0];

    let ct = 0;
    for(let i = 0; i < verts.length; ++i)
    {
      if(usedVerts[i])
      {
        ++ct;
        min = Math.min(min, verts[i]);
        max = Math.max(max, verts[i]);
      }
    }
    console.log({ct:ct/3, min, max});

    let nMin = -1;
    let nMax = 1;

    for(let i = 0; i < verts.length; ++i){
      if(usedVerts[i])
        verts[i] = nMin + (nMax - nMin)/(max - min)*(verts[i] - min);
    }
  }

  function norm(array){
    let max = array.reduce( (acc, val) => { return Math.max(acc, val); }, array[0]);
    let min = array.reduce( (acc, val) => { return Math.min(acc, val); }, array[0]);
    //let max = Math.max(...array);
    //let min = Math.min(...array);

    for(let i = 0; i < array.length; ++i){
      array[i] = (array[i] - min) * 2 / (max - min) - 1;
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
    let verts, faces, normals;

    // get the count of vertices and faces
    while (lines[row] !== 'end_header'){
      words = lines[row].split(/\s+/);
      if (words[0] == 'element'){
        if (words[1] === 'vertex'){
          vct = parseInt(words[2])
          words = lines[++row].split(' ');
          if (words[1] === 'float' || words[1] === 'float32'){
            verts = new Float32Array(vct*3);
            normals = new Float32Array(vct*3);
            console.log('verts: Float32Array');
          } else if (words[1] === 'double') {
            verts = new Float64Array(vct*3);
            normals = new Float64Array(vct*3);
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
            model.indices_type = gl.UNSIGNED_SHORT;
            console.log('indices: Uint8Array');
          } else if (words[3] === 'int16' || words[3] === 'uint16' || words[3] == 'int'){
            faces = new Uint16Array(fct*3);
            model.indices_type = gl.UNSIGNED_SHORT;
            console.log('indices: Uint16Array');
          } else if (words[3] === 'int32' || words[3] === 'uint32'){
            faces = new Uint32Array(fct*3);
            model.indices_type = gl.UNSIGNED_INT;
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

    norm(verts);
    //normUsed(verts, faces);

    // calculate the normals
    //let vfct = new Float32Array(vct);
    for (i = 0; i < faces.length;){
      let v1 = faces[i++];
      let v1x = verts[v1*3];
      let v1y = verts[v1*3+1];
      let v1z = verts[v1*3+2];

      let v2 = faces[i++];
      let v2x = verts[v2*3];
      let v2y = verts[v2*3+1];
      let v2z = verts[v2*3+2];

      let v3 = faces[i++];
      let v3x = verts[v3*3];
      let v3y = verts[v3*3+1];
      let v3z = verts[v3*3+2];

      let ux = v2x-v1x;
      let uy = v2y-v1y;
      let uz = v2z-v1z;

      let vx = v3x-v1x;
      let vy = v3y-v1y;
      let vz = v3z-v1z;

      let nx = uy*vz-uz*vy;
      let ny = uz*vx-ux*vz;
      let nz = ux*vy-uy*vx;

      normals[v1*3] += nx;
      normals[v1*3+1] += ny;
      normals[v1*3+2] += nz;

      normals[v2*3] += nx;
      normals[v2*3+1] += ny;
      normals[v2*3+2] += nz;

      normals[v3*3] += nx;
      normals[v3*3+1] += ny;
      normals[v3*3+2] += nz;

      //++vfct[v1];
      //++vfct[v2];
      //++vfct[v3];
    }

    //for (let i = 0; i < vfct.length; ++i){
    //normals[i*3] /= vfct[i];
    //normals[i*3+1] /= vfct[i];
    //normals[i*3+2] /= vfct[i];
    //}

    normalize(normals);

    // random vertex colors
    let colors = new Float32Array(verts.length);
    for(i = 0; i < verts.length; ++i)
    {
      colors[i*3+0]=(1);
      colors[i*3+1]=(1);
      colors[i*3+2]=(1);
    }

    model.vertices = verts;
    model.indices = faces;
    model.normals = normals;
    model.vertexColors = colors;

    model.center = {
      x: 0.0,
      y: 0.0,
      z: -2.0
    };

    model.scale = {
      x: 1.0,
      y: 1.0,
      z: 1.0
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
    console.log("LOADING: ", filename);
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
