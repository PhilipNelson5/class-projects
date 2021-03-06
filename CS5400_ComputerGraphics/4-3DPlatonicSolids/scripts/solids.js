// ------------------------------------------------------------------
//
// This is the solids object.
// It provides all the platonic solids
//
// ------------------------------------------------------------------
function dcopy(src) {
  let copy = {};
  for (let prop in src) {
    if (src.hasOwnProperty(prop)) {
      copy[prop] = src[prop];
    }
  }
  return copy;
}

Engine.objects = (function() {
  'use strict';

  let vertices = new Float32Array([
    // 0 -- tetrahedron
    0              , 1   , 0              , // 0
    0              , -1/3, -Math.sqrt(8/9), // 1
    Math.sqrt(2/3) , -1/3, Math.sqrt(2/9) , // 2
    // 1
    0              , -1/3, -Math.sqrt(8/9), // 3
    Math.sqrt(2/3) , -1/3, Math.sqrt(2/9) , // 4
    -Math.sqrt(2/3), -1/3, Math.sqrt(2/9) , // 5
    // 2
    Math.sqrt(2/3) , -1/3, Math.sqrt(2/9) , // 6
    -Math.sqrt(2/3), -1/3, Math.sqrt(2/9) , // 7
    0              , 1   , 0              , // 8
    // 3
    -Math.sqrt(2/3), -1/3, Math.sqrt(2/9) , // 9
    0              , 1   , 0              , // 10
    0              , -1/3, -Math.sqrt(8/9), // 11
    // 4 -- octahedron
    .5,  0,  .5,             // 12
    -.5, 0, -.5,             // 13
    .5,  0, -.5,             // 14
    // 5
    .5,  0,  .5,             // 15
    -.5, 0, -.5,             // 16
    -.5, 0,  .5,             // 17
    // 6 - top
    0,   Math.sqrt(2/3),  0, // 18
    -.5, 0, -.5,             // 19
    .5,  0, -.5,             // 20
    // 7
    0,   Math.sqrt(2/3),  0, // 21
    .5, 0,  -.5,             // 22
    .5,  0,  .5,             // 23
    // 8
    0,   Math.sqrt(2/3),  0, // 24
    -.5, 0,  .5,             // 25
    .5,  0,  .5,             // 26
    // 9
    0,   Math.sqrt(2/3),  0, // 27
    -.5, 0, -.5,             // 28
    -.5, 0,  .5,             // 29
    // 10 - bottom
    0,   -Math.sqrt(2/3), 0, // 30
    -.5, 0, -.5,             // 31
    .5,  0, -.5,             // 32
    // 11
    0,   -Math.sqrt(2/3), 0, // 33
    .5, 0,  -.5,             // 34
    .5,  0,  .5,             // 35
    // 12
    0,   -Math.sqrt(2/3), 0, // 36
    -.5, 0,  .5,             // 37
    .5,  0,  .5,             // 38
    // 13
    0,   -Math.sqrt(2/3), 0, // 39
    -.5, 0, -.5,             // 40
    -.5, 0,  .5,             // 41
    // 14 -- hexahedron - top
    .5, .5, .5,    // 42
    .5, .5, -.5,   // 43
    -.5, .5, -.5,  // 44
    -.5, .5, .5,   // 45
    // 15 - front
    .5, .5, -.5,   // 46
    .5, -.5, -.5,  // 47
    -.5, -.5, -.5, // 48
    -.5, .5, -.5,  // 49
    // 16 - left
    -.5, .5, -.5,  // 50
    -.5, -.5, -.5, // 51
    -.5, -.5, .5,  // 52
    -.5, .5, .5,   // 53
    // 17 - back
    -.5, .5, .5,   // 54
    -.5, -.5, .5,  // 55
    .5, -.5, .5,   // 56
    .5, .5, .5,    // 57
    // 18 - right
    .5, .5, .5,    // 58
    .5, -.5, .5,   // 59
    .5, -.5, -.5,  // 60
    .5, .5, -.5,   // 61
    // 19 - bottom
    -.5, -.5, .5,  // 62
    -.5, -.5, -.5, // 63
    .5, -.5, -.5,  // 64
    .5, -.5, .5,   // 65
  ]);

  let vertexColors = new Float32Array([
    // 0 -- tetrahedron
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    // 1 - bottom
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    //3
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    // 3
    1.0, 1.0, 0.0,
    1.0, 1.0, 0.0,
    1.0, 1.0, 0.0,
    // 4 -- octahedron
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    // 5
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    // 6 - top
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    // 7
    0.0, 1.0, 1.0,
    0.0, 1.0, 1.0,
    0.0, 1.0, 1.0,
    // 8
    1.0, 1.0, 0.0,
    1.0, 1.0, 0.0,
    1.0, 1.0, 0.0,
    // 9
    1.0, 0.0, 1.0,
    1.0, 0.0, 1.0,
    1.0, 0.0, 1.0,
    // 10 - bottom
    1.0, 0.0, 1.0,
    1.0, 0.0, 1.0,
    1.0, 0.0, 1.0,
    // 11
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    // 12
    0.0, 1.0, 1.0,
    0.0, 1.0, 1.0,
    0.0, 1.0, 1.0,
    // 13
    1.0, 1.0, 0.0,
    1.0, 1.0, 0.0,
    1.0, 1.0, 0.0,
    // 14 -- hexahedron - top
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    1.0, 0.0, 0.0,
    // 15 - front
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 1.0, 0.0,
    // 16 - left
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    0.0, 0.0, 1.0,
    // 17 - back
    1.0, 1.0, 0.0,
    1.0, 1.0, 0.0,
    1.0, 1.0, 0.0,
    1.0, 1.0, 0.0,
    // 18 - right
    0.0, 1.0, 1.0,
    0.0, 1.0, 1.0,
    0.0, 1.0, 1.0,
    0.0, 1.0, 1.0,
    // 19 - bottom
    1.0, 0.0, 1.0,
    1.0, 0.0, 1.0,
    1.0, 0.0, 1.0,
    1.0, 0.0, 1.0,
  ]);

  function indices_tetrahedron(){
      return new Uint16Array([
        0, 1, 2,
        3, 4, 5,
        6, 7, 8,
        9, 10, 11,
      ]);
  }

  function indices_octahedron(){
    return new Uint16Array([
        // - center
        12, 13, 14,
        15, 16, 17,
        // - top
        18, 19, 20,
        21, 22, 23,
        24, 25, 26,
        27, 28, 29,
        // - bottom
        30, 31, 32,
        33, 34, 35,
        36, 37, 38,
        39, 40, 41,
      ]);
    }

  function indices_hexahedron(){
    return new Uint16Array([
        // - top
        42, 43, 44,
        42, 44, 45,
        // - front
        46, 47, 48,
        46, 48, 49,
        // - left
        50, 51, 52,
        50, 52, 53,
        // - back
        54, 55, 56,
        54, 56, 57,
        // - right
        58, 59, 60,
        58, 60, 61,
        // - bottom
        62, 63, 64,
        62, 64, 65,
      ]);
  }

  function make_solid(opts, type){
    let solid = dcopy(opts);
    switch (type) {
      case api.Solids.TETRAHEDRON:
        //solid.indices = indices_tetrahedron();
        solid.type = api.Solids.TETRAHEDRON;
        break;
      case api.Solids.OCTAHEDRON:
        //solid.indices = indices_octahedron();
        solid.type = api.Solids.OCTAHEDRON;
        break;
      case api.Solids.HEXAHEDRON:
        //solid.indices = indices_hexahedron();
        solid.type = api.Solids.HEXAHEDRON;
        break;
    }
    return solid;
  }

  const api = {
    vertices,
    vertexColors,
    make_solid,
    indices_tetrahedron,
    indices_octahedron,
    indices_hexahedron,
  };

  Object.defineProperty(api, 'Solids', {
    value: Object.freeze({
      TETRAHEDRON: 0,
      OCTAHEDRON: 1,
      HEXAHEDRON: 2,
    }),
    writable: false
  });

  console.log('objects...');
  return api;
}());
