Engine.main = (function(graphics, objs, glUtils) {
  'use strict';

  let canvas = document.getElementById('canvas-main');
  let gl = canvas.getContext('webgl');

  let solids = [];
  solids.push(objs.make_solid({
    center:{x:-1, y:.55, z:-4},
    scale:{x:.75, y:.75, z:.75},
    xRotDir:-1,
    yRotDir:1,
    zRotDir:1,
    update:function(th){
      this.center.x+=.05;
      if (this.center.x > 9)
        this.center.x = -9;
    },
  }, objs.Solids.TETRAHEDRON));

  solids.push(objs.make_solid({
    center:{x:0, y:0, z:-3},
    scale:{x:.25, y:.25, z:.25},
    xRotDir:1,
    yRotDir:-1,
    zRotDir:1,
    update:function(th){
      this.center.y+=.075;
      if (this.center.y > 7){
        this.center.y = -7;
        this.center.x = Math.random()*6-3;
      }
    },
  }, objs.Solids.OCTAHEDRON));

  let a = .03;
  (function addHexa(){
    solids.push(
      objs.make_solid({
        center:{x:0, y:0, z:-1.5},
        scale:{x:.5, y:.5, z:.5},
        xRotDir:1,
        yRotDir:1,
        zRotDir:-1,
        v:0,
        update:function(dt){
          this.v += a * dt;
          this.center.z -= this.v;
          if(this.center.z < -15){
            this.center.z = Math.random()*5;
            this.center.x = (Math.random()*2+8) * Math.cos(Math.random()*2*Math.PI);
            this.center.y = (Math.random()*2+8) * Math.sin(Math.random()*2*Math.PI)
            this.center.xRotDir = (Math.random()*2)-1;
            this.center.yRotDir = (Math.random()*2)-1;
            this.center.zRotDir = (Math.random()*2)-1;
            this.v = 0;
            this.scale.x = this.scale.y = this.scale.z = Math.random();
            if(solids.length < 1000){
              addHexa();
              addHexa();
              addHexa();
            }
          }
        },
      }, objs.Solids.HEXAHEDRON)
    );
  })();

  let buffers = {};

  buffers.vertexBuffer = glUtils.createBuffer(gl,
    objs.vertices,
    gl.ARRAY_BUFFER,
    gl.STATIC_DRAW);

  buffers.vertexColorBuffer = glUtils.createBuffer(gl,
    objs.vertexColors,
    gl.ARRAY_BUFFER,
    gl.STATIC_DRAW);

  let solidIndexBuffers = [];
  solidIndexBuffers[objs.Solids.TETRAHEDRON] = glUtils.createBuffer(gl,
    objs.indices_tetrahedron(),
    gl.ELEMENT_ARRAY_BUFFER,
    gl.STATIC_DRAW);
  solidIndexBuffers[objs.Solids.OCTAHEDRON] = glUtils.createBuffer(gl,
    objs.indices_octahedron(),
    gl.ELEMENT_ARRAY_BUFFER,
    gl.STATIC_DRAW);
  solidIndexBuffers[objs.Solids.HEXAHEDRON] = glUtils.createBuffer(gl,
    objs.indices_hexahedron(),
    gl.ELEMENT_ARRAY_BUFFER,
    gl.STATIC_DRAW);

  let vertexShaderSource = `
  uniform mat4 matRotate;
  uniform mat4 matScale;
  uniform mat4 matTranslate;
  uniform mat4 matProject;
  attribute vec4 aPosition;
  attribute vec4 aColor;
  varying vec4 vColor;
  void main()
  {
    gl_Position =
      matProject
      * matTranslate
      * matRotate
      * matScale
      * aPosition;

    vColor = aColor;
  }`;

  let vertexShader = glUtils.createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);

  let fragmentShaderSource = `
  precision lowp float;
  varying vec4 vColor;
  void main()
  {
    gl_FragColor = vColor;
  }`;

  let fragmentShader = glUtils.createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);

  let shaderProgram = glUtils.createProgram(gl, vertexShader, fragmentShader);

  gl.useProgram(shaderProgram);

  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.vertexBuffer);

  let position = gl.getAttribLocation(shaderProgram, 'aPosition');
  gl.vertexAttribPointer(position, 3, gl.FLOAT, false, objs.vertices.BYTES_PER_ELEMENT * 3, 0);
  gl.enableVertexAttribArray(position);
  gl.bindBuffer(gl.ARRAY_BUFFER, buffers.vertexColorBuffer);

  let color = gl.getAttribLocation(shaderProgram, 'aColor');
  gl.vertexAttribPointer(color, 3, gl.FLOAT, false, objs.vertexColors.BYTES_PER_ELEMENT * 3, 0);
  gl.enableVertexAttribArray(color);

  //------------------------------------------------------------------
  //
  // Scene updates go here.
  //
  //------------------------------------------------------------------
  let th = 0;
  let matRotateLoc = gl.getUniformLocation(shaderProgram, 'matRotate');
  let matScaleLoc = gl.getUniformLocation(shaderProgram, 'matScale');
  let matTranslateLoc = gl.getUniformLocation(shaderProgram, 'matTranslate');
  let matProjectLoc = gl.getUniformLocation(shaderProgram, 'matProject');

  setTimeout(() => {
    gl.uniformMatrix4fv(matProjectLoc, false,
      graphics.project_perspective(1, 1, 1, 14)
    );
    console.log("switch");
  }, 7000);

  function update(dt) {
    dt/=1000;
    th += .5 * dt;
    for(let i = 0; i < solids.length; ++i){
      solids[i].update(dt);
    }
    gl.clearColor(
      0.3921568627450980392156862745098,
      0.58431372549019607843137254901961,
      0.92941176470588235294117647058824, 1.0);

    gl.clearDepth(1.0);
    gl.depthFunc(gl.LEQUAL);
    gl.enable(gl.DEPTH_TEST);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
  }

  //------------------------------------------------------------------
  //
  // Rendering code goes here
  //
  //------------------------------------------------------------------
  gl.uniformMatrix4fv(matProjectLoc, false,
    transposeMatrix4x4(graphics.project_parallel(2, 2, 1, 20)));
  // transposeMatrix4x4(graphics.project_perspective(1, 1, 1, 10)));
  function render() {

    for(let i = 0; i < solids.length; ++i){
      let rotationMatComp = graphics.mat4Multiply(
        graphics.x_axis_rotate(solids[i].xRotDir * th),
        graphics.y_axis_rotate(solids[i].yRotDir * th),
        graphics.z_axis_rotate(solids[i].zRotDir * th)
      );

      gl.uniformMatrix4fv(matRotateLoc, false,
        transposeMatrix4x4(rotationMatComp));

      gl.uniformMatrix4fv(matScaleLoc, false,
        transposeMatrix4x4(graphics.scale(
          solids[i].scale.x,
          solids[i].scale.y,
          solids[i].scale.z)));

      gl.uniformMatrix4fv(matTranslateLoc, false,
        transposeMatrix4x4(graphics.translate(
          solids[i].center.x,
          solids[i].center.y,
          solids[i].center.z)));

      switch(solids[i].type){
        case objs.Solids.TETRAHEDRON:
          gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, solidIndexBuffers[objs.Solids.TETRAHEDRON]);
          gl.drawElements(gl.TRIANGLES, 12, gl.UNSIGNED_SHORT, 0);
          break;
        case objs.Solids.OCTAHEDRON:
          gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, solidIndexBuffers[objs.Solids.OCTAHEDRON]);
          gl.drawElements(gl.TRIANGLES, 30, gl.UNSIGNED_SHORT, 0);
          break;
        case objs.Solids.HEXAHEDRON:
          gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, solidIndexBuffers[objs.Solids.HEXAHEDRON]);
          gl.drawElements(gl.TRIANGLES, 36, gl.UNSIGNED_SHORT, 0);
          break;
      }
      //gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, solids[i].indexBuffer);
      //gl.drawElements(gl.TRIANGLES, solids[i].indices.length, gl.UNSIGNED_SHORT, 0);
    }
  }

  //------------------------------------------------------------------
  //
  // This is the animation loop.
  //
  //------------------------------------------------------------------
  let prevTime = performance.now();
  function animationLoop(time) {

    let dt = time - prevTime;
    prevTime = time;

    update(dt);
    render();

    requestAnimationFrame(animationLoop);
  }

  /**
   * Transpose a 4x4 matrix
   *
   * @param {array} m - a 4x4 matrix represented as a single array
   * @return a new transposed matrix represented as an array
   */
  function transposeMatrix4x4(m) {
    return [
      m[0], m[4], m[8], m[12],
      m[1], m[5], m[9], m[13],
      m[2], m[6], m[10], m[14],
      m[3], m[7], m[11], m[15]
    ];
  }

  console.log('initializing...');
  requestAnimationFrame(animationLoop);

}(Engine.graphics, Engine.objects, Engine.glUtils));
