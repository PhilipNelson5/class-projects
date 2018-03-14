// ------------------------------------------------------------------
//
//
// ------------------------------------------------------------------
let canvas = document.getElementById('canvas-main'),
  context = canvas.getContext('2d');


MyGame.graphics = (function() {
  'use strict';

  //
  // Place a 'clear' function on the Canvas prototype, this makes it a part
  // of the canvas, rather than making a function that calls and does it.
  CanvasRenderingContext2D.prototype.clear = function() {
    this.save();
    this.setTransform(1, 0, 0, 1, 0, 0);
    this.clearRect(0, 0, canvas.width, canvas.height);
    this.restore();
  };

  //------------------------------------------------------------------
  //
  // Public method that changes the canvas size to match the screen
  //
  //------------------------------------------------------------------
  function sizeCanvasToScreen()
  {
    canvas.width = window.innerWidth*.85;
    canvas.height = window.innerHeight*.85;
  }

  function drawImage(center, size, rotation, alpha, image) {
    context.save();
    context.translate(center.x, center.y);
    context.rotate(rotation);
    context.translate(-center.x, -center.y);
    context.globalAlpha = alpha;

    context.drawImage(
      image,
      center.x - size / 2,
      center.y - size / 2,
      size, size);

    context.restore();
  }

  function write(spec) {
    context.font = spec.font;
    context.fillText(spec.text, spec.x, spec.y);
  }

  function writeLowerRight(spec) {
    context.font = spec.font;
    let width = context.measureText(spec.text).width;
    let height = context.measureText('M').width;
    write({
      font: spec.font,
      text: spec.text,
      x: spec.x-width, y: spec.y-height,
    });
  }

  //------------------------------------------------------------------
  //
  // Public method that allows the client code to clear the canvas.
  //
  //------------------------------------------------------------------
  function clear() {
    context.clear();
  }

  //------------------------------------------------------------------
  //
  // This is used to create a texture object that can be used by client
  // code for rendering.
  //
  //------------------------------------------------------------------
  function Texture(spec) {
    let that = {},
      image = new Image();

    // Load the image, set the draw function once it is ready
    image.onload = function() {
      that.draw = function() {
        context.save();

        context.translate(spec.center.x, spec.center.y);
        context.rotate(spec.rotation);
        context.translate(-spec.center.x, -spec.center.y);

        context.drawImage(
          image,
          spec.center.x - spec.width/2,
          spec.center.y - spec.height/2,
          spec.width, spec.height);

        context.restore();
      }
    };
    image.src = spec.image;

    that.draw = function() {};

    spec.render = that;

    return spec;
  }

  return {
    clear : clear,
    Texture : Texture,
    drawImage: drawImage,
    write: write,
    writeLowerRight: writeLowerRight,
    sizeCanvasToScreen : sizeCanvasToScreen,
    size : {w:canvas.width, h:canvas.height}

  };
}());
