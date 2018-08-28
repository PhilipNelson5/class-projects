// ------------------------------------------------------------------
//
// This object is in charge of drawing on the canvas
//
// ------------------------------------------------------------------
let canvas = document.getElementById('canvas-main'),
  context = canvas.getContext('2d');


Engine.graphics = (function() {
  'use strict';

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

  /**
   * center   : the center of the image
   * size     : size in pixels of the image
   * rotation : rotation in radians
   * alpha    : alpha of the image
   * image    : the image object
   */
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

  /**
   * write text given the top right coordinate
   * spec {
   * font : '#px serif' // the font size and font name
   * text : 'text'      // the text to be written
   * x    : #px         // the x location
   * y    : #px         // the y location
   * }
   */
  function write(spec) {
    context.font = spec.font;
    context.fillText(spec.text, spec.x, spec.y);
  }

  /**
   * write text given the lower right coordinate
   * spec {
   * font : '#px serif' // the font size and font name
   * text : 'text'      // the text to be written
   * x    : #px         // the x location
   * y    : #px         // the y location
   * }
   */
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
    let image = new Image();

    // Load the image, set the draw function once it is ready
    image.onload = function() {
      console.log(spec.image + " loaded");
      spec.draw = function() {
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

    spec.draw = function() {};

    return spec;
  }

  function initialize() {
    let canvas = document.getElementById('canvas-main');
    canvas.width = 1777;
    canvas.height = 1000;

    canvas.style.top = 0+'px';
    canvas.style.left = 0+'px';

    // canvas.style.width = '100%';
    // canvas.style.height = '100%';

    // resizeCanvas(canvas)

    // window.addEventListener('resize', () => resizeCanvas(canvas))
  }

  function resizeCanvas(canvas) {
      let r = 1920 / 1080;
      let w = window.innerWidth;
      let h = window.innerHeight;
      if(r > w/h) { // width is the lower bound
        canvas.style.width = w + 'px';
        canvas.style.height = w/r + 'px';
      } else { // height is the lower bound
        canvas.style.width = h*r + 'px';
        canvas.style.height = h + 'px';
      }
      console.log({sw:canvas.style.width, sh:canvas.style.height, pw:canvas.width, ph:canvas.height});
    }

  return {
    initialize,
    clear,
    Texture,
    drawImage,
    write,
    writeLowerRight,
    sizeCanvasToScreen,
  };

}());
