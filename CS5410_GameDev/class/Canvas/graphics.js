MyGame.graphics = (function() {
  let canvas = document.getElementById('canvas-main');
  let context = canvas.getContext('2d');

  function clear() {
    context.clearRect(0, 0, canvas.width, canvas.height);
  }

  function Rectangle(spec) {
    let that = {};

    that.updateRotation = function(angle) {
      spec.rotation += angle;
    };

    that.draw = function() {
      canvas.save();
      context.translate(spec.x + spec.width / 2, spec.y + spec.height / 2);
      context.rotate(spec.rotation);
      context.translate(
          -(spec.x + spec.width / 2), -(spec.y + spec.height / 2));

      context.fillStyle = spec.strokeStyle;
      context.fillRect(spec.x, spec.y, spec.width, spec.height);

      context.strokeStyle = spec.strokeStyle;
      context.strokeRect(spec.x, spec.y, spec.width, spec.height);

      canvas.restore();
    }
  }

  function Triangle(spec) {}

  return {Rectangle: Rectangle, Triangle: Triangle, clear: clear};
})();
