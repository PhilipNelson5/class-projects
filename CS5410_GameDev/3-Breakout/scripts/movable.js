MyGame.movable = (function() {
  'use strict';

  function MakeMovable(obj)
  {
    obj.rotateRight = function(dt) {
      obj.rotation += obj.rotateRate * dt * .001;
    };

    obj.rotateLeft = function(dt) {
      obj.rotation -= obj.rotateRate * dt * .001;
    };

    obj.moveLeft = function(dt) {
      if(obj.center.x - obj.width/2 <= 0) return;
      obj.center.x -= obj.moveRate * dt * .001;
    };

    obj.moveRight = function(dt) {
      if(obj.center.x + obj.width/2 >= gwidth) return;
      obj.center.x += obj.moveRate * dt * .001;
    };

    obj.moveUp = function(dt) {
      obj.center.y -= obj.moveRate * dt * .001;
    };

    obj.moveDown = function(dt) {
      obj.center.y += obj.moveRate * dt * .001;
    };

    obj.moveTo = function(center) {
      obj.center = center;
    };
  }

  return {
    MakeMovable : MakeMovable,
  };
}());
