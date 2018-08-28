Engine.screens['help'] = (function(game) {
  'use strict';

  function initialize() {
    document.getElementById('id-help-back').addEventListener(
      'click',
      function() { game.showScreen('main-menu'); });
  }

  function run() {
    // there isn't anything to do.
  }

  return {
    initialize : initialize,
    run : run
  };
}(Engine.game));
