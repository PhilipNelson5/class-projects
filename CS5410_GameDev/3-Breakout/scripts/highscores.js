MyGame.screens['high-scores'] = (function(game) {
  'use strict';

  function initialize() {
    document.getElementById('id-high-scores-back').addEventListener(
      'click',
      function() { game.showScreen('main-menu'); });

    document.getElementById('id-clear-high-scores').addEventListener(
      'click',
      function() { MyGame.persistence.clear(); MyGame.persistence.report() });
  }

  function run() {
    MyGame.persistence.report();
  }

  return {
    initialize : initialize,
    run : run
  };
}(MyGame.game));
