Engine.screens['high-scores'] = (function(game) {
  'use strict';

  function initialize() {
    document.getElementById('id-high-scores-back').addEventListener(
      'click',
      function() { game.showScreen('main-menu'); });

    document.getElementById('id-clear-high-scores').addEventListener(
      'click',
      function() { Engine.persistence.clear(); Engine.persistence.report() });
  }

  function run() {
    Engine.persistence.report();
  }

  return {
    initialize : initialize,
    run : run
  };
}(Engine.game));
