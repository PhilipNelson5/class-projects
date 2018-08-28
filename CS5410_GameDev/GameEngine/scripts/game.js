// ------------------------------------------------------------------
//
// This is the game engine object.
// Everything about the game is an attribute of Engine
//
// ------------------------------------------------------------------

Engine.game = (function(screens) {
  'use strict';

  // -----------------------------------------------------------------
  //
  // showScreen is used to change to a new active screen.
  //
  // -----------------------------------------------------------------
  function showScreen(id) {
    let screen = 0;
    let active = null;

    // Remove the active state from all screens.
    active = document.getElementsByClassName('active');
    for (screen = 0; screen < active.length; screen++) {
      active[screen].classList.remove('active');
    }

    // Tell the screen run
    screens[id].run();

    // Set the new screen active
    document.getElementById(id).classList.add('active');
  }

  // -----------------------------------------------------------------
  //
  // Perform the one-time game initialization.
  //
  // -----------------------------------------------------------------
  function initialize() {
    let screen = null;

    // Initialize all the screens
    for (screen in screens) {
      if (screens.hasOwnProperty(screen)) {
        screens[screen].initialize();
      }
    }

    // Make the main-menu screen active
    showScreen('main-menu');
  }

  return {
    initialize : initialize,
    showScreen : showScreen
  };

}(Engine.screens));
