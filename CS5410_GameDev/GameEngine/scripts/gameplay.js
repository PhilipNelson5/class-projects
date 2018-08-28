Engine.screens['game-play'] = (function(game, graphics, input, movable, particles) {
  'use strict';

  let mouseCapture = false,
    mouse = input.Mouse(),
    keyboard = input.Keyboard(),
    ship = null,
    cancelNextRequest = false,
    lastTimeStamp;

  function quit() {
    // cancel next render request
    cancelNextRequest = true;
    // return to the main menu
    game.showScreen('main-menu');
  }

  function initialize() {
    console.log('game initializing...');

    graphics.initialize();

    ship = graphics.Texture( {
      image : 'textures/spaceship.png',
      center : { x : 100, y : 100 },
      width : 50, height : 50/.52715,
      rotation : 0,
      moveRate : 200,       // pixels per second
      rotateRate : 3.14159  // Radians per second
    });

    movable.makeMovable(ship);

    // Create the keyboard input handler and register the keyboard commands
    keyboard.registerCommand(KeyEvent.DOM_VK_A, ship.moveLeft);
    keyboard.registerCommand(KeyEvent.DOM_VK_D, ship.moveRight);
    keyboard.registerCommand(KeyEvent.DOM_VK_W, ship.moveUp);
    keyboard.registerCommand(KeyEvent.DOM_VK_S, ship.moveDown);
    keyboard.registerCommand(KeyEvent.DOM_VK_Q, ship.rotateLeft);
    keyboard.registerCommand(KeyEvent.DOM_VK_E, ship.rotateRight);
    keyboard.registerCommand(KeyEvent.DOM_VK_ESCAPE, quit)

    mouse = input.Mouse();
    mouse.registerCommand('mousedown', function(e) {
      mouseCapture = true;
      // ship.moveTo({x : e.clientX, y : e.clientY});
      particles.gravity({
        position : {x: e.clientX, y: e.clientY},
      });
    });

    mouse.registerCommand('mouseup', function(e) {
      mouseCapture = false;
    });

    mouse.registerCommand('mousemove', function(e) {
      if (mouseCapture) {
        // ship.moveTo({x : e.clientX, y : e.clientY});
      particles.gravity({
        position : {x: e.clientX, y: e.clientY},
      });
      }
    });
  }

  function update(dt) {
    keyboard.update(dt);
    mouse.update(dt);
    particles.update(dt);
  }

  function render() {
    graphics.clear();
    ship.draw();
    particles.render();
  }

  //------------------------------------------------------------------
  //
  // This is the Game Loop function!
  //
  //------------------------------------------------------------------
  function gameLoop(time) {

    update(time - lastTimeStamp);
    lastTimeStamp = time;

    render();

    if (!cancelNextRequest) {
      requestAnimationFrame(gameLoop);
    }
  }

  function run() {
    lastTimeStamp = performance.now();

    // Start the animation loop
    cancelNextRequest = false;
    requestAnimationFrame(gameLoop);
  }

  return {
    initialize : initialize,
    run : run
  };
}(Engine.game, Engine.graphics, Engine.input, Engine.movable, Engine.ParticleSystem));
