MyGame.screens['game-play'] = (function(game, graphics, input, movable, particles) {
  'use strict';

  let mouseCapture = false,
    myMouse = input.Mouse(),
    myKeyboard = input.Keyboard(),
    balls = [],
    bar = null,
    lives = [],
    background = null,
    cancelNextRequest = false,
    lastTimeStamp,
    bricks = [],
    score = 0,
    bricksRemoved = 0,
    originSpeed = 300,
    originWidth = 127
  ;

  function setSize() {
    let canvas = document.getElementById('canvas-main');
    gwidth = canvas.width;
    gheight = canvas.height;
  }

  function initObjects() {
    console.log("initing objects");
    setSize();

    bar.center.x = gwidth/2;
    bar.center.y = gheight*9/10;
    bar.width = originWidth;
    bar.newWidth = originWidth;

    balls.length = 0;
    spawnBall();

    let cols = 14;
    let rows = 8;
    let dropDown = 75;
    let width = gwidth/(cols+1);
    let height = width/3;
    let gap = width/cols;

    // -----------------------------------------------------------------
    // Generate Bricks
    // -----------------------------------------------------------------
    for(let i = 1, j = 1; i <= rows; ++i, j += (i%2)?1:0)
    {
      let color = j;
      let source = './images/brick' + j + '.png';
      bricks.push([]);
      for(let j = 1; j < 2*cols; j+=2)
      {
        bricks[i-1].push(graphics.Texture( {
          image : source,
          center : { x : j*(width/2+gap/2), y : i*(height+gap)+dropDown},
          width : width, height : height,
          keep : true,
          id : i*8+j,
          row : i,
          color: color,
          destroy : function() {
            this.keep = false;
            particles.explodeBrick({
              position: {x: this.center.x, y:this.center.y},
              fill: {w: width, h: height},
              texture: source,
            });

          },
        }));
      }
    }

    // -----------------------------------------------------------------
    // Generate Lives
    // -----------------------------------------------------------------
    let numLives = 3;
    lives = [];
    for(let i = 0; i < numLives; ++i) {
      lives.push(makeLifeBar(i*50 + 50, 25));
    }

    score = 0;
    bricksRemoved = 0

    background = graphics.Texture( {
      image : './images/background.jpg',
      center : { x : window.innerWidth/2, y : window.innerHeight/2 },
      width : window.innerWidth, height : window.innerHeight,
    });

  }

  function makeLifeBar(x , y) {
    return graphics.Texture( {
      image : './images/bar.png',
      center : { x : x, y : y },
      width : 50, height : 24,
      newWidth : 127,
      growRate : 50,
      rotation : 0,
      vel : { x: 175 , y : 160 },
      moveRate : 300,
      destroy : () => {},
    });

  }

  function spawnBall() {
    balls.push(graphics.Texture( {
      image : './images/ball3.png',
      center : { x : bar.center.x, y : gheight*17/20 },
      width : 25, height : 25,
      rotation : 0,
      vel : { x: .5 , y : -.5 },
      speed : originSpeed,
      rotateRate : 3.14159,
      destroy : () => {},
    }));
  }

  function initialize() {
    console.log('game initializing...');

    bar = graphics.Texture( {
      image : './images/bar.png',
      center : { x : 0, y : 0 },
      width : originWidth, height : 24,
      newWidth : originWidth,
      growRate : 50,
      rotation : 0,
      vel : { x: 175 , y : 160 },
      rotateRate : 3.14159,
      moveRate : 300,
      destroy : () => {},
    });

    movable.MakeMovable(bar);

    // Create the keyboard input handler and register the keyboard commands
    myKeyboard.registerCommand(KeyEvent.DOM_VK_A, bar.moveLeft);
    myKeyboard.registerCommand(KeyEvent.DOM_VK_D, bar.moveRight);
    myKeyboard.registerCommand(KeyEvent.DOM_VK_LEFT, bar.moveLeft);
    myKeyboard.registerCommand(KeyEvent.DOM_VK_RIGHT, bar.moveRight);
    myKeyboard.registerCommand(KeyEvent.DOM_VK_Q, exitToMain);
    myKeyboard.registerCommand(KeyEvent.DOM_VK_ESCAPE, exitToMain);

  }

  function exitToMain() {

    // Stop the game loop by canceling the request
    // for the next animation frame
    cancelNextRequest = true;

    // Then, return to the main menu
    game.showScreen('main-menu');
  }

  function countDown(from) {
    if(from === 0) {
      cancelNextRequest = false;
      lastTimeStamp = performance.now();
      requestAnimationFrame(gameLoop);
      return;
    }
    cancelNextRequest = true;
    render();
    graphics.write({
      font: '75px serif',
      text: ''+from,
      x: gwidth/2, y: gheight/2,
    });

    window.setTimeout(() => countDown(from - 1), 1000);
  }

  // ------------------------------------------------------------------
  //
  // U P D A T E
  //
  // ------------------------------------------------------------------
  function detectCollisionsWall(ball) {
    let fail = false;
    let rad = ball.width/2;

    if(ball.center.x + rad >= gwidth) {
      ball.vel.x *= -1;
      ball.center.x = gwidth-ball.width/1.9;
      return false;
    }

    if(ball.center.x - rad <= 0) {
      ball.vel.x *= -1;
      ball.center.x = 0+ball.width/1.9;
      return false;
    }

    if(ball.center.y - rad <= 0) {
      ball.vel.y *= -1;
      ball.center.y = 0+ball.width/1.9;
      return false;
    }

    if(ball.center.y + rad >= gheight+ball.width) {
      return true;
    }
  }

  function updatePos(obj, dt){
    obj.center.x += obj.vel.x * obj.speed * dt * .001;
    obj.center.y += obj.vel.y * obj.speed * dt * .001;
  }

  // TODO: Only shrinks objects
  function updateSize(obj, dt) {
    if(obj.width === obj.newWidth)
      return;

    if(obj.width < obj.newWidth)
      obj.width = obj.newWidth;

    obj.width += obj.growRate * dt * .001 * -1;
  }

  function detectCollisionsBar(ball, bar) {
    if(isCollidingCircleRect(ball, bar))
    {
      let newX = (ball.center.x - bar.center.x)/((bar.width+ball.width)/2);
      ball.vel.x = newX > .9 ? .9 : newX;
      ball.vel.y = Math.abs(ball.vel.x) - 1;
    }
  }

  function isCollidingCircleRect(c, r) {
    let rad = c.width/2;
    let wid = r.width/2;
    let hig = r.height/2;

    if ( Math.abs(c.center.x - r.center.x) <= Math.abs(rad+wid) &&
      Math.abs(c.center.y - r.center.y) <= Math.abs(rad+hig) )
      return true;

    return false;
  }

  function detectCollisions(ball, obj) {
    if(isCollidingCircleRect(ball, obj)) {

      obj.destroy();

      if(obj.row === 1) {
        bar.newWidth = 64;
      }

      ++bricksRemoved;

      switch(bricksRemoved) {
        case 4:
        case 12:
        case 36:
        case 62:
          for(let b of balls)
            b.speed += 75;
      }

      let oldScore = Math.floor(score/100);
      switch(obj.color) {
        case 1:
          score += 5;
          break;
        case 2:
          score += 3;
          break;
        case 3:
          score += 2;
          break;
        case 4:
          score += 1;
          break;
      }

      if(bricks[obj.row-1].length === 1)
        score += 25;
      let newScore = Math.floor(score/100);

      if(oldScore != newScore) {
        spawnBall();
      }

      return true;
    }
    return false;
  }

  function renderScore() {
    let txt = "Score: " + score;

    graphics.writeLowerRight({
      font: '45px serif',
      text: txt,
      x: gwidth-15, y: gheight+15,
    });
  }

  function gameOver() {
    cancelNextRequest = true;
    MyGame.persistence.newScore(score);
    render();
    graphics.write({
      font: '100px serif',
      text: 'Game Over!',
      x: gwidth/2, y: gheight/2,
    });
    setTimeout(exitToMain, 500);
  }

  function update(dt) {
    myKeyboard.update(dt);
    myMouse.update(dt);

    for(let ball of balls) {

      // lose a ball
      if(detectCollisionsWall(ball)) {
        balls.splice(balls.indexOf(ball), 1);

        // lose a life
        if(balls.length === 0) {
          lives.pop();
          bar.width = originWidth;
          bar.newWidth = originWidth;
          // game over
          if(lives.length === 0){
            gameOver();
          }
          bricksRemoved = 0;
          countDown(4);
          spawnBall();
          continue;
        }
      }
      detectCollisionsBar(ball, bar);

      let didCollide = false;
      let win = true;
      for (let i = 0; i < bricks.length; ++i)
      {
        let keepMe = [];
        for (let j = 0; j < bricks[i].length; ++j)
        {
          didCollide = detectCollisions(ball, bricks[i][j]) || didCollide;
          if(bricks[i][j].keep)
            keepMe.push(bricks[i][j]);
        }
        bricks[i].length = 0;
        bricks[i] = keepMe;
        if(keepMe.length != 0)
          win = false;
      }

      if(win) {
        gameOver();
        return;
      }

      if(didCollide)
        ball.vel.y *= -1;

      updatePos(ball, dt);
      updateSize(bar, dt);
      particles.update(dt);
    }
  }

  function render() {
    graphics.clear();
    background.render.draw();
    for(let ball of balls)
      ball.render.draw();
    bar.render.draw();
    for(let list of bricks)
      for (let brick of list)
        brick.render.draw();
    for(let life of lives)
      life.render.draw();
    particles.render();
    renderScore();
  }

  // -----------------------------------------------------------------
  //
  // This is the Game Loop function!
  //
  // -----------------------------------------------------------------
  function gameLoop(time) {
    let dt = time - lastTimeStamp;

    update(dt);
    lastTimeStamp = time;

    render();

    if (!cancelNextRequest) {
      requestAnimationFrame(gameLoop);
    }
  }

  function run() {
    bricks = [];

    graphics.sizeCanvasToScreen();

    // give objects initial locations
    initObjects();

    lastTimeStamp = performance.now();

    // Start the animation loop
    cancelNextRequest = false;
    graphics.clear();
    particles.clear();
    countDown(4);
    requestAnimationFrame(gameLoop);
  }

  return {
    initialize : initialize,
    run : run
  };
}(MyGame.game, MyGame.graphics, MyGame.input, MyGame.movable, MyGame.ParticleSystem));
