// context.fillStyle = 'rgba(0, 0, 255, 1)';
// context.fillRect(100, 100, 100, 100);
//
// context.strokeStyle='rgba(255, 0, 0, 1)';
// context.strokeRect(100, 100, 100, 100);
//
// context.beginPath();
// context.moveTo(250,0);
// context.lineTo(500, 250);
// context.lineTo(0, 250);
// context.closePath();
//
// context.fillStyle='rgba(255, 0, 0, .5)';
// context.fill();
// context.strokeStyle-'rgba(0, 255, 0, 1)';
// context.stroke();

MyGame.main = (function(graphics) {

  console.log("game initialized");

  let myRect = graphics.Rectangle({
    x:100,
    y:100,
    width:100,
    height:100,
    rotation:0,
    fillStyle:'rgba(0,255,0,1)',
    strokeStyle:'rgba(255,0,0,1)'
  });

  function update(){
    myRect.updateRotation(.01);

  }

  function render() {
    graphics.clear();
    myRect.draw();
  }

  function gameLoop(){
    update()
    render()

    requestAnamationFrame(gameLoop);
  }

  requestAnamationFrame(gameLoop);
}) (MyGame.graphics)
