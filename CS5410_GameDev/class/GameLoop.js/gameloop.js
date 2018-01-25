let prevTime = performance.now()

function processInput(dTime){
}

function update(dTime){
}

function render(dTime){
  console.log(dTime);
}

function gameloop() {
  let curTime = performance.now();
  let dTime = curTime - prevTime;
  prevTime = curTime;

  processInput(dTime);
  update(dTime);
  render(dTime);

  requestAnimationFrame(gameloop);
}

gameloop();
