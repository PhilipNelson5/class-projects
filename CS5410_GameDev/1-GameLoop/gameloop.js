let prevTime = performance.now()
let quit = false;
let fps = 0;

let events = [];

function makeEvent (n, i, c, f) {
  event = {
    name:n,
    interval:i,
    count:c,
    exec:f,
    elapsed:0
  };

  return event;
}

avgFPS = {
  arr:[],
  total:0,
  get avg() { return this.total/this.n; },
  n:60,
  next : function(newVal) {
    this.total -= this.arr.shift();
    this.total += newVal;
    this.arr.push(newVal);
  }
}

for(i = 0; i < avgFPS.n; ++i)
  avgFPS.arr.push(0);

showFPS = makeEvent(
  "Show FPS",
  100,
  100,
  function(){
    document.getElementById('fps').innerHTML="FPS: " + avgFPS.avg.toFixed(2);
  });

events.push(showFPS);

window.addEventListener('keydown', function(event) {
  // console.log(event.keyCode)
  if(event.keyCode == 32)
    quit = true;
}, false);

function processInput(dTime){
}

function update(dTime){
  fps = 1/dTime*1000;
  avgFPS.next(fps);
  events.forEach(function(e) {
    if(e.elapsed >= e.interval)
    {
      e.elapsed=0;
      e.exec();
    }
    else
      e.elapsed+=dTime;
  });
}

function render(dTime){
  //console.log(dTime);
}

function gameloop() {
  let curTime = performance.now();
  let dTime = curTime - prevTime;
  prevTime = curTime;

  processInput(dTime);
  update(dTime);
  render(dTime);

  if(!quit)
    requestAnimationFrame(gameloop);
}
