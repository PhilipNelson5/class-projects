let canvas = document.getElementById('canvas-main');
let context = canvas.getContext('2d');

function clear(){
  context.clearRect(0,0,canvas.width, canvas.height);
}

context.fillStyle = 'rgba(0, 0, 255, 1)';
context.fillRect(100, 100, 100, 100);

context.strokeStyle='rgba(255, 0, 0, 1)';
context.strokeRect(100, 100, 100, 100);

context.beginPath();
context.moveTo(250,0);
context.lineTo(500, 250);
context.lineTo(0, 250);
context.closePath();

context.fillStyle='rgba(255, 0, 0, .5)';
context.fill();
context.strokeStyle-'rgba(0, 255, 0, 1)';
context.stroke();

