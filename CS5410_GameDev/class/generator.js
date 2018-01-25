function makeCircle (r) {
  circle = {
    radius:r
  };

  Object.defineProperty(circle, 'area', {
    value: 3.14*circle.radius*circle.radius,
    writable: false,
    enumerable: true,
    configurable: true
  });

  return circle;
}

function report(shape){
  console.log('--report--')
  for(var property in shape) { // reflection
    console.log(property + ": " + shape[property]);
  }
}

c1 = makeCircle(1);
c2 = makeCircle(5);

report(c1);
report(c2);
