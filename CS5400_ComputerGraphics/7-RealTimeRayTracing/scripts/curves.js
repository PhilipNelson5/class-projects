let curveSegments = 10;     // Initial setting for how many line segments to use in curve rendering
let du = 1 / curveSegments;
let BEZ = [[[]]];
let C = [];
let U = [[]];

function factorial(n) {
  let fact = 1;
  for (let i = 1; i <= n; ++i) {
    fact *= i;
  }

  return fact;
}

/**
 * @param controls {
 *   p0: {x: , y: } The first control point
 *   p1: {x: , y: } The second control point
 *   s2: {x: , y: } The third control point
 *   s3: {x: , y: } The fourth control point
 * }
 * @param showPoints  Boolean flag to show the intersection of line segments
 * @param showLine    Boolean flag to show the drawn curve
 * @param showControl Boolean flag to show the control points
 * @param color       Color of the line
 */
function drawCurveBezier(controls, segment) {
  let px, py;
  px = controls.p0.x * BEZ[0][segment]
    + controls.p1.x * BEZ[1][segment]
    + controls.p2.x * BEZ[2][segment]
    + controls.p3.x * BEZ[3][segment];

  py = controls.p0.y * BEZ[0][segment]
    + controls.p1.y * BEZ[1][segment]
    + controls.p2.y * BEZ[2][segment]
    + controls.p3.y * BEZ[3][segment];

  return {px,py};
}

function setCurveSegments(segments) {
  curveSegments = segments;
  du = 1 / curveSegments;
  for (let k = 0; k <= 3; ++k) {
    let u = 0;
    BEZ[k] = [];
    C[k] = factorial(3) / factorial(k) / factorial(3 - k);
    for (let i = 0; i <= curveSegments; ++i, u += du) {
      BEZ[k][i] = C[k] * Math.pow(u, k) * Math.pow((1 - u), (3 - k));
      U[i] = [1];
      for (let p = 1; p <= 3; ++p) {
        U[i][p] = U[i][p - 1] * u;
      }
    }
  }
}
