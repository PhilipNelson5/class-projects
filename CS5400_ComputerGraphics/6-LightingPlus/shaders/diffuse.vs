//
// Environment
uniform mat4 uAspect;
uniform mat4 uProjection;
uniform mat4 uView;
uniform mat4 uModel;
uniform mat4 uNormal;

//
// Geometry
attribute vec4 aColor;
attribute vec4 aNormal;
attribute vec4 aPosition;

//
// Output
varying vec4 vNormal;
varying vec4 vPosition;
varying vec4 vColor;

void main()
{
  gl_Position =
    uAspect
    * uProjection 
    * uView
    * uModel
    * aPosition;

  //mat4 invModelVew = transpose(inverse(uView*uModel));
  //transformedNormal = invModelVew * aNormal;

  vNormal = uNormal * aNormal;
  vPosition = uModel * uView * aPosition;
  vColor = aColor;

}
