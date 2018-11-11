//
// Environment
uniform mat4 uAspect;
uniform mat4 uProjection;
uniform mat4 uView;
uniform mat4 uModel;
uniform mat4 uNormal;

//
// Geometry
attribute vec4 aNormal;
attribute vec4 aPosition;

//
// Output
varying vec3 vNormal;
varying vec3 vPosition;

void main()
{
  gl_Position =
    uAspect
    * uProjection 
    * uView
    * uModel
    * aPosition;

  vNormal = (uNormal * aNormal).xyz;
  vPosition = (uModel * uView * aPosition).xyz;

}
