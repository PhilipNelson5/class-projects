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
attribute vec4 aColor;

//
// Output
varying vec3 vNormal;
varying vec3 vPosition;
varying vec3 vColor;

void main()
{
  gl_Position =
    uAspect
    * uProjection 
    * uView
    * uModel
    * aPosition;

  vNormal = normalize((uNormal * aNormal).xyz);
  vPosition = (uModel * uView * aPosition).xyz;
  vColor = aColor.xyz;

}
