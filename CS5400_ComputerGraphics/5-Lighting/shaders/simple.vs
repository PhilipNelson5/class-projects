//
// Environment
uniform mat4 uAspect;
uniform mat4 uProjection;
uniform mat4 uView;
uniform mat4 uModel;
uniform mat4 uNormal;

//
// Light
uniform vec4 uLightPos0;
uniform vec4 uLightColor0;

uniform vec4 uLightPos1;
uniform vec4 uLightColor1;

uniform vec4 uLightPos2;
uniform vec4 uLightColor2;

//
// Geometry
attribute vec4 aPosition;
attribute vec4 aNormal;
attribute vec4 aColor;

//
// Output
varying vec4 vColor;

void main()
{
  gl_Position =
    uAspect
    * uProjection 
    * uView
    * uModel
    * aPosition;

  vec4 ambientLight = vec4(.25, .25, .25, 1);

  vec4 transformedNormal = aNormal * uNormal;

  vec4 L0 = normalize((uModel * uView * aPosition) - uLightPos0);
  vec4 L1 = normalize((uModel * uView * aPosition) - uLightPos1);
  vec4 L2 = normalize((uModel * uView * aPosition) - uLightPos2);

  vec4 ambient = ambientLight * aColor;

  vec4 diffuse0 = dot(transformedNormal, L0) * uLightColor0 * aColor;
  vec4 diffuse1 = dot(transformedNormal, L1) * uLightColor1 * aColor;
  vec4 diffuse2 = dot(transformedNormal, L2) * uLightColor2 * aColor;
  vec4 diffuse = diffuse0 + diffuse1 + diffuse2;

  vColor = diffuse;
}
