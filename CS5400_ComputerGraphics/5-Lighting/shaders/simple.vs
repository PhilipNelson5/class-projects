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
attribute vec4 aColor;
attribute vec4 aNormal;
attribute vec4 aPosition;

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

  vec4 ambientLight = vec4(.1, .1, .1, 1);
  vec4 ambient = ambientLight * aColor;

  //mat4 invModelVew = transpose(inverse(uView*uModel));
  //transformedNormal = invModelVew * aNormal;

  vec4 transformedNormal = uNormal * aNormal;

  vec4 L0 = normalize(uLightPos0 - (uModel * uView * aPosition));
  vec4 L1 = normalize(uLightPos1 - (uModel * uView * aPosition));
  vec4 L2 = normalize(uLightPos2 - (uModel * uView * aPosition));

  vec4 diffuse0 = dot(transformedNormal, L0) * uLightColor0 * aColor;
  vec4 diffuse1 = dot(transformedNormal, L1) * uLightColor1 * aColor;
  vec4 diffuse2 = dot(transformedNormal, L2) * uLightColor2 * aColor;
  vec4 diffuse = diffuse0 + diffuse1 + diffuse2;

  vColor = clamp(ambient + diffuse, 0.0, 1.0);
  //vColor = transformedNormal;
  vColor.a = 1.0;
}
