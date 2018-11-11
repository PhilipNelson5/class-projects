precision highp float;

//
// Light
uniform vec4 uLightPos0;
uniform vec4 uLightColor0;

uniform vec4 uLightPos1;
uniform vec4 uLightColor1;

uniform vec4 uLightPos2;
uniform vec4 uLightColor2;

//
// Input
varying vec4 vNormal;
varying vec4 vPosition;
varying vec4 vColor;

void main()
{
  vec4 ambientLight = vec4(.2, .2, .2, 1);
  vec4 ambient = ambientLight * vColor;

  vec4 L0 = normalize(uLightPos0 - vPosition);
  vec4 L1 = normalize(uLightPos1 - vPosition);
  vec4 L2 = normalize(uLightPos2 - vPosition);

  vec4 diffuse0 = dot(vNormal, L0) * uLightColor0 * vColor;
  vec4 diffuse1 = dot(vNormal, L1) * uLightColor1 * vColor;
  vec4 diffuse2 = dot(vNormal, L2) * uLightColor2 * vColor;
  vec4 diffuse = diffuse0 + diffuse1 + diffuse2;

  gl_FragColor = clamp(ambient + diffuse, 0.0, 1.0);
  gl_FragColor.a = 1.0;
}
