precision highp float;

//
// Viewer Location
uniform vec4 uEye;

//
// Material Property
uniform vec4 uShine;

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

  //vec4 R0 = reflect(L0, normalize(vNormal));
  //vec4 R1 = reflect(L1, normalize(vNormal));
  //vec4 R2 = reflect(L2, normalize(vNormal));

  vec4 R0 = 2.0*dot(vNormal,L0)*vNormal-L0;
  vec4 R1 = 2.0*dot(vNormal,L1)*vNormal-L1;
  vec4 R2 = 2.0*dot(vNormal,L2)*vNormal-L2;

  vec4 V = uEye - vPosition;

  vec4 specular0 = (vColor+uShine) * uLightColor0 * dot(V, R0);
  vec4 specular1 = (vColor+uShine) * uLightColor1 * dot(V, R1);
  vec4 specular2 = (vColor+uShine) * uLightColor2 * dot(V, R2);
  vec4 specular = specular0 + specular1 + specular2;

  gl_FragColor = clamp(ambient + diffuse + specular0, 0.0, 1.0);
  gl_FragColor.a = 1.0;
}
