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
  vec3 ambientLight = vec3(.2, .2, .2);
  vec3 ambient = ambientLight * vColor.xyz;

  vec3 L0 = normalize(uLightPos0 - vPosition).xyz;
  vec3 L1 = normalize(uLightPos1 - vPosition).xyz;
  vec3 L2 = normalize(uLightPos2 - vPosition).xyz;

  vec3 diffuse0 = dot(vNormal.xyz, L0) * uLightColor0.xyz * vColor.xyz;
  vec3 diffuse1 = dot(vNormal.xyz, L1) * uLightColor1.xyz * vColor.xyz;
  vec3 diffuse2 = dot(vNormal.xyz, L2) * uLightColor2.xyz * vColor.xyz;
  vec3 diffuse = diffuse0 + diffuse1 + diffuse2;

  //vec4 R0 = reflect(-L0, normalize(vNormal));
  //vec4 R1 = reflect(-L1, normalize(vNormal));
  //vec4 R2 = reflect(-L2, normalize(vNormal));

  vec3 R0 = 2.0 * dot(vNormal.xyz, L0) * normalize(vNormal.xyz - L0);
  vec3 R1 = 2.0 * dot(vNormal.xyz, L1) * normalize(vNormal.xyz - L1);
  vec3 R2 = 2.0 * dot(vNormal.xyz, L2) * normalize(vNormal.xyz - L2);

  vec3 V = uEye.xyz - vPosition.xyz;

  vec3 specular0 = uShine.rgb * uLightColor0.rgb * pow(dot(V.xyz, R0.xyz), uShine.a);
  vec3 specular1 = uShine.rgb * uLightColor1.rgb * pow(dot(V.xyz, R1.xyz), uShine.a);
  vec3 specular2 = uShine.rgb * uLightColor2.rgb * pow(dot(V.xyz, R2.xyz), uShine.a);
  vec3 specular = specular0 + specular1 + specular2;

  gl_FragColor = vec4(clamp(ambient + diffuse + specular0, 0.0, 1.0), 1.0);
  //gl_FragColor.a = 1.0;
}
