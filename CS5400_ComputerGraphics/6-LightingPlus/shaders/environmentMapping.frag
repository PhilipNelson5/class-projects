precision mediump float;

uniform samplerCube uSampler;

uniform bool uReflection;
uniform float uRefractiveIndex;

// Viewer Location
uniform vec3 uEye;

// Input
varying vec3 vPosition;
varying vec3 vNormal;

void main()
{
  vec3 r;
  if(uReflection)
  {
    r = reflect(normalize(vPosition - uEye), vNormal);
  }
  else
  {
    r = refract(normalize(vPosition - uEye), vNormal, 1.0/uRefractiveIndex);
  }
  gl_FragColor = textureCube(uSampler, r);
}
