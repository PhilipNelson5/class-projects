precision mediump float;

uniform samplerCube uSampler;

//
//Input
varying vec4 vTexCoord;

void main()
{
  gl_FragColor = textureCube(uSampler, vTexCoord.xyz);
}
