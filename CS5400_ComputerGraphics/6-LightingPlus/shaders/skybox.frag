precision mediump float;

uniform samplerCube skybox;

//
//Input
varying vec3 vTexCoord;

void main()
{
  gl_FragColor = textureCube(skybox, vTexCoord);
  //gl_FragColor = vec4(vTexCoord.xyz, 1.0);
}
