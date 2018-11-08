//
// Environment
uniform mat4 uAspect;
uniform mat4 uProjection;
uniform mat4 uModel;
uniform mat4 uView;

//
// Geometry
attribute vec4 aPosition;

//
// Output
varying vec4 vTexCoord;

void main()
{
  mat4 view = mat4(mat3(uView));

  gl_Position =
    uAspect
    * uProjection 
    * view
    * uModel
    * aPosition;

  vTexCoord = gl_Position;
}
