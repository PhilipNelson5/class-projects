//
// Environment
uniform mat4 uAspect;
uniform mat4 uProjection;
uniform mat4 uView;
uniform mat4 uModel;

//
// Geometry
attribute vec4 aPosition;
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

    vColor = aColor;
}
