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
    mat4 mFinal = uAspect * uProjection * uView * uModel;
    gl_Position = mFinal * aPosition;
    vColor = aColor;
}
