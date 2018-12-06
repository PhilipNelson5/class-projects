precision highp float;

const int MAX_STACK_SIZE = 5;
const int STACK_POS_0 = 0;
const int STACK_POS_1 = 1;
const int STACK_POS_2 = 2;
const int STACK_POS_3 = 3;
const int STACK_POS_4 = 4;

const int SPHERE = 0;
const int PLANE = 1;

const int MATERIAL_DIFFUSE =    0;
const int MATERIAL_SPECULAR =   1;
const int MATERIAL_REFLECTIVE = 2;
const int MATERIAL_MIXTURE =    3;

// ------------------------------------------------------------------
//
// Structure declarations
//
// ------------------------------------------------------------------
struct Intersection
{
  bool didIntersect;
  float t;
  int material;
  vec3 color;
  vec3 normal;
};

struct Ray
{
  vec3 o;
  vec3 d;
};

struct StackItem
{
  Ray ray;
  vec3 color;
  int material;
};

struct Stack
{
  StackItem items[MAX_STACK_SIZE];
  int top;
} stack;

struct Sphere
{
  vec3 c;
  float r;
  vec3 color;
  int material;
};

struct Plane
{
  vec3 a;
  vec3 n;
  vec3 color;
  int material;
};

// Uniforms
uniform float uOffsetX;
uniform float uOffsetY;
uniform vec3 uEye;
float epsilon = 0.001;
uniform float uSeed;
uniform float uResolution;
uniform bool uMultiRay;

//
// Geometry
Sphere sky = Sphere(
    vec3(0.0, 0.0, 0.0),
    100.0,
    vec3(0.0, 1.0, 0.0),
    -1);
uniform Sphere uSphereDiffuse;
uniform Sphere uSphereReflective;
uniform Sphere uSphereMixture;
uniform Plane uPlane;

//
// Light
uniform vec3 uLightPos;

varying vec4 vPosition;

bool stackEmpty()
{
  return stack.top == STACK_POS_0;
}

bool stackPush(StackItem item)
{
  if (stack.top == (MAX_STACK_SIZE - 1)) return false;
  if (stack.top == STACK_POS_0) {
    stack.items[STACK_POS_0] = item;
  } else if (stack.top == STACK_POS_1) {
    stack.items[STACK_POS_1] = item;
  } else if (stack.top == STACK_POS_2) {
    stack.items[STACK_POS_2] = item;
  } else if (stack.top == STACK_POS_3) {
    stack.items[STACK_POS_3] = item;
  } else if (stack.top == STACK_POS_4) {
    stack.items[STACK_POS_4] = item;
  }

  stack.top++;
  return true;
}

StackItem stackPop()
{
  stack.top--;
  if (stack.top == STACK_POS_0) {
    return stack.items[STACK_POS_0];
  } else if (stack.top == STACK_POS_1) {
    return stack.items[STACK_POS_1];
  } else if (stack.top == STACK_POS_2) {
    return stack.items[STACK_POS_2];
  } else if (stack.top == STACK_POS_3) {
    return stack.items[STACK_POS_3];
  } else if (stack.top == STACK_POS_4) {
    return stack.items[STACK_POS_4];
  }
  // Danger Will Robinson, no return if stack underflow!!
}

//--------------------
// Random Function
//--------------------
//float num = rand(gl_FragCoord.xy/800.0);
float rand (vec2 st, float seed) {
  return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * seed);
}

//------------------------------------------------------------------------------
//
// Intersection with a plane
//
//------------------------------------------------------------------------------
Intersection iPlane(Ray r, Plane p)
{
  float denom = dot(r.d, p.n);
  float numer = dot((p.a - r.o), p.n);

  if(denom == 0.0 || (denom == 0.0 && numer == 0.0))
    return Intersection(false, 0.0, -1, vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0));

  float t = numer / denom;

  return Intersection(true, t, p.material, p.color, p.n);
}

//------------------------------------------------------------------------------
//
// Intersection with a sphere
//
//------------------------------------------------------------------------------
Intersection iSphere(Ray r, Sphere s)
{
  vec3 ro_sc = r.o - s.c;
  float A = dot(r.d, r.d);
  float B = 2.0 * dot(r.d, ro_sc);
  float C = dot(ro_sc, ro_sc) - (s.r * s.r);
  float descrim = (B * B) - (4.0 * A * C);

  if(descrim < 0.0)
    return Intersection(false, 0.0, -1, vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0));

  float t;

  if(descrim == 0.0)
  {
    t = dot(-r.d, (r.o - s.c)) / A / 2.0;
  }
  else
  {
    float t1 = (-B + sqrt(descrim)) / (A * 2.0);
    float t2 = (-B - sqrt(descrim)) / (A * 2.0);

    t = t1 < t2 ? t1 : t2;
  }


  vec3 iloc = r.o + r.d * t;
  vec3 normal = normalize(iloc - s.c);
  return Intersection(true, t, s.material, s.color, normal);
}

bool shadowIntersect(Ray r)
{
  Intersection i1 = iPlane(r, uPlane);
  Intersection i2 = iSphere(r, uSphereDiffuse);
  Intersection i3 = iSphere(r, uSphereReflective);
  Intersection i4 = iSphere(r, uSphereMixture);

  return (i1.didIntersect && i1.t > 0.0)
    || (i2.didIntersect && i2.t > 0.0)
    || (i3.didIntersect && i3.t > 0.0)
    || (i4.didIntersect && i4.t > 0.0);
}

//------------------------------------------------------------------------------
//
// Intersection with the scene geometry
//
//------------------------------------------------------------------------------
Intersection intersectScene(Ray r)
{
  Intersection i1 = iPlane(r, uPlane);
  Intersection i2 = iSphere(r, uSphereDiffuse);
  Intersection i3 = iSphere(r, uSphereReflective);
  Intersection i4 = iSphere(r, uSphereMixture);

  Intersection close;

  if((i1.didIntersect && i1.t > 0.0)
      || (i2.didIntersect && i2.t > 0.0)
      || (i3.didIntersect && i3.t > 0.0)
      || (i4.didIntersect && i4.t > 0.0)
    )
  {

    if(i1.didIntersect && i1.t > 0.0)
      close = i1;
    if(i2.didIntersect && i2.t > 0.0)
      close = i2;
    if(i3.didIntersect && i3.t > 0.0)
      close = i3;
    if(i4.didIntersect && i4.t > 0.0)
      close = i4;

    if(i1.didIntersect && i1.t > 0.0 && i1.t < close.t)
      close = i1;
    if(i2.didIntersect && i2.t > 0.0 && i2.t < close.t)
      close = i2;
    if(i3.didIntersect && i3.t > 0.0 && i3.t < close.t)
      close = i3;
    if(i3.didIntersect && i4.t > 0.0 && i4.t < close.t)
      close = i4;

  }
  else
  {
    Intersection iSky = iSphere(r, sky);
    vec3 iloc = r.o + r.d * iSky.t;
    vec3 norm = normalize(iloc) +.75;
    close = Intersection(false, 0.0, -1, vec3(0.0, 0.0, 0.0), norm);
  }

  // return nearest intersection
  return close;
}

//------------------------------------------------------------------------------
//
// Cast a ray into the scene
//
//------------------------------------------------------------------------------
vec3 castRay(Ray ray)
{
  stackPush(StackItem(ray, vec3(0.0, 0.0, 0.0), -1));
  StackItem item;
  for (int stackTop = 0; stackTop < MAX_STACK_SIZE; stackTop++)
  {
    if (stackEmpty()) 
      break;

    item = stackPop();
    Ray r = item.ray;
    Intersection inter = intersectScene(r);

    if(inter.didIntersect)
    {
      if(inter.material == MATERIAL_SPECULAR || inter.material == MATERIAL_DIFFUSE)
      {
        vec3 o = r.o + r.d * inter.t;
        vec3 d = normalize(uLightPos - o);
        Ray shadow = Ray(o, d);
        shadow.o += shadow.d * epsilon;

        if(!shadowIntersect(shadow))
        {
          vec3 diffuse = dot(inter.normal, d) * inter.color;
          if(inter.material == MATERIAL_DIFFUSE)
          {
            return item.color + diffuse;
          }

          vec3 reflected = reflect(-d, inter.normal);
          vec3 V = normalize(uEye - o);
          vec3 specular = pow(dot(V, reflected), 100.0) * vec3(1.0, 1.0, 1.0);

          if(item.material == MATERIAL_MIXTURE)
          {
            return item.color + 0.2 * (diffuse + specular);
          }

          return item.color + diffuse + specular;
        }
        else
          return vec3(0.0, 0.0, 0.0);
      }

      if(inter.material == MATERIAL_REFLECTIVE)
      {
        vec3 o = r.o + r.d * inter.t;
        vec3 d = normalize(reflect(r.d, inter.normal));
        stackPush(StackItem(Ray(o, d), item.color, MATERIAL_REFLECTIVE));
      }

      if(inter.material == MATERIAL_MIXTURE)
      {
        vec3 o = r.o + r.d * inter.t;
        vec3 d = normalize(uLightPos - o);
        Ray shadow = Ray(o, d);
        shadow.o += shadow.d * epsilon;

        if(!shadowIntersect(shadow))
        {
          vec3 diffuse = dot(inter.normal, d) * inter.color;
          vec3 reflected = reflect(-d, inter.normal);
          vec3 V = normalize(uEye - o);
          vec3 specular = pow(dot(V, reflected), 100.0) * vec3(1.0, 1.0, 1.0);
          item.color += 0.8 * (diffuse + specular);
        }
        d = normalize(reflect(r.d, inter.normal));
        stackPush(StackItem(Ray(o, d), item.color, MATERIAL_MIXTURE));
      }
    }
    else
    {
      return inter.normal;
    }
  }
  return vec3(0.0, 0.0, 0.0);
}

//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------
void main()
{
  stack.top = STACK_POS_0;

  vec3 pxlCenter = vec3(vPosition.x + uOffsetX, vPosition.y - uOffsetY, 0);

  if(uMultiRay)
  {
    float num1 = rand(gl_FragCoord.xy/uResolution, uSeed + 0.0) * .25;
    float num2 = rand(gl_FragCoord.xy/uResolution, uSeed + 500.0) * .25;
    float num3 = rand(gl_FragCoord.xy/uResolution, uSeed + 1000.0) * .25;
    float num4 = rand(gl_FragCoord.xy/uResolution, uSeed + 1500.0) * .25;

    vec3 d1 = normalize(vec3(pxlCenter.x + uOffsetX + num1 * uOffsetX, pxlCenter.y + uOffsetY + num4 * uOffsetY, pxlCenter.z) - uEye);
    vec3 d2 = normalize(vec3(pxlCenter.x + uOffsetX - num2 * uOffsetX, pxlCenter.y + uOffsetY + num1 * uOffsetY, pxlCenter.z) - uEye);
    vec3 d3 = normalize(vec3(pxlCenter.x + uOffsetX - num3 * uOffsetX, pxlCenter.y + uOffsetY - num2 * uOffsetY, pxlCenter.z) - uEye);
    vec3 d4 = normalize(vec3(pxlCenter.x + uOffsetX + num4 * uOffsetX, pxlCenter.y + uOffsetY - num3 * uOffsetY, pxlCenter.z) - uEye);

    Ray r1 = Ray(uEye, d1);
    Ray r2 = Ray(uEye, d2);
    Ray r3 = Ray(uEye, d3);
    Ray r4 = Ray(uEye, d4);

    vec3 color1 = castRay(r1);
    vec3 color2 = castRay(r2);
    vec3 color3 = castRay(r3);
    vec3 color4 = castRay(r4);

    gl_FragColor.rgb = (color1 + color2 + color3 + color4)/4.0;
  }
  else
  {
    vec3 d = normalize(pxlCenter - uEye);
    Ray r = Ray(uEye, d);
    vec3 color = castRay(r);
    gl_FragColor.rgb = color;
  }

  gl_FragColor.a = 1.0;
  //gl_FragColor = vec4(vec3(rand(gl_FragCoord.xy/800.0)), 1.0);
}
