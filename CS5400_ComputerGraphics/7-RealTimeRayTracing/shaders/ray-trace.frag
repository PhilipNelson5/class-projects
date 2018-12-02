precision highp float;

const int MAX_STACK_SIZE = 5;
const int STACK_POS_0 = 0;
const int STACK_POS_1 = 1;
const int STACK_POS_2 = 2;
const int STACK_POS_3 = 3;
const int STACK_POS_4 = 4;

const int SPHERE = 0;
const int PLANE = 1;

const int MATERIAL_DIFFUSE = 0;

// ------------------------------------------------------------------
//
// Structure declarations
//
// ------------------------------------------------------------------
struct StackItem
{
  float data;
  vec3 color;
};

struct Stack
{
  StackItem items[MAX_STACK_SIZE];
  int top;
} stack;

struct Ray
{
  vec3 o;
  vec3 d;
};

struct Sphere
{
  vec3 c;
  float r;
  vec3 color;
  int material;
};

struct Intersection
{
  bool didIntersect;
  float t;
  int material;
  vec3 color;
  vec3 normal;
};

//
uniform float uOffsetX;
uniform float uOffsetY;
uniform vec3 uEye;

//
// Geometry
uniform Sphere uSphereDiffuse;

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

//------------------------------------------------------------------------------
//
// Intersection with a sphere
//
//------------------------------------------------------------------------------
Intersection iSphere(Ray r, Sphere s)
{
  float A = dot(r.d, r.d);
  float B = 2.0 * dot(r.d, (r.o - s.c));
  float C = dot(r.o - s.c, r.o - s.c) - s.r * s.r;
  float descrim = B * B - 4.0 * A * C;

  if(descrim < 0.0)
    return Intersection(false, 0.0, -1, vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0));


  if(descrim == 0.0)
  {
    float t = dot(-r.d, (r.o - s.c)) / dot(r.d, r.d);
    vec3 normal = normalize((r.o + r.d * t) - s.c);
    return Intersection(true, t, s.material, s.color, normal);
  }

  float t1 = dot(-r.d, (r.o - s.c)) + sqrt(descrim) / dot(r.d, r.d);
  float t2 = dot(-r.d, (r.o - s.c)) - sqrt(descrim) / dot(r.d, r.d);

  if(t1 > t2)
  {
    vec3 normal = normalize((r.o + r.d * t1) - s.c);
    return Intersection(true, t1, s.material, s.color, normal);
  }

  vec3 normal = normalize((r.o + r.d * t2) - s.c);
  return Intersection(true, t2, s.material, s.color, normal);
}

//------------------------------------------------------------------------------
//
// Intersection with the scene geometry
//
//------------------------------------------------------------------------------
Intersection intersectScene(Ray r)
{
  Intersection i1 = iSphere(r, uSphereDiffuse);

  // return nearest intersection
  return i1;
}

//------------------------------------------------------------------------------
//
// Cast a ray into the scene
//
//------------------------------------------------------------------------------
vec3 castRay(Ray r)
{
  Intersection inter = intersectScene(r);

  if(inter.didIntersect)
  {
    if(inter.material == MATERIAL_DIFFUSE)
    {
      vec3 o = r.o + r.d * inter.t;
      vec3 d = normalize(uLightPos - o);
      Ray shadow = Ray(o, d);
      if(!intersectScene(shadow).didIntersect)
      {
        vec3 loc = r.o + r.d * inter.t;
        vec3 light = normalize(loc - uLightPos);
        vec3 diffuse = dot(inter.normal, light) * vec3(1.0, 1.0, 1.0) *  inter.color;
        return diffuse;
      }
      else
        return inter.color * .2;
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
  vec3 d = normalize(pxlCenter - uEye);
  Ray r = Ray(uEye, d);

  vec3 color = castRay(r);

  gl_FragColor.rgb = color;
  gl_FragColor.a = 1.0;
}
