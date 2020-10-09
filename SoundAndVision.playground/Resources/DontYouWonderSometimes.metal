#include <metal_stdlib>
using namespace metal;

#define numberOfLayers 16.0
#define iterations 23


#define layerThickness 0.02
#define  nbLayers 7

#define PI 3.1415926535

float sphere(float3 p, float radius) {
  return length(p) - radius;
}


float sat(float c) {
  return clamp(c, 0.0, 1.0);
}

float radians(float d) {
  return (PI * d) / 180.0;
}

float2 minfloatSelect(float2 a, float2 b) {
  return a.x<b.x?a:b;
}

float hash(float n) {
  return fract(sin(n) * 43758.5453);
}

float noise(float3 x) {
  float3 p = floor(x);
  float3 f = fract(x);

  f = f * f * (3.0 - 2.0 * f);

  float n = p.x + p.y * 57.0 + 113.0 * p.z;

  float res = mix(mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
                      mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y),
                  mix(mix( hash(n+113.0), hash(n+114.0),f.x),
                      mix( hash(n+170.0), hash(n+171.0),f.x),f.y),f.z);
  return res;
}

float2 map(float3 p, float time) {
  float slide = time / 1.0;
  float fr = fract(slide);
  int   fl = int(floor(slide));
  float2  vd = float2(100.0, -1.0);
  float cnoise = noise(p * 2. + time / 8.0) / 3.0;
  for( int i = 0; i < nbLayers; i++) {
    float m = fmod(float(i + fl), float(nbLayers));
    float r = 0.6 - m * layerThickness + ( 1.0 - fr) * layerThickness;
    float d = sphere(p, r);
    d = abs(d) - layerThickness / 2.0;
    float o =  - 4. * fract( (time + float(i)) / float(nbLayers));
  d = max(d, 1.5 + p.x  + o + cnoise);
  vd = minfloatSelect(float2(d, float(i)), vd);
  }
  return vd;
}


float3 calculateNormal(float3 p, float time) {
  const float h = 0.001;
  const float2 k = float2(1.0, -1.0);
  return normalize( k.xyy * map(p + k.xyy * h, time).x +
                    k.yyx * map(p + k.yyx * h, time).x +
                    k.yxy * map(p + k.yxy * h, time).x +
                    k.xxx * map(p + k.xxx * h, time).x );
}

float calcAO(float3 pos, float3 nor, float time) {
  float occ = 0.0;
  float sca = 1.0;
  for(int i=0; i<5; i++) {
    float hr = 0.01 + 0.12*float(i)/12.0;
    float3 aopos =  nor * hr + pos;
    float dd = map( aopos, time).x;
    occ += -(dd-hr)*sca;
    sca *= 0.95;
  }
  return sat(1.0 - 4. * occ);
}


float3 Render(float3 ro, float3 rd, float3 cd, float dist, float time) {
  float t = 0.5;
  float d;
  float m = 0.0;
  for(int i=0; i < 1024; i++) {
    float3  p = ro + t * rd;
    float2  h = map(p, time);
    t += h.x * 0.7;
    d = dot(t * rd, cd);
    m = h.y;
    if( abs(h.x) < 0.0001 || d > dist ) break;
  }

  float3 col = float3(0.9, 0.8, 0.7);

  if(d < dist) {
    float3 light = float3(0.0, 4.0, 2.0);
    float3 p = ro + t*rd;
    float3 n = calculateNormal(p, time);
    float3 v = normalize(ro - p);
    float3 l = normalize(light - p);
    float3 h = normalize(l + v);

    float3 diffcol = normalize(float3(1.0 + sin(m * 0.7 + 1.3), 1. + sin(m * 1.3 + 4.45), 1. + sin(m * 1.9 + 2.3)));
    float3 speccol = float3(1.,1.,1.);
    float3 ambcol = diffcol;
    float ao = calcAO(p, n, time);

    col = sat(dot(n,l)) * diffcol;
    col += pow(sat(dot(n, h)), 40.0) * speccol * 0.5;
    col += 0.2 * ambcol;
    col *= ao;
  }
  return col;
}

float3x3 setCamera(float3 ro, float3 ta) {
  float3 cw = normalize(ta - ro);
  float3 up = float3(0, 1, 0);
  float3 cu = normalize( cross(cw,up) );
  float3 cv = normalize( cross(cu,cw) );
  return float3x3(cu, cv, cw);
}

float3 desat(float3 c, float a) {
  float l = dot(c, float3(1. / 3.));
  return mix(c, float3(l), a);
}

kernel void boutSoundAndVision(texture2d<float, access::write> o[[texture(0)]],
                               constant float &time [[buffer(0)]],
                               constant float &frame [[buffer(1)]],
                               constant float2 *touchEvent [[buffer(2)]],
                               constant int &numberOfTouches [[buffer(3)]],
                               ushort2 gid [[thread_position_in_grid]]) {

  int width = o.get_width();
  int height = o.get_height();
  float2 res = float2(width, height);
  float2 co = float2(gid.xy);

  // NDJ
  float3 total = float3(0.0);

  float2 rook[4];
  rook[0] = float2( 1.0 / 8.0, 3.0 / 8.0);
  rook[1] = float2( 3.0 / 8.0, -1.0 / 8.0);
  rook[2] = float2(-1.0 / 8.0, -3. / 8.0);
  rook[3] = float2(-3.0 / 8.0, 1.0 / 8.0);

  for(int n = 0; n < 4; ++n) {
    // coordinates
    float2 o = rook[n];
    float2 p = (-res.xy + 2.0 * (co + o)) / res.y;

    // camera
    float theta  = radians(360.0) + time * 0.1;
    float phi  = radians(45.0) + radians(90.0) * - 1.0;
    float3 ro = 2.7 * float3( sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta));
    float3 ta = float3(0, -0.2, 0);

    // camera to world space
    float3x3 ca = setCamera(ro, ta);
    float3 rd =  ca * normalize(float3(p, 1.5));
    float3 col = Render(ro, rd, ca[2], 20.0, time);

    total += col;
  }
  total /= 4.0;
  total = desat(total, -0.4);
  total = pow(total, float3(1.0 / 2.2));

  o.write(float4(total, 1), gid);
}
