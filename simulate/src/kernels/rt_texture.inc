#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <hip/hip_runtime.h>
#include "hipgl/types.h"          // TextureState
#include "hipgl/texture_mip.h"    // TextureMipLevel, MAX_TEXTURE_MIP_LEVELS

namespace hipgl {
namespace physics {

// v_rcp_f32 + one Newton step (hipgl_rcp_nr).
__device__ inline float rt_rcp_nr(float x) {
    float r = __builtin_amdgcn_rcpf(x);
    r = __builtin_fmaf(__builtin_fmaf(-x, r, 1.0f), r, r);
    return r;
}

// c / 255.0f for an 8-bit c, bit-identical to the IEEE division for all 256
// inputs (checked exhaustively): q = c * fl(1/255), one exact-residual
// correction. 3 VALU instead of the ~10 of the general division sequence.
__device__ __forceinline__ float rt_div255(unsigned char c) {
    const float x = (float)c;
    constexpr float r = 1.0f / 255.0f;
    const float q = x * r;
    return __builtin_fmaf(__builtin_fmaf(-q, 255.0f, x), r, q);
}

__device__ inline void rt_load_texel(const TextureState& ts, int texid, int px, int py,
                                     float& r, float& g, float& b) {
    const int pidx = ts.tex_offset[texid] + (py * ts.tex_width[texid] + px) * 3;
    r = rt_div255(ts.tex_data[pidx]);
    g = rt_div255(ts.tex_data[pidx + 1]);
    b = rt_div255(ts.tex_data[pidx + 2]);
}

// Level-0 GL_REPEAT + GL_LINEAR, texel centres at (i+0.5)/w, rows top-first.
__device__ inline void rt_bilinear_2d(int texid, float u, float v, const TextureState& ts,
                                      float& cr, float& cg, float& cb) {
    cr = 1.0f; cg = 1.0f; cb = 1.0f;
    if (texid < 0 || texid >= ts.ntexture || !ts.tex_data ||
        !ts.tex_width || !ts.tex_height || !ts.tex_offset) return;
    const int tw = ts.tex_width[texid];
    const int th = ts.tex_height[texid];
    if (tw <= 0 || th <= 0) return;
    u = u - floorf(u);
    v = v - floorf(v);
    const float fx = u * (float)tw - 0.5f;
    const float fy = v * (float)th - 0.5f;
    const float fx0 = floorf(fx);
    const float fy0 = floorf(fy);
    const float dx = fx - fx0;
    const float dy = fy - fy0;
    int x0 = (int)fx0 % tw; if (x0 < 0) x0 += tw;
    int y0 = (int)fy0 % th; if (y0 < 0) y0 += th;
    int x1 = x0 + 1; if (x1 >= tw) x1 -= tw;
    int y1 = y0 + 1; if (y1 >= th) y1 -= th;
    float r00, g00, b00, r10, g10, b10, r01, g01, b01, r11, g11, b11;
    rt_load_texel(ts, texid, x0, y0, r00, g00, b00);
    rt_load_texel(ts, texid, x1, y0, r10, g10, b10);
    rt_load_texel(ts, texid, x0, y1, r01, g01, b01);
    rt_load_texel(ts, texid, x1, y1, r11, g11, b11);
    const float w00 = (1.0f - dx) * (1.0f - dy);
    const float w10 = dx * (1.0f - dy);
    const float w01 = (1.0f - dx) * dy;
    const float w11 = dx * dy;
    cr = r00*w00 + r10*w10 + r01*w01 + r11*w11;
    cg = g00*w00 + g10*w10 + g01*w01 + g11*w11;
    cb = b00*w00 + b10*w10 + b01*w01 + b11*w11;
}

// GL cube-map face table: +X -X +Y -Y +Z -Z = 0..5.
__device__ inline void rt_cube_select_face(float x, float y, float z, int& face, float& s, float& t) {
    const float ax = fabsf(x), ay = fabsf(y), az = fabsf(z);
    float sc, tc, ma;
    if (ax >= ay && ax >= az) {
        if (x > 0.0f) { face = 0; sc = -z; tc = -y; }
        else          { face = 1; sc =  z; tc = -y; }
        ma = ax;
    } else if (ay >= az) {
        if (y > 0.0f) { face = 2; sc =  x; tc =  z; }
        else          { face = 3; sc =  x; tc = -z; }
        ma = ay;
    } else {
        if (z > 0.0f) { face = 4; sc =  x; tc = -y; }
        else          { face = 5; sc = -x; tc = -y; }
        ma = az;
    }
    if (ma < 1e-20f) { face = 0; s = 0.5f; t = 0.5f; return; }
    s = 0.5f * (sc / ma + 1.0f);
    t = 0.5f * (tc / ma + 1.0f);
}

// Level-0 cube sample from a direction, CLAMP_TO_EDGE within the face.
__device__ inline void rt_cube_sample_dir(int texid, float dx, float dy, float dz,
                                          const TextureState& ts,
                                          float& cr, float& cg, float& cb) {
    cr = 1.0f; cg = 1.0f; cb = 1.0f;
    if (texid < 0 || texid >= ts.ntexture || !ts.tex_data ||
        !ts.tex_width || !ts.tex_height || !ts.tex_offset) return;
    const int w = ts.tex_width[texid];
    if (w <= 0) return;
    int face; float s, t;
    rt_cube_select_face(dx, dy, dz, face, s, t);
    const float fx = s * (float)w - 0.5f;
    const float fy = t * (float)w - 0.5f;
    const float fx0 = floorf(fx), fy0 = floorf(fy);
    const float ddx = fx - fx0, ddy = fy - fy0;
    int x0 = (int)fx0, y0 = (int)fy0;
    int x1 = x0 + 1,   y1 = y0 + 1;
    x0 = x0 < 0 ? 0 : (x0 > w - 1 ? w - 1 : x0);
    x1 = x1 < 0 ? 0 : (x1 > w - 1 ? w - 1 : x1);
    y0 = y0 < 0 ? 0 : (y0 > w - 1 ? w - 1 : y0);
    y1 = y1 < 0 ? 0 : (y1 > w - 1 ? w - 1 : y1);
    const int base = ts.tex_offset[texid] + face * w * w * 3;
    const unsigned char* T = ts.tex_data;
    const int p00 = base + (y0 * w + x0) * 3, p10 = base + (y0 * w + x1) * 3;
    const int p01 = base + (y1 * w + x0) * 3, p11 = base + (y1 * w + x1) * 3;
    const float w00 = (1.0f - ddx) * (1.0f - ddy);
    const float w10 = ddx * (1.0f - ddy);
    const float w01 = (1.0f - ddx) * ddy;
    const float w11 = ddx * ddy;
    cr = (T[p00+0]*w00 + T[p10+0]*w10 + T[p01+0]*w01 + T[p11+0]*w11) / 255.0f;
    cg = (T[p00+1]*w00 + T[p10+1]*w10 + T[p01+1]*w01 + T[p11+1]*w11) / 255.0f;
    cb = (T[p00+2]*w00 + T[p10+2]*w10 + T[p01+2]*w01 + T[p11+2]*w11) / 255.0f;
}

// Bilinear sample of one pyramid level (2D: repeat, cube: clamp in face).
__device__ inline void rt_sample_mip_level(const TextureState& ts, const TextureMipLevel& level,
                                           bool cube, int face, float u, float v,
                                           float& r, float& g, float& b) {
    const int w=level.width, h=cube ? w : level.height;
    if (!cube) { u-=floorf(u); v-=floorf(v); }
    const float x=u*w-.5f, y=v*h-.5f;
    const float xf=floorf(x), yf=floorf(y);
    const float dx=x-xf, dy=y-yf;
    int x0=(int)xf, x1=x0+1, y0=(int)yf, y1=y0+1;
    if (cube) {
        x0=max(0,min(w-1,x0)); x1=max(0,min(w-1,x1));
        y0=max(0,min(h-1,y0)); y1=max(0,min(h-1,y1));
    } else {
        if (x0<0) x0+=w;
        if (x1>=w) x1-=w;
        if (y0<0) y0+=h;
        if (y1>=h) y1-=h;
    }
    const int base=level.offset+(cube ? face*w*h*3 : 0);
    const unsigned char* data=ts.tex_data+base;
    const int p00=(y0*w+x0)*3, p10=(y0*w+x1)*3;
    const int p01=(y1*w+x0)*3, p11=(y1*w+x1)*3;
    const float w00=(1.f-dx)*(1.f-dy), w10=dx*(1.f-dy);
    const float w01=(1.f-dx)*dy, w11=dx*dy;
    constexpr float inv255=1.f/255.f;
    r=(data[p00]*w00+data[p10]*w10+data[p01]*w01+data[p11]*w11)*inv255;
    g=(data[p00+1]*w00+data[p10+1]*w10+data[p01+1]*w01+data[p11+1]*w11)*inv255;
    b=(data[p00+2]*w00+data[p10+2]*w10+data[p01+2]*w01+data[p11+2]*w11)*inv255;
}

// Project a direction derivative onto the selected cube face's (s, t).
__device__ inline void rt_cube_project_gradient(int face, const float* tc, const float* d,
                                                float s, float t, float& du, float& dv) {
    float ma, dma, ds, dt;
    if (face<2) {
        const float sign=face==0 ? 1.f : -1.f;
        ma=fabsf(tc[0]); dma=sign*d[0]; ds=-sign*d[2]; dt=-d[1];
    } else if (face<4) {
        const float sign=face==2 ? 1.f : -1.f;
        ma=fabsf(tc[1]); dma=sign*d[1]; ds=d[0]; dt=sign*d[2];
    } else {
        const float sign=face==4 ? 1.f : -1.f;
        ma=fabsf(tc[2]); dma=sign*d[2]; ds=sign*d[0]; dt=-d[1];
    }
    const float inv=.5f*rt_rcp_nr(fmaxf(ma,1.e-20f));
    du=(ds-(2.f*s-1.f)*dma)*inv;
    dv=(dt-(2.f*t-1.f)*dma)*inv;
}

// llvmpipe's approximate log2 LOD from rho^2 (texel units), rho2 > 1.
__device__ inline float rt_llvmpipe_lod(float rho2) {
    const uint32_t bits=__float_as_uint(rho2);
    const float log_rho2=float(int((bits>>23)&255)-127)
        +float(bits&0x7fffff)*(1.f/8388608.f);
    return fminf(.5f*log_rho2,float(MAX_TEXTURE_MIP_LEVELS-1));
}

// Trilinear sample at a given LOD (lod >= 0) -- the tail of the raster's
// sample_triangle_mipmapped. lod <= 0 (or no pyramid) samples level 0.
__device__ inline void rt_sample_lod(const TextureState& ts, int texid, bool cube, int face,
                                     float u, float v, float lod,
                                     float& r, float& g, float& b) {
    const int l0=(int)lod, l1=min(l0+1,MAX_TEXTURE_MIP_LEVELS-1);
    const float f=lod-l0;
    const TextureMipLevel* levels=ts.tex_mip+texid*MAX_TEXTURE_MIP_LEVELS;
    rt_sample_mip_level(ts,levels[l0],cube,face,u,v,r,g,b);
    if (f>0.f && levels[l0].offset!=levels[l1].offset) {
        float r1,g1,b1;
        rt_sample_mip_level(ts,levels[l1],cube,face,u,v,r1,g1,b1);
        r+=(r1-r)*f; g+=(g1-g)*f; b+=(b1-b)*f;
    }
}

// Texture lookup with screen derivatives, i.e. sample_triangle_texture with
// the derivative computation hoisted out. tc = (u, v) for 2D, the direction
// for cube; tx / ty = d(tc)/d(screen x) and d(tc)/d(screen y) in sample-grid
// units (3 components for cube, 2 used for 2D). Returns level-0 bilinear on
// magnification (rho^2 <= 1) or when no pyramid exists, exactly like the
// raster. lod_bias is added to the llvmpipe LOD (0 = raster identical).
template <bool CUBE>
__device__ inline void rt_tex_sample_grad(const TextureState& ts, int texid, const float* tc,
                                          const float* tx, const float* ty, float lod_bias,
                                          float& r, float& g, float& b) {
    if (!ts.tex_mip || texid<0 || texid>=ts.ntexture) {
        if (CUBE) rt_cube_sample_dir(texid,tc[0],tc[1],tc[2],ts,r,g,b);
        else      rt_bilinear_2d(texid,tc[0],tc[1],ts,r,g,b);
        return;
    }
    int face=0;
    float u=tc[0], v=tc[1], ux=tx[0], vx=tx[1], uy=ty[0], vy=ty[1];
    if constexpr (CUBE) {
        rt_cube_select_face(tc[0],tc[1],tc[2],face,u,v);
        rt_cube_project_gradient(face,tc,tx,u,v,ux,vx);
        rt_cube_project_gradient(face,tc,ty,u,v,uy,vy);
    }
    const float w=ts.tex_width[texid], h=CUBE ? w : ts.tex_height[texid];
    ux*=w; uy*=w; vx*=h; vy*=h;
    const float rho2=fmaxf(ux*ux+vx*vx,uy*uy+vy*vy);
    float lod = rho2 > 1.f ? rt_llvmpipe_lod(rho2) : 0.f;
    lod = fminf(fmaxf(lod + lod_bias, 0.f), float(MAX_TEXTURE_MIP_LEVELS-1));
    if (!(lod > 0.f)) {
        // GL_LINEAR magnification at level zero (raster: rho2 <= 1).
        if (CUBE) rt_cube_sample_dir(texid,tc[0],tc[1],tc[2],ts,r,g,b);
        else      rt_bilinear_2d(texid,tc[0],tc[1],ts,r,g,b);
        return;
    }
    rt_sample_lod(ts,texid,CUBE,face,u,v,lod,r,g,b);
}

// ── llvmpipe float path for sRGB 2D textures (RT_TEX_SRGB_DATA, RT_SH_MESA_TEX) 
// Texels stay sRGB-encoded; each tap is decoded with Mesa's polynomial
// (lp_build_srgb_to_linear) before the float bilinear / trilinear filter.
__device__ __forceinline__ float rt_mesa_srgb_dec(unsigned char b) {
    const float x = (float)b;
    if (x <= 15.0f) return x * (1.0f / (12.6f * 255.0f));
    return 0.0023f + x * ((0.0030f / 255.0f) + x * ((0.6935f / (255.0f * 255.0f)) +
                                                    x * (0.3012f / (255.0f * 255.0f * 255.0f))));
}

__device__ inline void rt_srgb_level(const unsigned char* base, const TextureMipLevel& lv, float u, float v,
                                     float& r, float& g, float& b) {
    const int w = lv.width, h = lv.height;
    u -= floorf(u); v -= floorf(v);
    const float x = u * w - .5f, y = v * h - .5f;
    const float xf = floorf(x), yf = floorf(y);
    const float dx = x - xf, dy = y - yf;
    int x0 = (int)xf, x1 = x0 + 1, y0 = (int)yf, y1 = y0 + 1;
    if (x0 < 0) x0 += w;
    if (x1 >= w) x1 -= w;
    if (y0 < 0) y0 += h;
    if (y1 >= h) y1 -= h;
    const unsigned char* d = base + lv.offset;
    const int p00 = (y0 * w + x0) * 3, p10 = (y0 * w + x1) * 3, p01 = (y1 * w + x0) * 3, p11 = (y1 * w + x1) * 3;
    float c[3];
    for (int k = 0; k < 3; ++k) {
        const float a = rt_mesa_srgb_dec(d[p00 + k]), bb = rt_mesa_srgb_dec(d[p10 + k]);
        const float cc = rt_mesa_srgb_dec(d[p01 + k]), dd = rt_mesa_srgb_dec(d[p11 + k]);
        const float top = a + (bb - a) * dx, bot = cc + (dd - cc) * dx;
        c[k] = top + (bot - top) * dy;
    }
    r = c[0]; g = c[1]; b = c[2];
}

__device__ inline void rt_tex_sample_srgb(const TextureState& ts, const unsigned char* sdata,
                                          const TextureMipLevel* smip, int texid, const float* tc,
                                          const float* tx, const float* ty, float lod_bias,
                                          float& r, float& g, float& b) {
    const TextureMipLevel* levels = smip + texid * MAX_TEXTURE_MIP_LEVELS;
    const int w = levels[0].width, h = levels[0].height;
    const float ux = tx[0] * w, uy = ty[0] * w, vx = tx[1] * h, vy = ty[1] * h;
    const float rho2 = fmaxf(ux * ux + vx * vx, uy * uy + vy * vy);
    float lod = rho2 > 1.f ? rt_llvmpipe_lod(rho2) : 0.f;
    lod = fminf(fmaxf(lod + lod_bias, 0.f), float(MAX_TEXTURE_MIP_LEVELS - 1));
    const int l0 = (int)lod, l1 = min(l0 + 1, MAX_TEXTURE_MIP_LEVELS - 1);
    const float f = lod - l0;
    rt_srgb_level(sdata, levels[l0], tc[0], tc[1], r, g, b);
    if (f > 0.f && levels[l1].offset != levels[l0].offset) {
        float r1, g1, b1;
        rt_srgb_level(sdata, levels[l1], tc[0], tc[1], r1, g1, b1);
        r += (r1 - r) * f; g += (g1 - g) * f; b += (b1 - b) * f;
    }
    (void)ts;
}

// ── llvmpipe AoS emulation (RT_SH_MESA_TEX) ─────────────────────────────────
// MuJoCo's reference frames come from Mesa llvmpipe, which samples 8-bit
// unorm 2D textures (not sRGB, not seamless cube) on its fixed-point "AoS"
// path (gallivm lp_bld_sample_aos.c, Mesa 23.2): texel coordinates in 24.8
// fixed point (iround(s * w * 256) - 128; NPOT REPEAT takes fract(s) first),
// 8-bit weights, every lerp v0 + ((w * (v1 - v0) + 128) >> 8) (pmulhrsw),
// x before y, and the mip lerp weight trunc(lod_fpart * 256).
__device__ inline void rt_aos_axis(float s, int len, bool pot, int& i0, int& i1, int& frac) {
    int c;
    if (pot) {
        c = (int)rintf(s * (float)(len << 8)) - 128;
        i0 = (c >> 8) & (len - 1);
    } else {
        const float sf = s - floorf(s);
        c = (int)rintf((sf * (float)len) * 256.0f) - 128;
        i0 = c >> 8;
        if (i0 < 0) i0 = len - 1;
        i0 = min(i0, len - 1);
    }
    frac = c & 255;
    i1 = (i0 == len - 1) ? 0 : i0 + 1;
}

__device__ __forceinline__ int rt_lerp8(int w, int a, int b) {
    return a + ((w * (b - a) * 128 + 16384) >> 15);
}

// One level, REPEAT, LINEAR; returns 8-bit channels.
__device__ inline void rt_aos_level(const TextureState& ts, const TextureMipLevel& lv, bool potw, bool poth,
                                    float u, float v, int c[3]) {
    int x0, x1, fx, y0, y1, fy;
    rt_aos_axis(u, lv.width, potw, x0, x1, fx);
    rt_aos_axis(v, lv.height, poth, y0, y1, fy);
    const unsigned char* d = ts.tex_data + lv.offset;
    const int w = lv.width;
    const unsigned char* p00 = d + (y0 * w + x0) * 3;
    const unsigned char* p01 = d + (y0 * w + x1) * 3;
    const unsigned char* p10 = d + (y1 * w + x0) * 3;
    const unsigned char* p11 = d + (y1 * w + x1) * 3;
    for (int k = 0; k < 3; ++k) {
        const int top = rt_lerp8(fx, p00[k], p01[k]);
        const int bot = rt_lerp8(fx, p10[k], p11[k]);
        c[k] = rt_lerp8(fy, top, bot);
    }
}

// 2D GL_LINEAR_MIPMAP_LINEAR / GL_LINEAR(mag) on the AoS path. tx/ty as in
// rt_tex_sample_grad (sample-grid derivatives of (u, v)); same LOD.
__device__ inline void rt_tex_sample_aos(const TextureState& ts, int texid, const float* tc,
                                         const float* tx, const float* ty, float lod_bias,
                                         float& r, float& g, float& b) {
    const int w = ts.tex_width[texid], h = ts.tex_height[texid];
    const float ux = tx[0] * w, uy = ty[0] * w, vx = tx[1] * h, vy = ty[1] * h;
    const float rho2 = fmaxf(ux * ux + vx * vx, uy * uy + vy * vy);
    float lod = rho2 > 1.f ? rt_llvmpipe_lod(rho2) : 0.f;
    lod = fminf(fmaxf(lod + lod_bias, 0.f), float(MAX_TEXTURE_MIP_LEVELS - 1));
    const bool potw = (w & (w - 1)) == 0, poth = (h & (h - 1)) == 0;
    int c[3];
    if (!ts.tex_mip) {
        const TextureMipLevel l0{ts.tex_offset[texid], w, h};
        rt_aos_level(ts, l0, potw, poth, tc[0], tc[1], c);
    } else {
        const TextureMipLevel* levels = ts.tex_mip + texid * MAX_TEXTURE_MIP_LEVELS;
        const int l0 = (int)lod;
        const int l1 = min(l0 + 1, MAX_TEXTURE_MIP_LEVELS - 1);
        rt_aos_level(ts, levels[l0], potw, poth, tc[0], tc[1], c);
        const int wl = (int)((lod - (float)l0) * 256.0f);
        if (wl > 0 && levels[l1].offset != levels[l0].offset) {
            int c1[3];
            rt_aos_level(ts, levels[l1], potw, poth, tc[0], tc[1], c1);
            for (int k = 0; k < 3; ++k) c[k] = rt_lerp8(wl, c[k], c1[k]);
        }
    }
    r = c[0] * (1.0f / 255.0f); g = c[1] * (1.0f / 255.0f); b = c[2] * (1.0f / 255.0f);
}

// ── hardware-GL sampler (RT_SH_HW_TEX) ──────────────────────────────────────
// What a GPU GL driver does for MuJoCo's GL_LINEAR_MIPMAP_LINEAR textures:
// float bilinear taps, float trilinear blend, LOD = log2(rho) with
// rho = max(|d(u,v)/dx|, |d(u,v)/dy|) in texels (GL 4.6 eq. 8.7, no
// anisotropy), and the glGenerateMipmap box pyramid in hw_mip (level 0 is the
// model's). Cube maps: CLAMP_TO_EDGE within the face, as the raster.
__device__ inline void rt_hw_level(const unsigned char* d, int w, int h, bool cube, int face,
                                   float u, float v, float& r, float& g, float& b) {
    if (!cube) { u -= floorf(u); v -= floorf(v); }
    const float x = u * w - .5f, y = v * h - .5f;
    const float xf = floorf(x), yf = floorf(y);
    const float dx = x - xf, dy = y - yf;
    int x0 = (int)xf, x1 = x0 + 1, y0 = (int)yf, y1 = y0 + 1;
    if (cube) {
        x0 = max(0, min(w - 1, x0)); x1 = max(0, min(w - 1, x1));
        y0 = max(0, min(h - 1, y0)); y1 = max(0, min(h - 1, y1));
        d += (size_t)face * w * h * 3;
    } else {
        if (x0 < 0) x0 += w;
        if (x1 >= w) x1 -= w;
        if (y0 < 0) y0 += h;
        if (y1 >= h) y1 -= h;
    }
    const int p00 = (y0 * w + x0) * 3, p10 = (y0 * w + x1) * 3, p01 = (y1 * w + x0) * 3, p11 = (y1 * w + x1) * 3;
    const float w00 = (1.f - dx) * (1.f - dy), w10 = dx * (1.f - dy), w01 = (1.f - dx) * dy, w11 = dx * dy;
    constexpr float inv255 = 1.f / 255.f;
    r = (d[p00] * w00 + d[p10] * w10 + d[p01] * w01 + d[p11] * w11) * inv255;
    g = (d[p00 + 1] * w00 + d[p10 + 1] * w10 + d[p01 + 1] * w01 + d[p11 + 1] * w11) * inv255;
    b = (d[p00 + 2] * w00 + d[p10 + 2] * w10 + d[p01 + 2] * w01 + d[p11 + 2] * w11) * inv255;
}

// hw_data / hw_mip: RT-owned pyramid (level 0 entries index ts.tex_data), or
// hw_mip == nullptr to use the model's pyramid ts.tex_mip.
template <bool CUBE>
__device__ inline void rt_tex_sample_hw(const TextureState& ts, const unsigned char* hw_data,
                                        const TextureMipLevel* hw_mip, int texid, const float* tc,
                                        const float* tx, const float* ty, float lod_bias,
                                        float& r, float& g, float& b) {
    int face = 0;
    float u = tc[0], v = tc[1], ux = tx[0], vx = tx[1], uy = ty[0], vy = ty[1];
    if constexpr (CUBE) {
        rt_cube_select_face(tc[0], tc[1], tc[2], face, u, v);
        rt_cube_project_gradient(face, tc, tx, u, v, ux, vx);
        rt_cube_project_gradient(face, tc, ty, u, v, uy, vy);
    }
    const int w0 = ts.tex_width[texid], h0 = CUBE ? w0 : ts.tex_height[texid];
    ux *= w0; uy *= w0; vx *= h0; vy *= h0;
    const float rho2 = fmaxf(ux * ux + vx * vx, uy * uy + vy * vy);
    float lod = rho2 > 1.f ? .5f * log2f(rho2) : 0.f;
    lod = fminf(fmaxf(lod + lod_bias, 0.f), float(MAX_TEXTURE_MIP_LEVELS - 1));
    const TextureMipLevel* L = hw_mip ? hw_mip + texid * MAX_TEXTURE_MIP_LEVELS
                             : (ts.tex_mip ? ts.tex_mip + texid * MAX_TEXTURE_MIP_LEVELS : nullptr);
    if (!L) {
        rt_hw_level(ts.tex_data + ts.tex_offset[texid], w0, h0, CUBE, face, u, v, r, g, b);
        return;
    }
    const int l0 = (int)lod, l1 = min(l0 + 1, MAX_TEXTURE_MIP_LEVELS - 1);
    const float f = lod - (float)l0;
    const unsigned char* d0 = (hw_mip && l0 > 0) ? hw_data : ts.tex_data;
    rt_hw_level(d0 + L[l0].offset, L[l0].width, CUBE ? L[l0].width : L[l0].height, CUBE, face, u, v, r, g, b);
    if (f > 0.f && (L[l1].width != L[l0].width || L[l1].height != L[l0].height)) {
        float r1, g1, b1;
        const unsigned char* d1 = hw_mip ? hw_data : ts.tex_data;
        rt_hw_level(d1 + L[l1].offset, L[l1].width, CUBE ? L[l1].width : L[l1].height, CUBE, face, u, v, r1, g1, b1);
        r += (r1 - r) * f; g += (g1 - g) * f; b += (b1 - b) * f;
    }
}

}  // namespace physics
}  // namespace hipgl
