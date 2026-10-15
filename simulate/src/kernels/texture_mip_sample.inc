// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Included inside hipgl::kernels, after the level-zero samplers. Mip storage
// is optional so raw TextureState callers retain their existing base-level
// behavior. LOD follows the projected texture footprint, independently of AA.

HIPGL_DEVICE inline void sample_mip_level(const TextureState& ts,
    const TextureMipLevel& level, bool cube, int face, float u, float v,
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
        // Wrapped normalized coordinates put the neighbors in [-1, extent].
        // Conditional wrap avoids a variable integer divide for every tap.
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

// A direction's derivative must be projected using the SAME selected face.
// Finite-differencing remapped face UVs would invent a huge footprint at seams.
HIPGL_DEVICE inline void cube_project_gradient(int face, const float* tc,
    const float* d, float s, float t, float& du, float& dv) {
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
    const float inv=.5f*hipgl_rcp_nr(fmaxf(ma,1.e-20f));
    du=(ds-(2.f*s-1.f)*dma)*inv;
    dv=(dt-(2.f*t-1.f)*dma)*inv;
}

template <bool CUBE>
HIPGL_DEVICE inline bool sample_triangle_mipmapped(
    const TriangleSetup& tri, int texid, float E0, float E1, float E2,
    const float* tc, int pixel_idx, int fb_width, int fb_height,
    const TextureState& ts, float& r, float& g, float& b) {
    if (!ts.tex_mip || texid<0 || texid>=ts.ntexture) return false;
    // llvmpipe's 2D footprint uses the GL lower-left corner of a 2x2 quad.
    // Cube directions use fine derivatives: X along this pixel's row and Y
    // along its column. HP's +screen Y is opposite to GL's +Y.
    const int px=pixel_idx%fb_width, py=pixel_idx/fb_width;
    const float ox=-(px&1), oy=(fb_height-1-py)&1;
    float bary[3]={E1,E2,E0};
    float wx[3], wy[3], wqx[3], wqy[3];
    float dqx=0.f, dqy=0.f, ddx=0.f, ddy=0.f;
#pragma unroll
    for (int i=0;i<3;++i) {
        const int e=(i+1)%3;
        const float iw=tri.v[i].inv_w;
        wx[i]=tri.edge_eq[e][0]*iw;
        wy[i]=-tri.edge_eq[e][1]*iw;
        wqx[i]=bary[i]*iw+ox*wx[i]-(CUBE ? 0.f : oy*wy[i]);
        wqy[i]=bary[i]*iw+(CUBE ? 0.f : ox*wx[i])-oy*wy[i];
        dqx+=wqx[i]; dqy+=wqy[i]; ddx+=wx[i]; ddy+=wy[i];
    }
    const float iqx=hipgl_rcp_nr(dqx), iqy=hipgl_rcp_nr(dqy);
    const float ix=hipgl_rcp_nr(dqx+ddx), iy=hipgl_rcp_nr(dqy+ddy);
    float tx[3]={0.f}, ty[3]={0.f};
#pragma unroll
    for (int k=0;k<(CUBE?3:2);++k) {
        float nqx=0.f, nqy=0.f, nx=0.f, ny=0.f;
#pragma unroll
        for (int i=0;i<3;++i) {
            const float v=tri.v[i].texcoord[k];
            nqx+=wqx[i]*v; nqy+=wqy[i]*v; nx+=wx[i]*v; ny+=wy[i]*v;
        }
        tx[k]=(nqx+nx)*ix-nqx*iqx;
        ty[k]=(nqy+ny)*iy-nqy*iqy;
    }
    int face=0;
    float u=tc[0], v=tc[1], ux=tx[0], vx=tx[1], uy=ty[0], vy=ty[1];
    if constexpr (CUBE) {
        cube_select_face(tc[0],tc[1],tc[2],face,u,v);
        cube_project_gradient(face,tc,tx,u,v,ux,vx);
        cube_project_gradient(face,tc,ty,u,v,uy,vy);
    }
    const float w=ts.tex_width[texid], h=CUBE ? w : ts.tex_height[texid];
    ux*=w; uy*=w; vx*=h; vy*=h;
    const float rho2=fmaxf(ux*ux+vx*vx,uy*uy+vy*vy);
    if (!(rho2>1.f)) return false;  // GL_LINEAR magnification at level zero.
    // Match llvmpipe's exponent/mantissa log approximation. Its mip weights
    // differ measurably from log2f, even with the same derivatives and levels.
    // rho2>1 here, so denormals and negative logarithms need no special case.
    const uint32_t bits=__float_as_uint(rho2);
    const float log_rho2=float(int((bits>>23)&255)-127)
        +float(bits&0x7fffff)*(1.f/8388608.f);
    const float lod=fminf(.5f*log_rho2,float(MAX_TEXTURE_MIP_LEVELS-1));
    const int l0=(int)lod, l1=min(l0+1,MAX_TEXTURE_MIP_LEVELS-1);
    const float f=lod-l0;
    const TextureMipLevel* levels=ts.tex_mip+texid*MAX_TEXTURE_MIP_LEVELS;
    sample_mip_level(ts,levels[l0],CUBE,face,u,v,r,g,b);
    if (f>0.f && levels[l0].offset!=levels[l1].offset) {
        float r1,g1,b1;
        sample_mip_level(ts,levels[l1],CUBE,face,u,v,r1,g1,b1);
        r+=(r1-r)*f; g+=(g1-g)*f; b+=(b1-b)*f;
    }
    return true;
}

HIPGL_DEVICE inline void sample_triangle_texture(
    const TriangleSetup& tri, int texid, float E0, float E1, float E2,
    const float* tc, int pixel_idx, int fb_width, int fb_height,
    const TextureState& ts, float& r, float& g, float& b) {
    const bool cube=ts.tex_iscube && texid>=0 && texid<ts.ntexture && ts.tex_iscube[texid];
    if (cube) {
        if (!sample_triangle_mipmapped<true>(tri,texid,E0,E1,E2,tc,pixel_idx,
                                             fb_width,fb_height,ts,r,g,b))
            cube_sample_dir(texid,tc[0],tc[1],tc[2],ts,r,g,b);
    } else {
        if (!sample_triangle_mipmapped<false>(tri,texid,E0,E1,E2,tc,pixel_idx,
                                              fb_width,fb_height,ts,r,g,b))
            bilinear_sample_texture(texid,tc[0],tc[1],ts,r,g,b);
    }
}
