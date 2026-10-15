// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Same contraction policy as rt_trace.hip (see there).
#ifdef RT_PARITY_FP
#pragma clang fp contract(off)
#else
#pragma clang fp contract(on)
#endif

#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include "hipgl/physics/ray_tracer.h"
#include "rt_math.inc"
#include "rt_frustum.inc"

namespace hipgl {
namespace physics {

namespace {

// Conservative growth of every world box. Analytic surfaces lie exactly on
// their AABB faces (an axis-aligned box IS its AABB), so an unpadded slab test
// can reject a true face hit by one ulp of rounding and punch pinholes.
constexpr float RT_AABB_REL = 1e-4f;
constexpr float RT_AABB_ABS = 1e-5f;

// One (slot, item) of K1: the RtInst of render geom `item`, or the RtCam of
// selected camera item - nrg.
__device__ __forceinline__ void rt_inst_item(const RtSceneArgs& a, int slot, int item)
{
    const int w = a.active[slot];

    if (item < a.nrg) {
        // ── instance ───────────────────────────────────────────────────────
        const RtGeom& g = a.geom[item];
        const bool site = (g.flags & RT_F_SITE) != 0;
        const float* gp;
        const float* gq;
        if (site) {
            const size_t sb = (size_t)w * a.nsite + g.pad0;
            gp = a.site_xpos + sb * 3;
            gq = a.site_xquat + sb * 4;
        } else {
            const size_t gb = (size_t)w * a.ngeom + g.gid;
            gp = a.geom_xpos + gb * 3;
            gq = a.geom_xquat + gb * 4;
        }
        RtInst I;
        const float q[4] = {gq[0], gq[1], gq[2], gq[3]};
        rt_quat_to_mat(q, I.R);
        I.p[0] = gp[0]; I.p[1] = gp[1]; I.p[2] = gp[2];
        I.li = item; I.pad = w;   // pad: world index (per-world site rgba lookup in shading)

        float c[3], e[3];
        if (g.type == GEOM_SPHERE) {
            // p +- r exactly (rotation-invariant), then the common pad.
            for (int k = 0; k < 3; k++) { c[k] = I.p[k]; e[k] = g.size[0]; }
        } else {
            // c = p + R (lo+hi)/2 ; e_i = sum_j |R_ij| (hi-lo)_j / 2
            float lc[3], le[3];
            for (int k = 0; k < 3; k++) {
                lc[k] = 0.5f * (g.llo[k] + g.lhi[k]);
                le[k] = 0.5f * (g.lhi[k] - g.llo[k]);
            }
            for (int i = 0; i < 3; i++) {
                c[i] = I.p[i] + I.R[i*3+0] * lc[0] + I.R[i*3+1] * lc[1] + I.R[i*3+2] * lc[2];
                e[i] = fabsf(I.R[i*3+0]) * le[0] + fabsf(I.R[i*3+1]) * le[1] + fabsf(I.R[i*3+2]) * le[2];
            }
        }
        for (int k = 0; k < 3; k++) {
            const float ek = e[k] * (1.0f + RT_AABB_REL) + RT_AABB_ABS;
            I.lo[k] = c[k] - ek;
            I.hi[k] = c[k] + ek;
        }
        if (site) {
            // Hidden in this world unless opaque (MuJoCo: alpha < 0.995 is the
            // blended pass; robosuite hides sites with alpha <= 0): a point
            // box at 1e30 fails every slab test with tmax <= RT_MAXVAL (an
            // inverted box would not: the slab test is min/max symmetric).
            const float alpha = (a.site_rgba && w < a.site_rgba_nw)
                ? a.site_rgba[((size_t)w * a.nsite + g.pad0) * 4 + 3] : g.rgba[3];
            if (!(alpha >= 0.995f))
                for (int k = 0; k < 3; k++) { I.lo[k] = 1e30f; I.hi[k] = 1e30f; }
        }
        a.inst[(size_t)slot * a.nrg + item] = I;
        return;
    }

    // ── camera ─────────────────────────────────────────────────────────────
    // Same composition as the raster's build_camera_state (bind.cpp): the body
    // pose is read for ANY body index >= 0, including the world body 0 (whose
    // xpos/xquat are the identity), so RT and raster agree bit-for-bit on the
    // inputs. Only a negative body (never produced by the parser) falls back
    // to the identity.
    const int ci = item - a.nrg;
    const int c = a.cam_ids[ci];
    const int b = a.cam_body[c];
    float bx = 0.f, by = 0.f, bz = 0.f, bqw = 1.f, bqx = 0.f, bqy = 0.f, bqz = 0.f;
    if (b >= 0) {
        const size_t bb = (size_t)w * a.nbody + b;
        bx = a.xpos[bb*3+0]; by = a.xpos[bb*3+1]; bz = a.xpos[bb*3+2];
        bqw = a.xquat[bb*4+0]; bqx = a.xquat[bb*4+1]; bqy = a.xquat[bb*4+2]; bqz = a.xquat[bb*4+3];
    }
    const float lx = a.cam_pos[c*3+0], ly = a.cam_pos[c*3+1], lz = a.cam_pos[c*3+2];
    // eye = pb + qrot(qb, cam_pos), written exactly as build_camera_state.
    const float tw_x = 2.0f*(bqy*lz - bqz*ly), tw_y = 2.0f*(bqz*lx - bqx*lz), tw_z = 2.0f*(bqx*ly - bqy*lx);
    RtCam C;
    C.eye[0] = lx + bqw*tw_x + (bqy*tw_z - bqz*tw_y) + bx;
    C.eye[1] = ly + bqw*tw_y + (bqz*tw_x - bqx*tw_z) + by;
    C.eye[2] = lz + bqw*tw_z + (bqx*tw_y - bqy*tw_x) + bz;
    // q = qb (x) cam_quat
    const float cqw = a.cam_quat[c*4+0], cqx = a.cam_quat[c*4+1];
    const float cqy = a.cam_quat[c*4+2], cqz = a.cam_quat[c*4+3];
    const float q[4] = {
        bqw*cqw - bqx*cqx - bqy*cqy - bqz*cqz,
        bqw*cqx + bqx*cqw + bqy*cqz - bqz*cqy,
        bqw*cqy - bqx*cqz + bqy*cqw + bqz*cqx,
        bqw*cqz + bqx*cqy - bqy*cqx + bqz*cqw};
    // Rc columns are the camera x (right), y (up), z (back) axes; the raster's
    // look_at(eye, eye + qrot(q,-z), qrot(q,y)) builds the same basis.
    rt_quat_to_mat(q, C.Rc);
    C.tanh_ = tanf(0.5f * a.cam_fovy[c]);
    C.pad[0] = C.pad[1] = C.pad[2] = 0.0f;
    a.cam[(size_t)slot * a.ncam + ci] = C;
}

__global__ void __launch_bounds__(256)
rt_inst_kernel(RtSceneArgs a)
{
    const int items = a.nrg + a.ncam;
    const long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)a.nslot * items) return;
    const int slot = (int)(tid / items);
    const int item = (int)(tid - (long long)slot * items);
    rt_inst_item(a, slot, item);
}

// K1 + the per-(slot, camera) view cull in one dispatch: one workgroup per
// slot writes that slot's instances and cameras, then (after the barrier) one
// wave per camera compacts the instances meeting the camera's whole view
// frustum into vlist (rt_view_cull, the same lists rt_view_cull_kernel wrote).
constexpr int RT_INST_CULL_BLOCK = 256;
__global__ void __launch_bounds__(RT_INST_CULL_BLOCK)
rt_inst_cull_kernel(RtSceneArgs a)
{
    const int slot = (int)blockIdx.x;
    const int items = a.nrg + a.ncam;
    for (int item = (int)threadIdx.x; item < items; item += RT_INST_CULL_BLOCK)
        rt_inst_item(a, slot, item);
    __syncthreads();
    const int wave = (int)threadIdx.x >> 6;
    for (int zc = wave; zc < a.ncam; zc += RT_INST_CULL_BLOCK / 64) {
        const size_t k = (size_t)slot * a.ncam + zc;
        const RtCam& C = a.cam[k];
        const float3 eye = make_float3(C.eye[0], C.eye[1], C.eye[2]);
        const RtTileFrustum fr = rt_tile_frustum(C.Rc, C.tanh_, a.W, a.H, 0, a.W, 0, a.H, a.clip, a.znear, a.zfar);
        rt_view_cull(fr, eye, a.inst + (size_t)slot * a.nrg, a.nrg, a.vlist + k * a.nrg, a.vcount + k);
    }
}

// One glGenerateMipmap level (see RayTracer::ensure_hw_mips): float bilinear
// at the destination texel centre (a 2x2 box for an even source size, centred
// scaling for an odd one), rounded once to nearest-even.
__global__ void __launch_bounds__(256)
rt_hw_mip_kernel(const unsigned char* __restrict__ src, int sw, int sh, int faces,
                 unsigned char* __restrict__ dst, int dw, int dh)
{
    const long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long per = (long long)dw * dh;
    if (k >= per * faces) return;
    const int f = (int)(k / per), r = (int)(k % per), y = r / dw, x = r % dw;
    // source coordinate of the destination centre: (2i+1) s / (2d) - 1/2
    const float sx = (float)((2.0 * x + 1.0) * sw / (2.0 * dw) - 0.5);
    const float sy = (float)((2.0 * y + 1.0) * sh / (2.0 * dh) - 0.5);
    const float xf = floorf(sx), yf = floorf(sy);
    const float fx = sx - xf, fy = sy - yf;
    const int x0 = max(0, min(sw - 1, (int)xf)), x1 = max(0, min(sw - 1, (int)xf + 1));
    const int y0 = max(0, min(sh - 1, (int)yf)), y1 = max(0, min(sh - 1, (int)yf + 1));
    const unsigned char* S = src + (size_t)f * sw * sh * 3;
    unsigned char* D = dst + (size_t)f * dw * dh * 3 + (size_t)r * 3;
    for (int c = 0; c < 3; ++c) {
        const float a = S[((size_t)y0 * sw + x0) * 3 + c], b = S[((size_t)y0 * sw + x1) * 3 + c];
        const float cc = S[((size_t)y1 * sw + x0) * 3 + c], d = S[((size_t)y1 * sw + x1) * 3 + c];
        const float v = (a * (1.f - fx) + b * fx) * (1.f - fy) + (cc * (1.f - fx) + d * fx) * fy;
        D[c] = (unsigned char)fminf(255.f, rintf(v));
    }
}

}  // namespace

void rt_launch_hw_mip(const unsigned char* src, int sw, int sh, int faces,
                      unsigned char* dst, int dw, int dh, hipStream_t stream) {
    const long long n = (long long)dw * dh * faces;
    if (n <= 0) return;
    const int block = 256;
    hipLaunchKernelGGL(rt_hw_mip_kernel, dim3((unsigned)((n + block - 1) / block)), dim3(block), 0, stream,
                       src, sw, sh, faces, dst, dw, dh);
}

bool rt_launch_inst(const RtSceneArgs& a, hipStream_t stream, hipEvent_t start) {
    const long long n = (long long)a.nslot * (a.nrg + a.ncam);
    if (n <= 0) return false;
    // `start` (optional) is recorded by the dispatch itself (no separate
    // marker packet ahead of K1).
    if (a.vlist && a.vcount && a.ncam > 0) {
        hipExtLaunchKernelGGL(rt_inst_cull_kernel, dim3((unsigned)a.nslot), dim3(RT_INST_CULL_BLOCK), 0, stream,
                              start, nullptr, 0u, a);
        return true;
    }
    const int block = 256;
    const unsigned grid = (unsigned)((n + block - 1) / block);
    hipExtLaunchKernelGGL(rt_inst_kernel, dim3(grid), dim3(block), 0, stream, start, nullptr, 0u, a);
    return true;
}

}  // namespace physics
}  // namespace hipgl
