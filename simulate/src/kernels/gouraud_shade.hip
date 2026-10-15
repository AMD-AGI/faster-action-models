// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Gouraud (per-vertex) shading kernel -- matches MuJoCo's OpenGL fixed-function pipeline.
//
// MuJoCo uses GL_SMOOTH (Gouraud): lighting is computed at each vertex in eye space,
// colors are clamped to [0,1] at the vertex, then interpolated across the triangle.
// This replaces the GPU simulator's per-fragment Blinn-Phong, eliminating the 0.42x brightness hack.

#include <hip/hip_runtime.h>
#include "hipgl/types.h"

namespace hipgl {
namespace physics {
namespace kernels {

struct GouraudLight {
    float position[4];    // eye-space: (x,y,z,w); w=0 directional, w=1 positional
    float ambient[4];
    float diffuse[4];
    float specular[4];
    float spot_direction[3];
    float spot_cutoff;    // degrees (180 = omnidirectional)
    float spot_exponent;
    float constant_atten;
    float linear_atten;
    float quadratic_atten;
    int   enabled;
};

static constexpr int GOURAUD_MAX_LIGHTS = 8;

__global__ void kernel_gouraud_shade(
    const float* __restrict__ positions,   // [nverts*3] world space
    const float* __restrict__ normals_in,  // [nverts*3] world space
    float*       __restrict__ colors,      // [nverts*4] in: geom rgba, out: lit rgba (diffuse+ambient only)
    float*       __restrict__ spec_colors_out, // [nverts*3] out: separate specular RGB
    const float* __restrict__ specular,    // [nverts] per-vertex specular coeff
    const float* __restrict__ shininess,   // [nverts] per-vertex shininess (MuJoCo [0,1])
    int          nverts,
    const float* __restrict__ view_matrix,   // [16] column-major 4x4
    const float* __restrict__ normal_matrix, // [9] 3x3
    GouraudLight lights[GOURAUD_MAX_LIGHTS],
    int          nactive_lights,
    float        global_ambient)
{
    int vi = blockIdx.x * blockDim.x + threadIdx.x;
    if (vi >= nverts) return;

    // Read world-space position and normal
    float wx = positions[vi*3+0], wy = positions[vi*3+1], wz = positions[vi*3+2];
    float wn_x = normals_in[vi*3+0], wn_y = normals_in[vi*3+1], wn_z = normals_in[vi*3+2];

    // Transform to view/eye space
    float vx = view_matrix[0]*wx + view_matrix[4]*wy + view_matrix[8]*wz + view_matrix[12];
    float vy = view_matrix[1]*wx + view_matrix[5]*wy + view_matrix[9]*wz + view_matrix[13];
    float vz = view_matrix[2]*wx + view_matrix[6]*wy + view_matrix[10]*wz + view_matrix[14];

    float nx = normal_matrix[0]*wn_x + normal_matrix[3]*wn_y + normal_matrix[6]*wn_z;
    float ny = normal_matrix[1]*wn_x + normal_matrix[4]*wn_y + normal_matrix[7]*wn_z;
    float nz = normal_matrix[2]*wn_x + normal_matrix[5]*wn_y + normal_matrix[8]*wn_z;

    float nlen = sqrtf(nx*nx + ny*ny + nz*nz);
    if (nlen > 1e-8f) { nx /= nlen; ny /= nlen; nz /= nlen; }

    float cr = colors[vi*4+0], cg = colors[vi*4+1], cb = colors[vi*4+2], ca = colors[vi*4+3];

    // Per-vertex material
    float mat_spec = specular ? specular[vi] : 0.5f;
    // physics_renderer.hip already scales MuJoCo [0,1] shininess to OpenGL [0,128]
    float gl_shininess = shininess ? shininess[vi] : 64.0f;

    // MuJoCo uses GL_COLOR_MATERIAL: mat_ambient = mat_diffuse = rgba
    // Global ambient: GL_LIGHT_MODEL_AMBIENT * mat_ambient
    float lit_r = cr * global_ambient;
    float lit_g = cg * global_ambient;
    float lit_b = cb * global_ambient;
    float spec_r = 0.0f, spec_g = 0.0f, spec_b = 0.0f;

    for (int li = 0; li < GOURAUD_MAX_LIGHTS; li++) {
        if (!lights[li].enabled) continue;

        float lx, ly, lz;
        float atten = 1.0f;

        if (lights[li].position[3] == 0.0f) {
            lx = lights[li].position[0];
            ly = lights[li].position[1];
            lz = lights[li].position[2];
            float ll = sqrtf(lx*lx + ly*ly + lz*lz);
            if (ll > 1e-8f) { lx /= ll; ly /= ll; lz /= ll; }
        } else {
            lx = lights[li].position[0] - vx;
            ly = lights[li].position[1] - vy;
            lz = lights[li].position[2] - vz;
            float dist = sqrtf(lx*lx + ly*ly + lz*lz);
            if (dist > 1e-8f) { lx /= dist; ly /= dist; lz /= dist; }
            float denom = lights[li].constant_atten
                        + lights[li].linear_atten * dist
                        + lights[li].quadratic_atten * dist * dist;
            atten = (denom > 1e-8f) ? (1.0f / denom) : 1.0f;
        }

        // Spot light attenuation
        float spot_atten = 1.0f;
        if (lights[li].spot_cutoff < 180.0f && lights[li].position[3] != 0.0f) {
            float sdx = lights[li].spot_direction[0];
            float sdy = lights[li].spot_direction[1];
            float sdz = lights[li].spot_direction[2];
            float sdlen = sqrtf(sdx*sdx + sdy*sdy + sdz*sdz);
            if (sdlen > 1e-8f) { sdx /= sdlen; sdy /= sdlen; sdz /= sdlen; }
            float cos_angle = -(lx*sdx + ly*sdy + lz*sdz);
            float cos_cutoff = cosf(lights[li].spot_cutoff * 3.14159265f / 180.0f);
            if (cos_angle < cos_cutoff) {
                spot_atten = 0.0f;
            } else {
                spot_atten = powf(cos_angle, lights[li].spot_exponent);
            }
        }

        atten *= spot_atten;

        float raw_NdotL = nx*lx + ny*ly + nz*lz;
        float flip = (raw_NdotL < 0.0f) ? -1.0f : 1.0f;
        float fnx = nx * flip, fny = ny * flip, fnz = nz * flip;
        float NdotL = raw_NdotL * flip;

        lit_r += atten * (lights[li].ambient[0] * cr + lights[li].diffuse[0] * cr * NdotL);
        lit_g += atten * (lights[li].ambient[1] * cg + lights[li].diffuse[1] * cg * NdotL);
        lit_b += atten * (lights[li].ambient[2] * cb + lights[li].diffuse[2] * cb * NdotL);

        if (NdotL > 0.0f && mat_spec > 0.0f) {
            float vd = sqrtf(vx*vx + vy*vy + vz*vz);
            float ex, ey, ez;
            if (vd > 1e-8f) { ex = -vx/vd; ey = -vy/vd; ez = -vz/vd; }
            else             { ex = 0.0f;   ey = 0.0f;   ez = 1.0f;   }
            float hx = lx + ex, hy = ly + ey, hz = lz + ez;
            float hlen = sqrtf(hx*hx + hy*hy + hz*hz);
            if (hlen > 1e-8f) { hx /= hlen; hy /= hlen; hz /= hlen; }

            float NdotH = fnx*hx + fny*hy + fnz*hz;
            if (NdotH < 0.0f) NdotH = 0.0f;
            float spec_power = (gl_shininess > 0.0f) ? powf(NdotH, gl_shininess) : 1.0f;

            spec_r += atten * lights[li].specular[0] * mat_spec * spec_power;
            spec_g += atten * lights[li].specular[1] * mat_spec * spec_power;
            spec_b += atten * lights[li].specular[2] * mat_spec * spec_power;
        }
    }

    // Clamp diffuse+ambient to [0,1] at the vertex (key Gouraud behavior)
    lit_r = (lit_r > 1.0f) ? 1.0f : (lit_r < 0.0f ? 0.0f : lit_r);
    lit_g = (lit_g > 1.0f) ? 1.0f : (lit_g < 0.0f ? 0.0f : lit_g);
    lit_b = (lit_b > 1.0f) ? 1.0f : (lit_b < 0.0f ? 0.0f : lit_b);

    colors[vi*4+0] = lit_r;
    colors[vi*4+1] = lit_g;
    colors[vi*4+2] = lit_b;
    colors[vi*4+3] = ca;

    // Write separate specular (clamped to [0,1])
    if (spec_colors_out) {
        spec_colors_out[vi*3+0] = (spec_r > 1.0f) ? 1.0f : (spec_r < 0.0f ? 0.0f : spec_r);
        spec_colors_out[vi*3+1] = (spec_g > 1.0f) ? 1.0f : (spec_g < 0.0f ? 0.0f : spec_g);
        spec_colors_out[vi*3+2] = (spec_b > 1.0f) ? 1.0f : (spec_b < 0.0f ? 0.0f : spec_b);
    }
}

void launch_gouraud_shade(
    const float* d_positions,
    const float* d_normals,
    float* d_colors,
    float* d_spec_colors,
    const float* d_specular,
    const float* d_shininess,
    int nverts,
    const float* d_view_matrix,
    const float* d_normal_matrix,
    const GouraudLight* h_lights,
    int nactive_lights,
    float global_ambient,
    hipStream_t stream)
{
    if (nverts <= 0) return;

    GouraudLight* d_lights;
    hipMalloc(&d_lights, GOURAUD_MAX_LIGHTS * sizeof(GouraudLight));
    hipMemcpyAsync(d_lights, h_lights, GOURAUD_MAX_LIGHTS * sizeof(GouraudLight),
                   hipMemcpyHostToDevice, stream);

    const int block = 256;
    int grid = (nverts + block - 1) / block;
    hipLaunchKernelGGL(kernel_gouraud_shade, grid, block, 0, stream,
        d_positions, d_normals, d_colors, d_spec_colors, d_specular, d_shininess,
        nverts, d_view_matrix, d_normal_matrix,
        d_lights, nactive_lights, global_ambient);

    hipStreamSynchronize(stream);
    hipFree(d_lights);
}

} // namespace kernels
} // namespace physics
} // namespace hipgl
