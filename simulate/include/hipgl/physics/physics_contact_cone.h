#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

#if defined(__HIPCC__)
#define HIPGL_CONE_INLINE __host__ __device__ inline
#else
#define HIPGL_CONE_INLINE inline
#endif

namespace hipgl {
namespace physics {

// MuJoCo elliptic contact with normal, two sliding, and one torsional row.
// Friction rows use different units; friction[2] is a length, not the sliding
// coefficient. D must include MuJoCo's friction regularization rescaling:
// D[3] = D[1] * (friction[2] / friction[0])^2.
// Returns constraint cost; force = -gradient and hessian is row-major 4x4.
HIPGL_CONE_INLINE double elliptic_contact4(
    const double* x, const double* D, const double* friction,
    double impr_inv, double* force, int* state, double* hessian = nullptr)
{
    const double mu = friction[0] * impr_inv;
    const double mu2 = mu * mu;
    const double scale[4] = {mu, friction[0], friction[1], friction[2]};
    const double u[4] = {x[0] * scale[0], x[1] * scale[1],
                         x[2] * scale[2], x[3] * scale[3]};
    const double N = u[0];
    const double TT = u[1]*u[1] + u[2]*u[2] + u[3]*u[3];
    const double T = TT > 0.0 ? sqrt(TT) : 0.0;
    if (hessian) for (int i = 0; i < 16; ++i) hessian[i] = 0.0;

    if ((N >= mu*T) || (T <= 0.0 && N >= 0.0)) {
        if (state) *state = 0;
        if (force) for (int i = 0; i < 4; ++i) force[i] = 0.0;
        return 0.0;
    }
    if ((mu*N + T <= 0.0) || (T <= 0.0 && N < 0.0)) {
        if (state) *state = 1;
        double cost = 0.0;
        for (int i = 0; i < 4; ++i) {
            if (force) force[i] = -D[i]*x[i];
            if (hessian) hessian[i*4+i] = D[i];
            cost += 0.5*D[i]*x[i]*x[i];
        }
        return cost;
    }

    if (state) *state = 4;
    const double dm = D[0] / fmax(mu2 * (1.0 + mu2), 1e-20);
    const double nmt = N - mu*T;
    const double Tsafe = fmax(T, 1e-20);
    if (force) {
        force[0] = -dm*nmt*mu;
        for (int i = 1; i < 4; ++i)
            force[i] = (dm*nmt*mu/Tsafe) * u[i]*scale[i];
    }
    if (hessian) {
        // Factor the outer product using normalized tangential residuals.
        // Clamping T^3 changes this scale-invariant Hessian for T < ~1e-7.
        const double outer = mu*N/Tsafe;
        const double diag = mu2 - mu*N/Tsafe;
        hessian[0] = dm*mu2;
        for (int i = 1; i < 4; ++i) {
            const double v = (-mu/Tsafe*u[i]) * dm*scale[0]*scale[i];
            hessian[i] = v;
            hessian[i*4] = v;
            for (int j = 1; j < 4; ++j)
                hessian[i*4+j] = (outer*(u[i]/Tsafe)*(u[j]/Tsafe) + (i == j ? diag : 0.0))
                                  * dm*scale[i]*scale[j];
        }
    }
    return 0.5*dm*nmt*nmt;
}

} // namespace physics
} // namespace hipgl

#undef HIPGL_CONE_INLINE
