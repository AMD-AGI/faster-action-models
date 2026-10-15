// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Newton solve and M_hat solve for nv > 64 (RoboCasa-GR1 tabletop: nv 64-87).
// Included by physics_solver.hip; only built when KERNEL_MAXNV_VALUE > 64.
//
// kernel_newton_solve_t and kernel_mhat_solve_t give each dof one lane of a
// 64-lane wave (`active = t < nv`), so dofs >= 64 were never solved. These
// kernels run one wave per env too, but lane t owns dofs t, t + 64, ...
// and the M / H tiles live in LDS (2 x 96 x 97 doubles at KERNEL_MAXNV 96).
//
// They minimise the same objective as kernel_newton_solve_t,
//     0.5 a'Ma - a'qfrc_smooth + sum of constraint costs (same row types, same
//     elliptic cones via elliptic_contact4),
// with the same smooth force, warm-start choice, regularised Cholesky and stop
// tests. M is positive definite, so the objective is strictly convex and both
// kernels converge to the same unique minimiser; only the path differs: the
// line search here is a safeguarded 1-D Newton on the exact directional
// derivative instead of the LIBERO kernel's backtracking schedule.

constexpr int WIDE_BLK = 256;           // one workgroup of 4 waves per env (thread stride everywhere)
constexpr int WIDE_WAVES = WIDE_BLK / 64;

// Block-wide sum of N values (every thread gets the total): wave butterflies, then the 4 wave sums added
// in a fixed order, so the result is deterministic.
template <int N>
HIPGL_DEVICE __forceinline__ void wide_reduce(double (&v)[N]) {
    __shared__ double s_part[N][WIDE_WAVES];
    wf_reduce_sum_dn<N>(v);
    const int w = threadIdx.x >> 6;
    __syncthreads();
    if ((threadIdx.x & 63) == 0)
        for (int i = 0; i < N; i++) s_part[i][w] = v[i];
    __syncthreads();
    for (int i = 0; i < N; i++) {
        double t = s_part[i][0];
        for (int k = 1; k < WIDE_WAVES; k++) t += s_part[i][k];
        v[i] = t;
    }
}
HIPGL_DEVICE __forceinline__ double wide_reduce1(double x) { double v[1] = {x}; wide_reduce<1>(v); return v[0]; }
constexpr int WIDE_LS_MAXITER = 50;
// Sparse copy of J staged in LDS once per dispatch (J is fixed during the solve; a contact row
// touches only the dofs of its two bodies). Scenes past either cap use the dense global-J path.
constexpr int WIDE_MAXROWS = 1024;
constexpr int WIDE_NNZ = 6144;
constexpr double WIDE_LS_TOL = 1e-6;    // |phi'(alpha)| <= tol * |phi'(0)|

// Constraint cost, its first and second derivative along jv at step alpha
// (x_c + alpha * jv_c), summed over this lane's rows and reduced across the
// wave. force/state (optional) receive the per-row force and state at that
// point, with the codes kernel_newton_solve_t uses (0 off, 1 quadratic,
// 2 / 3 frictionloss linear zone, 4 elliptic middle zone).
HIPGL_DEVICE inline void wide_constraint_eval(
    int t, int nc, int cone_type, double impr_inv,
    const double* x, const double* jv, double alpha,
    const double* D, const double* lo, const double* hi,
    const int* par, const int* sib, const double* mu, const bool* is_en,
    double* force, int* state, double& cost, double& d1, double& d2)
{
    double acc[3] = {0.0, 0.0, 0.0};
    for (int c = t; c < nc; c += WIDE_BLK) {
        const int par_c = par[c];
        if (cone_type == 1 && (par_c >= 0 || is_en[c])) continue;
        const double j = jv ? jv[c] : 0.0;
        const double xa = x[c] + alpha * j;
        const double Dc = D[c], lo_c = lo[c], hi_c = hi[c];
        double fo = 0.0;
        int st = 0;
        if (lo_c < -1e15 && hi_c > 1e15) {
            fo = -Dc * xa; st = 1;
        } else if (par_c < 0 && fabs(lo_c) < 1e-6 && hi_c > 1e15) {
            if (xa < 0.0) { fo = -Dc * xa; st = 1; }
        } else if (par_c < 0) {
            const double f = fabs(hi_c);
            const double rf = (Dc > 1e-10) ? f / Dc : 0.0;
            if      (xa <= -rf) { fo =  f; st = 2; acc[0] += f * (fabs(xa) - 0.5 * rf); }
            else if (xa >=  rf) { fo = -f; st = 3; acc[0] += f * (fabs(xa) - 0.5 * rf); }
            else                { fo = -Dc * xa; st = 1; }
        } else {
            if (xa < 0.0) { fo = -Dc * xa; st = 1; }
        }
        if (st == 1) { acc[0] += 0.5 * Dc * xa * xa; acc[2] += Dc * j * j; }
        acc[1] -= fo * j;
        if (force) force[c] = fo;
        if (state) state[c] = st;
    }
    if (cone_type == 1) {
        for (int c = t; c < nc; c += WIDE_BLK) {
            const int p = par[c], s = sib[c];
            if (p < 0 || p >= nc || s < 0 || s >= nc || c >= s) continue;
            const int tor = sib[p];
            const bool c4 = (tor >= 0 && tor < nc);
            const int rows[4] = {p, c, s, c4 ? tor : p};
            const int nr = c4 ? 4 : 3;
            double xr[4], dr[4], fr[3], jr[4], fo[4], h[16];
            int st;
            for (int i = 0; i < 4; ++i) {
                jr[i] = (i < nr && jv) ? jv[rows[i]] : 0.0;
                xr[i] = (i < nr) ? x[rows[i]] + alpha * jr[i] : 0.0;
                dr[i] = (i < nr) ? D[rows[i]] : 0.0;
            }
            // 3-row cones: both tangents use row c's coefficient (as the
            // LIBERO kernel does); the zero 4th row leaves N, T, the cost and
            // the 3x3 Hessian block unchanged.
            fr[0] = mu[c];
            fr[1] = c4 ? mu[s] : mu[c];
            fr[2] = c4 ? mu[tor] : 1.0;
            acc[0] += elliptic_contact4(xr, dr, fr, impr_inv, fo, &st, h);
            for (int i = 0; i < nr; ++i) {
                acc[1] -= fo[i] * jr[i];
                for (int k = 0; k < nr; ++k) acc[2] += jr[i] * h[i * 4 + k] * jr[k];
                if (force) force[rows[i]] = fo[i];
                if (state) state[rows[i]] = st;
            }
        }
    }
    wide_reduce<3>(acc);
    cost = acc[0]; d1 = acc[1]; d2 = acc[2];
}

// In-place regularised Cholesky of the nv x nv LDS tile A (lower triangle),
// pivot clamp as mw_chol_factor_lds: s < delta -> delta.
HIPGL_DEVICE inline void wide_chol(double* A, int nv, int t, double delta)
{
    for (int k = 0; k < nv; k++) {
        double s = A[k * LDS_STRIDE + k];
        if (s < delta) s = delta;
        const double dkk = sqrt(s);
        __syncthreads();
        if (t == 0) A[k * LDS_STRIDE + k] = dkk;
        for (int r = k + 1 + t; r < nv; r += WIDE_BLK) A[r * LDS_STRIDE + k] /= dkk;
        __syncthreads();
        // trailing update: wave w takes rows k+1+w, k+1+w+4, ...; its lanes the columns of that row
        {
            const int w = t >> 6, lane = t & 63;
            for (int r = k + 1 + w; r < nv; r += WIDE_WAVES) {
                const double Lrk = A[r * LDS_STRIDE + k];
                for (int j = k + 1 + lane; j <= r; j += 64)
                    A[r * LDS_STRIDE + j] -= Lrk * A[j * LDS_STRIDE + k];
            }
        }
        __syncthreads();
    }
}

// x <- (L L')^-1 x for the factor wide_chol left in A; x is an LDS vector.
HIPGL_DEVICE inline void wide_chol_solve(const double* A, int nv, int t, double* x)
{
    for (int k = 0; k < nv; k++) {
        if (t == 0) x[k] /= A[k * LDS_STRIDE + k];
        __syncthreads();
        const double xk = x[k];
        for (int r = k + 1 + t; r < nv; r += WIDE_BLK) x[r] -= A[r * LDS_STRIDE + k] * xk;
        __syncthreads();
    }
    for (int k = nv - 1; k >= 0; k--) {
        if (t == 0) x[k] /= A[k * LDS_STRIDE + k];
        __syncthreads();
        const double xk = x[k];
        for (int r = t; r < k; r += WIDE_BLK) x[r] -= A[k * LDS_STRIDE + r] * xk;
        __syncthreads();
    }
}

HIPGL_DEVICE inline double wide_max_diag(const double* A, int nv, int t)
{
    __shared__ double s_mx[WIDE_WAVES];
    double m = 0.0;
    for (int v = t; v < nv; v += WIDE_BLK) m = fmax(m, A[v * LDS_STRIDE + v]);
    for (int off = 32; off > 0; off >>= 1) m = fmax(m, wf_shfl_xor_d(m, off));
    __syncthreads();
    if ((t & 63) == 0) s_mx[t >> 6] = m;
    __syncthreads();
    m = s_mx[0];
    for (int k = 1; k < WIDE_WAVES; k++) m = fmax(m, s_mx[k]);
    return m;
}

// Joint index owning dof v (first match, same invalid-address break as the
// LIBERO kernel), or -1.
HIPGL_DEVICE inline int wide_dof_joint(int v, int nv, int njnt, const int* jnt_type, const int* jnt_dof_adr)
{
    for (int ji = 0; ji < njnt; ji++) {
        const int da = jnt_dof_adr[ji];
        if (da < 0 || da >= nv) break;
        const int jt = jnt_type[ji];
        const int nd = (jt == JOINT_FREE) ? 6 : (jt == JOINT_BALL) ? 3 : 1;
        if (v >= da && v < da + nd) return ji;
    }
    return -1;
}

HIPGL_GLOBAL void __launch_bounds__(WIDE_BLK, 1) kernel_newton_solve_wide(
    int num_envs, int nv,
    const double* __restrict__ solver_M,
    const double* __restrict__ qfrc_bias,
    const double* __restrict__ qfrc_fluid,
    const double* __restrict__ qfrc_applied_d,
    const double* __restrict__ ctrl_d,
    int nu,
    const int*    __restrict__ actuator_jnt,
    const int*    __restrict__ actuator_type,
    const double* __restrict__ actuator_gain,
    const double* __restrict__ actuator_kv,
    const double* __restrict__ actuator_biasprm,
    const double* __restrict__ actuator_forcerange,
    const double* __restrict__ actuator_ctrlrange,
    const int*    __restrict__ jnt_type,
    const int*    __restrict__ jnt_dof_adr,
    const int*    __restrict__ jnt_qpos_adr,
    const double* __restrict__ jnt_damping,
    const double* __restrict__ jnt_stiffness,
    int njnt,
    bool enable_damping,
    const double* __restrict__ qpos_d,
    const double* __restrict__ qvel_d,
    const double* __restrict__ qpos0_d,
    int nq,
    const double* __restrict__ constraint_J,
    const double* __restrict__ constraint_rhs,
    const double* __restrict__ constraint_D,
    const int*    __restrict__ nconstraint,
    double*       __restrict__ qacc,
    double*       __restrict__ constraint_lam,
    const double* __restrict__ constraint_lo,
    const double* __restrict__ constraint_hi,
    const int*    __restrict__ constraint_friction_parent,
    const int*    __restrict__ constraint_friction_sibling,
    const double* __restrict__ constraint_friction_mu,
    int cone_type,
    double impratio,
    bool warm_start,
    double* __restrict__ qacc_warmstart,
    double* __restrict__ ws_x,
    double* __restrict__ ws_force,
    int*    __restrict__ ws_state,
    double* __restrict__ ws_jv,
    bool*   __restrict__ ws_is_elliptic,
    double solver_tolerance,
    int max_iterations,
    int prof)   // HP_WIDE_PROF=1: env 0 prints iterations / line-search evaluations / rows
{
    const int env = blockIdx.x;
    const int t = threadIdx.x;
    if (env >= num_envs || HIPGL_WORLD_FROZEN(env) || nv > KERNEL_MAXNV) return;
    if (max_iterations <= 0 || max_iterations > NEWTON_MAXITER) max_iterations = NEWTON_MAXITER;

    // M is read from global (L2-resident, only used in products); LDS holds H and the sparse J.
    __shared__ double sH[KERNEL_MAXNV * LDS_STRIDE];
    __shared__ int    s_rp[WIDE_MAXROWS + 1];   // row c's entries: [s_rp[c], s_rp[c+1])
    __shared__ unsigned char s_ji[WIDE_NNZ];    // dof of each entry (nv <= KERNEL_MAXNV <= 255)
    __shared__ double s_jv[WIDE_NNZ];           // J value of each entry
    __shared__ double s_fc[WIDE_MAXROWS];       // per-row force of the current iterate (sparse path)
    __shared__ int    s_st[WIDE_MAXROWS];       // per-row state of the current iterate (sparse path)
    __shared__ unsigned char s_cr[WIDE_MAXROWS]; // 1: elliptic cone row (normal or friction)
    constexpr int WIDE_CONES = 384;
    __shared__ short  s_cone[WIDE_CONES][4];    // cone owner rows: normal, t1, t2, torsion (or -1)
    __shared__ int    s_ncone;
    __shared__ double s_a[KERNEL_MAXNV];    // qacc
    __shared__ double s_f[KERNEL_MAXNV];    // qfrc_smooth
    __shared__ double s_g[KERNEL_MAXNV];    // gradient
    __shared__ double s_s[KERNEL_MAXNV];    // search direction / scratch
    __shared__ double s_w[KERNEL_MAXNV];    // warm-start candidate / M*search
    __shared__ double s_act_force[MAX_ACTUATORS];
    __shared__ int    s_act_dof[MAX_ACTUATORS];

    int nc = nconstraint[env];
    if (nc > MAX_CONSTRAINTS) nc = MAX_CONSTRAINTS;
    const size_t cb = (size_t)env * MAX_CONSTRAINTS;
    const double* J = constraint_J + cb * nv;          // row-major [nc][nv]
    const double* rhs = constraint_rhs + cb;
    const double* Dv = constraint_D + cb;
    const double* lo = constraint_lo + cb;
    const double* hi = constraint_hi + cb;
    const int*    par = constraint_friction_parent + cb;
    const int*    sib = constraint_friction_sibling + cb;
    const double* mu = constraint_friction_mu + cb;
    double* g_x = ws_x + cb;
    double* g_force = ws_force + cb;
    int*    g_state = ws_state + cb;
    double* g_jv = ws_jv + cb;
    bool*   g_is_en = ws_is_elliptic + cb;
    const double* ev = qvel_d + (size_t)env * nv;
    const double* ep = qpos_d + (size_t)env * nq;
    const double impr_inv = 1.0 / sqrt(fmax(impratio, 1e-6));

    const double* Mg = solver_M + (size_t)env * KERNEL_MAXNV * KERNEL_MAXNV;
    // HP_WIDE_PROF: cycles per phase (0 setup, 1 sparse+warm, 2 residual/eval, 3 gradient, 4 H, 5 chol+solve, 6 line search)
    long long pc[7] = {0, 0, 0, 0, 0, 0, 0};
    long long pt = clock64();
#define WIDE_PT(i) do { if (prof) { const long long _n = clock64(); pc[i] += _n - pt; pt = _n; } } while (0)

    // ── qfrc_smooth: -bias + applied + fluid + passive + actuators ──
    for (int a = t; a < nu; a += WIDE_BLK) {
        const int jid = actuator_jnt[a];
        const int da = jnt_dof_adr[jid];
        double c = ctrl_d[env * nu + a];
        if (actuator_ctrlrange) {
            const double cl = actuator_ctrlrange[a * 2], ch = actuator_ctrlrange[a * 2 + 1];
            if (cl < ch) c = fmax(cl, fmin(ch, c));
        }
        const int atype = actuator_type ? actuator_type[a] : ACTUATOR_MOTOR;
        const double gain = actuator_gain[a];
        const double vel = ev[da];
        double af = 0.0;
        if (atype == ACTUATOR_MOTOR) af = gain * c;
        else if (atype == ACTUATOR_POSITION)
            af = gain * (c - ep[jnt_qpos_adr[jid]]) - (actuator_kv ? actuator_kv[a] : 0.0) * vel;
        else if (atype == ACTUATOR_VELOCITY) af = gain * (c - vel);
        else if (atype == ACTUATOR_GENERAL && actuator_biasprm)
            af = gain * c + actuator_biasprm[a * 3] + actuator_biasprm[a * 3 + 1] * ep[jnt_qpos_adr[jid]]
                 + actuator_biasprm[a * 3 + 2] * vel;
        if (actuator_forcerange) {
            const double fl = actuator_forcerange[a * 2], fh = actuator_forcerange[a * 2 + 1];
            if (fl < fh) af = fmax(fl, fmin(fh, af));
        }
        s_act_force[a] = af;
        s_act_dof[a] = da;
    }
    __syncthreads();
    for (int v = t; v < nv; v += WIDE_BLK) {
        double f = -qfrc_bias[env * nv + v] + qfrc_applied_d[env * nv + v];
        if (qfrc_fluid) f += qfrc_fluid[env * nv + v];
        const int jm = wide_dof_joint(v, nv, njnt, jnt_type, jnt_dof_adr);
        if (jm >= 0) {
            if (enable_damping && jnt_damping) f -= jnt_damping[jm] * ev[v];
            if (jnt_stiffness && qpos0_d && jnt_stiffness[jm] != 0.0) {
                const int qa = jnt_qpos_adr[jm] + (v - jnt_dof_adr[jm]);
                f -= jnt_stiffness[jm] * (ep[qa] - qpos0_d[qa]);
            }
        }
        for (int a = 0; a < nu; a++) if (s_act_dof[a] == v) f += s_act_force[a];
        s_f[v] = f;
    }

    // ── qacc_smooth = M^-1 qfrc_smooth ──
    for (int r = 0; r < nv; r++)
        for (int v = t; v < nv; v += WIDE_BLK) sH[r * LDS_STRIDE + v] = Mg[r * KERNEL_MAXNV + v];
    for (int v = t; v < nv; v += WIDE_BLK) s_a[v] = s_f[v];
    __syncthreads();
    wide_chol(sH, nv, t, fmax(wide_max_diag(sH, nv, t) * 1e-12, 1e-12));
    wide_chol_solve(sH, nv, t, s_a);

    double* env_qacc = qacc + (size_t)env * nv;
    double* env_lam = constraint_lam + cb;
    if (nc == 0) {
        for (int v = t; v < nv; v += WIDE_BLK) {
            env_qacc[v] = s_a[v];
            if (qacc_warmstart) qacc_warmstart[(size_t)env * nv + v] = s_a[v];
        }
        return;
    }

    // Elliptic-normal marks (one writer value: true), then lambda init.
    for (int c = t; c < nc; c += WIDE_BLK) { g_is_en[c] = false; env_lam[c] = 0.0; }
    __syncthreads();
    if (cone_type == 1)
        for (int c = t; c < nc; c += WIDE_BLK) {
            const int p = par[c];
            if (p >= 0 && p < nc) g_is_en[p] = true;
        }
    __syncthreads();

    WIDE_PT(0);
    // ── sparse J rows ──
    bool sparse = (nc <= WIDE_MAXROWS);
    if (sparse) {
        for (int c = t; c < nc; c += WIDE_BLK) {
            const double* Jc = J + (size_t)c * nv;
            int n = 0;
            for (int v = 0; v < nv; v++) n += (Jc[v] != 0.0);
            s_rp[c + 1] = n;
        }
        __syncthreads();
        if (t == 0) {
            s_rp[0] = 0;
            for (int c = 0; c < nc; c++) s_rp[c + 1] += s_rp[c];
        }
        __syncthreads();
        sparse = (s_rp[nc] <= WIDE_NNZ);
        // dispatch-constant row kinds and the cone list (owner = t1 row, as in wide_constraint_eval)
        for (int c = t; c < nc; c += WIDE_BLK)
            s_cr[c] = (cone_type == 1 && (par[c] >= 0 || g_is_en[c])) ? 1 : 0;
        if (t == 0) s_ncone = 0;
        __syncthreads();
        if (cone_type == 1 && t < 64) {
            int base = 0;
            for (int c0 = 0; c0 < nc; c0 += 64) {
                const int c = c0 + t;
                bool own = false;
                int p = -1, sb = -1;
                if (c < nc) {
                    p = par[c]; sb = sib[c];
                    own = !(p < 0 || p >= nc || sb < 0 || sb >= nc || c >= sb);
                }
                const unsigned long long mk = __ballot(own);
                if (own) {
                    const int q = base + __popcll(mk & ((1ull << t) - 1ull));
                    if (q < WIDE_CONES) {
                        const int tor = sib[p];
                        s_cone[q][0] = (short)p; s_cone[q][1] = (short)c; s_cone[q][2] = (short)sb;
                        s_cone[q][3] = (short)((tor >= 0 && tor < nc) ? tor : -1);
                    }
                }
                base += __popcll(mk);
            }
            if (t == 0) s_ncone = base;
        }
        __syncthreads();
        sparse = sparse && (s_ncone <= WIDE_CONES);
        if (sparse)
            for (int c = t; c < nc; c += WIDE_BLK) {
                const double* Jc = J + (size_t)c * nv;
                int k = s_rp[c];
                for (int v = 0; v < nv; v++)
                    if (Jc[v] != 0.0) { s_ji[k] = (unsigned char)v; s_jv[k] = Jc[v]; k++; }
            }
        __syncthreads();
    }
    // J_c . vec (same ascending-v order in both forms)
    auto rowdot = [&](int c, const double* vec) -> double {
        double r = 0.0;
        if (sparse) {
            for (int k = s_rp[c], e = s_rp[c + 1]; k < e; k++) r += s_jv[k] * vec[s_ji[k]];
        } else {
            const double* Jc = J + (size_t)c * nv;
            for (int v = 0; v < nv; v++) r += Jc[v] * vec[v];
        }
        return r;
    };

    // Total cost of a candidate in s_w: residual into g_x, quadratic + constraint cost.
    auto total_cost = [&](const double* a) -> double {
        for (int c = t; c < nc; c += WIDE_BLK) {
            g_x[c] = -rhs[c] + rowdot(c, a);
        }
        double q = 0.0;
        for (int v = t; v < nv; v += WIDE_BLK) {
            double Ma = 0.0;
            for (int j = 0; j < nv; j++) Ma += Mg[v * KERNEL_MAXNV + j] * a[j];
            q += 0.5 * a[v] * Ma - a[v] * s_f[v];
        }
        q = wide_reduce1(q);
        __syncthreads();
        double cc, d1, d2;
        wide_constraint_eval(t, nc, cone_type, impr_inv, g_x, nullptr, 0.0, Dv, lo, hi, par, sib, mu,
                             g_is_en, nullptr, nullptr, cc, d1, d2);
        return q + cc;
    };

    // ── Warm start: the lower-cost of qacc_warmstart and qacc_smooth ──
    if (warm_start && qacc_warmstart) {
        for (int v = t; v < nv; v += WIDE_BLK) s_w[v] = qacc_warmstart[(size_t)env * nv + v];
        __syncthreads();
        const double cw = total_cost(s_w);
        const double cs = total_cost(s_a);
        if (cw <= cs)
            for (int v = t; v < nv; v += WIDE_BLK) s_a[v] = s_w[v];
        __syncthreads();
    }

    WIDE_PT(1);
    double prev_cost = 1e30;
    int it_done = 0, ls_evals = 0, exit_code = 4;   // 1 grad, 2 no-descent/NaN, 3 improvement, 4 max iters
    double exit_sg = 0.0, exit_gn = 0.0;
    for (int it = 0; it < max_iterations; it++) {
        it_done = it + 1;
        // ── residual, forces, states at a ──
        for (int c = t; c < nc; c += WIDE_BLK) {
            g_x[c] = -rhs[c] + rowdot(c, s_a);
        }
        __syncthreads();
        double c0, d1u, d2u;
        double* const F = sparse ? s_fc : g_force;
        int* const S = sparse ? s_st : g_state;
        wide_constraint_eval(t, nc, cone_type, impr_inv, g_x, nullptr, 0.0, Dv, lo, hi, par, sib, mu,
                             g_is_en, F, S, c0, d1u, d2u);
        __syncthreads();

        WIDE_PT(2);
        // ── gradient = M a - qfrc - J' force ──
        for (int v = t; v < nv; v += WIDE_BLK) {
            double g = -s_f[v];
            for (int j = 0; j < nv; j++) g += Mg[v * KERNEL_MAXNV + j] * s_a[j];
            if (!sparse)
                for (int c = 0; c < nc; c++) g -= J[(size_t)c * nv + v] * g_force[c];
            s_g[v] = g;
        }
        if (sparse) {
            // rows in order; one row's entries are distinct dofs, so its lanes never collide
            __syncthreads();
            if (t < 64)   // rows in order on one wave (a row's entries are distinct dofs)
                for (int c = 0; c < nc; c++) {
                    const double fc = F[c];
                    if (fc == 0.0) continue;
                    for (int k = s_rp[c] + t, e = s_rp[c + 1]; k < e; k += 64) s_g[s_ji[k]] -= s_jv[k] * fc;
                    __builtin_amdgcn_wave_barrier();
                }
            __syncthreads();
        }
        double gn2 = 0.0;
        for (int v = t; v < nv; v += WIDE_BLK) gn2 += s_g[v] * s_g[v];
        gn2 = wide_reduce1(gn2);
        if (gn2 < solver_tolerance * solver_tolerance) { exit_code = 1; break; }
        exit_gn = sqrt(gn2);

        WIDE_PT(3);
        // ── H = M + J' (D | cone Hessian) J over the active rows; lane owns rows v ──
        for (int r = 0; r < nv; r++)
            for (int v = t; v < nv; v += WIDE_BLK) sH[r * LDS_STRIDE + v] = Mg[r * KERNEL_MAXNV + v];
        __syncthreads();
        for (int c = 0; c < nc; c++) {
            if (sparse ? (s_cr[c] != 0) : (cone_type == 1 && (par[c] >= 0 || g_is_en[c]))) continue;
            if (S[c] != 1) continue;
            const double Dc = Dv[c];
            if (sparse) {
                // lower triangle only: wide_chol / wide_chol_solve never read above the diagonal
                const int b0 = s_rp[c], n = s_rp[c + 1] - b0;
                for (int p = t; p < n * n; p += WIDE_BLK) {
                    const int i = p / n, j = p - i * n;
                    const int vi = s_ji[b0 + i], vj = s_ji[b0 + j];
                    if (vi >= vj) sH[vi * LDS_STRIDE + vj] += Dc * s_jv[b0 + i] * s_jv[b0 + j];
                }
                __syncthreads();
                continue;
            }
            const double* Jc = J + (size_t)c * nv;
            for (int v = t; v < nv; v += WIDE_BLK) {
                const double dj = Dc * Jc[v];
                if (dj == 0.0) continue;
                for (int w = 0; w < nv; w++) sH[v * LDS_STRIDE + w] += dj * Jc[w];
            }
        }
        if (cone_type == 1) {
            const int ncn = sparse ? s_ncone : nc;
            for (int q = 0; q < ncn; q++) {
                int c, p, s, tor;
                if (sparse) {
                    p = s_cone[q][0]; c = s_cone[q][1]; s = s_cone[q][2]; tor = s_cone[q][3];
                } else {
                    c = q; p = par[c]; s = sib[c];
                    if (p < 0 || p >= nc || s < 0 || s >= nc || c >= s) continue;
                    tor = sib[p];
                }
                const int st = S[p];
                if (st == 0) continue;
                const bool c4 = (tor >= 0 && tor < nc);
                const int nr = c4 ? 4 : 3;
                const int rows[4] = {p, c, s, c4 ? tor : p};
                double xr[4], dr[4], fr[3], h[16];
                for (int i = 0; i < 4; ++i) {
                    xr[i] = (i < nr) ? g_x[rows[i]] : 0.0;
                    dr[i] = (i < nr) ? Dv[rows[i]] : 0.0;
                }
                fr[0] = mu[c]; fr[1] = c4 ? mu[s] : mu[c]; fr[2] = c4 ? mu[tor] : 1.0;
                int st2;
                elliptic_contact4(xr, dr, fr, impr_inv, nullptr, &st2, h);
                if (sparse) {
                    for (int ra = 0; ra < nr; ++ra)
                        for (int rb = 0; rb < nr; ++rb) {
                            const double hab = h[ra * 4 + rb];
                            if (hab == 0.0) continue;
                            const int a0 = s_rp[rows[ra]], na = s_rp[rows[ra] + 1] - a0;
                            const int b0 = s_rp[rows[rb]], nb = s_rp[rows[rb] + 1] - b0;
                            for (int p = t; p < na * nb; p += WIDE_BLK) {
                                const int i = p / nb, j = p - i * nb;
                                const int vi = s_ji[a0 + i], vj = s_ji[b0 + j];
                                if (vi >= vj) sH[vi * LDS_STRIDE + vj] += hab * s_jv[a0 + i] * s_jv[b0 + j];
                            }
                            __syncthreads();
                        }
                    continue;
                }
                for (int v = t; v < nv; v += WIDE_BLK) {
                    double b[4] = {0.0, 0.0, 0.0, 0.0};
                    bool any = false;
                    for (int i = 0; i < nr; ++i) {
                        const double Jiv = J[(size_t)rows[i] * nv + v];
                        if (Jiv == 0.0) continue;
                        any = true;
                        for (int k = 0; k < nr; ++k) b[k] += Jiv * h[i * 4 + k];
                    }
                    if (!any) continue;
                    for (int w = 0; w < nv; w++) {
                        double hw = 0.0;
                        for (int k = 0; k < nr; ++k) hw += b[k] * J[(size_t)rows[k] * nv + w];
                        sH[v * LDS_STRIDE + w] += hw;
                    }
                }
            }
        }
        __syncthreads();

        WIDE_PT(4);
        // ── search = -H^-1 grad ──
        wide_chol(sH, nv, t, fmax(wide_max_diag(sH, nv, t) * 1e-12, 1e-12));
        for (int v = t; v < nv; v += WIDE_BLK) s_s[v] = -s_g[v];
        __syncthreads();
        wide_chol_solve(sH, nv, t, s_s);

        WIDE_PT(5);
        // ── line search along s: phi(alpha) = quad(a + alpha s) + C(x + alpha J s) ──
        double sg = 0.0, sMs = 0.0, bad = 0.0;
        for (int v = t; v < nv; v += WIDE_BLK) {
            double Ms = 0.0;
            for (int j = 0; j < nv; j++) Ms += Mg[v * KERNEL_MAXNV + j] * s_s[j];
            sMs += s_s[v] * Ms;
            sg += s_s[v] * s_g[v];
            if (isnan(s_s[v]) || isinf(s_s[v])) bad = 1.0;
        }
        for (int c = t; c < nc; c += WIDE_BLK) {
            const double jv = rowdot(c, s_s);
            g_jv[c] = jv;
        }
        double red[3] = {sg, sMs, bad};
        wide_reduce<3>(red);
        sg = red[0]; sMs = red[1];
        if (red[2] > 0.0 || !(sg < 0.0)) { exit_code = 2; exit_sg = sg; break; }   // NaN search or no descent
        // quadratic part: q(alpha) - q(0) = alpha * (s'(Ma - f)) + 0.5 alpha^2 s'Ms, with
        // s'(Ma - f) = s'grad + s'J'force(0) = sg - (constraint d1 at 0)
        __syncthreads();
        double cz, dz, d2z;
        wide_constraint_eval(t, nc, cone_type, impr_inv, g_x, g_jv, 0.0, Dv, lo, hi, par, sib, mu,
                             g_is_en, nullptr, nullptr, cz, dz, d2z);
        const double q1 = sg - dz;
        const double dphi0 = sg;
        double lo_a = 0.0, hi_a = 1e30, alpha = 1.0, ccost_a = 0.0;   // ccost_a: C at alpha
        for (int ls = 0; ; ls++) {
            double da, dda;
            wide_constraint_eval(t, nc, cone_type, impr_inv, g_x, g_jv, alpha, Dv, lo, hi, par, sib, mu,
                                 g_is_en, nullptr, nullptr, ccost_a, da, dda);
            const double d1 = q1 + alpha * sMs + da;
            const double d2 = sMs + dda;
            ls_evals++;
            if (fabs(d1) <= WIDE_LS_TOL * fabs(dphi0) || ls == WIDE_LS_MAXITER - 1) break;
            if (d1 < 0.0) lo_a = alpha; else hi_a = alpha;
            double an = (d2 > 0.0) ? alpha - d1 / d2 : -1.0;
            if (!(an > lo_a && an < hi_a))
                an = (hi_a < 1e29) ? 0.5 * (lo_a + hi_a) : 2.0 * alpha;
            alpha = an;
        }
        WIDE_PT(6);
        for (int v = t; v < nv; v += WIDE_BLK) s_a[v] += alpha * s_s[v];
        __syncthreads();

        // ── stop test on the cost improvement (as kernel_newton_solve_t) ──
        double qa = 0.0;
        for (int v = t; v < nv; v += WIDE_BLK) {
            double Ma = 0.0;
            for (int j = 0; j < nv; j++) Ma += Mg[v * KERNEL_MAXNV + j] * s_a[j];
            qa += 0.5 * s_a[v] * Ma - s_a[v] * s_f[v];
        }
        const double cur_cost = wide_reduce1(qa) + ccost_a;
        const double improvement = prev_cost - cur_cost;
        const double scale = fmax(fmax(fabs(prev_cost), fabs(cur_cost)), 1e-12);
        if (improvement >= 0.0 && improvement < solver_tolerance * scale && it >= 3) { exit_code = 3; break; }
        prev_cost = cur_cost;
    }

    if (prof && env == 0 && t == 0)
        printf("WIDE nnz %d ncone %d exit %d sg %.3e gnorm %.3e | nv %d nc %d iters %d ls_evals %d sparse %d kcyc setup %lld sp+warm %lld res %lld grad %lld H %lld chol %lld ls %lld\n",
               sparse ? s_rp[nc] : -1, sparse ? s_ncone : -1, exit_code, exit_sg, exit_gn, nv, nc, it_done, ls_evals, (int)sparse, pc[0] / 1000, pc[1] / 1000, pc[2] / 1000, pc[3] / 1000,
               pc[4] / 1000, pc[5] / 1000, pc[6] / 1000);
#undef WIDE_PT

    // ── final forces, qacc, warm start ──
    for (int c = t; c < nc; c += WIDE_BLK) {
        g_x[c] = -rhs[c] + rowdot(c, s_a);
    }
    __syncthreads();
    double cf, d1f, d2f;
    wide_constraint_eval(t, nc, cone_type, impr_inv, g_x, nullptr, 0.0, Dv, lo, hi, par, sib, mu,
                         g_is_en, g_force, nullptr, cf, d1f, d2f);
    __syncthreads();
    for (int c = t; c < nc; c += WIDE_BLK) env_lam[c] = g_force[c];
    for (int v = t; v < nv; v += WIDE_BLK) {
        env_qacc[v] = s_a[v];
        if (qacc_warmstart) qacc_warmstart[(size_t)env * nv + v] = s_a[v];
    }
}

// qacc <- (M + h*diag(damping))^-1 M qacc, the kernel_mhat_solve_t contract,
// via Cholesky (M + hD is SPD) instead of pivoted Gauss-Jordan.
HIPGL_GLOBAL void __launch_bounds__(WIDE_BLK, 1) kernel_mhat_solve_wide(
    int num_envs, int nv,
    const double* __restrict__ solver_M,
    const int*    __restrict__ jnt_type,
    const int*    __restrict__ jnt_dof_adr,
    const double* __restrict__ jnt_damping,
    int njnt,
    double timestep,
    bool enable_damping,
    double*       __restrict__ qacc)
{
    const int env = blockIdx.x;
    const int t = threadIdx.x;
    if (env >= num_envs || nv > KERNEL_MAXNV) return;
    __shared__ double sA[KERNEL_MAXNV * LDS_STRIDE];
    __shared__ double sr[KERNEL_MAXNV];
    const double* M = solver_M + (size_t)env * KERNEL_MAXNV * KERNEL_MAXNV;
    double* qa = qacc + (size_t)env * nv;
    for (int r = 0; r < nv; r++)
        for (int v = t; v < nv; v += WIDE_BLK) sA[r * LDS_STRIDE + v] = M[r * KERNEL_MAXNV + v];
    for (int v = t; v < nv; v += WIDE_BLK) {
        double s = 0.0;
        for (int j = 0; j < nv; j++) s += M[v * KERNEL_MAXNV + j] * qa[j];
        sr[v] = s;
    }
    __syncthreads();
    if (enable_damping)
        for (int v = t; v < nv; v += WIDE_BLK) {
            const int jm = wide_dof_joint(v, nv, njnt, jnt_type, jnt_dof_adr);
            if (jm >= 0) sA[v * LDS_STRIDE + v] += timestep * jnt_damping[jm];
        }
    __syncthreads();
    wide_chol(sA, nv, t, 1e-300);
    wide_chol_solve(sA, nv, t, sr);
    for (int v = t; v < nv; v += WIDE_BLK) qa[v] = sr[v];
}
