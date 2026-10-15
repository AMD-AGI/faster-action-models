// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// PhysicsData: batched per-environment state on GPU.

#include "hipgl/physics/physics_data.h"
#include <hip/hip_runtime.h>
#include <cstring>
#include <cstdio>
#include <vector>

namespace hipgl {
namespace physics {

PhysicsData::PhysicsData() : allocated_(false) {
    memset(&d_data_, 0, sizeof(d_data_));
}

PhysicsData::~PhysicsData() {
    free_device();
}

static void safe_free(void* p) { if (p) hipFree(p); }

template<typename T>
static bool alloc_zero(T** ptr, size_t count) {
    if (count == 0) { *ptr = nullptr; return true; }
    hipError_t err = hipMalloc(ptr, count * sizeof(T));
    if (err != hipSuccess) return false;
    return hipMemset(*ptr, 0, count * sizeof(T)) == hipSuccess;
}

bool PhysicsData::allocate(const PhysicsModelDesc& desc, int num_envs) {
    free_device();

    DevicePhysicsData& d = d_data_;
    d.num_envs = num_envs;
    d.nq = desc.nq;
    d.nv = desc.nv;
    d.nu = desc.nu;
    d.nbody = desc.nbody;
    d.ngeom = desc.ngeom;

    int ne = num_envs;
    int nq = desc.nq, nv = desc.nv, nu = desc.nu;
    int nb = desc.nbody, ng = desc.ngeom;

    // State
    if (!alloc_zero(&d.qpos,            ne * nq)) return false;
    if (!alloc_zero(&d.qvel,            ne * nv)) return false;
    if (!alloc_zero(&d.qpos_d,          ne * nq)) return false;
    if (!alloc_zero(&d.qvel_d,          ne * nv)) return false;
    if (!alloc_zero(&d.qacc,            ne * nv)) return false;
    if (!alloc_zero(&d.qacc_warmstart,  ne * nv)) return false;
    if (!alloc_zero(&d.ctrl,            ne * nu)) return false;
    if (!alloc_zero(&d.ctrl_d,          ne * nu)) return false;
    if (!alloc_zero(&d.qfrc_applied,    ne * nv)) return false;
    if (!alloc_zero(&d.qfrc_applied_d,  ne * nv)) return false;

    // Body state
    if (!alloc_zero(&d.xpos,  ne * nb * 3)) return false;
    if (!alloc_zero(&d.xquat, ne * nb * 4)) return false;
    if (!alloc_zero(&d.xpos_d,  ne * nb * 3)) return false;
    if (!alloc_zero(&d.xquat_d, ne * nb * 4)) return false;
    if (!alloc_zero(&d.jnt_xaxis_d, ne * desc.njnt * 3)) return false;
    if (!alloc_zero(&d.cvel,  ne * nb * 6)) return false;
    if (!alloc_zero(&d.xfrc,  ne * nb * 6)) return false;

    // Geom state
    if (!alloc_zero(&d.geom_xpos,  ne * ng * 3)) return false;
    if (!alloc_zero(&d.geom_xquat, ne * ng * 4)) return false;
    if (!alloc_zero(&d.geom_xpos_d,  ne * ng * 3)) return false;
    if (!alloc_zero(&d.geom_xquat_d, ne * ng * 4)) return false;
    if (!alloc_zero(&d.geom_xmat_d,  ne * ng * 9)) return false;
    if (!alloc_zero(&d.geom_aabb,  ne * ng))      return false;

    // Contacts
    if (!alloc_zero(&d.contacts, ne * MAX_CONTACTS)) return false;
    if (!alloc_zero(&d.ncon,     ne))                return false;
    if (!alloc_zero(&d.body_pair_ncon, ne * nb * nb)) return false;

    // Constraint solver workspace
    size_t jac_size = (size_t)ne * MAX_CONSTRAINTS * nv;
    if (!alloc_zero(&d.constraint_J,   jac_size))           return false;
    if (!alloc_zero(&d.constraint_rhs, ne * MAX_CONSTRAINTS)) return false;
    if (!alloc_zero(&d.constraint_lo,  ne * MAX_CONSTRAINTS)) return false;
    if (!alloc_zero(&d.constraint_hi,  ne * MAX_CONSTRAINTS)) return false;
    if (!alloc_zero(&d.constraint_lam, ne * MAX_CONSTRAINTS)) return false;
    if (!alloc_zero(&d.constraint_D,   ne * MAX_CONSTRAINTS)) return false;
    if (!alloc_zero(&d.constraint_friction_parent, ne * MAX_CONSTRAINTS)) return false;
    if (!alloc_zero(&d.constraint_friction_sibling, ne * MAX_CONSTRAINTS)) return false;
    if (!alloc_zero(&d.constraint_friction_mu, ne * MAX_CONSTRAINTS)) return false;
    if (!alloc_zero(&d.nconstraint,    ne))                   return false;

    // Pre-computed inverse mass/inertia
    if (!alloc_zero(&d.body_inv_mass,    ne * nb))     return false;
    if (!alloc_zero(&d.body_inv_inertia, ne * nb * 3)) return false;

    // RNEA bias force
    if (!alloc_zero(&d.qfrc_bias, ne * nv)) return false;

    // Fluid drag forces
    if (!alloc_zero(&d.qfrc_fluid, ne * nv)) return false;

    // Time
    if (!alloc_zero(&d.time, ne)) return false;

    // Solver workspace
    size_t nv2 = (size_t)KERNEL_MAXNV * KERNEL_MAXNV;
    if (!alloc_zero(&d.solver_M,          ne * nv2))     return false;
    if (!alloc_zero(&d.solver_H,          ne * nv2))     return false;
    if (!alloc_zero(&d.solver_rhs,        ne * KERNEL_MAXNV)) return false;
    if (!alloc_zero(&d.solver_M_inv_diag, ne * MAX_DOF)) return false;
    if (!alloc_zero(&d.solver_M_inv,     ne * nv2))     return false;

    // Newton per-constraint workspace
    size_t nc_total = (size_t)ne * MAX_CONSTRAINTS;
    if (!alloc_zero(&d.newton_x,            nc_total)) return false;
    if (!alloc_zero(&d.newton_force,        nc_total)) return false;
    if (!alloc_zero(&d.newton_state,        nc_total)) return false;
    if (!alloc_zero(&d.newton_jv,           nc_total)) return false;
    if (!alloc_zero(&d.newton_is_elliptic,  nc_total)) return false;

    // RNEA workspace
    if (!alloc_zero(&d.rnea_omega, ne * nb * 3)) return false;
    if (!alloc_zero(&d.rnea_alpha, ne * nb * 3)) return false;
    if (!alloc_zero(&d.rnea_acc,   ne * nb * 3)) return false;
    if (!alloc_zero(&d.rnea_fnet,  ne * nb * 3)) return false;
    if (!alloc_zero(&d.rnea_tnet,  ne * nb * 3)) return false;

    // Site workspace
    d.nsite = desc.nsite;
    if (desc.nsite > 0) {
        if (!alloc_zero(&d.site_xpos, ne * desc.nsite * 3)) return false;
        if (!alloc_zero(&d.site_xquat, ne * desc.nsite * 4)) return false;
        if (!alloc_zero(&d.site_xmat, ne * desc.nsite * 9)) return false;
    }

    allocated_ = true;
    return true;
}

void PhysicsData::reset(const PhysicsModelDesc& desc) {
    if (!allocated_) return;
    int ne = d_data_.num_envs;
    int nq = desc.nq, nv = desc.nv;

    // Set qpos to qpos0 for each env (float and double)
    std::vector<float> h_qpos(ne * nq);
    std::vector<double> h_qpos_d(ne * nq);
    for (int e = 0; e < ne; e++) {
        memcpy(&h_qpos[e * nq], desc.qpos0, nq * sizeof(float));
        for (int i = 0; i < nq; i++)
            h_qpos_d[e * nq + i] = (double)desc.qpos0[i];
    }
    hipMemcpy(d_data_.qpos, h_qpos.data(), ne * nq * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_data_.qpos_d, h_qpos_d.data(), ne * nq * sizeof(double), hipMemcpyHostToDevice);

    // Zero velocities, accelerations, controls, forces
    hipMemset(d_data_.qvel,            0, ne * nv * sizeof(float));
    hipMemset(d_data_.qvel_d,          0, ne * nv * sizeof(double));
    hipMemset(d_data_.qacc,            0, ne * nv * sizeof(double));
    hipMemset(d_data_.qacc_warmstart,  0, ne * nv * sizeof(double));
    hipMemset(d_data_.ctrl,            0, ne * desc.nu * sizeof(float));
    hipMemset(d_data_.ctrl_d,          0, ne * desc.nu * sizeof(double));
    hipMemset(d_data_.qfrc_applied,    0, ne * nv * sizeof(float));
    hipMemset(d_data_.qfrc_applied_d,  0, ne * nv * sizeof(double));
    hipMemset(d_data_.ncon,         0, ne * sizeof(int));
    hipMemset(d_data_.nconstraint,  0, ne * sizeof(int));
    hipMemset(d_data_.time,         0, ne * sizeof(float));

    // Initialize xquat to identity for each body in each env
    int nb = desc.nbody;
    std::vector<float> h_xquat(ne * nb * 4, 0.0f);
    for (int e = 0; e < ne; e++)
        for (int b = 0; b < nb; b++)
            h_xquat[(e * nb + b) * 4] = 1.0f; // w = 1
    hipMemcpy(d_data_.xquat, h_xquat.data(), ne * nb * 4 * sizeof(float), hipMemcpyHostToDevice);
}

void PhysicsData::reset_env(const PhysicsModelDesc& desc, int env_idx) {
    if (!allocated_ || env_idx < 0 || env_idx >= d_data_.num_envs) return;
    int nq = desc.nq, nv = desc.nv, nu = desc.nu, nb = desc.nbody;

    hipMemcpy(d_data_.qpos + env_idx * nq, desc.qpos0, nq * sizeof(float), hipMemcpyHostToDevice);
    {
        std::vector<double> h_qpos_d(nq);
        for (int i = 0; i < nq; i++) h_qpos_d[i] = (double)desc.qpos0[i];
        hipMemcpy(d_data_.qpos_d + env_idx * nq, h_qpos_d.data(), nq * sizeof(double), hipMemcpyHostToDevice);
    }
    hipMemset(d_data_.qvel + env_idx * nv, 0, nv * sizeof(float));
    hipMemset(d_data_.qvel_d + env_idx * nv, 0, nv * sizeof(double));
    hipMemset(d_data_.qacc + env_idx * nv, 0, nv * sizeof(double));
    hipMemset(d_data_.qacc_warmstart + env_idx * nv, 0, nv * sizeof(double));
    hipMemset(d_data_.ctrl + env_idx * nu, 0, nu * sizeof(float));
    hipMemset(d_data_.ctrl_d + env_idx * nu, 0, nu * sizeof(double));
    // The remaining buffers reset() clears per env. qfrc_applied is read by the
    // bias-force path and ncon/nconstraint gate the solver's constraint loop, so
    // a stale value survives into the new episode's very first substep.
    hipMemset(d_data_.qfrc_applied + env_idx * nv, 0, nv * sizeof(float));
    hipMemset(d_data_.qfrc_applied_d + env_idx * nv, 0, nv * sizeof(double));
    hipMemset(d_data_.ncon + env_idx, 0, sizeof(int));
    hipMemset(d_data_.nconstraint + env_idx, 0, sizeof(int));
    hipMemset(d_data_.time + env_idx, 0, sizeof(float));

    // Reset xquat to identity
    std::vector<float> h_xquat(nb * 4, 0.0f);
    for (int b = 0; b < nb; b++) h_xquat[b * 4] = 1.0f;
    hipMemcpy(d_data_.xquat + env_idx * nb * 4, h_xquat.data(), nb * 4 * sizeof(float), hipMemcpyHostToDevice);
}

void PhysicsData::read_qpos(float* host_buf) const {
    if (!allocated_) return;
    int total = d_data_.num_envs * d_data_.nq;
    hipMemcpy(host_buf, d_data_.qpos, total * sizeof(float), hipMemcpyDeviceToHost);
}

void PhysicsData::read_qvel(float* host_buf) const {
    if (!allocated_) return;
    int total = d_data_.num_envs * d_data_.nv;
    hipMemcpy(host_buf, d_data_.qvel, total * sizeof(float), hipMemcpyDeviceToHost);
}

void PhysicsData::read_xpos(float* host_buf) const {
    if (!allocated_) return;
    int total = d_data_.num_envs * d_data_.nbody * 3;
    hipMemcpy(host_buf, d_data_.xpos, total * sizeof(float), hipMemcpyDeviceToHost);
}

void PhysicsData::read_xquat(float* host_buf) const {
    if (!allocated_) return;
    int total = d_data_.num_envs * d_data_.nbody * 4;
    hipMemcpy(host_buf, d_data_.xquat, total * sizeof(float), hipMemcpyDeviceToHost);
}

void PhysicsData::read_qfrc_bias(double* host_buf) const {
    if (!allocated_) return;
    int total = d_data_.num_envs * d_data_.nv;
    hipMemcpy(host_buf, d_data_.qfrc_bias, total * sizeof(double), hipMemcpyDeviceToHost);
}

void PhysicsData::write_qpos(const float* host_buf) {
    if (!allocated_) return;
    int total = d_data_.num_envs * d_data_.nq;
    hipMemcpy(d_data_.qpos, host_buf, total * sizeof(float), hipMemcpyHostToDevice);
    std::vector<double> h_d(total);
    for (int i = 0; i < total; i++) h_d[i] = (double)host_buf[i];
    hipMemcpy(d_data_.qpos_d, h_d.data(), total * sizeof(double), hipMemcpyHostToDevice);
}

void PhysicsData::write_qpos_env(const float* host_buf, int env_idx) {
    if (!allocated_ || env_idx < 0 || env_idx >= d_data_.num_envs) return;
    int nq = d_data_.nq;
    hipMemcpy(d_data_.qpos + env_idx * nq, host_buf, nq * sizeof(float), hipMemcpyHostToDevice);
    std::vector<double> h_d(nq);
    for (int i = 0; i < nq; i++) h_d[i] = (double)host_buf[i];
    hipMemcpy(d_data_.qpos_d + env_idx * nq, h_d.data(), nq * sizeof(double), hipMemcpyHostToDevice);
}

void PhysicsData::write_qvel(const float* host_buf) {
    if (!allocated_) return;
    int total = d_data_.num_envs * d_data_.nv;
    hipMemcpy(d_data_.qvel, host_buf, total * sizeof(float), hipMemcpyHostToDevice);
    std::vector<double> h_d(total);
    for (int i = 0; i < total; i++) h_d[i] = (double)host_buf[i];
    hipMemcpy(d_data_.qvel_d, h_d.data(), total * sizeof(double), hipMemcpyHostToDevice);
}

void PhysicsData::write_qvel_env(const float* host_buf, int env_idx) {
    if (!allocated_ || env_idx < 0 || env_idx >= d_data_.num_envs) return;
    int nv = d_data_.nv;
    hipMemcpy(d_data_.qvel + env_idx * nv, host_buf, nv * sizeof(float), hipMemcpyHostToDevice);
    std::vector<double> h_d(nv);
    for (int i = 0; i < nv; i++) h_d[i] = (double)host_buf[i];
    hipMemcpy(d_data_.qvel_d + env_idx * nv, h_d.data(), nv * sizeof(double), hipMemcpyHostToDevice);
}

void PhysicsData::read_qpos_d(double* host_buf) const {
    if (!allocated_) return;
    int total = d_data_.num_envs * d_data_.nq;
    hipMemcpy(host_buf, d_data_.qpos_d, total * sizeof(double), hipMemcpyDeviceToHost);
}

void PhysicsData::read_qvel_d(double* host_buf) const {
    if (!allocated_) return;
    int total = d_data_.num_envs * d_data_.nv;
    hipMemcpy(host_buf, d_data_.qvel_d, total * sizeof(double), hipMemcpyDeviceToHost);
}

void PhysicsData::write_qpos_d(const double* host_buf) {
    if (!allocated_) return;
    int total = d_data_.num_envs * d_data_.nq;
    hipMemcpy(d_data_.qpos_d, host_buf, total * sizeof(double), hipMemcpyHostToDevice);
    std::vector<float> h_f(total);
    for (int i = 0; i < total; i++) h_f[i] = (float)host_buf[i];
    hipMemcpy(d_data_.qpos, h_f.data(), total * sizeof(float), hipMemcpyHostToDevice);
}

void PhysicsData::write_qpos_d_env(const double* host_buf, int env_idx) {
    if (!allocated_ || env_idx < 0 || env_idx >= d_data_.num_envs) return;
    int nq = d_data_.nq;
    hipMemcpy(d_data_.qpos_d + env_idx * nq, host_buf, nq * sizeof(double), hipMemcpyHostToDevice);
    std::vector<float> h_f(nq);
    for (int i = 0; i < nq; i++) h_f[i] = (float)host_buf[i];
    hipMemcpy(d_data_.qpos + env_idx * nq, h_f.data(), nq * sizeof(float), hipMemcpyHostToDevice);
}

void PhysicsData::write_qvel_d(const double* host_buf) {
    if (!allocated_) return;
    int total = d_data_.num_envs * d_data_.nv;
    hipMemcpy(d_data_.qvel_d, host_buf, total * sizeof(double), hipMemcpyHostToDevice);
    std::vector<float> h_f(total);
    for (int i = 0; i < total; i++) h_f[i] = (float)host_buf[i];
    hipMemcpy(d_data_.qvel, h_f.data(), total * sizeof(float), hipMemcpyHostToDevice);
}

void PhysicsData::write_qvel_d_env(const double* host_buf, int env_idx) {
    if (!allocated_ || env_idx < 0 || env_idx >= d_data_.num_envs) return;
    int nv = d_data_.nv;
    hipMemcpy(d_data_.qvel_d + env_idx * nv, host_buf, nv * sizeof(double), hipMemcpyHostToDevice);
    std::vector<float> h_f(nv);
    for (int i = 0; i < nv; i++) h_f[i] = (float)host_buf[i];
    hipMemcpy(d_data_.qvel + env_idx * nv, h_f.data(), nv * sizeof(float), hipMemcpyHostToDevice);
}

void PhysicsData::read_site_xpos(float* host_buf) const {
    if (!allocated_ || d_data_.nsite <= 0) return;
    int total = d_data_.num_envs * d_data_.nsite * 3;
    hipMemcpy(host_buf, d_data_.site_xpos, total * sizeof(float), hipMemcpyDeviceToHost);
}

void PhysicsData::read_site_xmat(float* host_buf) const {
    if (!allocated_ || d_data_.nsite <= 0) return;
    int total = d_data_.num_envs * d_data_.nsite * 9;
    hipMemcpy(host_buf, d_data_.site_xmat, total * sizeof(float), hipMemcpyDeviceToHost);
}

void PhysicsData::write_ctrl(const float* host_buf) {
    if (!allocated_) return;
    int total = d_data_.num_envs * d_data_.nu;
    hipMemcpy(d_data_.ctrl, host_buf, total * sizeof(float), hipMemcpyHostToDevice);
}

void PhysicsData::write_ctrl_env(const float* host_buf, int env_idx) {
    if (!allocated_ || env_idx < 0 || env_idx >= d_data_.num_envs) return;
    hipMemcpy(d_data_.ctrl + env_idx * d_data_.nu, host_buf,
              d_data_.nu * sizeof(float), hipMemcpyHostToDevice);
}

void PhysicsData::free_device() {
    if (!allocated_) return;
    DevicePhysicsData& d = d_data_;
    safe_free(d.qpos); safe_free(d.qvel); safe_free(d.qpos_d); safe_free(d.qvel_d);
    safe_free(d.qacc); safe_free(d.qacc_warmstart);
    safe_free(d.ctrl); safe_free(d.ctrl_d);
    safe_free(d.qfrc_applied); safe_free(d.qfrc_applied_d);
    safe_free(d.xpos); safe_free(d.xquat); safe_free(d.xpos_d); safe_free(d.xquat_d);
    safe_free(d.jnt_xaxis_d);
    safe_free(d.cvel); safe_free(d.xfrc);
    safe_free(d.geom_xpos); safe_free(d.geom_xquat);
    safe_free(d.geom_xpos_d); safe_free(d.geom_xquat_d); safe_free(d.geom_xmat_d);
    safe_free(d.geom_aabb);
    safe_free(d.contacts); safe_free(d.ncon); safe_free(d.body_pair_ncon);
    safe_free(d.constraint_J); safe_free(d.constraint_rhs);
    safe_free(d.constraint_lo); safe_free(d.constraint_hi);
    safe_free(d.constraint_lam); safe_free(d.constraint_D);
    safe_free(d.constraint_friction_parent); safe_free(d.constraint_friction_sibling); safe_free(d.constraint_friction_mu);
    safe_free(d.nconstraint);
    safe_free(d.body_inv_mass); safe_free(d.body_inv_inertia);
    safe_free(d.qfrc_bias);
    safe_free(d.qfrc_fluid);
    safe_free(d.time);
    safe_free(d.solver_M); safe_free(d.solver_H);
    safe_free(d.solver_rhs); safe_free(d.solver_M_inv_diag);
    safe_free(d.solver_M_inv);
    safe_free(d.newton_x); safe_free(d.newton_force); safe_free(d.newton_state);
    safe_free(d.newton_jv); safe_free(d.newton_is_elliptic);
    safe_free(d.rnea_omega); safe_free(d.rnea_alpha); safe_free(d.rnea_acc);
    safe_free(d.rnea_fnet); safe_free(d.rnea_tnet);
    safe_free(d.site_xpos); safe_free(d.site_xquat); safe_free(d.site_xmat);
    memset(&d, 0, sizeof(d));
    allocated_ = false;
}

} // namespace physics
} // namespace hipgl
