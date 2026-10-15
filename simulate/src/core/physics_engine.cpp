// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// PhysicsEngine: orchestrates the full physics step pipeline on GPU.
//
// Two solver paths:
//   SOLVER_NEWTON: parallel RNEA + parallel M-build + Newton (Cholesky) solve
//   SOLVER_PGS:    serial RNEA + apply forces + PGS + mass matrix solve (fallback)
//
// Phase 5: Fused FK (single kernel launch for all tree levels).
// Phase 6: Dual-stream physics/render overlap.

#include "hipgl/physics/physics_engine.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace hipgl {
namespace physics {

namespace kernels {
    // Kinematics
    void launch_forward_kinematics(int level_start, int level_count, int num_envs,
                                    const DevicePhysicsModel& model,
                                    const DevicePhysicsData& data,
                                    hipStream_t stream);
    void launch_forward_kinematics_fused(const DevicePhysicsModel& model,
                                          const DevicePhysicsData& data,
                                          int tree_depth,
                                          hipStream_t stream);
    void launch_forward_kinematics_fused_d(const DevicePhysicsModel& model,
                                            const DevicePhysicsData& data,
                                            int tree_depth,
                                            hipStream_t stream);
    void launch_integrate_pos_mj(const DevicePhysicsModel& model, const DevicePhysicsData& data,
                                 double dt, double* qpos_out, hipStream_t stream);
    void launch_geom_transforms(const DevicePhysicsModel& model,
                                 const DevicePhysicsData& data,
                                 hipStream_t stream);
    void launch_compute_aabbs(const DevicePhysicsModel& model,
                               const DevicePhysicsData& data,
                               hipStream_t stream);
    void launch_geom_transforms_and_aabbs(const DevicePhysicsModel& model,
                                           const DevicePhysicsData& data,
                                           hipStream_t stream);
    void launch_geom_transforms_d(const DevicePhysicsModel& model,
                                   const DevicePhysicsData& data,
                                   hipStream_t stream);
    // Integration / RNEA
    void launch_compute_inv_mass(const DevicePhysicsModel& model,
                                  DevicePhysicsData& data,
                                  hipStream_t stream);
    void launch_compute_bias_force(const DevicePhysicsModel& model,
                                    DevicePhysicsData& data,
                                    hipStream_t stream,
                                    const int* h_level_start_cached = nullptr);
    void launch_compute_fluid_forces(const DevicePhysicsModel& model,
                                      DevicePhysicsData& data,
                                      hipStream_t stream,
                                      const int* h_level_start_cached = nullptr);
    void launch_bias_and_fluid_forces(const DevicePhysicsModel& model,
                                       DevicePhysicsData& data,
                                       hipStream_t stream,
                                       const int* h_level_start_cached = nullptr);
    void launch_apply_forces(const DevicePhysicsModel& model,
                              const DevicePhysicsData& data,
                              const SolverOptions& opts,
                              hipStream_t stream);
    void launch_mass_matrix_solve(const DevicePhysicsModel& model,
                                   DevicePhysicsData& data,
                                   const SolverOptions& opts,
                                   hipStream_t stream);
    void launch_integrate(const DevicePhysicsModel& model,
                           DevicePhysicsData& data,
                           const SolverOptions& opts,
                           hipStream_t stream);
    // Collision
    void launch_broadphase(const DevicePhysicsModel& model,
                            DevicePhysicsData& data,
                            const SolverOptions& opts,
                            hipStream_t stream);
    void launch_broadphase_masked(const DevicePhysicsModel& model,
                                  DevicePhysicsData& data,
                                  const SolverOptions& opts,
                                  hipStream_t stream,
                                  const unsigned char* pair_mask);
    void launch_broadphase_compact(const DevicePhysicsModel& model,
                                   DevicePhysicsData& data,
                                   const SolverOptions& opts,
                                   hipStream_t stream,
                                   const unsigned char* pair_mask,
                                   const int* chunk_base, int nchunk);
    int broadphase_chunk_size();
    void launch_broadphase_pair_mask(const DevicePhysicsModel& model,
                                     unsigned char* pair_mask,
                                     hipStream_t stream);
    void launch_narrowphase(const DevicePhysicsModel& model,
                             DevicePhysicsData& data,
                             const SolverOptions& opts,
                             hipStream_t stream);
    // Constraints
    void launch_constraint_setup(const DevicePhysicsModel& model,
                                  DevicePhysicsData& data,
                                  const SolverOptions& opts,
                                  hipStream_t stream);
    // Solvers
    void launch_pgs_solve(const DevicePhysicsModel& model,
                           DevicePhysicsData& data,
                           const SolverOptions& opts,
                           hipStream_t stream);
    void launch_build_mass_matrix(const DevicePhysicsModel& model,
                                   DevicePhysicsData& data,
                                   const SolverOptions& opts,
                                   hipStream_t stream);
    void launch_invert_mass_matrix(const DevicePhysicsModel& model,
                                    DevicePhysicsData& data,
                                    hipStream_t stream);
    void launch_newton_solve(const DevicePhysicsModel& model,
                              DevicePhysicsData& data,
                              const SolverOptions& opts,
                              hipStream_t stream,
                              bool warm_start = false);
    void launch_mhat_solve(const DevicePhysicsModel& model,
                            DevicePhysicsData& data,
                            const SolverOptions& opts,
                            hipStream_t stream);
    void launch_clear_constraint_buffers(DevicePhysicsData& data,
                                          bool clear_lam,
                                          hipStream_t stream);
    void launch_site_transforms(const DevicePhysicsModel& model,
                                 const DevicePhysicsData& data,
                                 hipStream_t stream);
    bool launch_fk_pre_collision(const DevicePhysicsModel& model, DevicePhysicsData& data,
                                 const SolverOptions& opts, int tree_depth, bool zero_contacts,
                                 hipStream_t stream);
    bool launch_newton_solve_integ(const DevicePhysicsModel& model, DevicePhysicsData& data,
                                   const SolverOptions& opts, hipStream_t stream,
                                   bool warm_start, bool do_integrate);
    bool launch_fk_all_fused(const DevicePhysicsModel& model,
                             const DevicePhysicsData& data,
                             int tree_depth,
                             const SolverOptions& opts,
                             bool do_integrate,
                             bool zero_contacts,
                             int clear_constraints,
                             bool do_bias,
                             bool* bias_done,
                             hipStream_t stream);
    void launch_build_M_inv_diag_standalone(const DevicePhysicsModel& model,
                                             DevicePhysicsData& data,
                                             hipStream_t stream);
    void launch_gravcomp(const DevicePhysicsModel& model, DevicePhysicsData& data,
                         hipStream_t stream);
    bool constraint_setup_fused_ok(const DevicePhysicsModel& model,
                                   const DevicePhysicsData& data, const SolverOptions& opts);
    void launch_constraint_setup_fused(const DevicePhysicsModel& model,
                                       DevicePhysicsData& data, const SolverOptions& opts,
                                       hipStream_t stream);
    void launch_extract_M_diag_inv(const DevicePhysicsModel& model,
                                    DevicePhysicsData& data,
                                    hipStream_t stream);
    void launch_joint_pd(const DevicePhysicsModel& model, DevicePhysicsData& data, const DeviceJointPd& pd,
                         hipStream_t stream);
    // world skip: per-translation-unit freeze-mask setters (world_mask.h).
    void set_world_mask_kinematics(const unsigned char* const* src, hipStream_t stream);
    void set_world_mask_solver(const unsigned char* const* src, hipStream_t stream);
    void set_world_mask_collision(const unsigned char* const* src, hipStream_t stream);
    void set_world_mask_integrate(const unsigned char* const* src, hipStream_t stream);
}

PhysicsEngine::PhysicsEngine()
    : model_(nullptr), options_(default_solver_options()),
      gpu_id_(0), initialized_(false),
      stream_(nullptr), render_stream_(nullptr), physics_done_event_(nullptr) {}

PhysicsEngine::~PhysicsEngine() {
    if (world_mask_on_) clear_world_active();
    if (d_world_active_)  { hipFree(d_world_active_);  d_world_active_  = nullptr; }
    if (h_world_stage_)   { hipHostFree(h_world_stage_); h_world_stage_ = nullptr; }
    if (d_bp_pair_mask_)  { hipFree(d_bp_pair_mask_);  d_bp_pair_mask_  = nullptr; }
    if (d_bp_chunks_)     { hipFree(d_bp_chunks_);     d_bp_chunks_     = nullptr; }
    for (void* p : {(void*)lag_qpos_d_, (void*)lag_xpos_d_, (void*)lag_xquat_d_, (void*)lag_xpos_,
                    (void*)lag_xquat_, (void*)lag_site_xpos_, (void*)lag_site_xquat_,
                    (void*)lag_site_xmat_})
        if (p) hipFree(p);
    for (void* p : {(void*)jpd_.act, (void*)jpd_.dof, (void*)jpd_.qadr, (void*)jpd_.grp_start, (void*)jpd_.grp_len,
                    (void*)jpd_.kp, (void*)jpd_.kd, (void*)jpd_.gear, (void*)jpd_.lo, (void*)jpd_.hi,
                    (void*)jpd_.target})
        if (p) hipFree(p);
    jpd_ = DeviceJointPd{};
    if (d_osc_config_)    { hipFree(d_osc_config_);    d_osc_config_    = nullptr; }
    if (d_osc_env_state_) { hipFree(d_osc_env_state_); d_osc_env_state_ = nullptr; }
    if (d_osc_actions_)   { hipFree(d_osc_actions_);   d_osc_actions_   = nullptr; }
    if (osc_exec_cold_)  { hipGraphExecDestroy(osc_exec_cold_);  osc_exec_cold_  = nullptr; }
    if (osc_graph_cold_) { hipGraphDestroy(osc_graph_cold_);      osc_graph_cold_ = nullptr; }
    if (osc_exec_warm_)  { hipGraphExecDestroy(osc_exec_warm_);  osc_exec_warm_  = nullptr; }
    if (osc_graph_warm_) { hipGraphDestroy(osc_graph_warm_);      osc_graph_warm_ = nullptr; }
    if (step_graph_exec_k_)  { hipGraphExecDestroy(step_graph_exec_k_); step_graph_exec_k_ = nullptr; }
    if (step_graph_k_)       { hipGraphDestroy(step_graph_k_);          step_graph_k_ = nullptr; }
    if (step_graph_exec_)    { hipGraphExecDestroy(step_graph_exec_); step_graph_exec_ = nullptr; }
    if (step_graph_)         { hipGraphDestroy(step_graph_);          step_graph_ = nullptr; }
    if (physics_done_event_) { hipEventDestroy(physics_done_event_); physics_done_event_ = nullptr; }
    if (render_stream_)      { hipStreamDestroy(render_stream_);      render_stream_ = nullptr; }
    if (stream_)             { hipStreamDestroy(stream_);             stream_ = nullptr; }
}

bool PhysicsEngine::init(PhysicsModel* model, int num_envs, int gpu_id) {
    if (!model) return false;

    // Capacity guard. The solver kernels early-return when nv > KERNEL_MAXNV
    // (physics_solver.hip: `if (nv > KERNEL_MAXNV) return;`), which silently
    // produces a model that loads fine, steps fine, and never integrates any
    // constraint force. Real LIBERO models hit this: libero_object is nv=51
    // and 2 libero_10 tasks are nv=57, against KERNEL_MAXNV=48 — 12 of 40
    // tasks were running with a dead solver and no diagnostic. Fail loudly
    // here instead.
    //
    // Render-only use is exempt: PhysicsEngine::forward() runs forward
    // kinematics, geom transforms and site transforms — it never enters the
    // solver — so a render pipeline driven by externally-supplied qpos/qvel is
    // correct at any nv. Callers opt in with `PhysicsEngine::set_render_only()`
    // before init (or the `render_only` flag on create_engine), which downgrades
    // the nv check to a warning. Everything that integrates dynamics must leave
    // it off and take the hard failure.
    {
        const PhysicsModelDesc& d = model->desc();
        if (d.nv > KERNEL_MAXNV) {
            if (!render_only_) {
                fprintf(stderr,
                        "instinct_sim: model nv=%d exceeds KERNEL_MAXNV=%d — the dense "
                        "solver cannot run and would silently no-op. Raise "
                        "KERNEL_MAXNV in include/hipgl/physics/physics_types.h and "
                        "rebuild (note: increases solver VGPR/LDS pressure), or call "
                        "set_render_only(true) if this engine only drives rendering.\n",
                        d.nv, KERNEL_MAXNV);
                return false;
            }
            fprintf(stderr,
                    "instinct_sim: [render-only] model nv=%d exceeds KERNEL_MAXNV=%d; "
                    "forward kinematics and rendering are valid, but step() would "
                    "produce no constraint forces. Do not integrate with this engine.\n",
                    d.nv, KERNEL_MAXNV);
        }
        // nq is checked separately from nv and against the same MAX_DOF bound,
        // because PhysicsModelDesc::qpos0 is declared float[MAX_DOF] yet is
        // indexed by qpos_adr, which scales with nq — NOT nv. A free joint
        // contributes 7 qpos to 6 qvel, so a model that passes an nv<=128 check
        // can still reach nq=147 and write 19 floats past qpos0 during parsing.
        // That is silent host-memory corruption before any kernel runs.
        if (d.nq > MAX_DOF || d.nv > MAX_DOF || d.nbody > MAX_BODIES ||
            d.njnt > MAX_JOINTS || d.ngeom > MAX_GEOMS) {
            fprintf(stderr,
                    "instinct_sim: model exceeds a compile-time capacity "
                    "(nq=%d/%d nv=%d/%d nbody=%d/%d njnt=%d/%d ngeom=%d/%d)\n",
                    d.nq, MAX_DOF, d.nv, MAX_DOF, d.nbody, MAX_BODIES,
                    d.njnt, MAX_JOINTS, d.ngeom, MAX_GEOMS);
            return false;
        }
    }

    model_ = model;
    gpu_id_ = gpu_id;

    hipError_t err = hipSetDevice(gpu_id);
    if (err != hipSuccess) {
        fprintf(stderr, "instinct_sim: failed to set device %d\n", gpu_id);
        return false;
    }

    err = hipStreamCreate(&stream_);
    if (err != hipSuccess) {
        fprintf(stderr, "instinct_sim: failed to create physics stream\n");
        return false;
    }

    // Phase 6: create render stream and synchronization event
    err = hipStreamCreate(&render_stream_);
    if (err != hipSuccess) {
        fprintf(stderr, "instinct_sim: failed to create render stream\n");
        return false;
    }
    err = hipEventCreateWithFlags(&physics_done_event_, hipEventDisableTiming);
    if (err != hipSuccess) {
        fprintf(stderr, "instinct_sim: failed to create physics done event\n");
        return false;
    }

    if (!data_.allocate(model->desc(), num_envs)) {
        fprintf(stderr, "instinct_sim: failed to allocate data for %d envs\n", num_envs);
        return false;
    }

    data_.reset(model->desc());

    DevicePhysicsData& dd = const_cast<DevicePhysicsData&>(data_.device_data());
    kernels::launch_compute_inv_mass(model->device_model(), dd, stream_);

    forward_kinematics();
    geom_transforms();
    kernels::launch_site_transforms(model_->device_model(), data_.device_data(), stream_);

    hipStreamSynchronize(stream_);
    initialized_ = true;
    return true;
}

void PhysicsEngine::step() {
    if (!initialized_) return;

    const DevicePhysicsModel& dm = model_->device_model();
    DevicePhysicsData& dd = const_cast<DevicePhysicsData&>(data_.device_data());

    static const bool pre_en = [] { const char* e = getenv("HP_PRECOLL"); return !(e && e[0] == '0'); }();
    static const int pre_max = [] { const char* e = getenv("HP_PRECOLL_MAX_ENVS"); return e ? atoi(e) : 128; }();
    const bool pre_done = pre_en && dd.num_envs <= pre_max && !world_mask_on_ && options_.solver == SOLVER_NEWTON &&
                          !options_.zero_bias &&
                          kernels::launch_fk_pre_collision(dm, dd, options_, model_->desc().tree_depth,
                                                           options_.enable_contact, stream_);
    if (!pre_done) {
        // Double-precision FK (authoritative), then d2f copy for rendering/AABB
        kernels::launch_forward_kinematics_fused_d(dm, dd, model_->desc().tree_depth, stream_);
        geom_transforms();
    }

    if (options_.enable_contact) collision_detection_on(stream_, /*zero_counters=*/!pre_done);

    if (options_.solver == SOLVER_NEWTON) {
        // One-launch constraint setup (pyramidal models): gravcomp + contact/joint
        // rows + JT mirror, no buffer clear.  Same kernels' arithmetic and order.
        const bool fused_setup = !options_.zero_bias &&
                                 kernels::constraint_setup_fused_ok(dm, dd, options_);
        if (options_.zero_bias) {
            hipMemsetAsync(dd.qfrc_bias, 0,
                           (size_t)dd.num_envs * dm.nv * sizeof(double), stream_);
            kernels::launch_compute_fluid_forces(dm, dd, stream_, model_->desc().level_start);
        } else {
            if (!pre_done)
                kernels::launch_bias_and_fluid_forces(dm, dd, stream_, model_->desc().level_start);
            if (!fused_setup) kernels::launch_gravcomp(dm, dd, stream_);
        }
        // Build mass matrix BEFORE constraint setup so we can extract diag(M)
        if (!pre_done) kernels::launch_build_mass_matrix(dm, dd, options_, stream_);
        kernels::launch_extract_M_diag_inv(dm, dd, stream_);
        // joint PD actuators read this substep's M and bias (joint_pd.h)
        if (jpd_.n > 0) kernels::launch_joint_pd(dm, dd, jpd_, stream_);

        if (fused_setup)
            kernels::launch_constraint_setup_fused(dm, dd, options_, stream_);
        else
            constraint_setup(/*preserve_lambda=*/true);

        const bool integ_done = kernels::launch_newton_solve_integ(dm, dd, options_, stream_,
                                    /*warm_start=*/true, /*do_integrate=*/!world_mask_on_);
        // solver_M = M_hat = M + h*D. Newton's qacc satisfies M_hat*qacc = qfrc + J'*lam.
        // mhat_solve transforms to integration qacc: qacc_int = M_hat^-1 * M * qacc.
        if (options_.enable_damping)
            kernels::launch_mhat_solve(dm, dd, options_, stream_);
        if (!integ_done) kernels::launch_integrate(dm, dd, options_, stream_);
    } else {
        // PGS fallback path: use body-level M_inv_diag for constraint setup
        kernels::launch_build_M_inv_diag_standalone(dm, dd, stream_);
        constraint_setup(/*preserve_lambda=*/false);
        constraint_solve();
        integration();
    }

    // Phase 6: record event so render_stream_ can wait
    hipEventRecord(physics_done_event_, stream_);
}

#ifndef HP_STEP_UNROLL_DEFAULT
#define HP_STEP_UNROLL_DEFAULT 10
#endif

void PhysicsEngine::capture_step_graph() {
    if (graph_captured_) return;

    // Warm up with one regular step to prime all buffers
    step();
    hipStreamSynchronize(stream_);

    // Capture the step sequence into a graph
    hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal);
    step();
    hipStreamEndCapture(stream_, &step_graph_);

    hipGraphNode_t err_node;
    size_t log_size = 0;
    hipError_t err = hipGraphInstantiate(&step_graph_exec_, step_graph_, &err_node, nullptr, log_size);
    if (err != hipSuccess) {
        fprintf(stderr, "instinct_sim: graph instantiation failed, falling back to loop\n");
        if (step_graph_) { hipGraphDestroy(step_graph_); step_graph_ = nullptr; }
        step_graph_exec_ = nullptr;
        return;
    }
    graph_captured_ = true;

    int unroll = HP_STEP_UNROLL_DEFAULT;
    if (const char* e = getenv("HP_STEP_UNROLL")) unroll = atoi(e);
    if (unroll > 1) {
        hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal);
        for (int k = 0; k < unroll; k++) step();
        hipError_t ce = hipStreamEndCapture(stream_, &step_graph_k_);
        if (ce == hipSuccess && step_graph_k_ &&
            hipGraphInstantiate(&step_graph_exec_k_, step_graph_k_, &err_node, nullptr, 0) == hipSuccess) {
            step_unroll_ = unroll;
        } else {
            fprintf(stderr, "instinct_sim: unrolled step graph (K=%d) unavailable, using K=1\n", unroll);
            if (step_graph_k_) { hipGraphDestroy(step_graph_k_); step_graph_k_ = nullptr; }
            step_graph_exec_k_ = nullptr;
            (void)hipGetLastError();
        }
    }
}

void PhysicsEngine::step_n(int n) {
    if (!initialized_ || n <= 0) return;

    if (!graph_captured_) capture_step_graph();

    if (graph_captured_ && step_graph_exec_) {
        int i = 0;
        if (step_graph_exec_k_ && step_unroll_ > 1)
            for (; i + step_unroll_ <= n; i += step_unroll_)
                hipGraphLaunch(step_graph_exec_k_, stream_);
        for (; i < n; i++)
            hipGraphLaunch(step_graph_exec_, stream_);
        hipEventRecord(physics_done_event_, stream_);
    } else {
        for (int i = 0; i < n; i++) step();
    }
}

PhysicsEngine::PhaseTimings PhysicsEngine::step_profile() {
    if (!initialized_) return {0,0,0,0,0,0};

    const DevicePhysicsModel& dm = model_->device_model();
    DevicePhysicsData& dd = const_cast<DevicePhysicsData&>(data_.device_data());

    PhaseTimings t = {};
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    float ms;

    hipEventRecord(e0, stream_);
    forward_kinematics();
    hipEventRecord(e1, stream_);
    hipEventSynchronize(e1);
    hipEventElapsedTime(&ms, e0, e1); t.fk_ms = ms;

    hipEventRecord(e0, stream_);
    geom_transforms();
    hipEventRecord(e1, stream_);
    hipEventSynchronize(e1);
    hipEventElapsedTime(&ms, e0, e1); t.geom_ms = ms;

    hipEventRecord(e0, stream_);
    if (options_.enable_contact) collision_detection();
    hipEventRecord(e1, stream_);
    hipEventSynchronize(e1);
    hipEventElapsedTime(&ms, e0, e1); t.collision_ms = ms;

    hipEventRecord(e0, stream_);
    kernels::launch_bias_and_fluid_forces(dm, dd, stream_, model_->desc().level_start);
    kernels::launch_gravcomp(dm, dd, stream_);
    kernels::launch_build_mass_matrix(dm, dd, options_, stream_);
    kernels::launch_extract_M_diag_inv(dm, dd, stream_);
    if (jpd_.n > 0) kernels::launch_joint_pd(dm, dd, jpd_, stream_);
    constraint_setup();
    hipEventRecord(e1, stream_);
    hipEventSynchronize(e1);
    hipEventElapsedTime(&ms, e0, e1); t.constraint_ms = ms;

    hipEventRecord(e0, stream_);
    if (options_.solver == SOLVER_NEWTON) {
        kernels::launch_build_mass_matrix(dm, dd, options_, stream_);
        kernels::launch_newton_solve(dm, dd, options_, stream_);
        if (options_.enable_damping)
            kernels::launch_mhat_solve(dm, dd, options_, stream_);
    } else {
        constraint_solve();
        kernels::launch_bias_and_fluid_forces(dm, dd, stream_, model_->desc().level_start);
        kernels::launch_gravcomp(dm, dd, stream_);
        kernels::launch_apply_forces(dm, data_.device_data(), options_, stream_);
        kernels::launch_mass_matrix_solve(dm, dd, options_, stream_);
    }
    hipEventRecord(e1, stream_);
    hipEventSynchronize(e1);
    hipEventElapsedTime(&ms, e0, e1); t.solve_ms = ms;

    hipEventRecord(e0, stream_);
    kernels::launch_integrate(dm, dd, options_, stream_);
    hipEventRecord(e1, stream_);
    hipEventSynchronize(e1);
    hipEventElapsedTime(&ms, e0, e1); t.integrate_ms = ms;

    hipEventRecord(physics_done_event_, stream_);
    hipEventDestroy(e0);
    hipEventDestroy(e1);
    return t;
}

void PhysicsEngine::forward() {
    if (!initialized_) return;
    forward_kinematics();
    geom_transforms();
    kernels::launch_site_transforms(model_->device_model(), data_.device_data(), stream_);
    kernels::launch_bias_and_fluid_forces(model_->device_model(),
        const_cast<DevicePhysicsData&>(data_.device_data()),
        stream_, model_->desc().level_start);
    kernels::launch_gravcomp(model_->device_model(),
        const_cast<DevicePhysicsData&>(data_.device_data()), stream_);
    hipStreamSynchronize(stream_);
}

void PhysicsEngine::sync() {
    if (stream_) hipStreamSynchronize(stream_);
}

void PhysicsEngine::sync_fk() {
    if (!initialized_) return;
    forward_kinematics();
    hipStreamSynchronize(stream_);
}

void PhysicsEngine::sync_geom() {
    if (!initialized_) return;
    geom_transforms();
    hipStreamSynchronize(stream_);
}

void PhysicsEngine::forward_kinematics() {
    const DevicePhysicsModel& dm = model_->device_model();
    const PhysicsModelDesc& desc = model_->desc();
    const DevicePhysicsData& dd = data_.device_data();

    // Double-precision FK writes xpos_d/xquat_d then copies to float xpos/xquat
    kernels::launch_forward_kinematics_fused_d(dm, dd, desc.tree_depth, stream_);
}

void PhysicsEngine::geom_transforms() {
    const DevicePhysicsModel& dm = model_->device_model();
    const DevicePhysicsData& dd = data_.device_data();

    kernels::launch_geom_transforms_and_aabbs(dm, dd, stream_);
    // Double-precision geom transforms for collision and constraint pipeline
    kernels::launch_geom_transforms_d(dm, dd, stream_);
}

void PhysicsEngine::site_transforms() {
    if (!initialized_) return;
    kernels::launch_site_transforms(model_->device_model(), data_.device_data(), stream_);
}

void PhysicsEngine::step_and_sites() {
    step();
    site_transforms();
}

void PhysicsEngine::lagged_kinematics(double dt_back) {
    if (!initialized_) return;
    const DevicePhysicsModel& dm = model_->device_model();
    const DevicePhysicsData& dd = data_.device_data();
    const size_t ne = (size_t)dd.num_envs;
    if (!lag_qpos_d_) {
        auto alloc = [](void** p, size_t bytes) {
            if (hipMalloc(p, bytes > 0 ? bytes : 1) != hipSuccess)
                throw std::runtime_error("lagged_kinematics: hipMalloc failed");
        };
        alloc((void**)&lag_qpos_d_,  ne * dm.nq * sizeof(double));
        alloc((void**)&lag_xpos_d_,  ne * dm.nbody * 3 * sizeof(double));
        alloc((void**)&lag_xquat_d_, ne * dm.nbody * 4 * sizeof(double));
        alloc((void**)&lag_xpos_,    ne * dm.nbody * 3 * sizeof(float));
        alloc((void**)&lag_xquat_,   ne * dm.nbody * 4 * sizeof(float));
        alloc((void**)&lag_site_xpos_,  ne * dm.nsite * 3 * sizeof(float));
        alloc((void**)&lag_site_xquat_, ne * dm.nsite * 4 * sizeof(float));
        alloc((void**)&lag_site_xmat_,  ne * dm.nsite * 9 * sizeof(float));
    }
    kernels::launch_integrate_pos_mj(dm, dd, -dt_back, lag_qpos_d_, stream_);
    // the regular FK / site kernels, pointed at the lag buffers
    DevicePhysicsData ld = dd;
    ld.qpos_d = lag_qpos_d_;
    ld.xpos_d = lag_xpos_d_;
    ld.xquat_d = lag_xquat_d_;
    ld.xpos = lag_xpos_;
    ld.xquat = lag_xquat_;
    ld.jnt_xaxis_d = nullptr;
    ld.site_xpos = lag_site_xpos_;
    ld.site_xquat = lag_site_xquat_;
    ld.site_xmat = lag_site_xmat_;
    kernels::launch_forward_kinematics_fused_d(dm, ld, model_->desc().tree_depth, stream_);
    kernels::launch_site_transforms(dm, ld, stream_);
}

void PhysicsEngine::begin_lagged_render(double dt_back) {
    if (!initialized_) return;
    lagged_kinematics(dt_back);           // allocates lag_qpos_d_ and fills it with the stepped-back qpos
    const DevicePhysicsModel& dm = model_->device_model();
    DevicePhysicsData ld = data_.device_data();
    ld.qpos_d = lag_qpos_d_;              // FK reads the lagged qpos, writes the live pose buffers
    kernels::launch_forward_kinematics_fused_d(dm, ld, model_->desc().tree_depth, stream_);
    geom_transforms();
    kernels::launch_site_transforms(dm, data_.device_data(), stream_);
    hipStreamSynchronize(stream_);        // the renderer runs on its own streams
}

void PhysicsEngine::end_lagged_render() {
    if (!initialized_) return;
    hipDeviceSynchronize();               // every render stream is done reading the lagged poses
    forward_kinematics();
    geom_transforms();
    kernels::launch_site_transforms(model_->device_model(), data_.device_data(), stream_);
    hipStreamSynchronize(stream_);
}

namespace {
template <typename T>
void read_lag(hipStream_t s, const T* src, T* out, size_t n) {
    if (!src) throw std::runtime_error("read_lagged_*: call lagged_kinematics() first");
    hipStreamSynchronize(s);
    hipMemcpy(out, src, n * sizeof(T), hipMemcpyDeviceToHost);
}
}  // namespace

void PhysicsEngine::read_lagged_qpos(double* out) const {
    const auto& dd = data_.device_data();
    read_lag(stream_, lag_qpos_d_, out, (size_t)dd.num_envs * dd.nq);
}
void PhysicsEngine::read_lagged_xpos(double* out) const {
    const auto& dd = data_.device_data();
    read_lag(stream_, lag_xpos_d_, out, (size_t)dd.num_envs * dd.nbody * 3);
}
void PhysicsEngine::read_lagged_xquat(double* out) const {
    const auto& dd = data_.device_data();
    read_lag(stream_, lag_xquat_d_, out, (size_t)dd.num_envs * dd.nbody * 4);
}
void PhysicsEngine::read_lagged_site_xpos(float* out) const {
    const auto& dd = data_.device_data();
    if (dd.nsite <= 0) return;
    read_lag(stream_, lag_site_xpos_, out, (size_t)dd.num_envs * dd.nsite * 3);
}

void PhysicsEngine::collision_detection() {
    collision_detection_on(stream_, /*zero_counters=*/true);
}

void PhysicsEngine::collision_detection_on(void* s_, bool zero_counters) {
    hipStream_t s = (hipStream_t)s_;
    const DevicePhysicsModel& dm = model_->device_model();
    DevicePhysicsData& dd = const_cast<DevicePhysicsData&>(data_.device_data());

    if (zero_counters) {
        hipMemsetAsync(dd.ncon, 0, dd.num_envs * sizeof(int), s);
        if (dd.body_pair_ncon) {
            int nb = dd.nbody;
            hipMemsetAsync(dd.body_pair_ncon, 0,
                           dd.num_envs * nb * nb * sizeof(int), s);
        }
    }
    const unsigned char* mask = broadphase_pair_mask(s);
    if (mask && mask == d_bp_pair_mask_ && d_bp_chunks_ && bp_nchunk_ > 0)
        kernels::launch_broadphase_compact(dm, dd, options_, s, mask,
                                           d_bp_chunks_, bp_nchunk_);
    else
        kernels::launch_broadphase_masked(dm, dd, options_, s, mask);
    kernels::launch_narrowphase(dm, dd, options_, s);
}

const unsigned char* PhysicsEngine::broadphase_pair_mask(void* s_) {
    hipStream_t s = (hipStream_t)s_;
    const DevicePhysicsModel& dm = model_->device_model();
    const int ng = dm.ngeom;
    if (ng < 2 || !dm.geom_body || !dm.geom_contype || !dm.geom_conaffinity)
        return nullptr;
    if (d_bp_pair_mask_ && bp_pair_mask_ng_ == ng &&
        bp_pair_mask_key_ == (const void*)dm.geom_body)
        return d_bp_pair_mask_;
    // Build only outside stream capture (hipMalloc is not capturable); a
    // graph captured before the first eager collision call simply uses the
    // unmasked kernel.
    hipStreamCaptureStatus cs = hipStreamCaptureStatusNone;
    if (hipStreamIsCapturing(s, &cs) != hipSuccess || cs != hipStreamCaptureStatusNone)
        return nullptr;
    if (d_bp_pair_mask_ || d_bp_chunks_) {
        hipStreamSynchronize(s);
        if (d_bp_pair_mask_) hipFree(d_bp_pair_mask_);
        if (d_bp_chunks_)    hipFree(d_bp_chunks_);
        d_bp_pair_mask_ = nullptr;
        d_bp_chunks_    = nullptr;
        bp_nchunk_      = 0;
    }
    const size_t npairs = (size_t)ng * (ng - 1) / 2;
    if (hipMalloc(&d_bp_pair_mask_, npairs) != hipSuccess) {
        d_bp_pair_mask_ = nullptr;
        return nullptr;
    }
    kernels::launch_broadphase_pair_mask(dm, d_bp_pair_mask_, s);
    bp_pair_mask_ng_  = ng;
    bp_pair_mask_key_ = (const void*)dm.geom_body;

    // Chunk list for the compacted launch: the first pair index of every
    // chunk-aligned slice of the pair triangle holding >= 1 surviving pair.
    // Built once here (setup time, outside capture); on any failure the
    // engine keeps using the full-triangle masked launch.
    {
        const int chunk = kernels::broadphase_chunk_size();
        std::vector<unsigned char> h_mask(npairs);
        if (hipMemcpyAsync(h_mask.data(), d_bp_pair_mask_, npairs,
                           hipMemcpyDeviceToHost, s) == hipSuccess &&
            hipStreamSynchronize(s) == hipSuccess) {
            std::vector<int> bases;
            size_t nsurv = 0;
            for (size_t b = 0; b < npairs; b += (size_t)chunk) {
                size_t e = std::min(npairs, b + (size_t)chunk);
                size_t c = 0;
                for (size_t p = b; p < e; ++p) c += h_mask[p] ? 1 : 0;
                if (c) bases.push_back((int)b);
                nsurv += c;
            }
            if (!bases.empty() &&
                hipMalloc(&d_bp_chunks_, bases.size() * sizeof(int)) == hipSuccess) {
                if (hipMemcpy(d_bp_chunks_, bases.data(), bases.size() * sizeof(int),
                              hipMemcpyHostToDevice) == hipSuccess) {
                    bp_nchunk_ = (int)bases.size();
                } else {
                    hipFree(d_bp_chunks_);
                    d_bp_chunks_ = nullptr;
                }
            }
            if (getenv("HIPGL_BP_DEBUG"))
                fprintf(stderr, "instinct_sim broadphase: ngeom=%d pairs=%zu survivors=%zu "
                        "chunks=%d/%zu (chunk=%d)\n", ng, npairs, nsurv, bp_nchunk_,
                        (npairs + chunk - 1) / chunk, chunk);
        }
    }
    return d_bp_pair_mask_;
}

bool PhysicsEngine::osc_fused_head(bool do_integrate, bool zero_contacts,
                                   int clear_constraints, bool* cons_cleared,
                                   bool* bias_done) {
    const DevicePhysicsModel& dm = model_->device_model();
    DevicePhysicsData& dd = const_cast<DevicePhysicsData&>(data_.device_data());
    if (cons_cleared) *cons_cleared = false;
    if (bias_done) *bias_done = false;
    const bool can_clear = clear_constraints >= 0 && dd.num_envs > 0 &&
        dd.nconstraint && dd.constraint_friction_parent &&
        dd.constraint_friction_sibling && dd.constraint_friction_mu &&
        (clear_constraints == 0 || dd.constraint_lam);
    if (kernels::launch_fk_all_fused(dm, dd, model_->desc().tree_depth, options_,
                                     do_integrate, zero_contacts,
                                     can_clear ? clear_constraints : -1,
                                     /*do_bias=*/bias_done != nullptr, bias_done,
                                     stream_)) {
        if (cons_cleared) *cons_cleared = can_clear;
        return zero_contacts;
    }
    // Fallback: the unfused sequence.
    if (do_integrate) kernels::launch_integrate(dm, dd, options_, stream_);
    forward_kinematics();
    geom_transforms();
    kernels::launch_site_transforms(dm, data_.device_data(), stream_);
    return false;
}

void PhysicsEngine::constraint_setup(bool preserve_lambda) {
    const DevicePhysicsModel& dm = model_->device_model();
    DevicePhysicsData& dd = const_cast<DevicePhysicsData&>(data_.device_data());

    kernels::launch_clear_constraint_buffers(dd, !preserve_lambda, stream_);

    kernels::launch_constraint_setup(dm, dd, options_, stream_);
}

void PhysicsEngine::constraint_solve() {
    const DevicePhysicsModel& dm = model_->device_model();
    DevicePhysicsData& dd = const_cast<DevicePhysicsData&>(data_.device_data());

    kernels::launch_pgs_solve(dm, dd, options_, stream_);
}

void PhysicsEngine::integration() {
    const DevicePhysicsModel& dm = model_->device_model();
    DevicePhysicsData& dd = const_cast<DevicePhysicsData&>(data_.device_data());

    kernels::launch_bias_and_fluid_forces(dm, dd, stream_, model_->desc().level_start);
    kernels::launch_gravcomp(dm, dd, stream_);
    kernels::launch_apply_forces(dm, data_.device_data(), options_, stream_);
    kernels::launch_mass_matrix_solve(dm, dd, options_, stream_);
    kernels::launch_integrate(dm, dd, options_, stream_);
}

// ── GPU-side OSC: setup + step_with_osc ─────────────────────────────────────

void PhysicsEngine::set_osc_config(const OSCConfig& config) {
    if (!d_osc_config_) {
        hipMalloc(&d_osc_config_, sizeof(OSCConfig));
    }
    hipMemcpy(d_osc_config_, &config, sizeof(OSCConfig), hipMemcpyHostToDevice);
    osc_config_host_ = config;
    osc_configured_ = true;
}

const OSCConfig& PhysicsEngine::osc_config() const {
    if (!osc_configured_ || !d_osc_config_)
        throw std::runtime_error("OSC is not configured");
    return osc_config_host_;
}

void PhysicsEngine::initialize_osc_env_state(int env_idx, const OSCEnvState& state) {
    if (!initialized_ || !model_)
        throw std::runtime_error("initialize_osc_env_state requires an initialized engine");
    const OSCConfig& cfg = osc_config();
    const auto& desc = model_->desc();
    const int num_envs = data_.num_envs();
    if (env_idx < 0 || env_idx >= num_envs)
        throw std::out_of_range("OSC environment index is out of range");
    if (cfg.n_arm <= 0 || cfg.n_arm > OSC_MAX_ARM ||
        cfg.n_gripper < 0 || cfg.n_gripper > OSC_MAX_GRIP)
        throw std::invalid_argument("OSC config has invalid arm/gripper dimensions");
    if ((cfg.eef_site_idx >= 0 && cfg.eef_site_idx >= desc.nsite) ||
        (cfg.eef_site_idx < 0 && (cfg.eef_body_idx < 0 || cfg.eef_body_idx >= desc.nbody)))
        throw std::invalid_argument("OSC config has an invalid EEF index");
    for (int i = 0; i < cfg.n_arm; ++i) {
        if (cfg.arm_joint_indices[i] < 0 || cfg.arm_joint_indices[i] >= desc.njnt ||
            cfg.arm_dof_indices[i] < 0 || cfg.arm_dof_indices[i] >= desc.nv ||
            cfg.arm_qpos_indices[i] < 0 || cfg.arm_qpos_indices[i] >= desc.nq ||
            cfg.arm_actuator_indices[i] < 0 || cfg.arm_actuator_indices[i] >= desc.nu)
            throw std::invalid_argument("OSC config has an invalid arm index");
    }
    for (int i = 0; i < cfg.n_gripper; ++i) {
        if (cfg.gripper_actuator_indices[i] < 0 || cfg.gripper_actuator_indices[i] >= desc.nu)
            throw std::invalid_argument("OSC config has an invalid gripper index");
    }
    auto finite = [](const double* values, int count, const char* name) {
        for (int i = 0; i < count; ++i)
            if (!std::isfinite(values[i]))
                throw std::invalid_argument(std::string(name) + " must contain finite values");
    };
    const double gains[] = {cfg.kp_pos, cfg.kp_ori, cfg.kd_pos, cfg.kd_ori,
        cfg.kp_null, cfg.kd_null, cfg.action_scale_pos, cfg.action_scale_ori,
        cfg.gripper_speed};
    finite(gains, 9, "OSC config gains/scales");
    finite(cfg.arm_ctrl_limits, cfg.n_arm, "OSC config arm limits");
    if (cfg.has_gripper_ctrlrange)
        finite(cfg.gripper_ctrlrange, 2 * cfg.n_gripper, "OSC config gripper ranges");
    if (state.initialized != 1 ||
        (state.first_goal_cache_pending != 0 && state.first_goal_cache_pending != 1) ||
        (state.first_tick_cache_pending != 0 && state.first_tick_cache_pending != 1))
        throw std::invalid_argument("OSC reset state has invalid initialization/cache flags");
    finite(state.init_jpos, cfg.n_arm, "home");
    finite(state.goal_pos, 3, "goal_pos");
    finite(state.goal_ori, 9, "goal_ori");
    finite(state.gripper_current_action, OSC_MAX_GRIP, "gripper");
    finite(&state.gripper_direction, 1, "gripper_direction");
    if (state.first_goal_cache_pending || state.first_tick_cache_pending) {
        finite(state.cached_ee_pos, 3, "cache.ee_pos");
        finite(state.cached_ee_ori, 9, "cache.ee_ori");
        finite(state.cached_ee_vel, 6, "cache.ee_vel");
        finite(state.cached_joint_pos, cfg.n_arm, "cache.joint_pos");
        finite(state.cached_joint_vel, cfg.n_arm, "cache.joint_vel");
        for (int r = 0; r < 6; ++r)
            finite(state.cached_J_full + r * OSC_MAX_ARM, cfg.n_arm, "cache.J_full");
        for (int r = 0; r < cfg.n_arm; ++r)
            finite(state.cached_mass_matrix + r * OSC_MAX_ARM, cfg.n_arm, "cache.mass_matrix");
    }
    if ((d_osc_env_state_ || d_osc_actions_) && osc_num_envs_ != num_envs)
        throw std::runtime_error("OSC buffer count does not match the engine environment count");

    // All validation is complete. Do not reset actions, goals, warmstart, or
    // physics state for peers. Stable device storage also preserves HIP graphs.
    auto checked = [](hipError_t status) {
        if (status != hipSuccess)
            throw std::runtime_error(std::string("OSC state upload: ") + hipGetErrorString(status));
    };
    checked(hipStreamSynchronize(stream_));
    if (!d_osc_env_state_) {
        OSCEnvState* allocation = nullptr;
        checked(hipMalloc(&allocation, num_envs * sizeof(OSCEnvState)));
        hipError_t status = hipMemset(allocation, 0, num_envs * sizeof(OSCEnvState));
        if (status != hipSuccess) {
            hipFree(allocation);
            checked(status);
        }
        d_osc_env_state_ = allocation;
        osc_num_envs_ = num_envs;
    }
    checked(hipMemcpy(d_osc_env_state_ + env_idx, &state, sizeof(OSCEnvState),
                      hipMemcpyHostToDevice));
}

OSCEnvState PhysicsEngine::read_osc_env_state(int env_idx) const {
    if (!initialized_ || !d_osc_env_state_)
        throw std::runtime_error("OSC state is not allocated");
    if (env_idx < 0 || env_idx >= data_.num_envs() || env_idx >= osc_num_envs_)
        throw std::out_of_range("OSC environment index is out of range");
    OSCEnvState result = {};
    hipError_t status = hipStreamSynchronize(stream_);
    if (status == hipSuccess)
        status = hipMemcpy(&result, d_osc_env_state_ + env_idx,
                           sizeof(OSCEnvState), hipMemcpyDeviceToHost);
    if (status != hipSuccess)
        throw std::runtime_error(std::string("OSC state readback: ") + hipGetErrorString(status));
    return result;
}

void PhysicsEngine::set_osc_actions(const float* actions, int num_envs) {
    if (!d_osc_actions_ || osc_num_envs_ != num_envs) {
        if (d_osc_actions_) hipFree(d_osc_actions_);
        hipMalloc(&d_osc_actions_, num_envs * 7 * sizeof(float));
        osc_num_envs_ = num_envs;
    }
    if (!d_osc_env_state_ || osc_num_envs_ != num_envs) {
        if (d_osc_env_state_) hipFree(d_osc_env_state_);
        hipMalloc(&d_osc_env_state_, num_envs * sizeof(OSCEnvState));
        hipMemset(d_osc_env_state_, 0, num_envs * sizeof(OSCEnvState));
        osc_num_envs_ = num_envs;
    }
    hipMemcpyAsync(d_osc_actions_, actions, num_envs * 7 * sizeof(float),
                   hipMemcpyHostToDevice, stream_);
}

void PhysicsEngine::reset_osc_state() {
    if (d_osc_env_state_ && osc_num_envs_ > 0) {
        hipMemset(d_osc_env_state_, 0, osc_num_envs_ * sizeof(OSCEnvState));
    }
}

void PhysicsEngine::reset_osc_env_state(int env_idx) {
    if (d_osc_env_state_ && env_idx >= 0 && env_idx < osc_num_envs_) {
        OSCEnvState zero = {};
        hipMemcpy(d_osc_env_state_ + env_idx, &zero, sizeof(OSCEnvState),
                  hipMemcpyHostToDevice);
    }
}

void PhysicsEngine::set_osc_gripper_state(const double* state, int num_envs) {
    if (!state || num_envs <= 0) return;
    if (!d_osc_env_state_ || osc_num_envs_ != num_envs) {
        if (d_osc_env_state_) hipFree(d_osc_env_state_);
        hipMalloc(&d_osc_env_state_, num_envs * sizeof(OSCEnvState));
        hipMemset(d_osc_env_state_, 0, num_envs * sizeof(OSCEnvState));
        osc_num_envs_ = num_envs;
    }

    std::vector<OSCEnvState> host(num_envs);
    hipMemcpy(host.data(), d_osc_env_state_,
              num_envs * sizeof(OSCEnvState), hipMemcpyDeviceToHost);
    for (int env = 0; env < num_envs; ++env) {
        for (int grip = 0; grip < OSC_MAX_GRIP; ++grip) {
            host[env].gripper_current_action[grip] =
                state[env * OSC_MAX_GRIP + grip];
        }
    }
    hipMemcpy(d_osc_env_state_, host.data(),
              num_envs * sizeof(OSCEnvState), hipMemcpyHostToDevice);
}

bool PhysicsEngine::read_osc_env_state(OSCEnvState* state, int num_envs) const {
    if (!state || !d_osc_env_state_ || num_envs != osc_num_envs_) return false;
    return hipMemcpy(state, d_osc_env_state_,
                     num_envs * sizeof(OSCEnvState),
                     hipMemcpyDeviceToHost) == hipSuccess;
}

void PhysicsEngine::begin_osc_policy_step() {
    if (!initialized_ || !osc_configured_ || !d_osc_config_ ||
        !d_osc_env_state_ || !d_osc_actions_) {
        return;
    }
    kernels::launch_osc_process_actions(
        model_->device_model(), data_.device_data(), d_osc_config_,
        d_osc_env_state_, d_osc_actions_, stream_);
}

void PhysicsEngine::step_osc_substep(bool warm_start) {
    if (!initialized_ || !osc_configured_ || !d_osc_config_ ||
        !d_osc_env_state_) {
        return;
    }
    record_osc_substep(warm_start);
    forward_kinematics();
    geom_transforms();
    kernels::launch_site_transforms(
        model_->device_model(), data_.device_data(), stream_);
    hipEventRecord(physics_done_event_, stream_);
}

void PhysicsEngine::reset_warmstart() {
    if (!initialized_) return;
    DevicePhysicsData& dd = const_cast<DevicePhysicsData&>(data_.device_data());
    int nv = model_->desc().nv;
    int ne = dd.num_envs;
    hipMemsetAsync(dd.qacc_warmstart, 0, (size_t)ne * nv * sizeof(double), stream_);
    hipMemsetAsync(dd.constraint_lam, 0, (size_t)ne * MAX_CONSTRAINTS * sizeof(double), stream_);

    // The Newton solver's own scratch persists across steps too, and clearing only
    // qacc_warmstart + constraint_lam left it behind. That made a seeded state
    // history-dependent: writing the SAME qpos/qvel after 0/10/50/90 prior steps
    // gave 4 different trajectories, with the first difference at 2.65e-14 -- five
    // orders above one ULP (4.3e-19), so a genuine state leak rather than
    // floating-point non-associativity. `read_qacc` and `read_qfrc_bias` differed
    // immediately after seeding while qpos/qvel matched, which is what localized it.
    const size_t nc_total = (size_t)ne * MAX_CONSTRAINTS;
    if (dd.newton_x)           hipMemsetAsync(dd.newton_x,     0, nc_total * sizeof(double), stream_);
    if (dd.newton_force)       hipMemsetAsync(dd.newton_force, 0, nc_total * sizeof(double), stream_);
    if (dd.newton_state)       hipMemsetAsync(dd.newton_state, 0, nc_total * sizeof(*dd.newton_state), stream_);
    if (dd.newton_jv)          hipMemsetAsync(dd.newton_jv,    0, nc_total * sizeof(double), stream_);
    if (dd.newton_is_elliptic) hipMemsetAsync(dd.newton_is_elliptic, 0, nc_total * sizeof(bool), stream_);

    // qacc and qfrc_bias are recomputed every step, but they are READ BACK by
    // diagnostics between seeding and the first step, where a stale value is
    // indistinguishable from a real one. Zero them so the engine's reported state
    // after a reset reflects only what was seeded.
    if (dd.qacc)       hipMemsetAsync(dd.qacc,       0, (size_t)ne * nv * sizeof(double), stream_);
    if (dd.qfrc_bias)  hipMemsetAsync(dd.qfrc_bias,  0, (size_t)ne * nv * sizeof(double), stream_);
}

void PhysicsEngine::reset_warmstart_env(int env_idx) {
    if (!initialized_ || env_idx < 0 || env_idx >= data_.num_envs()) return;
    DevicePhysicsData& dd = const_cast<DevicePhysicsData&>(data_.device_data());
    int nv = model_->desc().nv;
    hipMemsetAsync(dd.qacc_warmstart + env_idx * nv, 0, nv * sizeof(double), stream_);
    hipMemsetAsync(dd.constraint_lam + env_idx * MAX_CONSTRAINTS, 0,
                   MAX_CONSTRAINTS * sizeof(double), stream_);
}

// Record one full substep (FK through integrate) on stream_ for graph capture or direct use.
void PhysicsEngine::record_osc_substep(bool warm_start, bool integrate_prev,
                                       bool integrate_tail) {
    const DevicePhysicsModel& dm = model_->device_model();
    DevicePhysicsData& dd = const_cast<DevicePhysicsData&>(data_.device_data());

    bool cons_cleared = false, bias_done = false;
    const bool counters_zeroed = osc_fused_head(integrate_prev, options_.enable_contact,
                                                /*clear_constraints=*/warm_start ? 0 : 1,
                                                &cons_cleared,
                                                options_.zero_bias ? nullptr : &bias_done);

    if (bias_done) {
        // qfrc_bias / qfrc_fluid already written by the fused head.
        kernels::launch_gravcomp(dm, dd, stream_);
    } else if (options_.zero_bias) {
        hipMemsetAsync(dd.qfrc_bias, 0,
                       (size_t)dd.num_envs * dm.nv * sizeof(double), stream_);
        kernels::launch_compute_fluid_forces(dm, dd, stream_, model_->desc().level_start);
    } else {
        kernels::launch_bias_and_fluid_forces(dm, dd, stream_,
                                              model_->desc().level_start);
        kernels::launch_gravcomp(dm, dd, stream_);
    }

    kernels::launch_build_mass_matrix(dm, dd, options_, stream_);
    kernels::launch_osc_controller(dm, dd, dd.solver_M,
                                    d_osc_config_, d_osc_env_state_, stream_);

    if (options_.enable_contact)
        collision_detection_on(stream_, /*zero_counters=*/!counters_zeroed);

    kernels::launch_extract_M_diag_inv(dm, dd, stream_);
    if (cons_cleared)
        kernels::launch_constraint_setup(dm, dd, options_, stream_);
    else
        constraint_setup(/*preserve_lambda=*/warm_start);
    kernels::launch_newton_solve(dm, dd, options_, stream_, warm_start);
    if (options_.enable_damping)
        kernels::launch_mhat_solve(dm, dd, options_, stream_);
    if (integrate_tail)
        kernels::launch_integrate(dm, dd, options_, stream_);
}

void PhysicsEngine::invalidate_osc_graphs() {
    if (!osc_graph_captured_) return;
    hipStreamSynchronize(stream_);
    if (osc_exec_cold_)  { hipGraphExecDestroy(osc_exec_cold_);  osc_exec_cold_  = nullptr; }
    if (osc_graph_cold_) { hipGraphDestroy(osc_graph_cold_); osc_graph_cold_ = nullptr; }
    if (osc_exec_warm_)  { hipGraphExecDestroy(osc_exec_warm_);  osc_exec_warm_  = nullptr; }
    if (osc_graph_warm_) { hipGraphDestroy(osc_graph_warm_); osc_graph_warm_ = nullptr; }
    osc_graph_captured_ = false;
}

void PhysicsEngine::capture_osc_step_graphs() {
    if (osc_graph_captured_) return;
    if (!osc_configured_ || !d_osc_config_ || !d_osc_env_state_) return;

    // No prime step -- buffers were warmed by the normal loop that just ran.
    // Stream must be idle (caller syncs before calling this).

    hipGraphNode_t err_node;
    size_t log_size = 0;

    hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal);
    record_osc_substep(false, /*integrate_prev=*/false, /*integrate_tail=*/false);
    hipStreamEndCapture(stream_, &osc_graph_cold_);

    hipError_t err = hipGraphInstantiate(&osc_exec_cold_, osc_graph_cold_,
                                         &err_node, nullptr, log_size);
    if (err != hipSuccess) {
        fprintf(stderr, "instinct_sim: OSC cold graph instantiation failed\n");
        if (osc_graph_cold_) { hipGraphDestroy(osc_graph_cold_); osc_graph_cold_ = nullptr; }
        osc_exec_cold_ = nullptr;
        return;
    }

    hipStreamBeginCapture(stream_, hipStreamCaptureModeGlobal);
    record_osc_substep(true, /*integrate_prev=*/true, /*integrate_tail=*/false);
    hipStreamEndCapture(stream_, &osc_graph_warm_);

    err = hipGraphInstantiate(&osc_exec_warm_, osc_graph_warm_,
                               &err_node, nullptr, log_size);
    if (err != hipSuccess) {
        fprintf(stderr, "instinct_sim: OSC warm graph instantiation failed\n");
        if (osc_graph_warm_) { hipGraphDestroy(osc_graph_warm_); osc_graph_warm_ = nullptr; }
        osc_exec_warm_ = nullptr;
        if (osc_exec_cold_)  { hipGraphExecDestroy(osc_exec_cold_);  osc_exec_cold_  = nullptr; }
        if (osc_graph_cold_) { hipGraphDestroy(osc_graph_cold_); osc_graph_cold_ = nullptr; }
        return;
    }

    osc_graph_captured_ = true;
}

// ── world skip: per-world freeze mask ───────────────────────────────────────────

// Source of the "no mask" pointer value. The async symbol copies read their
// source when they execute, so both possible sources (this constant and the
// d_world_active_ member, fixed after its allocation) must outlive them.
static const unsigned char* const kNoWorldMask = nullptr;

void PhysicsEngine::push_world_mask() {
    const unsigned char* const* src = world_mask_on_ ? &d_world_active_ : &kNoWorldMask;
    kernels::set_world_mask_kinematics(src, stream_);
    kernels::set_world_mask_solver(src, stream_);
    kernels::set_world_mask_collision(src, stream_);
    kernels::set_world_mask_integrate(src, stream_);
}

void PhysicsEngine::set_world_active(const unsigned char* mask, int n) {
    if (!initialized_)
        throw std::runtime_error("set_world_active requires an initialized engine");
    const int ne = data_.num_envs();
    if (n != ne)
        throw std::runtime_error("set_world_active: mask length != num_envs");
    if (!d_world_active_) {
        hipMalloc((void**)&d_world_active_, (size_t)ne);
        hipHostMalloc((void**)&h_world_stage_, (size_t)ne);
    }
    hipStreamSynchronize(stream_);
    for (int i = 0; i < ne; ++i) h_world_stage_[i] = mask[i] ? 1 : 0;
    hipMemcpyAsync(d_world_active_, h_world_stage_, (size_t)ne, hipMemcpyHostToDevice, stream_);
    world_mask_on_ = true;
    push_world_mask();
}

void PhysicsEngine::clear_world_active() {
    if (!world_mask_on_) return;
    world_mask_on_ = false;
    push_world_mask();          // ordered on stream_ after the masked step's kernels
}

void PhysicsEngine::step_with_osc(int n_substeps) {
    if (!initialized_ || !osc_configured_ || !d_osc_config_ || !d_osc_env_state_)
        return;

    const DevicePhysicsModel& dm = model_->device_model();
    DevicePhysicsData& dd = const_cast<DevicePhysicsData&>(data_.device_data());

    kernels::launch_osc_process_actions(dm, dd, d_osc_config_,
                                        d_osc_env_state_, d_osc_actions_, stream_);

    if (osc_graph_captured_ && osc_exec_cold_ && osc_exec_warm_) {
        hipGraphLaunch(osc_exec_cold_, stream_);
        for (int sub = 1; sub < n_substeps; sub++)
            hipGraphLaunch(osc_exec_warm_, stream_);
    } else {
        // First call: run normally, then capture graphs for future calls
        for (int sub = 0; sub < n_substeps; sub++)
            record_osc_substep(sub > 0, /*integrate_prev=*/sub > 0,
                               /*integrate_tail=*/false);

        // Last substep's integrate + final FK for observation readout
        // (before syncing for capture)
        osc_fused_head(/*do_integrate=*/n_substeps > 0, /*zero_contacts=*/false);
        hipEventRecord(physics_done_event_, stream_);

        hipStreamSynchronize(stream_);
        capture_osc_step_graphs();
        return;
    }

    // Last substep's integrate + final FK + sites for observation readout
    // (the cold graph always runs, so there is always a pending integrate).
    osc_fused_head(/*do_integrate=*/true, /*zero_contacts=*/false);

    hipEventRecord(physics_done_event_, stream_);
}

} // namespace physics
} // namespace hipgl


namespace hipgl { namespace physics {

// ── GPU joint-position PD controller (joint_pd.h) ──────────────────────────────
void PhysicsEngine::set_joint_pd(int n, const int* act, const int* dof, const int* qadr, const int* grp_start,
                                 const int* grp_len, const double* kp, const double* kd, const double* gear,
                                 const double* lo, const double* hi) {
    if (n < 0 || n > 64) throw std::runtime_error("set_joint_pd: 0 <= n <= 64");
    if (graph_captured_) throw std::runtime_error("set_joint_pd: configure before the first step_n");
    auto up_i = [&](int*& d, const int* h) { if (d) hipFree(d); d = nullptr;
        if (n) { hipMalloc(&d, n * sizeof(int)); hipMemcpy(d, h, n * sizeof(int), hipMemcpyHostToDevice); } };
    auto up_d = [&](double*& d, const double* h) { if (d) hipFree(d); d = nullptr;
        if (n) { hipMalloc(&d, n * sizeof(double)); hipMemcpy(d, h, n * sizeof(double), hipMemcpyHostToDevice); } };
    up_i(jpd_.act, act); up_i(jpd_.dof, dof); up_i(jpd_.qadr, qadr); up_i(jpd_.grp_start, grp_start);
    up_i(jpd_.grp_len, grp_len);
    up_d(jpd_.kp, kp); up_d(jpd_.kd, kd); up_d(jpd_.gear, gear); up_d(jpd_.lo, lo); up_d(jpd_.hi, hi);
    if (jpd_.target) hipFree(jpd_.target);
    jpd_.target = nullptr;
    const int ne = data_.device_data().num_envs;
    if (n) { hipMalloc(&jpd_.target, (size_t)ne * n * sizeof(double)); hipMemset(jpd_.target, 0, (size_t)ne * n * sizeof(double)); }
    jpd_.n = n;
}

void PhysicsEngine::set_joint_pd_targets(const double* targets) {
    if (jpd_.n <= 0) return;
    const int ne = data_.device_data().num_envs;
    hipMemcpyAsync(jpd_.target, targets, (size_t)ne * jpd_.n * sizeof(double), hipMemcpyHostToDevice, stream_);
}

}}  // namespace hipgl::physics
