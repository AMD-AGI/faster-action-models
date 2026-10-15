#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "hipgl/physics/joint_pd.h"

#include "physics_types.h"
#include "physics_model.h"
#include "physics_data.h"
#include "physics_osc.h"

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

namespace hipgl {
namespace physics {

// Orchestrates the full physics step pipeline on GPU.
//
// Pipeline (Newton solver path):
//   1. Forward kinematics  (qpos -> body world transforms) [fused single launch]
//   2. Geom transform      (body transforms + geom offsets -> geom world poses)
//   3. Collision detection  (broadphase AABB + narrowphase primitives)
//   4. Constraint setup     (contacts + limits + equality -> Jacobian rows) [parallel]
//   5. RNEA bias force      (level-by-level parallel forward + backward)
//   6. Mass matrix build    (parallel, 1 thread per entry)
//   7. Newton H build       (M + J^T D J, parallel)
//   8. Newton solve         (Cholesky factor + solve, 1 thread per env)
//   9. Integration          (qacc -> qvel -> qpos)
//
// Pipeline (PGS fallback):
//   1-4. Same as above
//   5. RNEA bias force
//   6. Apply forces (actuators + constraint lambda -> net force per DOF)
//   7. Mass matrix solve (CRBA + LU, 1 thread per env)
//   8. Integration
class PhysicsEngine {
public:
    PhysicsEngine();
    ~PhysicsEngine();

    bool init(PhysicsModel* model, int num_envs, int gpu_id = 0);

    // Declare that this engine only drives rendering (forward kinematics, geom
    // and site transforms) from externally-supplied qpos/qvel, and never
    // integrates dynamics. Must be called BEFORE init(). This downgrades the
    // nv > KERNEL_MAXNV check from a hard failure to a warning, since the dense
    // solver is not on the render path. Calling step()/step_with_osc() on a
    // render-only engine whose nv exceeds KERNEL_MAXNV yields no constraint
    // forces — the exact silent failure the guard exists to prevent.
    void set_render_only(bool v) { render_only_ = v; }
    bool render_only() const { return render_only_; }

    void step();
    void step_n(int n);
    void forward();

    // GPU joint-position PD controller run inside every substep (joint_pd.h). Configure before the first
    // step_n (graph capture); targets are [num_envs * n] goals, updated in place each policy step.
    void set_joint_pd(int n, const int* act, const int* dof, const int* qadr, const int* grp_start,
                      const int* grp_len, const double* kp, const double* kd, const double* gear,
                      const double* lo, const double* hi);
    void set_joint_pd_targets(const double* targets);
    int joint_pd_n() const { return jpd_.n; }

    // GPU-side OSC: runs n_substeps of FK→bias→mass→OSC→collision→Newton→integrate
    // with zero host synchronisation.  Call set_osc_config() once, then
    // set_osc_actions() each policy step before step_with_osc().
    void step_with_osc(int n_substeps);
    // world skip: freeze worlds for the following step_with_osc() calls. mask has
    // num_envs bytes, nonzero = step. A frozen world keeps its state exactly
    // (see world_mask.h). clear_world_active() restores the unmasked kernels;
    // the mask is process-global, so clear it before stepping another engine.
    void set_world_active(const unsigned char* mask, int n);
    void clear_world_active();
    bool world_mask_on() const { return world_mask_on_; }
    void set_osc_config(const OSCConfig& config);
    void set_osc_actions(const float* actions, int num_envs);
    void reset_osc_state();
    void reset_osc_env_state(int env_idx);
    // Import one controller's reset state without advancing any environment.
    // Optional cache fields are consumed by the first goal/tick independently;
    // gravity compensation always comes from the current native simulation.
    void initialize_osc_env_state(int env_idx, const OSCEnvState& state);
    OSCEnvState read_osc_env_state(int env_idx) const;
    const OSCConfig& osc_config() const;
    void set_osc_gripper_state(const double* state, int num_envs);
    bool read_osc_env_state(OSCEnvState* state, int num_envs) const;

    // Diagnostic split-phase OSC stepping. These execute the same kernels as
    // step_with_osc(), but expose each physics substep for parity tracing.
    void begin_osc_policy_step();
    void step_osc_substep(bool warm_start);

    // Reset solver warmstart state (qacc_warmstart + constraint_lam).
    // Call after externally writing qpos/qvel to avoid stale solver state.
    void reset_warmstart();
    void reset_warmstart_env(int env_idx);

    // Invalidate cached HIP graphs so they are re-captured on the next
    // step_with_osc() call.  Must be called after changing solver options
    // (tolerance, iterations) for changes to take effect.
    void invalidate_osc_graphs();

    SolverOptions& options() { return options_; }
    const SolverOptions& options() const { return options_; }

    PhysicsData& data() { return data_; }
    const PhysicsData& data() const { return data_; }

    PhysicsModel* model() { return model_; }

    void sync();

    void sync_fk();
    void sync_geom();

    void forward_kinematics();
    void geom_transforms();
    void site_transforms();
    void step_and_sites();

    // Kinematics of the state stepped back by dt_back along qvel (MuJoCo mj_integratePos with
    // -dt_back), written to separate buffers; the live state is only read.  robosuite samples
    // eef pose / gripper qpos 2 substeps before the end of a policy step (HP_OBS_LAG).
    // Asynchronous on the physics stream; the read_lagged_* calls synchronize it.
    void lagged_kinematics(double dt_back);
    // Camera timing: robosuite renders its cameras from the kinematics of the state dt_back before the
    // final one. begin_lagged_render() points the LIVE pose buffers (body, geom, site) at that state for the
    // renderer; end_lagged_render() recomputes them from the true qpos (bit-identical to before).
    void begin_lagged_render(double dt_back);
    void end_lagged_render();
    void read_lagged_qpos(double* out) const;       // [num_envs * nq]
    void read_lagged_xpos(double* out) const;       // [num_envs * nbody * 3]
    void read_lagged_xquat(double* out) const;      // [num_envs * nbody * 4]
    void read_lagged_site_xpos(float* out) const;   // [num_envs * nsite * 3]

    struct PhaseTimings {
        double fk_ms, geom_ms, collision_ms, constraint_ms, solve_ms, integrate_ms;
    };

    PhaseTimings step_profile();

#ifdef __HIPCC__
    hipStream_t stream() const { return stream_; }
    hipStream_t render_stream() const { return render_stream_; }
    hipEvent_t  physics_done_event() const { return physics_done_event_; }
#endif

private:
    DeviceJointPd  jpd_;
    PhysicsModel*  model_;
    PhysicsData    data_;
    SolverOptions  options_;
    int            gpu_id_;
    bool           initialized_;
    bool           render_only_ = false;

#ifdef __HIPCC__
    hipStream_t    stream_;
    hipStream_t    render_stream_;
    hipEvent_t     physics_done_event_;
    hipGraph_t     step_graph_ = nullptr;
    hipGraphExec_t step_graph_exec_ = nullptr;
    bool           graph_captured_ = false;
    hipGraph_t     step_graph_k_ = nullptr;
    hipGraphExec_t step_graph_exec_k_ = nullptr;
    int            step_unroll_ = 1;

    hipGraph_t     osc_graph_cold_ = nullptr;
    hipGraphExec_t osc_exec_cold_  = nullptr;
    hipGraph_t     osc_graph_warm_ = nullptr;
    hipGraphExec_t osc_exec_warm_  = nullptr;
    bool           osc_graph_captured_ = false;
#else
    void*          stream_;
    void*          render_stream_;
    void*          physics_done_event_;
    void*          step_graph_ = nullptr;
    void*          step_graph_exec_ = nullptr;
    bool           graph_captured_ = false;
    void*          step_graph_k_ = nullptr;
    void*          step_graph_exec_k_ = nullptr;
    int            step_unroll_ = 1;

    void*          osc_graph_cold_ = nullptr;
    void*          osc_exec_cold_  = nullptr;
    void*          osc_graph_warm_ = nullptr;
    void*          osc_exec_warm_  = nullptr;
    bool           osc_graph_captured_ = false;
#endif

    void collision_detection();
    void constraint_setup(bool preserve_lambda = false);
    void constraint_solve();
    void integration();

    void capture_step_graph();
    void capture_osc_step_graphs();
    void record_osc_substep(bool warm_start, bool integrate_prev = false,
                            bool integrate_tail = true);
    // [integrate] + [zero contact counters] + FK + geom + site; returns true
    // if the contact counters were zeroed (i.e. collision memsets not needed).
    // clear_constraints: -1 = leave constraint buffers alone, 0/1 = also do
    // kernel_clear_constraint_buffers' work (1 = including lambda) in the
    // head; *cons_cleared reports whether that happened.
    bool osc_fused_head(bool do_integrate, bool zero_contacts,
                        int clear_constraints = -1, bool* cons_cleared = nullptr,
                        bool* bias_done = nullptr);
    // Collision detection on stream `s` (a hipStream_t); counter memsets only
    // when zero_counters (the fused head may already have zeroed them).
    void collision_detection_on(void* s, bool zero_counters);
    // Static broadphase pair mask (built once, outside graph capture);
    // nullptr -> unmasked broadphase.
    const unsigned char* broadphase_pair_mask(void* s);

    // lagged_kinematics() buffers (allocated on first use)
    double* lag_qpos_d_  = nullptr;
    double* lag_xpos_d_  = nullptr;
    double* lag_xquat_d_ = nullptr;
    float*  lag_xpos_    = nullptr;
    float*  lag_xquat_   = nullptr;
    float*  lag_site_xpos_  = nullptr;
    float*  lag_site_xquat_ = nullptr;
    float*  lag_site_xmat_  = nullptr;

    // OSC device state
    OSCConfig*   d_osc_config_    = nullptr;
    OSCEnvState* d_osc_env_state_ = nullptr;
    float*       d_osc_actions_   = nullptr;
    int          osc_num_envs_    = 0;
    bool         osc_configured_  = false;
    OSCConfig    osc_config_host_ = {};

    unsigned char* d_bp_pair_mask_     = nullptr;
    const void*    bp_pair_mask_key_   = nullptr;  // model geom_body the mask was built for
    int            bp_pair_mask_ng_    = 0;
    int*           d_bp_chunks_        = nullptr;  // non-empty chunk bases (compacted broadphase)
    int            bp_nchunk_          = 0;

    // world skip: per-world freeze mask (world_mask.h).
    unsigned char* d_world_active_     = nullptr;  // [num_envs] device, 1 = step
    unsigned char* h_world_stage_      = nullptr;  // pinned host staging for it
    bool           world_mask_on_      = false;
    void           push_world_mask();
};

} // namespace physics
} // namespace hipgl
