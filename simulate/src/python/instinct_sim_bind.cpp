// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// GPU Python bindings via pybind11.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <hip/hip_runtime.h>
#include <memory>
#include <climits>
#include <cstdlib>
#include <vector>
#include <mutex>
#include <unordered_map>
#include <thread>
#include <condition_variable>
#include <functional>
#include <exception>
#include "hipgl/physics/physics_model.h"
#include "hipgl/physics/physics_data.h"
#include "hipgl/physics/physics_engine.h"
#include "hipgl/physics/physics_osc.h"
#include "hipgl/physics/physics_renderer.h"
#include "hipgl/pipeline.h"
#include "hipgl/framebuffer.h"
#include "hipgl/math_types.h"
#include "hipgl/kernels.h"
#include "hipgl/planar_reflection.h"
#include "hipgl/intrinsic_transparency.h"
#include "hipgl/physics/physics_mjccd_worker.h"
#include "hipgl/physics/ray_tracer.h"

namespace {
struct HpHwQueuePlant {
    HpHwQueuePlant() {
        const char* hp = getenv("HP_HWQ");
        int n = hp ? atoi(hp) : 8;
        if (n <= 0) return;                       // HP_HWQ=0: leave ROCm default
        if (getenv("GPU_MAX_HW_QUEUES")) return;  // explicit external setting wins
        char buf[16];
        snprintf(buf, sizeof(buf), "%d", n);
        setenv("GPU_MAX_HW_QUEUES", buf, 0);
    }
};
static HpHwQueuePlant g_hp_hw_queue_plant;

// Resolution-aware worker (stream) count. With the HW queues unblocked, MORE
// concurrent worlds is a pure win at low resolution, where the per-world grids
// cannot fill the machine on their own, and a LOSS at r1024, where the same
// fine rasterizer is already instruction-issue-bound (VALUBusy 45.8 + SALUBusy
// 35.8 ~= 82%) and extra concurrent work only adds contention. Gate by the
// rendered (super-sampled) width, which is what actually sets the grid size.
// gfx950 (MI355X) has 256 CUs -> 8 x 32-bit mask words.
#define HP_CU_WORDS 8
inline int hp_w_lo() { static const int v = []{ const char* e = getenv("HP_W_LO"); return e ? atoi(e) : 4; }(); return v; }
inline int hp_w_cap() { static const int v = []{ const char* e = getenv("HP_W_CAP"); return e ? atoi(e) : 16; }(); return v; }
inline int hp_w_hi() { static const int v = []{ const char* e = getenv("HP_W_HI"); return e ? atoi(e) : 0;  }(); return v; }
inline bool hp_lowres(int render_width) {
    static const int th = []{ const char* e = getenv("HP_W_TH"); return e ? atoi(e) : 4096; }();
    return render_width > 0 && render_width <= th;
}
inline int hp_host_trace() {
    static const int v = getenv("HP_HOST_TRACE") ? atoi(getenv("HP_HOST_TRACE")) : 0;
    return v;
}
inline double hp_now_ms() {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}


static inline void hp_scrub_hole(void* base, size_t a, size_t b) {
    if (b > a) std::memset(static_cast<char*>(base) + a, 0, b - a);
}
static inline void hp_scrub_padding(hipgl::RenderState& s) {
    hp_scrub_hole(&s, offsetof(hipgl::RenderState, shadow_bias) + sizeof(float),
                      offsetof(hipgl::RenderState, tri_stats));
    hp_scrub_hole(&s, offsetof(hipgl::RenderState, reflect_rect)
                        + sizeof(hipgl::RenderState::reflect_rect),
                      sizeof(hipgl::RenderState));
}
static inline void hp_scrub_padding(hipgl::PlanarReflectionPlan& p) {
    hp_scrub_hole(&p, offsetof(hipgl::PlanarReflectionPlan, render_scene) + sizeof(bool),
                      offsetof(hipgl::PlanarReflectionPlan, reflected_state));
    hp_scrub_padding(p.reflected_state);
    hp_scrub_hole(&p, offsetof(hipgl::PlanarReflectionPlan, matrices)
                        + sizeof(hipgl::PlanarReflectionPlan::matrices),
                      sizeof(hipgl::PlanarReflectionPlan));
}

inline int hp_par_issue() {
    static const int v = getenv("HP_PAR_ISSUE") ? atoi(getenv("HP_PAR_ISSUE")) : 1;
    return v;
}
inline int hp_par_issue_min_envs() {
    static const int v = getenv("HP_PAR_MIN_ENVS") ? atoi(getenv("HP_PAR_MIN_ENVS")) : 4;
    return v;
}
inline int hp_solo_stream() {
    static const int v = getenv("HP_SOLO_STREAM") ? atoi(getenv("HP_SOLO_STREAM")) : 1;
    return v;
}
inline int hp_render_graph() {
    static const int v = getenv("HP_RENDER_GRAPH") ? atoi(getenv("HP_RENDER_GRAPH")) : 1;
    return v;
}
inline long long hp_graph_max_tiles() {
    static const long long v = getenv("HP_GRAPH_MAX_TILES")
        ? atoll(getenv("HP_GRAPH_MAX_TILES")) : (1LL << 40);
    return v;
}
inline int hp_par_issue_max_tiles() {
    static const int v = getenv("HP_PAR_MAX_TILES") ? atoi(getenv("HP_PAR_MAX_TILES")) : 1024;
    return v;
}

inline bool hp_share_attribs() {
    static const int v = getenv("HP_SHARE_ATTRIB") ? atoi(getenv("HP_SHARE_ATTRIB")) : 1;
    return v != 0;
}

inline bool hp_cumask_enabled(int render_width) {
    static const bool on = []{ const char* e = getenv("HP_CUMASK"); return e ? atoi(e) != 0 : false; }();
    return on && hp_lowres(render_width);
}

constexpr int HP_MAT_STRIDE = 82;
inline int hp_desired_workers(int render_width, int fallback) {
    const bool low = hp_lowres(render_width);
    int n = low ? hp_w_lo() : hp_w_hi();
    if (n <= 0) n = fallback;
    const int cap = hp_w_cap();
    return n < 1 ? 1 : (n > cap ? cap : n);
}
}  // namespace

#ifndef HP_DEFER_CONVERT
#define HP_DEFER_CONVERT 1
#endif
namespace hipgl { namespace kernels {
bool launch_rgba_f32_to_rgb_u8_ss_bg(
    const float* src_rgba, const float* src_depth, uint8_t* dst_rgb,
    int out_width, int out_height, int ss, bool flip_v, bool flip_h,
    float bg_r, float bg_g, float bg_b, float clear_depth, hipStream_t stream);
}}

// Forward declaration of Gouraud shading kernel launcher
namespace hipgl { namespace physics { namespace kernels {
struct GouraudLight {
    float position[4];
    float ambient[4];
    float diffuse[4];
    float specular[4];
    float spot_direction[3];
    float spot_cutoff;
    float spot_exponent;
    float constant_atten;
    float linear_atten;
    float quadratic_atten;
    int   enabled;
};
static constexpr int GOURAUD_MAX_LIGHTS = 8;
void launch_gouraud_shade(
    const float* d_positions, const float* d_normals,
    float* d_colors, float* d_spec_colors,
    const float* d_specular, const float* d_shininess,
    int nverts, const float* d_view_matrix, const float* d_normal_matrix,
    const GouraudLight* h_lights, int nactive_lights, float global_ambient,
    hipStream_t stream);
}}}

namespace py = pybind11;
using namespace hipgl;
using namespace hipgl::physics;

// Reset import is deliberately separate from action upload. Parsing only
// changes this local host record; no slot is mutated until every field passes.
static void copy_osc_reset_array(const py::dict& source, const char* key,
                                double* target, int rows, int cols = 0,
                                int target_stride = 0) {
    if (!source.contains(key))
        throw py::value_error(std::string("Missing OSC reset field: ") + key);
    auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(source[key]);
    if (!array)
        throw py::value_error(std::string("OSC reset field must be numeric: ") + key);
    auto buffer = array.request();
    if (buffer.ndim != (cols ? 2 : 1) || buffer.shape[0] != rows ||
        (cols && buffer.shape[1] != cols))
        throw py::value_error(std::string("Incorrect shape for OSC reset field: ") + key);
    const int width = cols ? cols : 1;
    if (!target_stride) target_stride = width;
    const double* values = static_cast<const double*>(buffer.ptr);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < width; ++c) {
            const double value = values[r * width + c];
            if (!std::isfinite(value))
                throw py::value_error(std::string("Nonfinite OSC reset field: ") + key);
            target[r * target_stride + c] = value;
        }
}

static py::array_t<double> osc_state_array(const double* source, int rows,
                                          int cols = 0, int source_stride = 0) {
    std::vector<py::ssize_t> shape = {rows};
    if (cols) shape.push_back(cols);
    py::array_t<double> result(shape);
    const int width = cols ? cols : 1;
    if (!source_stride) source_stride = width;
    double* values = result.mutable_data();
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < width; ++c)
            values[r * width + c] = source[r * source_stride + c];
    return result;
}

// Wraps a GPU uint8 buffer and exposes __cuda_array_interface__ for zero-copy
// PyTorch interop. The buffer is owned by CameraRenderer and stays valid until
// the next render_gpu() call or CameraRenderer destruction.
struct GpuImage {
    uint8_t* ptr;
    int height, width;

    py::dict cuda_array_interface() const {
        py::dict d;
        d["shape"] = py::make_tuple(height, width, 3);
        d["typestr"] = "|u1";
        d["data"] = py::make_tuple(reinterpret_cast<intptr_t>(ptr), false);
        d["version"] = 3;
        d["strides"] = py::none();
        return d;
    }
};

struct HpPinnedPool {
    std::mutex m;
    std::unordered_map<size_t, std::vector<void*>> free_blocks;
    std::unordered_map<void*, size_t> live;   // handed-out block -> bytes
    size_t live_bytes = 0;
};
static HpPinnedPool& hp_pin_pool() {
    // Deliberately leaked: array capsules may be released during interpreter
    // shutdown, after static destructors would have torn a plain static down.
    static HpPinnedPool* p = new HpPinnedPool();
    return *p;
}
static inline bool hp_pin_obs_enabled() {
    static const bool v = !(getenv("HP_PIN_OBS") && atoi(getenv("HP_PIN_OBS")) == 0);
    return v;
}
static inline size_t hp_pin_free_max() {
    static const size_t v = getenv("HP_PIN_FREE_MAX") ? (size_t)atoi(getenv("HP_PIN_FREE_MAX")) : 4;
    return v;
}
static inline size_t hp_pin_max_bytes() {
    static const size_t v = (size_t)(getenv("HP_PIN_MAX_MB") ? atoi(getenv("HP_PIN_MAX_MB")) : 1024)
                            * (size_t)1 << 20;
    return v;
}
// Returns nullptr when the pool is disabled/over its cap/allocation failed.
static void* hp_pin_acquire(size_t bytes) {
    if (!hp_pin_obs_enabled() || bytes == 0) return nullptr;
    HpPinnedPool& P = hp_pin_pool();
    std::lock_guard<std::mutex> lk(P.m);
    if (P.live_bytes + bytes > hp_pin_max_bytes()) return nullptr;
    void* blk = nullptr;
    auto it = P.free_blocks.find(bytes);
    if (it != P.free_blocks.end() && !it->second.empty()) {
        blk = it->second.back();
        it->second.pop_back();
    } else if (hipHostMalloc(&blk, bytes, hipHostMallocDefault) != hipSuccess) {
        (void)hipGetLastError();
        return nullptr;
    }
    P.live[blk] = bytes;
    P.live_bytes += bytes;
    return blk;
}
static void hp_pin_release(void* blk) {
    HpPinnedPool& P = hp_pin_pool();
    std::lock_guard<std::mutex> lk(P.m);
    auto it = P.live.find(blk);
    if (it == P.live.end()) return;
    const size_t bytes = it->second;
    P.live.erase(it);
    P.live_bytes -= bytes;
    auto& fl = P.free_blocks[bytes];
    if (fl.size() < hp_pin_free_max()) fl.push_back(blk);
    else hipHostFree(blk);
}

// Batched variant: contiguous (num_envs, H, W, 3) GPU buffer
struct GpuImageBatch {
    uint8_t* ptr;
    int num_envs, height, width;
    // Optional shared ownership of the device buffer (RayTracer outputs): keeps
    // the allocation alive while this view exists. Null for raster batches.
    std::shared_ptr<void> keep;

    py::dict cuda_array_interface() const {
        py::dict d;
        d["shape"] = py::make_tuple(num_envs, height, width, 3);
        d["typestr"] = "|u1";
        d["data"] = py::make_tuple(reinterpret_cast<intptr_t>(ptr), false);
        d["version"] = 3;
        d["strides"] = py::none();
        return d;
    }
};

// Helper: build RenderState from camera parameters and scene lights
static RenderState build_camera_state(
    const float* cam_pos, const float* cam_quat, float fovy,
    int width, int height,
    const float* body_xpos, const float* body_xquat, int body_idx,
    const PhysicsModelDesc& desc,
    const float* bg_color = nullptr,
    bool gradient_bg = false,
    const float* gradient_top = nullptr,
    const float* gradient_bot = nullptr,
    float override_znear = -1.0f,
    float override_zfar = -1.0f)
{
    RenderState state;
    memset(&state, 0, sizeof(state));
    state.depth_test_enabled = true;
    state.depth_write_enabled = true;
    {
        const char* _bc = getenv("HIPGL_BACKFACE_CULL");
        state.backface_cull_enabled = (_bc == nullptr) || (_bc[0] == '1');
        const char* _ts = getenv("HIPGL_TWO_SIDED");
        state.two_sided_lighting = (_ts && _ts[0] == '1');
        const char* _ta = getenv("HIPGL_MIN_TRI_AREA");
        if (_ta && _ta[0]) {
            float v = strtof(_ta, nullptr);
            if (v >= 0.0f) hipgl::kernels::set_min_tri_area(v);
        }
    }
    state.lighting_enabled = true;
    state.color_material_enabled = true;
    state.shade_model = 1;
    state.depth_func = DEPTH_LEQUAL;
    state.gouraud_shading = 1;  // Default: match MuJoCo's per-vertex Gouraud shading
    if (bg_color) {
        state.clear_color[0] = bg_color[0]; state.clear_color[1] = bg_color[1];
        state.clear_color[2] = bg_color[2]; state.clear_color[3] = bg_color[3];
    } else {
        state.clear_color[0] = 0.0f; state.clear_color[1] = 0.0f;
        state.clear_color[2] = 0.0f; state.clear_color[3] = 1.0f;
    }
    state.clear_depth = 1.0f;
    if (gradient_bg) {
        state.gradient_bg = 1;
        state.gradient_top[0] = gradient_top[0];
        state.gradient_top[1] = gradient_top[1];
        state.gradient_top[2] = gradient_top[2];
        state.gradient_bot[0] = gradient_bot[0];
        state.gradient_bot[1] = gradient_bot[1];
        state.gradient_bot[2] = gradient_bot[2];
    }

    state.viewport_x = 0; state.viewport_y = 0;
    state.viewport_w = width; state.viewport_h = height;

    // Body world transform
    float bx = body_xpos[body_idx*3], by = body_xpos[body_idx*3+1], bz = body_xpos[body_idx*3+2];
    float bqw = body_xquat[body_idx*4], bqx = body_xquat[body_idx*4+1];
    float bqy = body_xquat[body_idx*4+2], bqz = body_xquat[body_idx*4+3];

    // Camera offset in body frame -> world frame
    float lx = cam_pos[0], ly = cam_pos[1], lz = cam_pos[2];
    float tw_x = 2.0f*(bqy*lz - bqz*ly), tw_y = 2.0f*(bqz*lx - bqx*lz), tw_z = 2.0f*(bqx*ly - bqy*lx);
    float wx = lx + bqw*tw_x + (bqy*tw_z - bqz*tw_y) + bx;
    float wy = ly + bqw*tw_y + (bqz*tw_x - bqx*tw_z) + by;
    float wz = lz + bqw*tw_z + (bqx*tw_y - bqy*tw_x) + bz;

    state.camera_world_pos[0] = wx;
    state.camera_world_pos[1] = wy;
    state.camera_world_pos[2] = wz;
    // camera_forward is set after orientation is computed below

    // Camera orientation: body_quat * cam_quat
    float cqw = cam_quat[0], cqx = cam_quat[1], cqy = cam_quat[2], cqz = cam_quat[3];
    float rw = bqw*cqw - bqx*cqx - bqy*cqy - bqz*cqz;
    float rx = bqw*cqx + bqx*cqw + bqy*cqz - bqz*cqy;
    float ry = bqw*cqy - bqx*cqz + bqy*cqw + bqz*cqx;
    float rz = bqw*cqz + bqx*cqy - bqy*cqx + bqz*cqw;

    // Camera looks along -Z in MuJoCo convention
    float fwd[3] = {0, 0, -1};
    float t_x = 2.0f*(ry*fwd[2] - rz*fwd[1]), t_y = 2.0f*(rz*fwd[0] - rx*fwd[2]), t_z = 2.0f*(rx*fwd[1] - ry*fwd[0]);
    float fx = fwd[0] + rw*t_x + (ry*t_z - rz*t_y);
    float fy = fwd[1] + rw*t_y + (rz*t_x - rx*t_z);
    float fz = fwd[2] + rw*t_z + (rx*t_y - ry*t_x);
    state.camera_forward[0] = fx;
    state.camera_forward[1] = fy;
    state.camera_forward[2] = fz;

    // Up = rotate_quat(0,1,0) by (rw,rx,ry,rz)
    float up[3] = {0, 1, 0};
    t_x = 2.0f*(ry*up[2] - rz*up[1]); t_y = 2.0f*(rz*up[0] - rx*up[2]); t_z = 2.0f*(rx*up[1] - ry*up[0]);
    float ux = up[0] + rw*t_x + (ry*t_z - rz*t_y);
    float uy = up[1] + rw*t_y + (rz*t_x - rx*t_z);
    float uz = up[2] + rw*t_z + (rx*t_y - ry*t_x);

    math::Vec3 eye(wx, wy, wz);
    math::Vec3 center(wx + fx, wy + fy, wz + fz);
    math::Vec3 up_vec(ux, uy, uz);

    math::Mat4 view = math::look_at(eye, center, up_vec);
    float aspect = (float)width / (float)height;
    float extent = desc.stat_extent;
    float znear = (override_znear > 0.0f) ? override_znear : (desc.map_znear * extent);
    float zfar  = (override_zfar  > 0.0f) ? override_zfar  : (desc.map_zfar  * extent);
    math::Mat4 proj = math::perspective(fovy, aspect, znear, zfar);
    math::Mat4 mvp = proj * view;

    memcpy(state.model_view, &view.m[0][0], 16 * sizeof(float));
    memcpy(state.projection, &proj.m[0][0], 16 * sizeof(float));
    memcpy(state.mvp, &mvp.m[0][0], 16 * sizeof(float));

    view.upper3x3(state.normal_matrix);

    // Per-fragment lighting is now done in the rasterizer.
    state.lighting_enabled = true;

    // Headlight: directional along camera forward in eye space.
    // MuJoCo empirically uses a directional light — all floor pixels get
    // identical NdotL regardless of distance (confirmed by uniform brightness test).
    // In eye space, the camera forward direction is (0,0,-1), but MuJoCo's
    // convention is that the headlight direction points toward the viewer,
    // i.e. (0,0,1) with w=0 for directional.
    {
        Light& L = state.lights[0];
        L.enabled = desc.headlight_active;
        L.position[0] = 0; L.position[1] = 0;
        L.position[2] = 1; L.position[3] = 0.0f;
        for (int k = 0; k < 3; k++) L.ambient[k] = desc.headlight_ambient[k];
        L.ambient[3] = 1.0f;
        for (int k = 0; k < 3; k++) L.diffuse[k] = desc.headlight_diffuse[k];
        L.diffuse[3] = 1.0f;
        for (int k = 0; k < 3; k++) L.specular[k] = desc.headlight_specular[k];
        L.specular[3] = 1.0f;
        L.spot_cutoff = 180.0f;
        L.constant_atten = 1.0f;
    }

    int nli = 1;
    for (int i = 0; i < desc.nscene_light && nli < MAX_LIGHTS; i++) {
        const SceneLightDesc& sl = desc.scene_lights[i];
        Light& L = state.lights[nli++];
        L.enabled = 1;
        if (sl.directional) {
            float neg_dir[3] = {-sl.dir[0], -sl.dir[1], -sl.dir[2]};
            L.position[0] = state.model_view[0]*neg_dir[0] + state.model_view[4]*neg_dir[1] + state.model_view[8]*neg_dir[2];
            L.position[1] = state.model_view[1]*neg_dir[0] + state.model_view[5]*neg_dir[1] + state.model_view[9]*neg_dir[2];
            L.position[2] = state.model_view[2]*neg_dir[0] + state.model_view[6]*neg_dir[1] + state.model_view[10]*neg_dir[2];
            L.position[3] = 0.0f;
            L.spot_cutoff = 180.0f;
        } else {
            L.position[0] = state.model_view[0]*sl.pos[0] + state.model_view[4]*sl.pos[1] + state.model_view[8]*sl.pos[2] + state.model_view[12];
            L.position[1] = state.model_view[1]*sl.pos[0] + state.model_view[5]*sl.pos[1] + state.model_view[9]*sl.pos[2] + state.model_view[13];
            L.position[2] = state.model_view[2]*sl.pos[0] + state.model_view[6]*sl.pos[1] + state.model_view[10]*sl.pos[2] + state.model_view[14];
            L.position[3] = 1.0f;
            L.direction[0] = state.model_view[0]*sl.dir[0] + state.model_view[4]*sl.dir[1] + state.model_view[8]*sl.dir[2];
            L.direction[1] = state.model_view[1]*sl.dir[0] + state.model_view[5]*sl.dir[1] + state.model_view[9]*sl.dir[2];
            L.direction[2] = state.model_view[2]*sl.dir[0] + state.model_view[6]*sl.dir[1] + state.model_view[10]*sl.dir[2];
            L.spot_cutoff = sl.cutoff;
            L.spot_exponent = sl.exponent;
        }
        for (int k = 0; k < 3; k++) L.ambient[k] = sl.ambient[k];
        for (int k = 0; k < 3; k++) L.diffuse[k] = sl.diffuse[k];
        for (int k = 0; k < 3; k++) L.specular[k] = sl.specular[k];
        L.constant_atten = sl.attenuation[0];
        L.linear_atten = sl.attenuation[1];
        L.quadratic_atten = sl.attenuation[2];
    }

    return state;
}

struct FSink {
    float* p = nullptr;
    size_t n = 0;
    size_t cap = 0;          // capacity in FLOATS
    bool over = false;
    inline void put3(float a, float b, float c) {
        if (n + 3 > cap) { over = true; n += 3; return; }
        if (p) { p[n] = a; p[n + 1] = b; p[n + 2] = c; }
        n += 3;
    }
    inline void put4(float a, float b, float c, float d) {
        if (n + 4 > cap) { over = true; n += 4; return; }
        if (p) { p[n] = a; p[n + 1] = b; p[n + 2] = c; p[n + 3] = d; }
        n += 4;
    }
    inline size_t size() const { return n; }
    inline float* data() const { return p; }
    inline bool empty() const { return n == 0; }
    inline void reset(float* base, size_t capacity) {
        p = base; n = 0; cap = capacity; over = false;
    }
};

static inline void push_vertex(
    FSink& positions,
    FSink& normals,
    FSink& colors,
    const float* p,
    const float* rgba)
{
    positions.put3(p[0], p[1], p[2]);
    normals.put3(0.0f, 0.0f, 1.0f);
    colors.put4(rgba[0], rgba[1], rgba[2], rgba[3]);
}

static inline void push_triangle(
    FSink& positions,
    FSink& normals,
    FSink& colors,
    const float* a, const float* b, const float* c,
    const float* rgba)
{
    push_vertex(positions, normals, colors, a, rgba);
    push_vertex(positions, normals, colors, b, rgba);
    push_vertex(positions, normals, colors, c, rgba);
}

static inline void push_quad(
    FSink& positions,
    FSink& normals,
    FSink& colors,
    const float* a, const float* b, const float* c, const float* d,
    const float* rgba)
{
    push_triangle(positions, normals, colors, a, b, c, rgba);
    push_triangle(positions, normals, colors, a, c, d, rgba);
}

static inline void vec_add3(const float* a, const float* b, float* out) {
    out[0] = a[0] + b[0]; out[1] = a[1] + b[1]; out[2] = a[2] + b[2];
}

static inline void vec_sub3(const float* a, const float* b, float* out) {
    out[0] = a[0] - b[0]; out[1] = a[1] - b[1]; out[2] = a[2] - b[2];
}

static inline void vec_scale3(const float* a, float s, float* out) {
    out[0] = a[0] * s; out[1] = a[1] * s; out[2] = a[2] * s;
}

static inline void vec_madd3(const float* a, const float* b, float s, float* out) {
    out[0] = a[0] + b[0] * s;
    out[1] = a[1] + b[1] * s;
    out[2] = a[2] + b[2] * s;
}

struct RingTrig {
    static const int kSeg = 40;
    float c[kSeg + 1];
    float s[kSeg + 1];
    RingTrig() {
        for (int i = 0; i <= kSeg; i++) {
            const float a = (2.0f * 3.14159265358979f * i) / kSeg;
            c[i] = cosf(a);
            s[i] = sinf(a);
        }
    }
};
static const RingTrig g_ring_trig;

static void append_disc(
    const float* center,
    const float* u_axis,
    const float* v_axis,
    float radius,
    int segments,
    const float* rgba,
    FSink& positions,
    FSink& normals,
    FSink& colors)
{
    if (radius <= 0.0f || segments < 3) return;
    const bool tbl = (segments == RingTrig::kSeg);
    for (int i = 0; i < segments; i++) {
        float c0, s0, c1, s1;
        if (tbl) {
            c0 = g_ring_trig.c[i];     s0 = g_ring_trig.s[i];
            c1 = g_ring_trig.c[i + 1]; s1 = g_ring_trig.s[i + 1];
        } else {
            float a0 = (2.0f * 3.14159265358979f * i) / segments;
            float a1 = (2.0f * 3.14159265358979f * (i + 1)) / segments;
            c0 = cosf(a0); s0 = sinf(a0);
            c1 = cosf(a1); s1 = sinf(a1);
        }
        float p0[3], p1[3];
        for (int k = 0; k < 3; k++) {
            p0[k] = center[k] + radius * (c0 * u_axis[k] + s0 * v_axis[k]);
            p1[k] = center[k] + radius * (c1 * u_axis[k] + s1 * v_axis[k]);
        }
        push_triangle(positions, normals, colors, center, p0, p1, rgba);
    }
}

static void append_cylinder_solid_segment(
    const float* start,
    const float* end,
    const float* x_axis,
    const float* y_axis,
    float start_radius,
    float end_radius,
    const float* rgba,
    FSink& positions,
    FSink& normals,
    FSink& colors)
{
    if (start_radius <= 0.0f && end_radius <= 0.0f) return;
    const int segments = 40;
    if (start_radius > 0.0f) {
        append_disc(start, x_axis, y_axis, start_radius, segments, rgba, positions, normals, colors);
    }
    if (end_radius > 0.0f) {
        append_disc(end, x_axis, y_axis, end_radius, segments, rgba, positions, normals, colors);
    }

    for (int i = 0; i < segments; i++) {
        const float ca0 = g_ring_trig.c[i],     sa0 = g_ring_trig.s[i];
        const float ca1 = g_ring_trig.c[i + 1], sa1 = g_ring_trig.s[i + 1];
        float s0[3], s1[3], e1[3], e0[3];
        for (int k = 0; k < 3; k++) {
            float dir0 = ca0 * x_axis[k] + sa0 * y_axis[k];
            float dir1 = ca1 * x_axis[k] + sa1 * y_axis[k];
            s0[k] = start[k] + start_radius * dir0;
            s1[k] = start[k] + start_radius * dir1;
            e1[k] = end[k] + end_radius * dir1;
            e0[k] = end[k] + end_radius * dir0;
        }
        push_quad(positions, normals, colors, s0, s1, e1, e0, rgba);
    }
}

static void append_cylinder_inner_ring(
    const float* center,
    const float* x_axis,
    const float* y_axis,
    float radius,
    float thickness,
    const float* rgba,
    FSink& positions,
    FSink& normals,
    FSink& colors)
{
    if (radius <= 0.0f || thickness <= 0.0f) return;
    const int segments = 40;
    float outer = radius + 0.5f * thickness;
    float inner = std::max(0.0f, radius - 0.5f * thickness);
    for (int i = 0; i < segments; i++) {
        const float ca0 = g_ring_trig.c[i],     sa0 = g_ring_trig.s[i];
        const float ca1 = g_ring_trig.c[i + 1], sa1 = g_ring_trig.s[i + 1];
        float o0[3], o1[3], i0v[3], i1v[3];
        for (int k = 0; k < 3; k++) {
            float basis0 = ca0 * x_axis[k] + sa0 * y_axis[k];
            float basis1 = ca1 * x_axis[k] + sa1 * y_axis[k];
            o0[k] = center[k] + outer * basis0;
            o1[k] = center[k] + outer * basis1;
            i0v[k] = center[k] + inner * basis0;
            i1v[k] = center[k] + inner * basis1;
        }
        push_quad(positions, normals, colors, o0, o1, i1v, i0v, rgba);
    }
}

static bool site_overlay_visible(const SiteDesc& site, const float* rgba) {
    return !(site.group >= 3 || rgba[3] < 0.01f);
}

static bool any_site_overlay_visible(const PhysicsModelDesc& desc,
                                     const float* rgba) {
    for (int si = 0; si < desc.nsite; ++si) {
        if (site_overlay_visible(desc.sites[si], rgba + (size_t)si * 4)) return true;
    }
    return false;
}

static void emit_site_overlay(
    const SiteDesc& s,
    const float* rgba,
    const float* center,
    const float* m,
    const float* model_view,
    FSink& positions,
    FSink& normals,
    FSink& colors)
{
    float x_axis[3] = {m[0], m[3], m[6]};
    float y_axis[3] = {m[1], m[4], m[7]};
    float z_axis[3] = {m[2], m[5], m[8]};

    if (s.type == SITE_TYPE_SPHERE) {
        float sphere_rgba[4] = {rgba[0], rgba[1], rgba[2], rgba[3]};
        if (sphere_rgba[3] < 0.999f) {
            sphere_rgba[0] *= 0.65f;
            sphere_rgba[1] *= 0.65f;
            sphere_rgba[2] *= 0.65f;
            sphere_rgba[3] = fminf(1.0f, sphere_rgba[3] * 1.1f);
        }
        append_disc(center, x_axis, y_axis, s.size[0], 40, sphere_rgba,
                    positions, normals, colors);
    } else if (s.type == SITE_TYPE_CYLINDER || s.type == SITE_TYPE_CAPSULE) {
        bool flat_marker = s.size[1] <= s.size[0] * 0.25f;
        // Match MuJoCo's rendered site-cylinder convention observed in
        // `mjvScene`: the long axis is local +Z, while the circular
        // cross-section lies in the local X/Y plane.
        const float* cyl_x = x_axis;
        const float* cyl_y = y_axis;
        const float* cyl_z = z_axis;
        float seg_start[3], seg_end[3];
        float overlay_rgba[4] = {rgba[0], rgba[1], rgba[2], rgba[3]};
        float start_radius = s.size[0];
        float end_radius = s.size[0];
        if (flat_marker) {
            float zoff[3];
            vec_scale3(cyl_z, s.size[1], zoff);
            vec_sub3(center, zoff, seg_start);
            vec_add3(center, zoff, seg_end);
        } else {
            float axis_eye_z =
                model_view[2] * cyl_z[0] +
                model_view[6] * cyl_z[1] +
                model_view[10] * cyl_z[2];
            float sign = (axis_eye_z < 0.0f) ? 1.0f : -1.0f;
            seg_start[0] = center[0];
            seg_start[1] = center[1];
            seg_start[2] = center[2];
            seg_end[0] = center[0] + sign * cyl_z[0] * s.size[1];
            seg_end[1] = center[1] + sign * cyl_z[1] * s.size[1];
            seg_end[2] = center[2] + sign * cyl_z[2] * s.size[1];
            start_radius = s.size[0] * 0.52f;
            end_radius = s.size[0] * 0.03f;
            overlay_rgba[0] *= 0.22f;
            overlay_rgba[1] *= 0.22f;
            overlay_rgba[2] *= 0.22f;
            overlay_rgba[3] = fminf(1.0f, overlay_rgba[3] * 1.5f);
        }
        append_cylinder_solid_segment(seg_start, seg_end, cyl_x, cyl_y,
                                      start_radius, end_radius, overlay_rgba,
                                      positions, normals, colors);
        if (flat_marker) {
            float inner_thickness = std::max(s.size[0] * 0.08f, 0.0020f);
            append_cylinder_inner_ring(center, cyl_x, cyl_y, s.size[0] * 0.55f, inner_thickness, rgba,
                                       positions, normals, colors);
        }
    }
}

// Vertex count ONE VISIBLE site of this description contributes, obtained by
// running the emitter itself with a null sink. A pure function of (type,size):
// neither the pose nor the colour can move it (they only select `sign` and the
// colour scale, never a branch that emits). Exact by construction.
static int site_overlay_vertex_count(const SiteDesc& s) {
    FSink cp, cn, cc;
    const size_t big = (size_t)1 << 40;
    cp.reset(nullptr, big); cn.reset(nullptr, big); cc.reset(nullptr, big);
    const float center[3] = {0.0f, 0.0f, 0.0f};
    const float m[9] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
    const float rgba[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float mv[16] = {0.0f};
    emit_site_overlay(s, rgba, center, m, mv, cp, cn, cc);
    return (int)(cp.size() / 3);
}

// Everything a graph replay needs to REBUILD one overlay slot without
// re-deriving the render state it was captured under. Both members are frozen
// by the graph signature (`reflect_dim` and `model_view` come from the
// RenderState / reflection plan shadows, which must compare byte-equal before
// a replay is allowed), so a rebuild reproduces the captured pass exactly --
// only the site POSE it reads is allowed to have moved.
struct SiteSlotCtx {
    float model_view[16] = {0.0f};
    float reflect_dim = 0.0f;
    int   verts = 0;
    bool  used = false;
    bool  miss = false;   // slot was too small; the heap fallback ran
};

// Fill the three sinks with one world's overlay, in the original order.
static void build_site_overlay(
    const PhysicsModelDesc& desc,
    const float* env_site_xpos,
    const float* env_site_xmat,
    const float* env_site_rgba,
    const float* model_view,
    float reflect_dim,
    FSink& positions,
    FSink& normals,
    FSink& colors)
{
    for (int si = 0; si < desc.nsite; si++) {
        const SiteDesc& s = desc.sites[si];
        const float* rgba = env_site_rgba + (size_t)si * 4;
        if (!site_overlay_visible(s, rgba)) continue;
        emit_site_overlay(s, rgba, env_site_xpos + si * 3, env_site_xmat + si * 9,
                          model_view, positions, normals, colors);
    }
    // This unlit host-vertex path does not run Gouraud material processing.
    // Match renderGeomReflection's RGB modulation before alpha composition.
    if (reflect_dim > 0.0f && colors.data() && !colors.over) {
        float* c = colors.data();
        for (size_t v = 0; v < colors.size(); v += 4) {
            c[v] *= reflect_dim;
            c[v + 1] *= reflect_dim;
            c[v + 2] *= reflect_dim;
        }
    }
}

static void refresh_site_overlay_slot(
    const PhysicsModelDesc& desc,
    const float* env_site_xpos,
    const float* env_site_xmat,
    const float* env_site_rgba,
    const SiteSlotCtx& ctx,
    float* stage,
    int stage_verts)
{
    if (!ctx.used || !stage || stage_verts <= 0) return;
    const size_t cv = (size_t)stage_verts;
    FSink P, N, C;
    P.reset(stage, cv * 3);
    N.reset(stage + cv * 3, cv * 3);
    C.reset(stage + cv * 6, cv * 4);
    build_site_overlay(desc, env_site_xpos, env_site_xmat, env_site_rgba,
                       ctx.model_view, ctx.reflect_dim, P, N, C);
}

static void draw_site_overlays(
    Pipeline& pipeline,
    const RenderState& base_state,
    const PhysicsModelDesc& desc,
    const float* env_site_xpos,
    const float* env_site_xmat,
    const float* env_site_rgba,
    float* stage = nullptr,
    int stage_verts = 0,
    SiteSlotCtx* ctx = nullptr,
    float* d_slot = nullptr)
{
    if (ctx) { ctx->used = false; ctx->miss = false; }
    // DIAGNOSTIC ABLATION ONLY (HP_SITE_ABL=1): drop the overlay draw to price
    // it. Never on by default -- the overlay is visible geometry.
    static const int hp_site_abl = getenv("HP_SITE_ABL") ? atoi(getenv("HP_SITE_ABL")) : 0;
    if (hp_site_abl) return;
    if (desc.nsite <= 0 || !env_site_xpos || !env_site_xmat || !env_site_rgba) return;

    bool any_vis = false;
    for (int si = 0; si < desc.nsite && !any_vis; si++)
        any_vis = site_overlay_visible(desc.sites[si], env_site_rgba + (size_t)si * 4);
    if (!any_vis) return;

    RenderState site_state = base_state;
    float reflect_dim = 0.0f;
    if (site_state.reflect_dim > 0.0f) {
        reflect_dim = site_state.reflect_dim;
        site_state.reflect_dim = 0.0f;
    }

    // Per-thread heap fallback: used by the single-world render path (which
    // has no slot) and by the impossible-but-guarded case of a slot that is
    // too small. It never writes past its capacity either; on a miss it grows
    // to the EXACT count the count-mode pass reported and retries.
    static thread_local std::vector<float> heap;
    FSink P, N, C;
    float* base = (stage && stage_verts > 0) ? stage : nullptr;
    size_t cap_v = base ? (size_t)stage_verts : (size_t)desc.nsite * 1024;
    bool missed = false;
    for (;;) {
        float* b = base;
        if (!b) {
            if (heap.size() < cap_v * 10) heap.resize(cap_v * 10);
            b = heap.data();
        }
        P.reset(b,                 cap_v * 3);
        N.reset(b + cap_v * 3,     cap_v * 3);
        C.reset(b + cap_v * 6,     cap_v * 4);
        build_site_overlay(desc, env_site_xpos, env_site_xmat, env_site_rgba,
                           base_state.model_view, reflect_dim, P, N, C);
        if (!P.over && !N.over && !C.over) break;
        const size_t need = P.size() / 3;
        cap_v = need > cap_v ? need : cap_v * 2;
        if (base) missed = true;
        base = nullptr;          // never spill into a neighbouring pinned slot
    }

    if (P.empty()) return;
    if (ctx) {
        ctx->miss = missed;
        if (!missed && stage && stage_verts > 0) {
            memcpy(ctx->model_view, base_state.model_view, 16 * sizeof(float));
            ctx->reflect_dim = reflect_dim;
            ctx->verts = (int)(P.size() / 3);
            ctx->used = true;
        }
    }

    site_state.texture_2d_enabled = false;
    site_state.lighting_enabled = false;
    site_state.depth_test_enabled = true;
    site_state.depth_write_enabled = true;
    site_state.depth_func = DEPTH_LESS;
    site_state.blend_enabled = true;
    site_state.blend_src = BLEND_SRC_ALPHA;
    site_state.blend_dst = BLEND_ONE_MINUS_SRC_ALPHA;
    pipeline.set_render_state(site_state);

    const int nvert = static_cast<int>(P.size() / 3);
    if (d_slot && base == stage && stage && stage_verts > 0) {
        TextureState site_tex = {};
        hipMemcpyAsync(d_slot, stage, (size_t)stage_verts * 10 * sizeof(float),
                       hipMemcpyHostToDevice, pipeline.stream());
        pipeline.draw_arrays_device(PRIM_TRIANGLES,
                                    d_slot,
                                    d_slot + (size_t)stage_verts * 3,
                                    nullptr, nullptr,
                                    d_slot + (size_t)stage_verts * 6,
                                    nvert, site_tex);
        return;
    }
    pipeline.draw_arrays(PRIM_TRIANGLES,
                         P.data(),
                         N.data(),
                         nullptr,
                         C.data(),
                         nvert);
}

struct GpuArray {
    void* ptr = nullptr;
    std::vector<ssize_t> shape;
    std::string typestr;          // numpy typestr, e.g. "|u1", "<f4", "<i4"
    size_t itemsize = 1;
    py::object owner;             // the producing RayTracer (informational)
    std::shared_ptr<void> keep;   // device allocation ownership

    size_t count() const {
        size_t n = 1;
        for (ssize_t s : shape) n *= (size_t)s;
        return n;
    }
    py::tuple shape_tuple() const {
        py::tuple t(shape.size());
        for (size_t i = 0; i < shape.size(); ++i) t[i] = py::int_(shape[i]);
        return t;
    }
    py::dict cuda_array_interface() const {
        py::dict d;
        d["shape"] = shape_tuple();
        d["typestr"] = typestr;
        d["data"] = py::make_tuple(reinterpret_cast<intptr_t>(ptr), false);
        d["version"] = 3;
        d["strides"] = py::none();
        return d;
    }
    py::array numpy() const {
        py::array arr(py::dtype(typestr), shape);
        const size_t bytes = count() * itemsize;
        if (bytes) {
            hipError_t err = hipMemcpy(arr.mutable_data(), ptr, bytes, hipMemcpyDeviceToHost);
            if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
        }
        return arr;
    }
};

struct RayTracerPy {
    hipgl::physics::RayTracer rt;
    PhysicsModel* model = nullptr;
    py::object model_owner;       // keeps the model (and its device arrays) alive

    void check(PhysicsEngine& engine) const {
        if (!rt.initialized() || !model) throw std::runtime_error("RayTracer not initialized");
        if (engine.model() != model)
            throw std::runtime_error("RayTracer and engine models do not match; reinitialize");
    }
    // Slot table from `active` (None = every world of the batch).
    static std::vector<int> slots(py::object active, int num_envs) {
        std::vector<int> w;
        if (active.is_none()) {
            w.resize(num_envs);
            for (int i = 0; i < num_envs; ++i) w[i] = i;
            return w;
        }
        std::vector<unsigned char> seen((size_t)num_envs, 0);
        for (auto h : active) {
            const int e = h.cast<int>();
            if (e < 0 || e >= num_envs) throw std::runtime_error("RayTracer: active world outside the batch");
            if (!seen[e]) { seen[e] = 1; w.push_back(e); }
        }
        if (w.empty()) throw std::runtime_error("RayTracer: empty active world set");
        return w;
    }
    // Synchronous render of one (camera list, slot list) request.
    void run(PhysicsEngine& engine, const std::vector<int>& cams, const std::vector<int>& worlds,
             const int* out_rows, int nout, int mask, bool flip_v, bool flip_h) {
        check(engine);
        if (worlds.empty() || cams.empty()) throw std::runtime_error("RayTracer: empty request");
        hipgl::physics::RtRequest req;
        req.cam_ids = cams.data(); req.ncam = (int)cams.size();
        req.worlds = worlds.data(); req.nslot = (int)worlds.size();
        req.out_rows = out_rows; req.nout = nout;
        req.out_mask = mask; req.flip_v = flip_v; req.flip_h = flip_h;
        hipStream_t s = engine.stream();
        rt.render(model->device_model(), engine.data().device_data(), s, req);
        hipError_t err = hipStreamSynchronize(s);
        if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
    }
};

namespace {
struct PinnedStage {
    void* p = nullptr; size_t n = 0;
    void* get(size_t want) {
        if (want > n) {
            if (p) hipHostFree(p);
            size_t sz = std::max(want, (size_t)1 << 20);
            if (hipHostMalloc(&p, sz, hipHostMallocDefault) != hipSuccess) { p = nullptr; n = 0;
                throw std::runtime_error("hipHostMalloc failed"); }
            n = sz;
        }
        return p;
    }
};
PinnedStage g_stage_con, g_stage_lam;
}  // namespace

PYBIND11_MODULE(instinct_sim, m) {
    m.doc() = "HIP-based GPU rigid body physics for AMD Instinct™ GPUs";

    // ── SolverOptions ───────────────────────────────────────────────────────
    py::enum_<SolverType>(m, "SolverType")
        .value("PGS", SOLVER_PGS)
        .value("NEWTON", SOLVER_NEWTON);

    py::class_<SolverOptions>(m, "SolverOptions")
        .def(py::init<>())
        .def_readwrite("solver", &SolverOptions::solver)
        .def_readwrite("iterations", &SolverOptions::iterations)
        .def_readwrite("tolerance", &SolverOptions::tolerance)
        .def_readwrite("contact_slip", &SolverOptions::contact_slip)
        .def_readwrite("contact_margin", &SolverOptions::contact_margin)
        .def_readwrite("enable_contact", &SolverOptions::enable_contact)
        .def_readwrite("enable_limits", &SolverOptions::enable_limits)
        .def_readwrite("enable_gravity", &SolverOptions::enable_gravity)
        .def_readwrite("enable_damping", &SolverOptions::enable_damping)
        .def_readwrite("zero_bias", &SolverOptions::zero_bias)
        .def_readwrite("impratio", &SolverOptions::impratio)
        .def_readwrite("max_contacts_per_body_pair", &SolverOptions::max_contacts_per_body_pair)
        .def_readwrite("free_joint_damping", &SolverOptions::free_joint_damping);

    // ── PhysicsModel ────────────────────────────────────────────────────────
    py::class_<PhysicsModel>(m, "PhysicsModel")
        .def(py::init<>())
        .def("load_from_xml", &PhysicsModel::load_from_xml, py::arg("xml_string"), py::arg("meshdir") = "")
        .def_static("from_xml", [](const std::string& xml) {
            auto model = std::make_unique<PhysicsModel>();
            if (!model->load_from_xml(xml)) throw std::runtime_error("Failed to parse MJCF XML");
            return model;
        }, py::arg("xml_string"))
        .def_property_readonly("nbody", &PhysicsModel::nbody)
        .def_property_readonly("njnt", &PhysicsModel::njnt)
        .def_property_readonly("ngeom", &PhysicsModel::ngeom)
        .def_property_readonly("nq", &PhysicsModel::nq)
        .def_property_readonly("nv", &PhysicsModel::nv)
        .def_property_readonly("nu", &PhysicsModel::nu)
        .def_property_readonly("ncamera", [](const PhysicsModel& m) { return m.desc().ncamera; })
        .def_property_readonly("nsite", &PhysicsModel::nsite)
        .def_property_readonly("nequality", [](const PhysicsModel& m) { return m.desc().nequality; })
        .def_property_readonly("timestep", [](const PhysicsModel& m) { return m.desc().timestep; })
        .def_property_readonly("gravity", [](const PhysicsModel& m) {
            return py::make_tuple(m.desc().gravity[0], m.desc().gravity[1], m.desc().gravity[2]);
        })
        .def("body_name", [](const PhysicsModel& m, int idx) -> std::string {
            if (idx < 0 || idx >= m.nbody()) return "";
            return m.desc().bodies[idx].name;
        }, py::arg("idx"))
        .def("joint_name", [](const PhysicsModel& m, int idx) -> std::string {
            if (idx < 0 || idx >= m.njnt()) return "";
            return m.desc().joints[idx].name;
        }, py::arg("idx"))
        .def("camera_name", [](const PhysicsModel& m, int idx) -> std::string {
            if (idx < 0 || idx >= m.desc().ncamera) return "";
            return m.desc().cameras[idx].name;
        }, py::arg("idx"))
        .def("body_name2id", [](const PhysicsModel& m, const std::string& name) -> int {
            for (int i = 0; i < m.nbody(); i++)
                if (name == m.desc().bodies[i].name) return i;
            return -1;
        }, py::arg("name"))
        .def("joint_name2id", [](const PhysicsModel& m, const std::string& name) -> int {
            for (int i = 0; i < m.njnt(); i++)
                if (name == m.desc().joints[i].name) return i;
            return -1;
        }, py::arg("name"))
        .def("camera_name2id", [](const PhysicsModel& m, const std::string& name) -> int {
            for (int i = 0; i < m.desc().ncamera; i++)
                if (name == m.desc().cameras[i].name) return i;
            return -1;
        }, py::arg("name"))
        .def("geom_name", [](const PhysicsModel& m, int idx) -> std::string {
            if (idx < 0 || idx >= m.ngeom()) return "";
            return m.desc().geoms[idx].name;
        }, py::arg("idx"))
        .def("geom_name2id", [](const PhysicsModel& m, const std::string& name) -> int {
            for (int i = 0; i < m.ngeom(); i++)
                if (name == m.desc().geoms[i].name) return i;
            return -1;
        }, py::arg("name"))
        .def("geom_body_id", [](const PhysicsModel& m, int idx) -> int {
            if (idx < 0 || idx >= m.ngeom()) return -1;
            return m.desc().geoms[idx].body;
        }, py::arg("idx"))
        .def("geom_type", [](const PhysicsModel& m, int idx) -> int {
            if (idx < 0 || idx >= m.ngeom()) return -1;
            return (int)m.desc().geoms[idx].type;
        }, py::arg("idx"))
        .def("geom_texid", [](const PhysicsModel& m, int idx) -> int {
            if (idx < 0 || idx >= m.ngeom()) return -1;
            return m.desc().geom_texid[idx];
        }, py::arg("idx"))
        .def("geom_group", [](const PhysicsModel& m, int idx) -> int {
            if (idx < 0 || idx >= m.ngeom()) return -1;
            return m.desc().geoms[idx].group;
        }, py::arg("idx"))
        // DIAGNOSTIC: hide one geom by dropping its alpha under the renderer's
        // skip threshold. Lets an occlusion hypothesis be tested by removing
        // the suspected occluder and seeing whether the hidden surface
        // reappears -- which no amount of reading the depth test can settle.
        // The renderer caches its vertex layout, so re-init it after calling.
        .def("debug_set_geom_hidden", [](PhysicsModel& m, int idx, bool hidden) {
            if (idx < 0 || idx >= m.ngeom()) return false;
            auto& g = m.mutable_desc_for_debug().geoms[idx];
            g.rgba[3] = hidden ? 0.0f : 1.0f;
            return true;
        }, py::arg("idx"), py::arg("hidden"))
        .def("site_name", [](const PhysicsModel& m, int idx) -> std::string {
            if (idx < 0 || idx >= m.nsite()) return "";
            return m.desc().sites[idx].name;
        }, py::arg("idx"))
        .def("site_name2id", [](const PhysicsModel& m, const std::string& name) -> int {
            for (int i = 0; i < m.nsite(); i++)
                if (name == m.desc().sites[i].name) return i;
            return -1;
        }, py::arg("name"))
        // Joint property accessors for OSC controller
        .def("joint_type", [](const PhysicsModel& m, int idx) -> int {
            if (idx < 0 || idx >= m.njnt()) return -1;
            return (int)m.desc().joints[idx].type;
        }, py::arg("idx"))
        .def("joint_body", [](const PhysicsModel& m, int idx) -> int {
            if (idx < 0 || idx >= m.njnt()) return -1;
            return m.desc().joints[idx].body;
        }, py::arg("idx"))
        .def("joint_qpos_adr", [](const PhysicsModel& m, int idx) -> int {
            if (idx < 0 || idx >= m.njnt()) return -1;
            return m.desc().joints[idx].qpos_adr;
        }, py::arg("idx"))
        .def("joint_dof_adr", [](const PhysicsModel& m, int idx) -> int {
            if (idx < 0 || idx >= m.njnt()) return -1;
            return m.desc().joints[idx].dof_adr;
        }, py::arg("idx"))
        .def("joint_axis", [](const PhysicsModel& m, int idx) -> py::tuple {
            if (idx < 0 || idx >= m.njnt()) return py::make_tuple(0,0,0);
            auto& j = m.desc().joints[idx];
            return py::make_tuple(j.axis[0], j.axis[1], j.axis[2]);
        }, py::arg("idx"))
        .def("joint_pos", [](const PhysicsModel& m, int idx) -> py::tuple {
            if (idx < 0 || idx >= m.njnt()) return py::make_tuple(0,0,0);
            auto& j = m.desc().joints[idx];
            return py::make_tuple(j.pos[0], j.pos[1], j.pos[2]);
        }, py::arg("idx"))
        // Body property accessors
        .def("body_parent", [](const PhysicsModel& m, int idx) -> int {
            if (idx < 0 || idx >= m.nbody()) return -1;
            return m.desc().bodies[idx].parent;
        }, py::arg("idx"))
        .def("body_pos", [](const PhysicsModel& m, int idx) -> py::tuple {
            if (idx < 0 || idx >= m.nbody()) throw py::index_error("body index out of range");
            const auto& b = m.desc().bodies[idx];
            return py::make_tuple(b.pos_d[0], b.pos_d[1], b.pos_d[2]);
        }, py::arg("idx"))
        .def("body_quat", [](const PhysicsModel& m, int idx) -> py::tuple {
            if (idx < 0 || idx >= m.nbody()) throw py::index_error("body index out of range");
            const auto& b = m.desc().bodies[idx];
            return py::make_tuple(b.quat_d[0], b.quat_d[1], b.quat_d[2], b.quat_d[3]);
        }, py::arg("idx"))
        .def("set_fixed_body_poses", [](PhysicsModel& m, py::array body_ids,
                  py::array_t<double, py::array::c_style | py::array::forcecast> positions,
                  py::array_t<double, py::array::c_style | py::array::forcecast> quaternions) {
            const char kind = body_ids.dtype().kind();
            if (body_ids.ndim() != 1 || (kind != 'i' && kind != 'u')) {
                throw py::value_error("body_ids must be a one-dimensional integer array");
            }
            const auto count = body_ids.shape(0);
            if (positions.ndim() != 2 || positions.shape(0) != count || positions.shape(1) != 3 ||
                quaternions.ndim() != 2 || quaternions.shape(0) != count || quaternions.shape(1) != 4) {
                throw py::value_error("positions and quaternions must have shapes (N, 3) and (N, 4)");
            }
            auto ids64 = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(body_ids);
            if (!ids64) throw py::value_error("body_ids cannot be represented as integers");
            std::vector<int> ids;
            ids.reserve(count);
            for (py::ssize_t row = 0; row < count; ++row) {
                const int64_t id = ids64.data()[row];
                if (id <= 0 || id >= m.nbody()) throw py::value_error("body id is out of range or world");
                ids.push_back(static_cast<int>(id));
            }
            std::vector<double> pos(positions.data(), positions.data() + count * 3);
            std::vector<double> quat(quaternions.data(), quaternions.data() + count * 4);
            m.set_fixed_body_poses(ids, pos, quat);
        }, py::arg("body_ids"), py::arg("positions"), py::arg("quaternions"),
           "Update model-wide fixed-body local poses (unit wxyz quaternions), preserving GPU allocations. "
           "Exclude concurrent stepping; reset stale contacts/warmstarts and call engine.forward() afterward.")
        .def("set_fixed_body_poses_env", [](PhysicsModel& m, int num_envs, int env, py::array body_ids,
                  py::array_t<double, py::array::c_style | py::array::forcecast> positions,
                  py::array_t<double, py::array::c_style | py::array::forcecast> quaternions) {
            const char kind = body_ids.dtype().kind();
            if (body_ids.ndim() != 1 || (kind != 'i' && kind != 'u')) {
                throw py::value_error("body_ids must be a one-dimensional integer array");
            }
            const auto count = body_ids.shape(0);
            if (positions.ndim() != 2 || positions.shape(0) != count || positions.shape(1) != 3 ||
                quaternions.ndim() != 2 || quaternions.shape(0) != count || quaternions.shape(1) != 4) {
                throw py::value_error("positions and quaternions must have shapes (N, 3) and (N, 4)");
            }
            auto ids64 = py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(body_ids);
            if (!ids64) throw py::value_error("body_ids cannot be represented as integers");
            std::vector<int> ids;
            ids.reserve(count);
            for (py::ssize_t row = 0; row < count; ++row) {
                const int64_t id = ids64.data()[row];
                if (id <= 0 || id >= m.nbody()) throw py::value_error("body id is out of range or world");
                ids.push_back(static_cast<int>(id));
            }
            std::vector<double> pos(positions.data(), positions.data() + count * 3);
            std::vector<double> quat(quaternions.data(), quaternions.data() + count * 4);
            m.set_fixed_body_poses_env(num_envs, env, ids, pos, quat);
        }, py::arg("num_envs"), py::arg("env"), py::arg("body_ids"), py::arg("positions"), py::arg("quaternions"),
           "Per-world set_fixed_body_poses: move fixed bodies in world `env` only. The first call "
           "allocates per-world poses for num_envs worlds. Same caller contract as set_fixed_body_poses.")
        .def_property_readonly("body_pose_nenv", &PhysicsModel::body_pose_nenv)
        .def("read_body_pose_env", [](const PhysicsModel& m, int env) -> py::tuple {
            // Device readback of the per-world fixed-pose storage FK reads for
            // world `env` (FP32 pos, FP32 quat, FP64 pos, FP64 quat).
            const auto& dm = m.device_model();
            if (!dm.body_pos_env_d || env < 0 || env >= dm.body_pose_nenv)
                throw py::value_error("per-world body poses not allocated or env out of range");
            const size_t nb = (size_t)m.nbody();
            py::array_t<float> pf({(py::ssize_t)nb, (py::ssize_t)3}), qf({(py::ssize_t)nb, (py::ssize_t)4});
            py::array_t<double> pd({(py::ssize_t)nb, (py::ssize_t)3}), qd({(py::ssize_t)nb, (py::ssize_t)4});
            hipMemcpy(pf.mutable_data(), dm.body_pos_env + env * nb * 3, nb * 3 * sizeof(float), hipMemcpyDeviceToHost);
            hipMemcpy(qf.mutable_data(), dm.body_quat_env + env * nb * 4, nb * 4 * sizeof(float), hipMemcpyDeviceToHost);
            hipMemcpy(pd.mutable_data(), dm.body_pos_env_d + env * nb * 3, nb * 3 * sizeof(double), hipMemcpyDeviceToHost);
            hipMemcpy(qd.mutable_data(), dm.body_quat_env_d + env * nb * 4, nb * 4 * sizeof(double), hipMemcpyDeviceToHost);
            return py::make_tuple(pf, qf, pd, qd);
        }, py::arg("env"))
        .def("body_mass", [](const PhysicsModel& m, int idx) -> float {
            if (idx < 0 || idx >= m.nbody()) return 0.0f;
            return m.desc().bodies[idx].mass;
        }, py::arg("idx"))
        .def("body_inertia", [](const PhysicsModel& m, int idx) -> py::tuple {
            if (idx < 0 || idx >= m.nbody()) return py::make_tuple(0,0,0);
            auto& b = m.desc().bodies[idx];
            return py::make_tuple(b.inertia[0], b.inertia[1], b.inertia[2]);
        }, py::arg("idx"))
        .def("body_ipos", [](const PhysicsModel& m, int idx) -> py::tuple {
            if (idx < 0 || idx >= m.nbody()) return py::make_tuple(0,0,0);
            auto& b = m.desc().bodies[idx];
            return py::make_tuple(b.ipos[0], b.ipos[1], b.ipos[2]);
        }, py::arg("idx"))
        .def("body_iquat", [](const PhysicsModel& m, int idx) -> py::tuple {
            if (idx < 0 || idx >= m.nbody()) return py::make_tuple(1,0,0,0);
            auto& b = m.desc().bodies[idx];
            return py::make_tuple(b.iquat[0], b.iquat[1], b.iquat[2], b.iquat[3]);
        }, py::arg("idx"))
        .def("joint_limited", [](const PhysicsModel& m, int idx) -> bool {
            if (idx < 0 || idx >= m.njnt()) throw py::index_error("joint index out of range");
            return m.desc().joints[idx].limited != 0;
        }, py::arg("idx"))
        .def("joint_armature", [](const PhysicsModel& m, int idx) -> float {
            if (idx < 0 || idx >= m.njnt()) return 0.0f;
            return m.desc().joints[idx].armature;
        }, py::arg("idx"))
        .def("joint_damping", [](const PhysicsModel& m, int idx) -> float {
            if (idx < 0 || idx >= m.njnt()) return 0.0f;
            return m.desc().joints[idx].damping;
        }, py::arg("idx"))
        .def("read_gpu_jnt_armature", [](const PhysicsModel& m) -> py::array_t<float> {
            int nj = m.njnt();
            const auto& dm = m.device_model();
            py::array_t<float> arr(nj);
            if (dm.jnt_armature) {
                hipMemcpy(arr.mutable_data(), dm.jnt_armature, nj * sizeof(float), hipMemcpyDeviceToHost);
            } else {
                memset(arr.mutable_data(), 0, nj * sizeof(float));
            }
            return arr;
        })
        // Double-precision model shadows that load_from_xml fills by widening the float parse.  The env
        // overwrites them with the compiled MjModel's doubles so collision / constraint setup sees exactly
        // MuJoCo's values (float rounding of e.g. a 5 cm box half-size moves a resting contact by nm).
        .def("model_array_d", [](PhysicsModel& m, const std::string& name, py::object values) -> py::object {
            auto& dm = m.mutable_device_model();
            const int ng = m.ngeom(), nj = m.njnt(), nb = m.nbody();
            double* ptr = nullptr; int count = 0;
            if      (name == "geom_pos")          { ptr = dm.geom_pos_d;          count = ng * 3; }
            else if (name == "geom_quat")         { ptr = dm.geom_quat_d;         count = ng * 4; }
            else if (name == "geom_size")         { ptr = dm.geom_size_d;         count = ng * 3; }
            else if (name == "geom_friction")     { ptr = dm.geom_friction_d;     count = ng * 3; }
            else if (name == "geom_solref")       { ptr = dm.geom_solref_d;       count = ng * 2; }
            else if (name == "geom_solimp")       { ptr = dm.geom_solimp_d;       count = ng * 5; }
            else if (name == "jnt_range")         { ptr = dm.jnt_range_d;         count = nj * 2; }
            else if (name == "jnt_stiffness")     { ptr = dm.jnt_stiffness_d;     count = nj; }
            else if (name == "jnt_damping")       { ptr = dm.jnt_damping_d;       count = nj; }
            else if (name == "jnt_armature")      { ptr = dm.jnt_armature_d;      count = nj; }
            else if (name == "jnt_frictionloss")  { ptr = dm.jnt_frictionloss_d;  count = nj; }
            else if (name == "jnt_solref_lim")    { ptr = dm.jnt_solref_lim_d;    count = nj * 2; }
            else if (name == "jnt_solimp_lim")    { ptr = dm.jnt_solimp_lim_d;    count = nj * 5; }
            else if (name == "jnt_pos")           { ptr = dm.jnt_pos_d;           count = nj * 3; }
            else if (name == "jnt_axis")          { ptr = dm.jnt_axis_d;          count = nj * 3; }
            else if (name == "body_mass")         { ptr = dm.body_mass_d;         count = nb; }
            else if (name == "body_inertia")      { ptr = dm.body_inertia_d;      count = nb * 3; }
            else if (name == "body_ipos")         { ptr = dm.body_ipos_d;         count = nb * 3; }
            else if (name == "body_iquat")        { ptr = dm.body_iquat_d;        count = nb * 4; }
            else if (name == "body_pos")          { ptr = dm.body_pos_d;          count = nb * 3; }
            else if (name == "body_quat")         { ptr = dm.body_quat_d;         count = nb * 4; }
            else throw py::value_error("model_array_d: unknown array " + name);
            // Per-world fixed-body poses are seeded from body_pos_d / body_quat_d on first use; writing the
            // model-wide pose after that would leave the worlds' copies stale.
            if (!values.is_none() && (name == "body_pos" || name == "body_quat") && dm.body_pos_env_d)
                throw py::value_error("model_array_d: per-world body poses already exist; set " + name + " first");
            if (!ptr) throw py::value_error("model_array_d: " + name + " is not allocated");
            hipDeviceSynchronize();
            if (values.is_none()) {
                py::array_t<double> out(count);
                hipMemcpy(out.mutable_data(), ptr, count * sizeof(double), hipMemcpyDeviceToHost);
                return out;
            }
            auto arr = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(values);
            if (!arr || arr.size() != count)
                throw py::value_error("model_array_d: " + name + " needs " + std::to_string(count) + " values");
            hipMemcpy(ptr, arr.data(), count * sizeof(double), hipMemcpyHostToDevice);
            hipDeviceSynchronize();
            return py::none();
        }, py::arg("name"), py::arg("values") = py::none(),
           "Read (values=None) or overwrite a double-precision model shadow, flattened. Exclude concurrent stepping.")
        .def("gpu_jnt_armature_ptr", [](const PhysicsModel& m) -> uintptr_t {
            return reinterpret_cast<uintptr_t>(m.device_model().jnt_armature);
        })
        .def("set_textures", [](PhysicsModel& m,
                                py::array_t<uint8_t> tex_data,
                                py::list tex_widths,
                                py::list tex_heights,
                                py::list geom_texids,
                                py::object geom_texrepeat_obj,
                                py::object tex_iscube_obj) {
            auto& desc = const_cast<PhysicsModelDesc&>(m.desc());
            auto buf = tex_data.request();
            int ntex = py::len(tex_widths);
            desc.ntexture = ntex;

            delete[] desc.texture_data;
            desc.texture_data_size = (int)buf.size;
            desc.texture_data = new unsigned char[buf.size];
            memcpy(desc.texture_data, buf.ptr, buf.size);

            int offset = 0;
            for (int i = 0; i < ntex && i < PhysicsModelDesc::MAX_TEXTURES; i++) {
                int w = tex_widths[i].cast<int>();
                int h = tex_heights[i].cast<int>();
                desc.textures[i].width = w;
                desc.textures[i].height = h;
                desc.textures[i].data_offset = offset;
                desc.textures[i].is_cube = 0;
                offset += w * h * 3;
            }

            if (!tex_iscube_obj.is_none()) {
                py::list ic = tex_iscube_obj.cast<py::list>();
                int n = (int)py::len(ic);
                for (int i = 0; i < n && i < ntex &&
                                i < PhysicsModelDesc::MAX_TEXTURES; i++) {
                    desc.textures[i].is_cube = ic[i].cast<int>() ? 1 : 0;
                }
            } else {
                // No metadata: stay on the 2D path (the pre-cube behaviour,
                // which is safe) but say so, because any cube-mapped geom will
                // be mis-sampled and that must not be silent.
                for (int i = 0; i < ntex && i < PhysicsModelDesc::MAX_TEXTURES; i++) {
                    if (desc.textures[i].height == 6 * desc.textures[i].width) {
                        fprintf(stderr, "[set_textures] texture %d is %dx%d "
                                "(6 stacked faces) but no tex_iscube metadata "
                                "supplied -- treating as 2D. Pass tex_iscube "
                                "from extract_textures().\n",
                                i, desc.textures[i].width, desc.textures[i].height);
                    }
                }
            }

            int ng = py::len(geom_texids);
            for (int i = 0; i < ng && i < MAX_GEOMS; i++) {
                desc.geom_texid[i] = geom_texids[i].cast<int>();
                desc.geom_texrepeat[i * 2 + 0] = 1.0f;
                desc.geom_texrepeat[i * 2 + 1] = 1.0f;
            }

            if (!geom_texrepeat_obj.is_none()) {
                auto tr = geom_texrepeat_obj.cast<py::array_t<float>>();
                auto tr_buf = tr.request();
                float* tr_ptr = (float*)tr_buf.ptr;
                int n = (int)(tr_buf.size / 2);
                for (int i = 0; i < n && i < MAX_GEOMS; i++) {
                    desc.geom_texrepeat[i * 2 + 0] = tr_ptr[i * 2 + 0];
                    desc.geom_texrepeat[i * 2 + 1] = tr_ptr[i * 2 + 1];
                }
            }

            m.upload_textures();
        }, py::arg("tex_data"), py::arg("tex_widths"), py::arg("tex_heights"),
           py::arg("geom_texids"), py::arg("geom_texrepeat") = py::none(),
           py::arg("tex_iscube") = py::none())
        .def("set_mesh_texcoords", [](PhysicsModel& m, py::array_t<float> texcoords) {
            auto& desc = const_cast<PhysicsModelDesc&>(m.desc());
            auto buf = texcoords.request();
            int n = (int)(buf.size / 2);
            if (n != desc.mesh_total_verts) {
                fprintf(stderr, "[set_mesh_texcoords] WARNING: got %d UVs but need %d\n",
                        n, desc.mesh_total_verts);
            }
            int count = std::min(n, desc.mesh_total_verts);
            if (count <= 0) return;
            if (!desc.mesh_texcoord) {
                desc.mesh_texcoord = new float[desc.mesh_total_verts * MESH_UV_COMPONENTS];
                memset(desc.mesh_texcoord, 0, desc.mesh_total_verts * MESH_UV_COMPONENTS * sizeof(float));
            }
            memcpy(desc.mesh_texcoord, buf.ptr, count * MESH_UV_COMPONENTS * sizeof(float));
            // Re-upload to GPU
            auto& dm = const_cast<DevicePhysicsModel&>(m.device_model());
            if (dm.mesh_texcoord) {
                hipMemcpy(dm.mesh_texcoord, desc.mesh_texcoord,
                          desc.mesh_total_verts * MESH_UV_COMPONENTS * sizeof(float), hipMemcpyHostToDevice);
            } else {
                hipMalloc(&dm.mesh_texcoord, desc.mesh_total_verts * MESH_UV_COMPONENTS * sizeof(float));
                hipMemcpy(dm.mesh_texcoord, desc.mesh_texcoord,
                          desc.mesh_total_verts * MESH_UV_COMPONENTS * sizeof(float), hipMemcpyHostToDevice);
            }
            (void)count;
        }, py::arg("texcoords"));

    // ── PhysicsData ─────────────────────────────────────────────────────────
    py::class_<PhysicsData>(m, "PhysicsData")
        .def_property_readonly("num_envs", &PhysicsData::num_envs)
        .def("read_qpos", [](const PhysicsData& d) {
            const auto& dd = d.device_data();
            py::array_t<float> arr({dd.num_envs, dd.nq});
            d.read_qpos(arr.mutable_data());
            return arr;
        })
        .def("read_qvel", [](const PhysicsData& d) {
            const auto& dd = d.device_data();
            py::array_t<float> arr({dd.num_envs, dd.nv});
            d.read_qvel(arr.mutable_data());
            return arr;
        })
        .def("read_xpos", [](const PhysicsData& d) {
            const auto& dd = d.device_data();
            py::array_t<float> arr({dd.num_envs, dd.nbody, 3});
            d.read_xpos(arr.mutable_data());
            return arr;
        })
        // Debug: GEOM (not body) world pose, straight off the device buffers the
        // renderer reads. Needed to separate a pose/parsing bug from a
        // camera/projection bug from a rasterisation bug -- comparing world
        // coordinates against MuJoCo's data.geom_xpos/geom_xmat requires hip's
        // own geom poses, not body poses composed in Python.
        .def("read_geom_xpos", [](const PhysicsData& d) {
            const auto& dd = d.device_data();
            py::array_t<float> arr({dd.num_envs, dd.ngeom, 3});
            hipMemcpy(arr.mutable_data(), dd.geom_xpos,
                      (size_t)dd.num_envs * dd.ngeom * 3 * sizeof(float),
                      hipMemcpyDeviceToHost);
            return arr;
        })
        .def("read_geom_xquat", [](const PhysicsData& d) {
            const auto& dd = d.device_data();
            py::array_t<float> arr({dd.num_envs, dd.ngeom, 4});
            hipMemcpy(arr.mutable_data(), dd.geom_xquat,
                      (size_t)dd.num_envs * dd.ngeom * 4 * sizeof(float),
                      hipMemcpyDeviceToHost);
            return arr;
        })
        .def("read_xpos_d", [](const PhysicsData& d) {
            const auto& dd = d.device_data();
            py::array_t<double> arr({dd.num_envs, dd.nbody, 3});
            hipMemcpy(arr.mutable_data(), dd.xpos_d,
                      (size_t)dd.num_envs * dd.nbody * 3 * sizeof(double), hipMemcpyDeviceToHost);
            return arr;
        })
        .def("read_xquat_d", [](const PhysicsData& d) {
            const auto& dd = d.device_data();
            py::array_t<double> arr({dd.num_envs, dd.nbody, 4});
            hipMemcpy(arr.mutable_data(), dd.xquat_d,
                      (size_t)dd.num_envs * dd.nbody * 4 * sizeof(double), hipMemcpyDeviceToHost);
            return arr;
        })
        .def("read_xquat", [](const PhysicsData& d) {
            const auto& dd = d.device_data();
            py::array_t<float> arr({dd.num_envs, dd.nbody, 4});
            d.read_xquat(arr.mutable_data());
            return arr;
        })
        .def("read_site_xpos", [](const PhysicsData& d) {
            const auto& dd = d.device_data();
            if (dd.nsite <= 0) return py::array_t<float>();
            py::array_t<float> arr({dd.num_envs, dd.nsite, 3});
            d.read_site_xpos(arr.mutable_data());
            return arr;
        })
        .def("read_site_xmat", [](const PhysicsData& d) {
            const auto& dd = d.device_data();
            if (dd.nsite <= 0) return py::array_t<float>();
            py::array_t<float> arr({dd.num_envs, dd.nsite, 9});
            d.read_site_xmat(arr.mutable_data());
            return arr;
        })
        .def("write_qpos", [](PhysicsData& d, py::array_t<float> qpos) {
            d.write_qpos(static_cast<const float*>(qpos.request().ptr));
        }, py::arg("qpos"))
        .def("write_qpos_env", [](PhysicsData& d, py::array_t<float> qpos, int env_idx) {
            d.write_qpos_env(static_cast<const float*>(qpos.request().ptr), env_idx);
        }, py::arg("qpos"), py::arg("env_idx"))
        .def("write_qvel", [](PhysicsData& d, py::array_t<float> qvel) {
            d.write_qvel(static_cast<const float*>(qvel.request().ptr));
        }, py::arg("qvel"))
        .def("write_qvel_env", [](PhysicsData& d, py::array_t<float> qvel, int env_idx) {
            d.write_qvel_env(static_cast<const float*>(qvel.request().ptr), env_idx);
        }, py::arg("qvel"), py::arg("env_idx"))
        .def("read_qpos_d", [](const PhysicsData& d) {
            const auto& dd = d.device_data();
            py::array_t<double> arr({dd.num_envs, dd.nq});
            d.read_qpos_d(arr.mutable_data());
            return arr;
        })
        .def("read_qvel_d", [](const PhysicsData& d) {
            const auto& dd = d.device_data();
            py::array_t<double> arr({dd.num_envs, dd.nv});
            d.read_qvel_d(arr.mutable_data());
            return arr;
        })
        .def("write_qpos_d", [](PhysicsData& d, py::array_t<double> qpos) {
            d.write_qpos_d(static_cast<const double*>(qpos.request().ptr));
        }, py::arg("qpos"))
        .def("write_qpos_d_env", [](PhysicsData& d, py::array_t<double> qpos, int env_idx) {
            d.write_qpos_d_env(static_cast<const double*>(qpos.request().ptr), env_idx);
        }, py::arg("qpos"), py::arg("env_idx"))
        .def("write_qvel_d", [](PhysicsData& d, py::array_t<double> qvel) {
            d.write_qvel_d(static_cast<const double*>(qvel.request().ptr));
        }, py::arg("qvel"))
        .def("write_qvel_d_env", [](PhysicsData& d, py::array_t<double> qvel, int env_idx) {
            d.write_qvel_d_env(static_cast<const double*>(qvel.request().ptr), env_idx);
        }, py::arg("qvel"), py::arg("env_idx"))
        .def("write_qacc_warmstart_d", [](PhysicsData& d, py::array_t<double> qacc) {
            auto& dd = d.mutable_device_data();
            int total = dd.num_envs * dd.nv;
            hipMemcpy(dd.qacc_warmstart, static_cast<const double*>(qacc.request().ptr),
                      total * sizeof(double), hipMemcpyHostToDevice);
        }, py::arg("qacc"), "Write qacc_warmstart from host (shape: num_envs x nv)")
        .def("write_qacc_warmstart_d_env", [](PhysicsData& d, py::array_t<double> qacc, int env_idx) {
            auto& dd = d.mutable_device_data();
            int nv = dd.nv;
            hipMemcpy(dd.qacc_warmstart + env_idx * nv,
                      static_cast<const double*>(qacc.request().ptr),
                      nv * sizeof(double), hipMemcpyHostToDevice);
        }, py::arg("qacc"), py::arg("env_idx"),
           "Write qacc_warmstart for one env from host (shape: nv)")
        .def("read_qfrc_bias", [](const PhysicsData& d) {
            const auto& dd = d.device_data();
            py::array_t<double> arr({dd.num_envs, dd.nv});
            d.read_qfrc_bias(arr.mutable_data());
            return arr;
        })
        .def("read_qfrc_fluid", [](const PhysicsData& d) {
            const auto& dd = d.device_data();
            py::array_t<double> arr({dd.num_envs, dd.nv});
            if (dd.qfrc_fluid) {
                hipMemcpy(arr.mutable_data(), dd.qfrc_fluid,
                          (size_t)dd.num_envs * dd.nv * sizeof(double), hipMemcpyDeviceToHost);
            }
            return arr;
        })
        .def("read_qacc", [](const PhysicsData& d) {
            const auto& dd = d.device_data();
            int ne = dd.num_envs, nv = dd.nv;
            py::array_t<double> arr({ne, nv});
            hipMemcpy(arr.mutable_data(), dd.qacc,
                      (size_t)ne * nv * sizeof(double), hipMemcpyDeviceToHost);
            return arr;
        })
        .def("write_ctrl", [](PhysicsData& d, py::array_t<double> ctrl) {
            auto buf = ctrl.request();
            const double* src = static_cast<const double*>(buf.ptr);
            auto& dd = d.mutable_device_data();
            int total = dd.num_envs * dd.nu;
            hipMemcpy(dd.ctrl_d, src, total * sizeof(double), hipMemcpyHostToDevice);
            std::vector<float> h_f(total);
            for (int i = 0; i < total; i++) h_f[i] = (float)src[i];
            hipMemcpy(dd.ctrl, h_f.data(), total * sizeof(float), hipMemcpyHostToDevice);
        }, py::arg("ctrl"))
        .def("write_ctrl_env", [](PhysicsData& d, py::array_t<double> ctrl, int env_idx) {
            auto buf = ctrl.request();
            const double* src = static_cast<const double*>(buf.ptr);
            auto& dd = d.mutable_device_data();
            int nu = dd.nu;
            hipMemcpy(dd.ctrl_d + env_idx * nu, src, nu * sizeof(double), hipMemcpyHostToDevice);
            std::vector<float> h_f(nu);
            for (int i = 0; i < nu; i++) h_f[i] = (float)src[i];
            hipMemcpy(dd.ctrl + env_idx * nu, h_f.data(), nu * sizeof(float), hipMemcpyHostToDevice);
        }, py::arg("ctrl"), py::arg("env_idx"))
        .def("read_constraint_lam", [](const PhysicsData& d, int max_rows) -> py::tuple {
            // Solved constraint forces (first max_rows slots per world) and the per-world row
            // count -- the light subset of read_constraint_debug (which also copies J).
            hipDeviceSynchronize();
            const auto& dd = d.device_data();
            const int ne = dd.num_envs;
            const int rows = std::max(0, std::min(max_rows, (int)MAX_CONSTRAINTS));
            py::array_t<double> lam({ne, rows});
            py::array_t<int> ncons({ne});
            hipMemcpy(ncons.mutable_data(), dd.nconstraint, ne * sizeof(int), hipMemcpyDeviceToHost);
            if (rows > 0)
                hipMemcpy2D(lam.mutable_data(), rows * sizeof(double), dd.constraint_lam,
                            MAX_CONSTRAINTS * sizeof(double), rows * sizeof(double), ne,
                            hipMemcpyDeviceToHost);
            return py::make_tuple(lam, ncons);
        }, py::arg("max_rows") = 256)
        .def("read_constraint_debug", [](const PhysicsData& d) -> py::dict {
            hipDeviceSynchronize();
            const auto& dd = d.device_data();
            int ne = dd.num_envs;
            std::vector<int> h_ncon(ne);
            hipMemcpy(h_ncon.data(), dd.nconstraint, ne * sizeof(int), hipMemcpyDeviceToHost);

            int max_c = 0;
            for (int i = 0; i < ne; i++) max_c = std::max(max_c, h_ncon[i]);

            size_t D_size = (size_t)ne * MAX_CONSTRAINTS;
            std::vector<double> h_D(D_size, 0.0);
            std::vector<double> h_rhs(D_size, 0.0);
            hipMemcpy(h_D.data(), dd.constraint_D, D_size * sizeof(double), hipMemcpyDeviceToHost);
            hipMemcpy(h_rhs.data(), dd.constraint_rhs, D_size * sizeof(double), hipMemcpyDeviceToHost);

            size_t Minv_size = (size_t)ne * MAX_DOF;
            std::vector<double> h_Minv(Minv_size, 0.0);
            hipMemcpy(h_Minv.data(), dd.solver_M_inv_diag, Minv_size * sizeof(double), hipMemcpyDeviceToHost);

            py::dict result;
            py::array_t<int> ncon_arr(ne);
            memcpy(ncon_arr.mutable_data(), h_ncon.data(), ne * sizeof(int));
            result["nconstraint"] = ncon_arr;

            py::array_t<double> D_arr({ne, (int)MAX_CONSTRAINTS});
            memcpy(D_arr.mutable_data(), h_D.data(), D_size * sizeof(double));
            result["constraint_D"] = D_arr;

            py::array_t<double> rhs_arr({ne, (int)MAX_CONSTRAINTS});
            memcpy(rhs_arr.mutable_data(), h_rhs.data(), D_size * sizeof(double));
            result["constraint_rhs"] = rhs_arr;

            py::array_t<double> Minv_arr({ne, (int)MAX_DOF});
            memcpy(Minv_arr.mutable_data(), h_Minv.data(), Minv_size * sizeof(double));
            result["M_inv_diag"] = Minv_arr;

            int nv = dd.nv;

            size_t M_size = (size_t)ne * KERNEL_MAXNV * KERNEL_MAXNV;
            std::vector<double> h_M(M_size, 0.0);
            hipMemcpy(h_M.data(), dd.solver_M, M_size * sizeof(double), hipMemcpyDeviceToHost);
            py::array_t<double> M_arr({ne, (int)KERNEL_MAXNV, (int)KERNEL_MAXNV});
            memcpy(M_arr.mutable_data(), h_M.data(), M_size * sizeof(double));
            result["solver_M"] = M_arr;
            size_t J_size = (size_t)ne * MAX_CONSTRAINTS * nv;
            std::vector<double> h_J(J_size, 0.0);
            hipMemcpy(h_J.data(), dd.constraint_J, J_size * sizeof(double), hipMemcpyDeviceToHost);
            py::array_t<double> J_arr({ne, (int)MAX_CONSTRAINTS, nv});
            memcpy(J_arr.mutable_data(), h_J.data(), J_size * sizeof(double));
            result["constraint_J"] = J_arr;

            std::vector<double> h_lo(D_size, 0.0);
            std::vector<double> h_hi(D_size, 0.0);
            std::vector<double> h_lam(D_size, 0.0);
            std::vector<int> h_fp(D_size, -1);
            std::vector<int> h_fs(D_size, -1);
            std::vector<double> h_fmu(D_size, 0.0);
            hipMemcpy(h_lo.data(), dd.constraint_lo, D_size * sizeof(double), hipMemcpyDeviceToHost);
            hipMemcpy(h_hi.data(), dd.constraint_hi, D_size * sizeof(double), hipMemcpyDeviceToHost);
            hipMemcpy(h_lam.data(), dd.constraint_lam, D_size * sizeof(double), hipMemcpyDeviceToHost);
            hipMemcpy(h_fp.data(), dd.constraint_friction_parent, D_size * sizeof(int), hipMemcpyDeviceToHost);
            hipMemcpy(h_fs.data(), dd.constraint_friction_sibling, D_size * sizeof(int), hipMemcpyDeviceToHost);
            hipMemcpy(h_fmu.data(), dd.constraint_friction_mu, D_size * sizeof(double), hipMemcpyDeviceToHost);

            py::array_t<double> lo_arr({ne, (int)MAX_CONSTRAINTS});
            memcpy(lo_arr.mutable_data(), h_lo.data(), D_size * sizeof(double));
            result["constraint_lo"] = lo_arr;

            py::array_t<double> hi_arr({ne, (int)MAX_CONSTRAINTS});
            memcpy(hi_arr.mutable_data(), h_hi.data(), D_size * sizeof(double));
            result["constraint_hi"] = hi_arr;

            py::array_t<double> lam_arr({ne, (int)MAX_CONSTRAINTS});
            memcpy(lam_arr.mutable_data(), h_lam.data(), D_size * sizeof(double));
            result["constraint_lam"] = lam_arr;

            py::array_t<int> fp_arr({ne, (int)MAX_CONSTRAINTS});
            memcpy(fp_arr.mutable_data(), h_fp.data(), D_size * sizeof(int));
            result["constraint_friction_parent"] = fp_arr;

            py::array_t<int> fs_arr({ne, (int)MAX_CONSTRAINTS});
            memcpy(fs_arr.mutable_data(), h_fs.data(), D_size * sizeof(int));
            result["constraint_friction_sibling"] = fs_arr;

            py::array_t<double> fmu_arr({ne, (int)MAX_CONSTRAINTS});
            memcpy(fmu_arr.mutable_data(), h_fmu.data(), D_size * sizeof(double));
            result["constraint_friction_mu"] = fmu_arr;

            size_t nv_size = (size_t)ne * nv;
            std::vector<double> h_bias(nv_size, 0.0);
            std::vector<double> h_qacc(nv_size, 0.0);
            hipMemcpy(h_bias.data(), dd.qfrc_bias, nv_size * sizeof(double), hipMemcpyDeviceToHost);
            hipMemcpy(h_qacc.data(), dd.qacc, nv_size * sizeof(double), hipMemcpyDeviceToHost);
            py::array_t<double> bias_arr({ne, nv});
            memcpy(bias_arr.mutable_data(), h_bias.data(), nv_size * sizeof(double));
            result["qfrc_bias"] = bias_arr;
            py::array_t<double> qacc_arr({ne, nv});
            memcpy(qacc_arr.mutable_data(), h_qacc.data(), nv_size * sizeof(double));
            result["qacc"] = qacc_arr;

            return result;
        })
        .def("read_contacts", [](const PhysicsData& d) -> py::dict {
            const auto& dd = d.device_data();
            int ne = dd.num_envs;
            std::vector<int> h_ncon(ne);
            hipMemcpy(h_ncon.data(), dd.ncon, ne * sizeof(int), hipMemcpyDeviceToHost);

            int max_c = 0;
            for (int i = 0; i < ne; i++) max_c = std::max(max_c, h_ncon[i]);
            if (max_c > MAX_CONTACTS) max_c = MAX_CONTACTS;

            size_t total = (size_t)ne * MAX_CONTACTS;
            std::vector<ContactPair> h_con(total);
            hipMemcpy(h_con.data(), dd.contacts, total * sizeof(ContactPair), hipMemcpyDeviceToHost);

            py::dict result;
            py::array_t<int> ncon_arr(ne);
            memcpy(ncon_arr.mutable_data(), h_ncon.data(), ne * sizeof(int));
            result["ncon"] = ncon_arr;

            py::array_t<int> geom1({ne, max_c}), geom2({ne, max_c});
            py::array_t<int> body1({ne, max_c}), body2({ne, max_c});
            py::array_t<double> pos({ne, max_c, 3}), nrm({ne, max_c, 3});
            py::array_t<double> depth({ne, max_c}), friction({ne, max_c});
            auto g1 = geom1.mutable_unchecked<2>();
            auto g2 = geom2.mutable_unchecked<2>();
            auto b1 = body1.mutable_unchecked<2>();
            auto b2 = body2.mutable_unchecked<2>();
            auto pp = pos.mutable_unchecked<3>();
            auto nn = nrm.mutable_unchecked<3>();
            auto dd_arr = depth.mutable_unchecked<2>();
            auto ff = friction.mutable_unchecked<2>();
            for (int e = 0; e < ne; e++) {
                int nc = std::min(h_ncon[e], max_c);
                for (int c = 0; c < nc; c++) {
                    const auto& cp = h_con[e * MAX_CONTACTS + c];
                    g1(e,c) = cp.geom1; g2(e,c) = cp.geom2;
                    b1(e,c) = cp.body1; b2(e,c) = cp.body2;
                    for (int k = 0; k < 3; k++) { pp(e,c,k) = cp.pos[k]; nn(e,c,k) = cp.normal[k]; }
                    dd_arr(e,c) = cp.depth; ff(e,c) = cp.friction;
                }
            }
            result["geom1"] = geom1; result["geom2"] = geom2;
            result["body1"] = body1; result["body2"] = body2;
            result["pos"] = pos; result["normal"] = nrm;
            result["depth"] = depth; result["friction"] = friction;
            return result;
        });

    // ── PhysicsEngine ───────────────────────────────────────────────────────
    py::class_<PhysicsEngine>(m, "PhysicsEngine")
        .def("set_mujoco_convex", [](PhysicsEngine& e, py::dict d, int queue_cap_per_env, int workers) {
            // Route box-mesh / mesh-mesh collisions through the MuJoCo 3.9 mjc_Convex port
            // (physics_mjccd.h), using MuJoCo's own mesh data and geom local poses.
            DevicePhysicsModel& dm = e.model()->mutable_device_model();
            DevicePhysicsData& dd = e.data().mutable_device_data();
            auto up_i = [&](const char* key, int*& dst, py::ssize_t expect) {
                auto a = d[key].cast<py::array_t<int, py::array::c_style | py::array::forcecast>>();
                if (expect >= 0 && a.size() != expect)
                    throw std::runtime_error(std::string("set_mujoco_convex: bad size for ") + key);
                if (dst) hipFree(dst);
                size_t nbytes = std::max<size_t>(1, (size_t)a.size()) * sizeof(int);
                if (hipMalloc(&dst, nbytes) != hipSuccess) throw std::runtime_error("hipMalloc failed");
                if (a.size()) hipMemcpy(dst, a.data(), a.size() * sizeof(int), hipMemcpyHostToDevice);
            };
            auto up_d = [&](const char* key, double*& dst, py::ssize_t expect) {
                auto a = d[key].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
                if (expect >= 0 && a.size() != expect)
                    throw std::runtime_error(std::string("set_mujoco_convex: bad size for ") + key);
                if (dst) hipFree(dst);
                size_t nbytes = std::max<size_t>(1, (size_t)a.size()) * sizeof(double);
                if (hipMalloc(&dst, nbytes) != hipSuccess) throw std::runtime_error("hipMalloc failed");
                if (a.size()) hipMemcpy(dst, a.data(), a.size() * sizeof(double), hipMemcpyHostToDevice);
            };
            auto up_f = [&](const char* key, float*& dst) {
                auto a = d[key].cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
                if (dst) hipFree(dst);
                size_t nbytes = std::max<size_t>(1, (size_t)a.size()) * sizeof(float);
                if (hipMalloc(&dst, nbytes) != hipSuccess) throw std::runtime_error("hipMalloc failed");
                if (a.size()) hipMemcpy(dst, a.data(), a.size() * sizeof(float), hipMemcpyHostToDevice);
            };
            const int ng = dm.ngeom;
            up_i("geom_mj_mesh", dm.geom_mj_mesh, ng);
            up_d("geom_mj_pos", dm.geom_mj_pos, ng * 3);
            up_d("geom_mj_quat", dm.geom_mj_quat, ng * 4);
            up_d("geom_mj_size", dm.geom_mj_size, ng * 3);
            up_d("geom_mj_rbound", dm.geom_mj_rbound, ng);
            up_i("mesh_vertadr", dm.mj_mesh_vertadr, -1);  up_i("mesh_vertnum", dm.mj_mesh_vertnum, -1);
            up_f("mesh_vert", dm.mj_mesh_vert);
            up_i("mesh_graphadr", dm.mj_mesh_graphadr, -1); up_i("mesh_graph", dm.mj_mesh_graph, -1);
            up_i("mesh_polynum", dm.mj_mesh_polynum, -1);   up_i("mesh_polyadr", dm.mj_mesh_polyadr, -1);
            up_d("mesh_polynormal", dm.mj_mesh_polynormal, -1);
            up_i("mesh_polyvertadr", dm.mj_mesh_polyvertadr, -1); up_i("mesh_polyvertnum", dm.mj_mesh_polyvertnum, -1);
            up_i("mesh_polyvert", dm.mj_mesh_polyvert, -1);
            up_i("mesh_polymapadr", dm.mj_mesh_polymapadr, -1); up_i("mesh_polymapnum", dm.mj_mesh_polymapnum, -1);
            up_i("mesh_polymap", dm.mj_mesh_polymap, -1);
            dm.mjc_ccd_iterations = d["ccd_iterations"].cast<int>();
            dm.mjc_ccd_tolerance = d["ccd_tolerance"].cast<double>();
            dm.mjc_disable_nativeccd = d["disable_nativeccd"].cast<int>();
            dm.mjc_disable_multiccd = d["disable_multiccd"].cast<int>();
            if (dm.mjc_ccd_iterations < 1 || dm.mjc_ccd_iterations > MJCCD_MAX_ITERATIONS)
                throw std::runtime_error("set_mujoco_convex: ccd_iterations outside [1, MJCCD_MAX_ITERATIONS]; rebuild with a larger MJCCD_MAX_ITERATIONS");
            // queue + per-worker scratch
            if (dd.mjc_queue) hipFree(dd.mjc_queue);
            if (dd.mjc_queue_count) hipFree(dd.mjc_queue_count);
            if (dd.mjc_scratch) hipFree(dd.mjc_scratch);
            if (dd.mjc_fallback_count) hipFree(dd.mjc_fallback_count);
            dd.mjc_queue_cap = dd.num_envs * queue_cap_per_env;
            if (const char* ev = std::getenv("HP_MJC_WORKERS")) { int v = std::atoi(ev); if (v > 0) workers = v; }
            dd.mjc_workers = workers;
            const MjcAuxLayout aux = mjc_aux_layout(workers, dd.mjc_queue_cap);
            if (hipMalloc(&dd.mjc_queue, (size_t)dd.mjc_queue_cap * 3 * sizeof(int)) != hipSuccess ||
                hipMalloc(&dd.mjc_queue_count, sizeof(int)) != hipSuccess ||
                hipMalloc(&dd.mjc_fallback_count, sizeof(int)) != hipSuccess ||
                hipMalloc(&dd.mjc_scratch, aux.total) != hipSuccess)
                throw std::runtime_error("set_mujoco_convex: queue/scratch allocation failed");
            hipMemset(dd.mjc_queue_count, 0, sizeof(int));
            hipMemset(dd.mjc_fallback_count, 0, sizeof(int));
            hipMemset((char*)dd.mjc_scratch + aux.n_off, 0, aux.total - aux.n_off);
            dm.mjc_enabled = 1;
            return (size_t)workers * MJC_FAN_WAVES * sizeof(MjcWorkerScratch);
        }, py::arg("data"), py::arg("queue_cap_per_env") = 256, py::arg("workers") = 512,
           "Use MuJoCo's convex collision (mjc_Convex port) for box-mesh and mesh-mesh pairs. "
           "Returns the scratch bytes allocated.")
        .def("mujoco_convex_queue_count", [](PhysicsEngine& e) {
            // Debug: queued box-mesh/mesh-mesh pairs of the LAST collision pass (mjc_queue_count).
            const DevicePhysicsData& dd = e.data().device_data();
            int n = 0;
            if (dd.mjc_queue_count) { hipDeviceSynchronize(); hipMemcpy(&n, dd.mjc_queue_count, sizeof(int), hipMemcpyDeviceToHost); }
            return n;
        })
        .def("mujoco_convex_queue_hist", [](PhysicsEngine& e, bool reset) {
            // Debug: histogram of queued pairs per collision pass (bin = count, last bin = overflow).
            // Only filled when HP_MJC_QHIST=1 was set when the collision graph was captured.
            const DevicePhysicsData& dd = e.data().device_data();
            std::vector<int> h(MJC_QHIST_BINS, 0);
            if (!dd.mjc_scratch) return h;
            const MjcAuxLayout aux = mjc_aux_layout(dd.mjc_workers, dd.mjc_queue_cap);
            hipDeviceSynchronize();
            hipMemcpy(h.data(), (char*)dd.mjc_scratch + aux.h_off, MJC_QHIST_BINS * sizeof(int), hipMemcpyDeviceToHost);
            if (reset) hipMemset((char*)dd.mjc_scratch + aux.h_off, 0, MJC_QHIST_BINS * sizeof(int));
            return h;
        }, py::arg("reset") = true)
        .def("mujoco_convex_recomputes", [](PhysicsEngine& e) {
            // Debug: multiCCD fan-out reruns recomputed serially (warm start not proven equivalent).
            const DevicePhysicsData& dd = e.data().device_data();
            int n = 0;
            if (!dd.mjc_scratch) return n;
            const MjcAuxLayout aux = mjc_aux_layout(dd.mjc_workers, dd.mjc_queue_cap);
            hipDeviceSynchronize();
            hipMemcpy(&n, (char*)dd.mjc_scratch + aux.r_off, sizeof(int), hipMemcpyDeviceToHost);
            return n;
        })
        .def("mujoco_convex_fallbacks", [](PhysicsEngine& e) {
            const DevicePhysicsData& dd = e.data().device_data();
            int n = 0;
            if (dd.mjc_fallback_count) hipMemcpy(&n, dd.mjc_fallback_count, sizeof(int), hipMemcpyDeviceToHost);
            return n;
        })
        .def(py::init<>())
        .def("init", &PhysicsEngine::init, py::arg("model"), py::arg("num_envs"), py::arg("gpu_id") = 0)
        .def("step", &PhysicsEngine::step)
        .def("step_n", &PhysicsEngine::step_n, py::arg("n"))
        .def("set_joint_pd", [](PhysicsEngine& e, py::dict d) {
            // GPU joint-position PD controller (joint_pd.h); keys: act, dof, qadr, grp_start, grp_len (int),
            // kp, kd, gear, lo, hi (float), all of length n
            auto I = [&](const char* k) { return d[k].cast<py::array_t<int, py::array::c_style | py::array::forcecast>>(); };
            auto D = [&](const char* k) { return d[k].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(); };
            auto act = I("act"), dof = I("dof"), qadr = I("qadr"), gs = I("grp_start"), gl = I("grp_len");
            auto kp = D("kp"), kd = D("kd"), gear = D("gear"), lo = D("lo"), hi = D("hi");
            const int n = (int)act.size();
            for (auto* a : {&dof, &qadr, &gs, &gl}) if ((int)a->size() != n) throw std::runtime_error("set_joint_pd: length mismatch");
            for (auto* a : {&kp, &kd, &gear, &lo, &hi}) if ((int)a->size() != n) throw std::runtime_error("set_joint_pd: length mismatch");
            e.set_joint_pd(n, act.data(), dof.data(), qadr.data(), gs.data(), gl.data(), kp.data(), kd.data(),
                           gear.data(), lo.data(), hi.data());
        }, py::arg("config"))
        .def("set_joint_pd_targets", [](PhysicsEngine& e, py::array_t<double, py::array::c_style | py::array::forcecast> t) {
            if ((int)t.size() != e.data().device_data().num_envs * e.joint_pd_n())
                throw std::runtime_error("set_joint_pd_targets: expected num_envs * n values");
            e.set_joint_pd_targets(t.data());
        }, py::arg("targets"))
        .def("forward", &PhysicsEngine::forward)
        .def("forward_kinematics", &PhysicsEngine::forward_kinematics)
        .def("geom_transforms", &PhysicsEngine::geom_transforms)
        .def("site_transforms", &PhysicsEngine::site_transforms,
             "Compute site world positions/orientations from body transforms (lightweight)")
        .def("step_and_sites", &PhysicsEngine::step_and_sites,
             "Run one physics step + site transforms (no extra FK/geom overhead)")
        .def("sync", &PhysicsEngine::sync)
        .def("query", [](PhysicsEngine& e) {
            return hipStreamQuery(e.stream()) == hipSuccess;
        }, "True once all work queued on the physics stream has completed (non-blocking).")
        .def("read_state", [](PhysicsEngine& e) {
            const auto& dd = e.data().device_data();
            hipStream_t s = e.stream();
            py::array_t<double> q({dd.num_envs, dd.nq});
            py::array_t<double> xp({dd.num_envs, dd.nbody, 3});
            py::array_t<double> xq({dd.num_envs, dd.nbody, 4});
            hipMemcpyAsync(q.mutable_data(), dd.qpos_d, (size_t)dd.num_envs * dd.nq * sizeof(double),
                           hipMemcpyDeviceToHost, s);
            hipMemcpyAsync(xp.mutable_data(), dd.xpos_d, (size_t)dd.num_envs * dd.nbody * 3 * sizeof(double),
                           hipMemcpyDeviceToHost, s);
            hipMemcpyAsync(xq.mutable_data(), dd.xquat_d, (size_t)dd.num_envs * dd.nbody * 4 * sizeof(double),
                           hipMemcpyDeviceToHost, s);
            hipError_t err = hipStreamSynchronize(s);
            if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
            return py::make_tuple(q, xp, xq);
        }, "sync this engine's stream, then (qpos_d [W,nq], xpos_d [W,nbody,3], xquat_d [W,nbody,4]).")
        .def("write_ctrl_stream", [](PhysicsEngine& e, py::array_t<double, py::array::c_style | py::array::forcecast> ctrl) {
            auto& dd = e.data().mutable_device_data();
            const int total = dd.num_envs * dd.nu;
            if (ctrl.size() != total) throw std::runtime_error("write_ctrl_stream: bad size");
            hipStream_t s = e.stream();
            const double* src = ctrl.data();
            std::vector<float> h_f(total);
            for (int i = 0; i < total; i++) h_f[i] = (float)src[i];
            hipMemcpyAsync(dd.ctrl_d, src, total * sizeof(double), hipMemcpyHostToDevice, s);
            hipMemcpyAsync(dd.ctrl, h_f.data(), total * sizeof(float), hipMemcpyHostToDevice, s);
            hipError_t err = hipStreamSynchronize(s);
            if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
        }, py::arg("ctrl"), "PhysicsData.write_ctrl ordered on this engine's stream.")
        .def("read_contacts_lam", [](PhysicsEngine& e, int stride) -> py::tuple {
            // read_contacts + read_constraint_lam(stride * max ncon) in one stream-ordered pass,
            // copying only the first max-ncon contact rows of every world.
            const auto& dd = e.data().device_data();
            hipStream_t s = e.stream();
            const int ne = dd.num_envs;
            std::vector<int> h_ncon(ne);
            hipMemcpyAsync(h_ncon.data(), dd.ncon, ne * sizeof(int), hipMemcpyDeviceToHost, s);
            hipError_t err = hipStreamSynchronize(s);
            if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
            int max_c = 0;
            for (int i = 0; i < ne; i++) max_c = std::max(max_c, h_ncon[i]);
            if (max_c > MAX_CONTACTS) max_c = MAX_CONTACTS;
            ContactPair* h_con = static_cast<ContactPair*>(
                g_stage_con.get((size_t)ne * std::max(max_c, 1) * sizeof(ContactPair)));
            if (max_c > 0)
                hipMemcpy2DAsync(h_con, max_c * sizeof(ContactPair), dd.contacts,
                                 MAX_CONTACTS * sizeof(ContactPair), max_c * sizeof(ContactPair), ne,
                                 hipMemcpyDeviceToHost, s);
            const int rows = std::max(0, std::min(std::max(stride * max_c, 1), (int)MAX_CONSTRAINTS));
            py::array_t<double> lam({ne, rows});
            py::array_t<int> ncons({ne});
            hipMemcpyAsync(ncons.mutable_data(), dd.nconstraint, ne * sizeof(int), hipMemcpyDeviceToHost, s);
            double* h_lam = static_cast<double*>(g_stage_lam.get((size_t)ne * std::max(rows, 1) * sizeof(double)));
            if (rows > 0)
                hipMemcpy2DAsync(h_lam, rows * sizeof(double), dd.constraint_lam,
                                 MAX_CONSTRAINTS * sizeof(double), rows * sizeof(double), ne,
                                 hipMemcpyDeviceToHost, s);
            err = hipStreamSynchronize(s);
            if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
            if (rows > 0) memcpy(lam.mutable_data(), h_lam, (size_t)ne * rows * sizeof(double));

            py::dict result;
            py::array_t<int> ncon_arr(ne);
            memcpy(ncon_arr.mutable_data(), h_ncon.data(), ne * sizeof(int));
            result["ncon"] = ncon_arr;
            py::array_t<int> geom1({ne, max_c}), geom2({ne, max_c});
            py::array_t<int> body1({ne, max_c}), body2({ne, max_c});
            py::array_t<double> pos({ne, max_c, 3}), nrm({ne, max_c, 3});
            py::array_t<double> depth({ne, max_c}), friction({ne, max_c});
            auto g1 = geom1.mutable_unchecked<2>();
            auto g2 = geom2.mutable_unchecked<2>();
            auto b1 = body1.mutable_unchecked<2>();
            auto b2 = body2.mutable_unchecked<2>();
            auto pp = pos.mutable_unchecked<3>();
            auto nn = nrm.mutable_unchecked<3>();
            auto dd_arr = depth.mutable_unchecked<2>();
            auto ff = friction.mutable_unchecked<2>();
            for (int w = 0; w < ne; w++) {
                const int nc = std::min(h_ncon[w], max_c);
                for (int c = 0; c < max_c; c++) {
                    if (c >= nc) {   // same zero fill as a fresh numpy row is not guaranteed: clear it
                        g1(w,c) = g2(w,c) = b1(w,c) = b2(w,c) = 0;
                        for (int k = 0; k < 3; k++) { pp(w,c,k) = 0.0; nn(w,c,k) = 0.0; }
                        dd_arr(w,c) = 0.0; ff(w,c) = 0.0;
                        continue;
                    }
                    const auto& cp = h_con[(size_t)w * max_c + c];
                    g1(w,c) = cp.geom1; g2(w,c) = cp.geom2;
                    b1(w,c) = cp.body1; b2(w,c) = cp.body2;
                    for (int k = 0; k < 3; k++) { pp(w,c,k) = cp.pos[k]; nn(w,c,k) = cp.normal[k]; }
                    dd_arr(w,c) = cp.depth; ff(w,c) = cp.friction;
                }
            }
            result["geom1"] = geom1; result["geom2"] = geom2;
            result["body1"] = body1; result["body2"] = body2;
            result["pos"] = pos; result["normal"] = nrm;
            result["depth"] = depth; result["friction"] = friction;
            return py::make_tuple(result, lam, ncons);
        }, py::arg("stride"))
        .def("begin_lagged_render", &PhysicsEngine::begin_lagged_render, py::arg("dt_back"),
             "Point the live body/geom/site pose buffers at the state dt_back before the final one "
             "(robosuite's camera timing); pair with end_lagged_render().")
        .def("end_lagged_render", &PhysicsEngine::end_lagged_render,
             "Recompute the live pose buffers from the true qpos after a lagged render.")
        .def("lagged_kinematics", &PhysicsEngine::lagged_kinematics, py::arg("dt_back"),
             "FK + sites of the state stepped back by dt_back along qvel (mj_integratePos), on the "
             "GPU into separate buffers; the live state is untouched. Read with read_lagged_*.")
        .def("read_lagged_qpos", [](const PhysicsEngine& e) {
            const auto& dd = e.data().device_data();
            py::array_t<double> arr({dd.num_envs, dd.nq});
            e.read_lagged_qpos(arr.mutable_data());
            return arr;
        })
        .def("read_lagged_xpos", [](const PhysicsEngine& e) {
            const auto& dd = e.data().device_data();
            py::array_t<double> arr({dd.num_envs, dd.nbody, 3});
            e.read_lagged_xpos(arr.mutable_data());
            return arr;
        })
        .def("read_lagged_xquat", [](const PhysicsEngine& e) {
            const auto& dd = e.data().device_data();
            py::array_t<double> arr({dd.num_envs, dd.nbody, 4});
            e.read_lagged_xquat(arr.mutable_data());
            return arr;
        })
        .def("read_lagged_site_xpos", [](const PhysicsEngine& e) {
            const auto& dd = e.data().device_data();
            if (dd.nsite <= 0) return py::array_t<float>();
            py::array_t<float> arr({dd.num_envs, dd.nsite, 3});
            e.read_lagged_site_xpos(arr.mutable_data());
            return arr;
        })
        .def("step_profile", [](PhysicsEngine& e) -> py::dict {
            auto t = e.step_profile();
            py::dict d;
            d["fk_ms"] = t.fk_ms;
            d["geom_ms"] = t.geom_ms;
            d["collision_ms"] = t.collision_ms;
            d["constraint_ms"] = t.constraint_ms;
            d["solve_ms"] = t.solve_ms;
            d["integrate_ms"] = t.integrate_ms;
            d["total_ms"] = t.fk_ms + t.geom_ms + t.collision_ms + t.constraint_ms + t.solve_ms + t.integrate_ms;
            return d;
        }, "Run one profiled step, returns dict of per-phase timing in ms")
        .def_property("options",
             [](PhysicsEngine& e) -> SolverOptions& { return e.options(); },
             [](PhysicsEngine& e, const SolverOptions& o) { e.options() = o; })
        .def_property_readonly("data",
             [](PhysicsEngine& e) -> PhysicsData& { return e.data(); },
             py::return_value_policy::reference_internal)
        .def("reset", [](PhysicsEngine& e) {
            if (e.model()) e.data().reset(e.model()->desc());
        })
        .def("reset_env", [](PhysicsEngine& e, int env_idx) {
            if (e.model()) e.data().reset_env(e.model()->desc(), env_idx);
        }, py::arg("env_idx"))
        .def("invalidate_osc_graphs", &PhysicsEngine::invalidate_osc_graphs,
             "Invalidate cached HIP graphs; re-captured on next step_with_osc()")
        // ── GPU OSC ─────────────────────────────────────────────────────────
        .def("set_osc_config", [](PhysicsEngine& e, py::dict cfg) {
            OSCConfig c = {};
            auto arm_ji = cfg["arm_joint_indices"].cast<std::vector<int>>();
            auto arm_di = cfg["arm_dof_indices"].cast<std::vector<int>>();
            auto arm_qi = cfg["arm_qpos_indices"].cast<std::vector<int>>();
            auto arm_ai = cfg["arm_actuator_indices"].cast<std::vector<int>>();
            c.n_arm = (int)arm_ji.size();
            if (c.n_arm <= 0 || c.n_arm > OSC_MAX_ARM || arm_di.size() != arm_ji.size() ||
                arm_qi.size() != arm_ji.size() || arm_ai.size() != arm_ji.size())
                throw py::value_error("OSC arm index arrays must have matching lengths in [1, OSC_MAX_ARM]");
            for (int i = 0; i < c.n_arm; i++) {
                c.arm_joint_indices[i]    = arm_ji[i];
                c.arm_dof_indices[i]      = arm_di[i];
                c.arm_qpos_indices[i]     = arm_qi[i];
                c.arm_actuator_indices[i] = arm_ai[i];
            }
            if (cfg.contains("gripper_actuator_indices")) {
                auto grip = cfg["gripper_actuator_indices"].cast<std::vector<int>>();
                c.n_gripper = (int)grip.size();
                if (c.n_gripper > OSC_MAX_GRIP)
                    throw py::value_error("Too many OSC gripper actuators");
                for (int i = 0; i < c.n_gripper && i < OSC_MAX_GRIP; i++)
                    c.gripper_actuator_indices[i] = grip[i];
            }
            c.eef_site_idx = cfg.contains("eef_site_idx") ? cfg["eef_site_idx"].cast<int>() : -1;
            c.eef_body_idx = cfg.contains("eef_body_idx") ? cfg["eef_body_idx"].cast<int>() : 0;
            c.kp_pos = cfg.contains("kp_pos") ? cfg["kp_pos"].cast<double>() : 150.0;
            c.kp_ori = cfg.contains("kp_ori") ? cfg["kp_ori"].cast<double>() : 150.0;
            c.kd_pos = cfg.contains("kd_pos") ? cfg["kd_pos"].cast<double>() : 2.0*sqrt(150.0);
            c.kd_ori = cfg.contains("kd_ori") ? cfg["kd_ori"].cast<double>() : 2.0*sqrt(150.0);
            c.kp_null = cfg.contains("kp_null") ? cfg["kp_null"].cast<double>() : 10.0;
            c.kd_null = cfg.contains("kd_null") ? cfg["kd_null"].cast<double>() : 2.0*sqrt(10.0);
            c.action_scale_pos = cfg.contains("action_scale_pos") ? cfg["action_scale_pos"].cast<double>() : 0.05;
            c.action_scale_ori = cfg.contains("action_scale_ori") ? cfg["action_scale_ori"].cast<double>() : 0.5;
            if (cfg.contains("arm_ctrl_limits")) {
                auto lims = cfg["arm_ctrl_limits"].cast<std::vector<double>>();
                for (int i = 0; i < c.n_arm && i < (int)lims.size(); i++)
                    c.arm_ctrl_limits[i] = lims[i];
            } else {
                for (int i = 0; i < c.n_arm; i++) c.arm_ctrl_limits[i] = 80.0;
            }
            c.gripper_speed = cfg.contains("gripper_speed") ? cfg["gripper_speed"].cast<double>() : 0.01;
            c.has_gripper_ctrlrange = 0;
            if (cfg.contains("gripper_ctrlrange")) {
                auto gcr = cfg["gripper_ctrlrange"].cast<std::vector<double>>();
                if (!gcr.empty()) {
                    c.has_gripper_ctrlrange = 1;
                    for (int i = 0; i < (int)gcr.size() && i < OSC_MAX_GRIP*2; i++)
                        c.gripper_ctrlrange[i] = gcr[i];
                }
            }
            e.set_osc_config(c);
        }, py::arg("config"),
           "Configure GPU-side OSC controller. Pass dict with arm_joint_indices, "
           "arm_dof_indices, arm_qpos_indices, arm_actuator_indices, etc.")
        .def("set_osc_actions", [](PhysicsEngine& e,
                                    py::array_t<float, py::array::c_style> actions) {
            auto buf = actions.request();
            int ne = (int)buf.shape[0];
            e.set_osc_actions((const float*)buf.ptr, ne);
        }, py::arg("actions"),
           "Upload per-env OSC actions [num_envs, 7] (pos_delta[3], ori_delta[3], gripper)")
        .def("step_with_osc", &PhysicsEngine::step_with_osc, py::arg("n_substeps"),
             "Run n substeps with GPU-side OSC (zero host sync)")
        .def("set_world_active", [](PhysicsEngine& e,
                                    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> mask) {
            auto buf = mask.request();
            e.set_world_active((const unsigned char*)buf.ptr, (int)buf.size);
        }, py::arg("mask"),
           "world skip: freeze worlds (mask [num_envs] uint8, nonzero = step) for the next step_with_osc() calls")
        .def("clear_world_active", &PhysicsEngine::clear_world_active,
             "world skip: every world steps again (unmasked kernels)")
        .def("world_mask_on", &PhysicsEngine::world_mask_on)
        .def("reset_osc_state", &PhysicsEngine::reset_osc_state,
             "Reset all per-env OSC goal/gripper state")
        .def("reset_osc_env_state", &PhysicsEngine::reset_osc_env_state,
             py::arg("env_idx"), "Reset OSC state for one env")
        .def("initialize_osc_env_state", [](PhysicsEngine& e, int env_idx, py::dict state) {
            const int na = e.osc_config().n_arm;
            if (na <= 0 || na > OSC_MAX_ARM)
                throw py::value_error("OSC config has an invalid arm dimension");
            OSCEnvState imported = {};
            imported.initialized = 1;
            copy_osc_reset_array(state, "home", imported.init_jpos, na);
            copy_osc_reset_array(state, "goal_pos", imported.goal_pos, 3);
            copy_osc_reset_array(state, "goal_ori", imported.goal_ori, 3, 3);
            copy_osc_reset_array(state, "gripper", imported.gripper_current_action, OSC_MAX_GRIP);
            if (state.contains("cache") && !state["cache"].is_none()) {
                if (!py::isinstance<py::dict>(state["cache"]))
                    throw py::value_error("OSC reset cache must be a dictionary or None");
                const auto cache = state["cache"].cast<py::dict>();
                copy_osc_reset_array(cache, "ee_pos", imported.cached_ee_pos, 3);
                copy_osc_reset_array(cache, "ee_ori", imported.cached_ee_ori, 3, 3);
                copy_osc_reset_array(cache, "ee_vel", imported.cached_ee_vel, 6);
                copy_osc_reset_array(cache, "joint_pos", imported.cached_joint_pos, na);
                copy_osc_reset_array(cache, "joint_vel", imported.cached_joint_vel, na);
                copy_osc_reset_array(cache, "J_full", imported.cached_J_full, 6, na, OSC_MAX_ARM);
                copy_osc_reset_array(cache, "mass_matrix", imported.cached_mass_matrix, na, na, OSC_MAX_ARM);
                imported.first_goal_cache_pending = 1;
                imported.first_tick_cache_pending = 1;
            }
            e.initialize_osc_env_state(env_idx, imported);
        }, py::arg("env_idx"), py::arg("state"),
           "Import one OSC reset without integration. Required: home[n_arm], goal_pos[3], "
           "goal_ori[3,3], gripper[2]. Optional cache: ee_pos[3], ee_ori[3,3], ee_vel[6], "
           "joint_pos/joint_vel[n_arm], J_full[6,n_arm], mass_matrix[n_arm,n_arm]. "
           "Cache affects only the first goal and first torque tick; bias stays current.")
        .def("read_osc_env_state", [](const PhysicsEngine& e, int env_idx) {
            const int na = e.osc_config().n_arm;
            if (na <= 0 || na > OSC_MAX_ARM)
                throw py::value_error("OSC config has an invalid arm dimension");
            const OSCEnvState state = e.read_osc_env_state(env_idx);
            py::dict result, cache;
            result["home"] = osc_state_array(state.init_jpos, na);
            result["goal_pos"] = osc_state_array(state.goal_pos, 3);
            result["goal_ori"] = osc_state_array(state.goal_ori, 3, 3);
            result["gripper"] = osc_state_array(state.gripper_current_action, OSC_MAX_GRIP);
            result["gripper_direction"] = state.gripper_direction;
            result["initialized"] = bool(state.initialized);
            result["first_goal_cache_pending"] = bool(state.first_goal_cache_pending);
            result["first_tick_cache_pending"] = bool(state.first_tick_cache_pending);
            cache["ee_pos"] = osc_state_array(state.cached_ee_pos, 3);
            cache["ee_ori"] = osc_state_array(state.cached_ee_ori, 3, 3);
            cache["ee_vel"] = osc_state_array(state.cached_ee_vel, 6);
            cache["joint_pos"] = osc_state_array(state.cached_joint_pos, na);
            cache["joint_vel"] = osc_state_array(state.cached_joint_vel, na);
            cache["J_full"] = osc_state_array(state.cached_J_full, 6, na, OSC_MAX_ARM);
            cache["mass_matrix"] = osc_state_array(state.cached_mass_matrix, na, na, OSC_MAX_ARM);
            result["cache"] = cache;
            return result;
        }, py::arg("env_idx"), "Read one OSC state and both one-shot cache flags")
        .def("begin_osc_policy_step", &PhysicsEngine::begin_osc_policy_step,
             "Process the uploaded policy action without integrating")
        .def("step_osc_substep", &PhysicsEngine::step_osc_substep,
             py::arg("warm_start"), "Run one diagnostic OSC physics substep")
        .def("set_osc_gripper_state",
             [](PhysicsEngine& self, py::array_t<double, py::array::c_style | py::array::forcecast> state) {
                 auto buf = state.request();
                 if (buf.ndim != 2 || buf.shape[1] != OSC_MAX_GRIP) {
                     throw std::runtime_error(
                         "set_osc_gripper_state expects shape [num_envs, " +
                         std::to_string(OSC_MAX_GRIP) + "], got ndim=" +
                         std::to_string(buf.ndim));
                 }
                 self.set_osc_gripper_state(static_cast<const double*>(buf.ptr),
                                            (int)buf.shape[0]);
             },
             py::arg("state"),
             "Set the OSC gripper accumulator, shape [num_envs, OSC_MAX_GRIP]")
        // Readback exists so set_osc_gripper_state is testable round-trip; without it a
        // silently-ignored write is indistinguishable from a correct one.
        .def("read_osc_gripper_state",
             [](const PhysicsEngine& self, int num_envs) -> py::array_t<double> {
                 if (num_envs <= 0) throw std::runtime_error("num_envs must be > 0");
                 std::vector<OSCEnvState> host(num_envs);
                 if (!self.read_osc_env_state(host.data(), num_envs)) {
                     throw std::runtime_error(
                         "read_osc_env_state failed (OSC state not allocated? "
                         "call set_osc_config/set_osc_gripper_state first)");
                 }
                 py::array_t<double> out({num_envs, (int)OSC_MAX_GRIP});
                 auto o = out.mutable_unchecked<2>();
                 for (int e = 0; e < num_envs; ++e)
                     for (int g = 0; g < OSC_MAX_GRIP; ++g)
                         o(e, g) = host[e].gripper_current_action[g];
                 return out;
             },
             py::arg("num_envs"),
             "Read back the OSC gripper accumulator, shape [num_envs, OSC_MAX_GRIP]")
        .def("reset_warmstart", &PhysicsEngine::reset_warmstart,
             "Reset solver warmstart state (qacc_warmstart + constraint_lam) for all envs")
        .def("reset_warmstart_env", &PhysicsEngine::reset_warmstart_env,
             py::arg("env_idx"),
             "Reset solver warmstart state for one env");

    // ── PhysicsRenderer ─────────────────────────────────────────────────────
    py::class_<PhysicsRenderer>(m, "PhysicsRenderer")
        .def(py::init<>())
        .def("init", [](PhysicsRenderer& r, PhysicsModel& model) {
            return r.init(model.desc());
        }, py::arg("model"))
        .def("build_geometry", [](PhysicsRenderer& r, PhysicsEngine& engine, int env_idx) {
            r.build_geometry(engine.model()->device_model(),
                            engine.data().device_data(),
                            env_idx, engine.stream());
        }, py::arg("engine"), py::arg("env_idx") = 0)
        .def_property_readonly("total_vertices", &PhysicsRenderer::total_vertices)
        .def_property_readonly("geom_vertices", &PhysicsRenderer::geom_vertices)
        .def_property_readonly("site_vertex_offset", &PhysicsRenderer::site_vertex_offset)
        .def_property_readonly("site_vertices", &PhysicsRenderer::site_vertices);

    // ── GpuImage: GPU buffer with __cuda_array_interface__ ────────────────────
    py::class_<GpuImage>(m, "GpuImage")
        .def_readonly("height", &GpuImage::height)
        .def_readonly("width", &GpuImage::width)
        .def_property_readonly("shape", [](const GpuImage& g) {
            return py::make_tuple(g.height, g.width, 3);
        })
        .def_property_readonly("__cuda_array_interface__", &GpuImage::cuda_array_interface);

    py::class_<GpuImageBatch>(m, "GpuImageBatch")
        .def_readonly("num_envs", &GpuImageBatch::num_envs)
        .def_readonly("height", &GpuImageBatch::height)
        .def_readonly("width", &GpuImageBatch::width)
        .def_property_readonly("shape", [](const GpuImageBatch& g) {
            return py::make_tuple(g.num_envs, g.height, g.width, 3);
        })
        .def_property_readonly("__cuda_array_interface__", &GpuImageBatch::cuda_array_interface)
        .def("numpy", [](const GpuImageBatch& g) -> py::array_t<uint8_t> {
            size_t total = (size_t)g.num_envs * g.height * g.width * 3;
            if (void* blk = hp_pin_acquire(total)) {
                py::capsule owner(blk, [](void* p) { hp_pin_release(p); });
                hipError_t err = hipMemcpy(blk, g.ptr, total, hipMemcpyDeviceToHost);
                if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
                return py::array_t<uint8_t>({g.num_envs, g.height, g.width, 3},
                                            static_cast<uint8_t*>(blk), owner);
            }
            py::array_t<uint8_t> arr({g.num_envs, g.height, g.width, 3});
            hipMemcpy(arr.mutable_data(), g.ptr, total, hipMemcpyDeviceToHost);
            return arr;
        }, "Copy batch images from GPU to numpy (N, H, W, 3) uint8 array");

    // ── CameraRenderer: renders camera images from physics state ────────────
    struct CameraRenderer {
        Pipeline pipeline;
        Framebuffer* fb;
        PhysicsRenderer phys_renderer;
        int width, height;          // RENDER dims (ssaa-expanded)
        int ssaa = 1;
        int msaa = 1;
        int out_width = 0, out_height = 0;
        bool initialized;
        uint8_t* d_rgb;
        uint8_t* d_rgb_batch;
        int batch_capacity;
        float bg_color[4];
        float fog_start, fog_end;
        float fog_color[3];
        bool gradient_bg;
        float gradient_top[3];
        float gradient_bot[3];
        float znear, zfar;
        bool vis_transparent;  // Legacy request bit; never gates intrinsic alpha.
        std::vector<float> transparency_geom_pos;
        std::vector<float> transparency_rbound;
        bool geomgroup[6];    // MuJoCo MjvOption.geomgroup: per-group visibility
        bool gouraud_shading; // Per-vertex Gouraud shading (matches MuJoCo) vs per-fragment Blinn-Phong
        bool linear_output = false;     // set_linear_output: SAPIEN-style linear lighting + gamma
        float output_exposure = 1.0f;
        float output_inv_gamma = 1.0f;
        bool reflection_enabled = true;
        PlanarReflectionResources reflection_resources;
        bool shadow_enabled = false;
        Framebuffer* shadow_fb = nullptr;
        Pipeline shadow_pipeline;
        int shadow_size = 1024;

        // Mutable appearance belongs to this renderer, never PhysicsModel.
        // Python entry points retain the GIL; the mutex also makes a snapshot
        // and an update indivisible. Each render owns its copied snapshot until
        // all foreground/reflection draws and worker streams have completed.
        std::mutex site_state_mutex_;
        PhysicsModel* site_model_ = nullptr;
        PhysicsEngine* site_engine_ = nullptr;
        py::object site_model_owner_;
        py::object site_engine_owner_;
        int site_nsite_ = 0;
        int site_worlds_ = 0;
        std::vector<float> site_defaults_;
        std::vector<float> site_world_rgba_;

        static std::vector<float> copy_site_defaults(const PhysicsModelDesc& desc) {
            std::vector<float> result((size_t)desc.nsite * 4);
            for (int si = 0; si < desc.nsite; ++si) {
                for (int k = 0; k < 4; ++k) {
                    const float value = desc.sites[si].rgba[k];
                    if (!std::isfinite(value))
                        throw std::runtime_error("Model site RGBA must be finite");
                    result[(size_t)si * 4 + k] = value;
                }
            }
            return result;
        }

        void reset_site_state(PhysicsModel& model, std::vector<float> defaults) {
            py::object owner = py::cast(&model, py::return_value_policy::reference);
            std::lock_guard<std::mutex> lock(site_state_mutex_);
            // Release the old engine before releasing its retained model.
            site_engine_owner_ = py::object();
            site_engine_ = nullptr;
            site_model_owner_ = std::move(owner);
            site_model_ = &model;
            site_nsite_ = model.nsite();
            site_worlds_ = 0;
            site_defaults_ = std::move(defaults);
            site_world_rgba_.clear();
        }

        int validate_site_engine(PhysicsEngine& engine) const {
            if (!initialized || !site_model_)
                throw std::runtime_error("CameraRenderer not initialized");
            if (engine.model() != site_model_ || engine.model()->nsite() != site_nsite_)
                throw std::runtime_error("Renderer and engine models do not match; reinitialize renderer");
            const int worlds = engine.data().num_envs();
            if (worlds <= 0)
                throw std::runtime_error("Engine must have at least one environment");
            if (site_engine_ && site_engine_ != &engine)
                throw std::runtime_error("Renderer is bound to another engine; reinitialize renderer");
            if (site_engine_ && site_worlds_ != worlds)
                throw std::runtime_error("Engine environment count changed; reinitialize renderer");
            return worlds;
        }

        static void validate_site_world(int env_idx, int worlds) {
            if (env_idx < 0 || env_idx >= worlds)
                throw std::runtime_error("Invalid environment index");
        }

        std::vector<float> default_site_worlds(int worlds) const {
            std::vector<float> result((size_t)worlds * site_defaults_.size());
            for (int ei = 0; ei < worlds; ++ei) {
                std::copy(site_defaults_.begin(), site_defaults_.end(),
                          result.begin() + (size_t)ei * site_defaults_.size());
            }
            return result;
        }

        void commit_site_state(PhysicsEngine& engine, int worlds,
                               std::vector<float> values) {
            // Prepare the owner before modifying any state. A malformed update
            // never binds an engine, and a failed allocation leaves old bytes.
            py::object owner = site_engine_owner_;
            if (!site_engine_)
                owner = py::cast(&engine, py::return_value_policy::reference);
            site_world_rgba_.swap(values);
            site_engine_owner_ = std::move(owner);
            site_engine_ = &engine;
            site_worlds_ = worlds;
        }

        void set_site_rgba(PhysicsEngine& engine,
                           const py::array_t<float, py::array::c_style | py::array::forcecast>& rgba,
                           int env_idx, bool batch) {
            std::lock_guard<std::mutex> lock(site_state_mutex_);
            const int worlds = validate_site_engine(engine);
            if (!batch) validate_site_world(env_idx, worlds);
            const auto buffer = rgba.request();
            const bool valid_shape = batch
                ? (buffer.ndim == 3 && buffer.shape[0] == worlds &&
                   buffer.shape[1] == site_nsite_ && buffer.shape[2] == 4)
                : (buffer.ndim == 2 && buffer.shape[0] == site_nsite_ && buffer.shape[1] == 4);
            if (!valid_shape)
                throw std::runtime_error(batch ? "RGBA must have shape (num_envs, nsite, 4)"
                                               : "RGBA must have shape (nsite, 4)");
            std::vector<float> incoming((size_t)buffer.size);
            if (!incoming.empty())
                std::copy_n(static_cast<const float*>(buffer.ptr), incoming.size(), incoming.data());
            for (float value : incoming) {
                if (!std::isfinite(value))
                    throw std::runtime_error("Site RGBA must be finite after float32 conversion");
            }
            // Negative / above-one alpha carries LIBERO state; do not clamp it.
            if (batch) {
                commit_site_state(engine, worlds, std::move(incoming));
            } else {
                auto next = site_engine_ ? site_world_rgba_ : default_site_worlds(worlds);
                std::copy(incoming.begin(), incoming.end(),
                          next.begin() + (size_t)env_idx * site_defaults_.size());
                commit_site_state(engine, worlds, std::move(next));
            }
        }

        std::vector<float> site_snapshot(PhysicsEngine& engine, int first_env, int count) {
            std::lock_guard<std::mutex> lock(site_state_mutex_);
            const int worlds = validate_site_engine(engine);
            if (count == -1 && first_env == 0) count = worlds;
            if (first_env < 0 || count <= 0 || count > worlds || first_env > worlds - count)
                throw std::runtime_error("Invalid environment range");
            if (!site_engine_)
                commit_site_state(engine, worlds, default_site_worlds(worlds));
            const size_t stride = site_defaults_.size();
            return std::vector<float>(site_world_rgba_.begin() + (size_t)first_env * stride,
                                      site_world_rgba_.begin() + (size_t)(first_env + count) * stride);
        }

        void clear_site_rgba(PhysicsEngine& engine, const py::object& env_idx) {
            std::lock_guard<std::mutex> lock(site_state_mutex_);
            const int worlds = validate_site_engine(engine);
            int selected = -1;
            if (!env_idx.is_none()) {
                selected = env_idx.cast<int>();
                validate_site_world(selected, worlds);
            }
            auto next = (env_idx.is_none() || !site_engine_)
                ? default_site_worlds(worlds) : site_world_rgba_;
            if (selected >= 0)
                std::copy(site_defaults_.begin(), site_defaults_.end(),
                          next.begin() + (size_t)selected * site_defaults_.size());
            commit_site_state(engine, worlds, std::move(next));
        }

        struct IssuePool {
            std::vector<std::thread> th;
            std::mutex m;
            std::condition_variable cv_go, cv_done;
            const std::function<void(int)>* job = nullptr;
            int gen = 0;
            int active = 0;      // total slots this round (slot 0 = caller)
            int done = 0;        // helper slots finished
            bool quit = false;
            std::exception_ptr err;

            ~IssuePool() {
                {
                    std::lock_guard<std::mutex> lk(m);
                    quit = true; ++gen;
                }
                cv_go.notify_all();
                for (auto& t : th) if (t.joinable()) t.join();
            }

            void body(int id, int mygen) {
                for (;;) {
                    std::unique_lock<std::mutex> lk(m);
                    cv_go.wait(lk, [&]{ return gen != mygen; });
                    mygen = gen;
                    if (quit) return;
                    if (id >= active) continue;   // not needed this round
                    const std::function<void(int)>* fn = job;
                    lk.unlock();
                    try { (*fn)(id); }
                    catch (...) {
                        std::lock_guard<std::mutex> lk2(m);
                        if (!err) err = std::current_exception();
                    }
                    lk.lock();
                    const bool last = (++done == active - 1);
                    lk.unlock();
                    if (last) cv_done.notify_one();
                }
            }

            // Runs fn(0) on the CALLING thread and fn(1..slots-1) on helpers,
            // returns once every slot has finished. Exceptions are captured and
            // rethrown here so the caller's error handling is unchanged.
            void run(int slots, const std::function<void(int)>& fn) {
                if (slots <= 1) { fn(0); return; }
                for (int i = (int)th.size(); i < slots - 1; i++) {
                    const int id = i + 1;
                    const int g0 = gen;   // only the caller ever mutates gen
                    th.emplace_back([this, id, g0]{ body(id, g0); });
                }
                {
                    std::lock_guard<std::mutex> lk(m);
                    job = &fn; active = slots; done = 0; err = nullptr; ++gen;
                }
                cv_go.notify_all();
                std::exception_ptr local;
                try { fn(0); } catch (...) { local = std::current_exception(); }
                {
                    std::unique_lock<std::mutex> lk(m);
                    cv_done.wait(lk, [&]{ return done == active - 1; });
                    if (!local) local = err;
                    err = nullptr;
                    job = nullptr;
                }
                if (local) std::rethrow_exception(local);
            }
        };
        IssuePool issue_pool_;

        // Parallel render workers for multi-stream render_gpu_batch
        struct RenderWorker {
            hipStream_t stream = nullptr;
            Framebuffer* fb = nullptr;
            PhysicsRenderer* phys_renderer = nullptr;
            Pipeline* pipeline = nullptr;
            PlanarReflectionResources reflection_resources;
            hipGraphExec_t graph_exec = nullptr;

            RenderWorker() = default;
            ~RenderWorker() {
                if (graph_exec) hipGraphExecDestroy(graph_exec);
                delete pipeline;
                delete phys_renderer;
                delete fb;
                if (stream) hipStreamDestroy(stream);
            }
            RenderWorker(const RenderWorker&) = delete;
            RenderWorker& operator=(const RenderWorker&) = delete;
        };
        std::vector<std::unique_ptr<RenderWorker>> workers_;
        int num_workers_ = 6;

        std::vector<unsigned char> graph_sig_;
        std::vector<unsigned char> graph_sig_new_;
        // The three BIG per-world arrays are compared IN PLACE against these
        // retained shadows instead of being copied into the blob: at w=256
        // that is ~900 kB of memcpy per call that buys nothing, since a
        // mismatch is the rare case.
        std::vector<float> graph_site_shadow_;
        std::vector<RenderState> graph_state_shadow_;
        std::vector<PlanarReflectionPlan> graph_plan_shadow_;
        bool graph_sig_valid_ = false;
        bool graph_live_ = false;        // every worker holds a graph for it
        bool graph_failed_ = false;      // capture refused for graph_sig_
        long long gstat_calls_ = 0, gstat_hit_ = 0, gstat_capture_ = 0,
                  gstat_eager_ = 0, gstat_masked_ = 0, gstat_worlds_ = 0,
                  gstat_sigchg_ = 0;
        void graph_drop() {
            for (auto& up : workers_)
                if (up && up->graph_exec) {
                    hipGraphExecDestroy(up->graph_exec);
                    up->graph_exec = nullptr;
                }
            graph_live_ = false;
        }

        float* d_batch_matrices_ = nullptr;
        int batch_mat_cap_ = 0;
        float* h_batch_matrices_ = nullptr;
        int    h_batch_mat_cap_ = 0;
        hipEvent_t batch_mat_evt_ = nullptr;
        float* h_batch_mat_shadow_ = nullptr;
        int    batch_mat_shadow_envs_ = 0;
        bool   batch_mat_resident_ = false;
        hipEvent_t pose_evt_ = nullptr;
        hipStream_t solo_stream_ = nullptr;
        hipStream_t ensure_solo_stream() {
            if (!solo_stream_)
                if (hipStreamCreateWithFlags(&solo_stream_, hipStreamNonBlocking)
                        != hipSuccess)
                    solo_stream_ = nullptr;
            return solo_stream_;
        }
        std::vector<RenderState> batch_states_;
        std::vector<PlanarReflectionPlan> reflection_plans_;
        void ensure_batch_matrices(int num_envs) {
            if (num_envs > batch_mat_cap_) {
                if (d_batch_matrices_) hipFree(d_batch_matrices_);
                hipMalloc(&d_batch_matrices_, (size_t)num_envs * HP_MAT_STRIDE * sizeof(float));
                batch_mat_cap_ = num_envs;
                batch_mat_resident_ = false;   // fresh allocation holds nothing
            }
            if (num_envs > h_batch_mat_cap_) {
                float* fresh = nullptr;
                if (hipHostMalloc((void**)&fresh, (size_t)num_envs * HP_MAT_STRIDE * sizeof(float),
                                  hipHostMallocDefault) == hipSuccess) {
                    if (h_batch_matrices_) hipHostFree(h_batch_matrices_);
                    h_batch_matrices_ = fresh;
                    h_batch_mat_cap_ = num_envs;
                }
            }
            if (!batch_mat_evt_)
                hipEventCreateWithFlags(&batch_mat_evt_, hipEventDisableTiming);
            if (!pose_evt_)
                hipEventCreateWithFlags(&pose_evt_, hipEventDisableTiming);
            if (num_envs > batch_mat_shadow_envs_) {
                float* fresh = (float*)realloc(h_batch_mat_shadow_,
                                               (size_t)num_envs * HP_MAT_STRIDE * sizeof(float));
                if (fresh) { h_batch_mat_shadow_ = fresh; batch_mat_shadow_envs_ = num_envs; }
                else { batch_mat_shadow_envs_ = 0; }
                batch_mat_resident_ = false;
            }
        }

        // Reusable pinned readback storage. It owns bytes only, never cached
        // physics state: each batch refreshes all of its host-side dependencies.
        float* h_batch_poses_ = nullptr;
        size_t batch_pose_capacity_ = 0;
        void ensure_batch_poses(size_t count) {
            if (count <= batch_pose_capacity_) return;
            float* fresh = nullptr;
            hipError_t err = hipHostMalloc(&fresh, count * sizeof(float));
            if (err != hipSuccess)
                throw std::runtime_error(hipGetErrorString(err));
            // Every prior readback is drained before returning to the caller.
            if (h_batch_poses_) hipHostFree(h_batch_poses_);
            h_batch_poses_ = fresh;
            batch_pose_capacity_ = count;
        }

        float* h_site_stage_ = nullptr;
        size_t site_stage_floats_ = 0;
        int    site_stage_slots_ = 0;
        int    site_stage_verts_ = 0;     // per-slot capacity, in VERTICES
        bool   site_stage_broken_ = false;
        std::vector<SiteSlotCtx> site_slot_ctx_;
        const PhysicsModelDesc* site_vc_desc_ = nullptr;
        std::vector<int> site_vc_;

        // Per-site vertex counts for this model, derived once from the emitter.
        const int* site_vert_counts(const PhysicsModelDesc& desc) {
            if (site_vc_desc_ != &desc || (int)site_vc_.size() != desc.nsite) {
                site_vc_.assign(desc.nsite > 0 ? desc.nsite : 0, 0);
                for (int si = 0; si < desc.nsite; ++si)
                    site_vc_[si] = site_overlay_vertex_count(desc.sites[si]);
                site_vc_desc_ = &desc;
            }
            return site_vc_.empty() ? nullptr : site_vc_.data();
        }

        bool ensure_site_stage(int slots, int verts) {
            if (slots <= 0 || verts <= 0) return false;
            const size_t need = (size_t)slots * (size_t)verts * 10;
            if (h_site_stage_ && site_stage_verts_ == verts &&
                site_stage_slots_ >= slots && site_stage_floats_ >= need)
                return true;
            float* fresh = nullptr;
            if (hipHostMalloc(&fresh, need * sizeof(float)) != hipSuccess)
                return false;
            // Every prior batch is joined before render_gpu_batch returns, so
            // no captured copy can still be reading the old arena here.
            if (h_site_stage_) hipHostFree(h_site_stage_);
            h_site_stage_ = fresh;
            site_stage_floats_ = need;
            site_stage_slots_ = slots;
            site_stage_verts_ = verts;
            site_slot_ctx_.assign(slots, SiteSlotCtx{});
            return true;
        }

        float* site_slot(int idx) const {
            return h_site_stage_ + (size_t)idx * (size_t)site_stage_verts_ * 10;
        }

        // Per-worker device landing buffer for ONE overlay slot. Reused world
        // to world: the copy and the draw that reads it are consecutive on the
        // SAME worker stream, so the reuse is ordered by construction -- and
        // stays ordered inside a captured graph, which is a linear chain on
        // that stream.
        std::vector<float*> d_site_dev_;
        int d_site_dev_verts_ = 0;
        bool ensure_site_dev(int workers, int verts) {
            if (workers <= 0 || verts <= 0) return false;
            if ((int)d_site_dev_.size() == workers && d_site_dev_verts_ == verts) {
                for (float* q : d_site_dev_) if (!q) return false;
                return true;
            }
            for (float* q : d_site_dev_) if (q) hipFree(q);
            d_site_dev_.assign(workers, nullptr);
            d_site_dev_verts_ = 0;
            for (int i = 0; i < workers; i++) {
                if (hipMalloc(&d_site_dev_[i], (size_t)verts * 10 * sizeof(float))
                    != hipSuccess) return false;
            }
            d_site_dev_verts_ = verts;
            return true;
        }

        CameraRenderer() : fb(nullptr), width(0), height(0), initialized(false),
                           d_rgb(nullptr), d_rgb_batch(nullptr), batch_capacity(0),
                           fog_start(0.0f), fog_end(0.0f), gradient_bg(false),
                           znear(-1.0f), zfar(-1.0f), vis_transparent(false),
                           gouraud_shading(true), shadow_enabled(false),
                           shadow_fb(nullptr), shadow_size(1024), num_workers_(6) {
            // MuJoCo default: groups 0, 1, 2 visible
            for (int i = 0; i < 6; i++) geomgroup[i] = (i < 3);
            bg_color[0] = 0.0f; bg_color[1] = 0.0f;
            bg_color[2] = 0.0f; bg_color[3] = 1.0f;
            fog_color[0] = 0.0f; fog_color[1] = 0.0f; fog_color[2] = 0.0f;
            gradient_top[0] = gradient_top[1] = gradient_top[2] = 0.5f;
            gradient_bot[0] = gradient_bot[1] = gradient_bot[2] = 0.0f;
        }
        ~CameraRenderer() {
            // Drain every worker stream BEFORE any of them are destroyed.
            // ~RenderWorker does hipStreamDestroy then deletes the Framebuffer
            // that stream may still be writing; with 16 workers that reliably
            // corrupts the heap at process exit ("corrupted double-linked
            // list"). Rendering results are already correct at that point --
            // this is purely a teardown-ordering fault, and it stayed latent
            // because render_gpu_batch had no production caller.
            for (auto& w : workers_) {
                if (w && w->stream) hipStreamSynchronize(w->stream);
            }
            hipDeviceSynchronize();
            workers_.clear();  // destroy workers before fb
            if (solo_stream_) {
                hipStreamSynchronize(solo_stream_);
                hipStreamDestroy(solo_stream_);
                solo_stream_ = nullptr;
            }
            delete shadow_fb;
            delete fb;
            if (d_rgb) hipFree(d_rgb);
            if (d_rgb_batch) hipFree(d_rgb_batch);
            if (h_batch_poses_) hipHostFree(h_batch_poses_);
            if (h_site_stage_) hipHostFree(h_site_stage_);
            for (float* q : d_site_dev_) if (q) hipFree(q);
            if (h_batch_matrices_) hipHostFree(h_batch_matrices_);
            if (h_batch_mat_shadow_) { free(h_batch_mat_shadow_); h_batch_mat_shadow_ = nullptr; }
            if (batch_mat_evt_) hipEventDestroy(batch_mat_evt_);
            if (d_batch_matrices_) hipFree(d_batch_matrices_);
        }

        void ensure_batch_buffer(int num_envs) {
            if (num_envs <= batch_capacity) return;
            if (d_rgb_batch) hipFree(d_rgb_batch);
            size_t sz = (size_t)num_envs * width * height * 3;
            hipMalloc(&d_rgb_batch, sz);
            batch_capacity = num_envs;
        }

        void ensure_workers(const PhysicsModelDesc& desc, int num_envs = 16,
                            int want = -1) {
            // Cap by the envs actually being rendered, not just the 16 ceiling.
            // The caller already does actual_workers = min(num_workers_,
            // num_envs), so allocating 16 workers to render 4 envs built 12
            // full Framebuffer+PhysicsRenderer+Pipeline+stream sets that are
            // never used and still have to be torn down.
            int needed = std::min({want > 0 ? want : num_workers_,
                                   std::max(1, num_envs)});
            if (needed <= 1) return;  // single-stream: use original resources
            if ((int)workers_.size() >= needed) return;
            for (int i = (int)workers_.size(); i < needed; i++) {
                auto w = std::make_unique<RenderWorker>();
                bool made = false;
                if (hp_cumask_enabled(width)) {
                    uint32_t mask[HP_CU_WORDS];
                    for (int k = 0; k < HP_CU_WORDS; k++) mask[k] = 0xFFFFFFFFu;
                    const int drop = (i * 8) % (HP_CU_WORDS * 32);
                    mask[drop >> 5] &= ~(1u << (drop & 31));
                    made = (hipExtStreamCreateWithCUMask(&w->stream, HP_CU_WORDS,
                                                         mask) == hipSuccess);
                }
                if (!made) hipStreamCreate(&w->stream);
                w->fb = new Framebuffer(width, height, true, msaa);
                w->phys_renderer = new PhysicsRenderer();
                w->phys_renderer->init(desc, false);
                if (hp_share_attribs()) {
                    if (!w->phys_renderer->share_invariant_attribs_from(phys_renderer) &&
                        !workers_.empty())
                        w->phys_renderer->share_invariant_attribs_from(
                            *workers_.front()->phys_renderer);
                }
                w->pipeline = new Pipeline();
                w->pipeline->set_framebuffer(w->fb);
                w->pipeline->set_stream(w->stream);
                workers_.push_back(std::move(w));
            }
        }
    };

    auto do_render_common = [](CameraRenderer& cr, PhysicsEngine& engine,
                               int camera_idx, int env_idx) {
        if (!cr.initialized) throw std::runtime_error("CameraRenderer not initialized");
        using namespace hipgl::physics::kernels;

        const auto site_rgba = cr.site_snapshot(engine, env_idx, 1);
        const float* env_site_rgba = site_rgba.empty() ? nullptr : site_rgba.data();
        engine.sync();
        const PhysicsModelDesc& desc = engine.model()->desc();
        if (camera_idx < 0 || camera_idx >= desc.ncamera)
            throw std::runtime_error("Invalid camera index");

        const CameraDesc& cam = desc.cameras[camera_idx];

        int nb = desc.nbody;
        int nenvs = engine.data().num_envs();
        std::vector<float> h_xpos(nenvs * nb * 3), h_xquat(nenvs * nb * 4);
        engine.data().read_xpos(h_xpos.data());
        engine.data().read_xquat(h_xquat.data());
        std::vector<float> h_site_xpos, h_site_xmat;
        if (desc.nsite > 0) {
            h_site_xpos.resize(nenvs * desc.nsite * 3);
            h_site_xmat.resize(nenvs * desc.nsite * 9);
            engine.data().read_site_xpos(h_site_xpos.data());
            engine.data().read_site_xmat(h_site_xmat.data());
        }

        float* env_xpos = h_xpos.data() + env_idx * nb * 3;
        float* env_xquat = h_xquat.data() + env_idx * nb * 4;
        float* env_site_xpos = h_site_xpos.empty() ? nullptr : h_site_xpos.data() + env_idx * desc.nsite * 3;
        float* env_site_xmat = h_site_xmat.empty() ? nullptr : h_site_xmat.data() + env_idx * desc.nsite * 9;

        RenderState state = build_camera_state(
            cam.pos, cam.quat, cam.fovy,
            cr.width, cr.height,
            env_xpos, env_xquat, cam.body, desc, cr.bg_color,
            cr.gradient_bg, cr.gradient_top, cr.gradient_bot);
        const DevicePhysicsModel& device_model = engine.model()->device_model();
        TextureState texture_state = {
            device_model.tex_data,
            device_model.tex_width,
            device_model.tex_height,
            device_model.tex_offset,
            device_model.ntexture,
            device_model.tex_iscube,
            device_model.tex_mip
        };
        state.texture_2d_enabled = (texture_state.tex_data && texture_state.ntexture > 0);
        state.fog_start = cr.fog_start;
        state.fog_end = cr.fog_end;
        state.fog_color[0] = cr.fog_color[0];
        state.fog_color[1] = cr.fog_color[1];
        state.fog_color[2] = cr.fog_color[2];
        state.linear_output = cr.linear_output ? 1 : 0;
        state.output_exposure = cr.output_exposure;
        state.output_inv_gamma = cr.output_inv_gamma;
        state.gouraud_shading = cr.gouraud_shading ? 1 : 0;

        { unsigned m = 0; for (int g = 0; g < 3; g++) if (cr.geomgroup[g]) m |= (1u << g);
          cr.phys_renderer.set_active_groups(m, true); }
        cr.phys_renderer.build_geometry(
            device_model,
            engine.data().device_data(),
            env_idx, engine.stream());
        hipStreamSynchronize(engine.stream());

        int geom_verts = cr.phys_renderer.geom_vertices();

        // Shadow pass: render depth from headlight (camera-aligned directional)
        if (cr.shadow_enabled && cr.shadow_fb) {
            float extent = desc.stat_extent;
            float s_znear = (cr.znear > 0.0f) ? cr.znear : (desc.map_znear * extent);
            float s_zfar  = (cr.zfar  > 0.0f) ? cr.zfar  : (desc.map_zfar  * extent);
            float half_size = extent * 1.5f;
            math::Mat4 shadow_proj = math::ortho(-half_size, half_size,
                                                  -half_size, half_size,
                                                  s_znear, s_zfar);
            math::Mat4 shadow_view = math::Mat4::from_column_major(state.model_view);
            math::Mat4 shadow_mvp_mat = shadow_proj * shadow_view;

            RenderState shadow_state;
            memset(&shadow_state, 0, sizeof(shadow_state));
            shadow_state.depth_test_enabled = true;
            shadow_state.depth_write_enabled = true;
            shadow_state.backface_cull_enabled = false;
            shadow_state.lighting_enabled = false;
            shadow_state.depth_func = DEPTH_LEQUAL;
            shadow_state.clear_depth = 1.0f;
            shadow_state.viewport_x = 0;
            shadow_state.viewport_y = 0;
            shadow_state.viewport_w = cr.shadow_size;
            shadow_state.viewport_h = cr.shadow_size;
            memcpy(shadow_state.model_view, state.model_view, 16 * sizeof(float));
            shadow_proj.to_column_major(shadow_state.projection);
            shadow_mvp_mat.to_column_major(shadow_state.mvp);
            memcpy(shadow_state.normal_matrix, state.normal_matrix, 9 * sizeof(float));

            TextureState no_tex = {};
            cr.shadow_pipeline.set_render_state(shadow_state);
            cr.shadow_pipeline.clear(true, true);
            for (int g = 0; g < 3; g++) {
                if (!cr.geomgroup[g]) continue;
                int goff = cr.phys_renderer.group_offset(g);
                int gcnt = cr.phys_renderer.group_count(g);
                if (gcnt <= 0) continue;
                cr.shadow_pipeline.draw_arrays_device_phong(
                    PRIM_TRIANGLES,
                    cr.phys_renderer.device_positions() + goff * 3,
                    cr.phys_renderer.device_normals()   + goff * 3,
                    cr.phys_renderer.device_texcoords() + goff * TEXCOORD_COMPONENTS,
                    cr.phys_renderer.device_texids()    + goff,
                    cr.phys_renderer.device_colors()    + goff * 4,
                    cr.phys_renderer.device_specular()  + goff,
                    cr.phys_renderer.device_shininess() + goff,
                    cr.phys_renderer.device_emission()  + goff,
                    gcnt,
                    no_tex);
            }
            cr.shadow_pipeline.sync();

            // shadow_mvp for fragment shader: ortho proj with Y negated for rasterizer Y-flip
            math::Mat4 frag_shadow_mvp = shadow_proj;
            frag_shadow_mvp(1,1) = -frag_shadow_mvp(1,1);
            frag_shadow_mvp.to_column_major(state.shadow_mvp);
            state.shadow_enabled = 1;
            state.shadow_depth = cr.shadow_fb->device_fb().depth_buffer;
            state.shadow_w = cr.shadow_size;
            state.shadow_h = cr.shadow_size;
            state.shadow_bias = 0.005f;
        }

        const auto reflection = make_planar_reflection_plan(
            desc, state, env_xpos, env_xquat, cr.geomgroup, cr.reflection_enabled);
        const auto transparent_order = make_transparent_draw_order(
            desc, state, env_xpos, env_xquat, cr.geomgroup,
            cr.transparency_geom_pos.empty() ? nullptr : cr.transparency_geom_pos.data(),
            cr.transparency_rbound.empty() ? nullptr : cr.transparency_rbound.data());
        cr.pipeline.set_external_matrices(nullptr);
        cr.pipeline.set_deferred_bg(false);
        cr.pipeline.set_render_state(state);
        cr.pipeline.clear(true, true);
        draw_opaque_with_planar_reflection(
            cr.pipeline, *cr.fb, cr.phys_renderer, state, texture_state,
            cr.geomgroup, transparent_order, reflection, cr.reflection_resources,
            nullptr, nullptr, [&](const RenderState& reflected) {
                draw_site_overlays(cr.pipeline, reflected, desc, env_site_xpos, env_site_xmat, env_site_rgba);
            });

        draw_intrinsic_transparency(cr.pipeline, cr.phys_renderer, state, texture_state, transparent_order);

        draw_site_overlays(cr.pipeline, state, desc, env_site_xpos, env_site_xmat, env_site_rgba);
        cr.pipeline.sync();
    };

    py::class_<CameraRenderer>(m, "CameraRenderer")
        .def(py::init<>())
        .def("init", [](CameraRenderer& cr, PhysicsModel& model, int width, int height, int ssaa, int msaa) {
            if (ssaa < 1) throw std::runtime_error("ssaa must be >= 1");
            if (msaa != 1 && msaa != 4) throw std::runtime_error("msaa must be 1 or 4");
            if (msaa != 1 && ssaa != 1) throw std::runtime_error("native MSAA requires ssaa=1");
            auto site_defaults = CameraRenderer::copy_site_defaults(model.desc());
            // init invalidates compiled metadata even when this model object
            // was reloaded in place. Callers must resupply it after final init.
            cr.transparency_geom_pos.clear();
            cr.transparency_rbound.clear();
            if (cr.initialized) {
                // A new resolution or sample count needs matching worker
                // attachments, too. Drain them before replacing their storage.
                for (auto& w : cr.workers_)
                    if (w && w->stream) hipStreamSynchronize(w->stream);
                cr.pipeline.sync();
                cr.shadow_pipeline.sync();
                cr.workers_.clear();
                delete cr.fb; cr.fb = nullptr;
                delete cr.shadow_fb; cr.shadow_fb = nullptr;
                if (cr.d_rgb) { hipFree(cr.d_rgb); cr.d_rgb = nullptr; }
                if (cr.d_rgb_batch) { hipFree(cr.d_rgb_batch); cr.d_rgb_batch = nullptr; }
                cr.batch_capacity = 0;
                cr.initialized = false;
            }
            cr.msaa = msaa;
            cr.ssaa = ssaa;
            cr.out_width = width;
            cr.out_height = height;
            width *= ssaa; height *= ssaa;
            cr.width = width;
            cr.height = height;
            cr.fb = new Framebuffer(width, height, true, msaa);
            cr.pipeline.set_framebuffer(cr.fb);
            cr.phys_renderer.init(model.desc(), false);
            size_t rgb_size = (size_t)cr.out_width * cr.out_height * 3;
            hipMalloc(&cr.d_rgb, rgb_size);
            cr.shadow_fb = new Framebuffer(cr.shadow_size, cr.shadow_size, true);
            cr.shadow_pipeline.set_framebuffer(cr.shadow_fb);
            cr.reset_site_state(model, std::move(site_defaults));
            cr.initialized = true;
        }, py::arg("model"), py::arg("width") = 256, py::arg("height") = 256,
           py::arg("ssaa") = 1, py::arg("msaa") = 1,
           "Initialize the renderer; msaa=4 enables native four-sample antialiasing at ssaa=1.")
        .def("set_transparency_sort_metadata", [](CameraRenderer& cr,
                const py::array_t<float, py::array::c_style | py::array::forcecast>& geom_pos,
                const py::array_t<float, py::array::c_style | py::array::forcecast>& rbound) {
            if (!cr.initialized || !cr.site_model_)
                throw std::runtime_error("Initialize renderer before setting transparency metadata");
            const auto p = geom_pos.request();
            const auto r = rbound.request();
            const int ng = cr.site_model_->ngeom();
            if (p.ndim != 2 || p.shape[0] != ng || p.shape[1] != 3 ||
                r.ndim != 1 || r.shape[0] != ng)
                throw std::runtime_error("Expected geom_pos(ngeom,3), rbound(ngeom) in native geom order");
            const auto* pv = static_cast<const float*>(p.ptr);
            const auto* rv = static_cast<const float*>(r.ptr);
            std::vector<float> positions(pv, pv + ng * 3), radii(rv, rv + ng);
            for (float v : positions) if (!std::isfinite(v))
                throw std::runtime_error("Non-finite geom position");
            for (float v : radii) if (!std::isfinite(v) || v < 0)
                throw std::runtime_error("Invalid geom bounding radius");
            cr.transparency_geom_pos.swap(positions);
            cr.transparency_rbound.swap(radii);
        }, py::arg("geom_pos"), py::arg("rbound"),
           "Set static body-local sort centers and radii from the compiled model; no physics change.")
        .def("set_site_rgba", [](CameraRenderer& cr, PhysicsEngine& engine,
                const py::array_t<float, py::array::c_style | py::array::forcecast>& rgba, int env_idx) {
            cr.set_site_rgba(engine, rgba, env_idx, false);
        }, py::arg("engine"), py::arg("rgba"), py::arg("env_idx") = 0,
           "Copy finite (nsite, 4) float32 RGBA for one world. Signed alpha is preserved; model state is unchanged.")
        .def("set_site_rgba_batch", [](CameraRenderer& cr, PhysicsEngine& engine,
                const py::array_t<float, py::array::c_style | py::array::forcecast>& rgba) {
            cr.set_site_rgba(engine, rgba, 0, true);
        }, py::arg("engine"), py::arg("rgba"),
           "Atomically copy finite (num_envs, nsite, 4) RGBA for all worlds after complete validation.")
        .def("get_site_rgba", [](CameraRenderer& cr, PhysicsEngine& engine, int env_idx) {
            const auto values = cr.site_snapshot(engine, env_idx, 1);
            py::array_t<float> result({cr.site_nsite_, 4});
            if (!values.empty()) memcpy(result.mutable_data(), values.data(), values.size() * sizeof(float));
            return result;
        }, py::arg("engine"), py::arg("env_idx") = 0,
           "Return an owned (nsite, 4) float32 copy of this world's effective site RGBA.")
        .def("get_site_rgba_batch", [](CameraRenderer& cr, PhysicsEngine& engine) {
            const auto values = cr.site_snapshot(engine, 0, -1);
            py::array_t<float> result({engine.data().num_envs(), cr.site_nsite_, 4});
            if (!values.empty()) memcpy(result.mutable_data(), values.data(), values.size() * sizeof(float));
            return result;
        }, py::arg("engine"),
           "Return an owned (num_envs, nsite, 4) float32 copy of effective site RGBA.")
        .def("clear_site_rgba", [](CameraRenderer& cr, PhysicsEngine& engine, const py::object& env_idx) {
            cr.clear_site_rgba(engine, env_idx);
        }, py::arg("engine"), py::arg("env_idx") = py::none(),
           "Restore model-init site RGBA for one world, or all worlds when env_idx is None.")
        .def("set_bg_color", [](CameraRenderer& cr, float r, float g, float b, float a) {
            cr.bg_color[0] = r; cr.bg_color[1] = g;
            cr.bg_color[2] = b; cr.bg_color[3] = a;
        }, py::arg("r"), py::arg("g"), py::arg("b"), py::arg("a") = 1.0f)
        .def("set_linear_output", [](CameraRenderer& cr, bool enabled, float exposure, float gamma) {
            // SAPIEN-style output: sRGB texture decode, linear lighting, (exposure*c)^(1/gamma).
            cr.linear_output = enabled; cr.output_exposure = exposure;
            cr.output_inv_gamma = gamma > 0.f ? 1.f / gamma : 1.f;
        }, py::arg("enabled"), py::arg("exposure") = 1.0f, py::arg("gamma") = 2.2f)
        .def("set_fog", [](CameraRenderer& cr, float start, float end, float r, float g, float b) {
            cr.fog_start = start; cr.fog_end = end;
            cr.fog_color[0] = r; cr.fog_color[1] = g; cr.fog_color[2] = b;
        }, py::arg("start"), py::arg("end"), py::arg("r") = 0.0f, py::arg("g") = 0.0f, py::arg("b") = 0.0f)
        .def("set_clip_planes", [](CameraRenderer& cr, float znear, float zfar) {
            cr.znear = znear; cr.zfar = zfar;
        }, py::arg("znear"), py::arg("zfar"))
        .def("set_vis_transparent", [](CameraRenderer& cr, bool enabled) {
            cr.vis_transparent = enabled;
        }, py::arg("enabled"), "Legacy request bit; intrinsic alpha is always rendered. Global dynamic-geom fading is not implemented.")
        .def("debug_geom_verts", [](CameraRenderer& cr, int gid, int max_verts) {
            const auto& pr = cr.phys_renderer;
            int off = pr.geom_vert_offset_host(gid);
            int cnt = pr.geom_vert_count_host(gid);
            if (off < 0 || cnt <= 0) return py::dict();
            int n = cnt < max_verts ? cnt : max_verts;
            std::vector<int> tid(n);
            std::vector<float> tc((size_t)n * TEXCOORD_COMPONENTS);
            hipMemcpy(tid.data(), pr.device_texids() + off,
                      (size_t)n * sizeof(int), hipMemcpyDeviceToHost);
            hipMemcpy(tc.data(), pr.device_texcoords() + (size_t)off * TEXCOORD_COMPONENTS,
                      (size_t)n * TEXCOORD_COMPONENTS * sizeof(float), hipMemcpyDeviceToHost);
            py::dict out;
            out["vert_offset"] = off;
            out["vert_count"] = cnt;
            out["texids"] = py::cast(tid);
            out["texcoords"] = py::cast(tc);
            out["texcoord_components"] = TEXCOORD_COMPONENTS;
            return out;
        }, py::arg("gid"), py::arg("max_verts") = 12)
        // Capture the INTERPOLATED per-fragment attributes. Answers a question
        // the rendered image cannot: for a geom that draws untextured, did the
        // fragment stage receive the texcoord at all? Zeros mean the vertex
        // attribute never reached the fragment; a correct direction means the
        // sampling decision is at fault instead.
        .def("set_capture_attrs", [](CameraRenderer& cr, bool enabled) {
            if (cr.fb) cr.fb->set_capture_attrs(enabled);
        }, py::arg("enabled"))
        .def("read_attrs", [](CameraRenderer& cr) {
            if (!cr.fb || !cr.fb->capture_attrs()) return py::array_t<float>();
            // The render is issued on the pipeline's stream and the capture is
            // written by the same kernels, so the buffer is still being written
            // when this returns without a sync. hipMemcpy on the NULL stream
            // does not order against it.
            cr.pipeline.sync();
            int h = cr.fb->height(), w = cr.fb->width();
            py::array_t<float> arr({h, w, (int)FRAG_ATTR_COUNT});
            hipMemcpy(arr.mutable_data(), cr.fb->device_attrs(),
                      (size_t)h * w * FRAG_ATTR_COUNT * sizeof(float),
                      hipMemcpyDeviceToHost);
            return arr;
        })
        .def_static("frag_attr_layout", []() {
            py::dict d;
            d["pos"] = FRAG_ATTR_POS;
            d["normal"] = FRAG_ATTR_NORMAL;
            d["texcoord"] = FRAG_ATTR_TEXCOORD;
            d["color"] = FRAG_ATTR_COLOR;
            d["count"] = FRAG_ATTR_COUNT;
            d["vertex"] = FRAG_ATTR_VERTEX;
            d["texcoord_components"] = TEXCOORD_COMPONENTS;
            return d;
        })
        // DIAGNOSTIC: per-tile triangle counts. The binning kernel keeps
        // counting past MAX_TRIANGLES_PER_TILE while DROPPING the indices, so
        // any tile above the cap has silently lost triangles -- which looks
        // exactly like an occlusion or depth bug in the final image.
        // Map a captured FRAG_ATTR_VERTEX value back to a geom index.
        // The capture stores a DRAW-LOCAL vertex (the draw's pointers are
        // already offset by the group base), so the group offset is added back
        // here. Returns -1 when no fragment won or nothing matches, rather
        // than guessing.
        .def("debug_resolve_vertex", [](CameraRenderer& cr, py::array_t<float> local_v) {
            const auto& pr = cr.phys_renderer;
            auto buf = local_v.request();
            const float* src = static_cast<const float*>(buf.ptr);
            py::array_t<int> out(buf.size);
            int* dst = out.mutable_data();
            // Build a sorted span table once per call.
            struct Span { int lo, hi, gid; };
            std::vector<Span> spans;
            for (int g = 0; g < pr.ngeom(); g++) {
                int off = pr.geom_vert_offset_host(g);
                int cnt = pr.geom_vert_count_host(g);
                if (cnt > 0) spans.push_back({off, off + cnt, g});
            }
            std::sort(spans.begin(), spans.end(),
                      [](const Span& a, const Span& b) { return a.lo < b.lo; });
            // Which group each draw-local index belongs to is ambiguous from
            // the value alone, so try every visible group's base and accept a
            // hit only if exactly one lands inside a geom span. Ambiguous or
            // unmatched -> -1.
            for (ssize_t i = 0; i < buf.size; i++) {
                float f = src[i];
                if (!(f >= 0.0f)) { dst[i] = -1; continue; }   // NaN or -1
                int local = (int)f;
                int found = -1, hits = 0;
                for (int g = 0; g < 3; g++) {
                    if (!cr.geomgroup[g]) continue;
                    int v = local + pr.group_offset(g);
                    auto it = std::upper_bound(spans.begin(), spans.end(), v,
                        [](int val, const Span& s) { return val < s.lo; });
                    if (it == spans.begin()) continue;
                    --it;
                    if (v >= it->lo && v < it->hi) { found = it->gid; hits++; }
                }
                dst[i] = (hits == 1) ? found : -1;
            }
            return out;
        }, py::arg("local_vertex"))
        .def("debug_triangle", [](CameraRenderer& cr, int tri_idx) {
            TriangleSetup t{};
            py::dict out;
            if (!cr.pipeline.debug_read_triangle(tri_idx, &t)) return out;
            auto v3 = [](const float* a) { return py::make_tuple(a[0], a[1], a[2]); };
            py::list verts;
            for (int i = 0; i < 3; i++) {
                py::dict v;
                const auto& tv = t.v[i];
                // clip.x/clip.y and ndc[] no longer exist on TransformedVertex:
                // nothing outside this dump ever read them, and deleting them
                // took TriangleSetup from 496 B to 436 B. clip.z/clip.w survive
                // because the near-plane test needs them; ndc is recoverable
                // host-side from screen + viewport.
                v["clip_zw"] = py::make_tuple(tv.clip_zw[0], tv.clip_zw[1]);
                v["screen"] = py::make_tuple(tv.screen_x, tv.screen_y, tv.screen_z);
                v["inv_w"] = tv.inv_w;
                v["world_pos"] = v3(tv.world_pos);
                verts.append(v);
            }
            out["v"] = verts;
            out["bbox"] = py::make_tuple(t.bbox_min_x, t.bbox_min_y,
                                         t.bbox_max_x, t.bbox_max_y);
            out["degenerate"] = (int)t.degenerate;
            out["backfacing"] = (int)t.backfacing;
            out["first_vertex"] = t.first_vertex;
            return out;
        }, py::arg("tri_idx"))
        .def_static("set_min_tri_area", [](float v) {
            hipgl::kernels::set_min_tri_area(v);
        }, py::arg("area"))
        .def("debug_enable_tri_stats", [](CameraRenderer& cr, bool on) {
            cr.pipeline.debug_enable_tri_stats(on);
        }, py::arg("on"))
        // Per-triangle stage counters aggregated to GEOMS. Rows are
        // TRI_STAT_* in pipeline order, so reading a geom's row left to right
        // shows where its triangles stop surviving. Returns counts for the
        // LAST draw call issued -- the renderer draws one group per call, so
        // read it after rendering and pass the group whose geoms you want.
        .def("debug_tri_stats", [](CameraRenderer& cr, int group) {
            int n = cr.pipeline.debug_tri_stats_count();
            const auto& pr = cr.phys_renderer;
            py::dict out;
            if (n <= 0 || !cr.pipeline.debug_tri_stats_enabled()) return out;
            std::vector<int> raw((size_t)n * TRI_STAT_COUNT);
            cr.pipeline.debug_read_tri_stats(raw.data(), n);
            // Near-plane clipping reserves primary i and suffix N+i for
            // source triangle i. Both rows belong to vertices [3*i, 3*i+3)
            // LOCAL to the draw's base, including a clipped triangle's fan.
            // Add the group offset to map back to the source geom spans.
            int base = (group >= 0 && group < 3) ? pr.group_offset(group) : 0;
            py::dict per_geom;
            std::vector<std::array<long long, TRI_STAT_COUNT>> acc(pr.ngeom());
            for (auto& a : acc) a.fill(0);
            std::vector<long long> tris(pr.ngeom(), 0);
            for (int t = 0; t < n; t++) {
                int v = base + (t < n / 2 ? t : t - n / 2) * 3;
                int gid = -1;
                for (int g = 0; g < pr.ngeom(); g++) {
                    int off = pr.geom_vert_offset_host(g);
                    int cnt = pr.geom_vert_count_host(g);
                    if (cnt > 0 && v >= off && v < off + cnt) { gid = g; break; }
                }
                if (gid < 0) continue;
                tris[gid]++;
                for (int k = 0; k < TRI_STAT_COUNT; k++)
                    acc[gid][k] += raw[(size_t)t * TRI_STAT_COUNT + k];
            }
            for (int g = 0; g < pr.ngeom(); g++) {
                if (tris[g] == 0) continue;
                py::list row;
                for (int k = 0; k < TRI_STAT_COUNT; k++) row.append(acc[g][k]);
                per_geom[py::int_(g)] = row;
            }
            out["per_geom"] = per_geom;
            out["stat_names"] = py::make_tuple(
                "input", "clipped", "degenerate", "backface", "offscreen",
                "binned", "bin_overflow", "covered", "depth_pass");
            out["tri_count"] = n;
            return out;
        }, py::arg("group") = 1)
        .def("debug_tile_counts", [](CameraRenderer& cr) {
            int n = cr.pipeline.debug_num_tiles();
            py::array_t<int> arr(n > 0 ? n : 0);
            if (n > 0) cr.pipeline.debug_tile_counts(arr.mutable_data(), n);
            py::dict out;
            out["counts"] = arr;
            out["cap"] = MAX_TRIANGLES_PER_TILE;
            out["tile_size"] = TILE_SIZE;
            return out;
        })
        .def("debug_tile_contents", [](CameraRenderer& cr, int tile_idx, int max_entries) {
            // Bin CONTENTS for one tile, not just its count. Identical counts do
            // NOT imply identical contents: atomicAdd makes the total
            // deterministic while the slot assignment is not.
            if (max_entries <= 0) max_entries = MAX_TRIANGLES_PER_TILE;
            if (max_entries > MAX_TRIANGLES_PER_TILE) max_entries = MAX_TRIANGLES_PER_TILE;
            std::vector<int> idx(max_entries);
            std::vector<float> zs(max_entries);
            int count = 0;
            cr.pipeline.debug_tile_contents(tile_idx, idx.data(), zs.data(), max_entries, &count);
            int n = count < max_entries ? count : max_entries;
            if (n < 0) n = 0;
            py::array_t<int> aidx(n);
            py::array_t<float> az(n);
            for (int i = 0; i < n; i++) {
                aidx.mutable_data()[i] = idx[i];
                az.mutable_data()[i]   = zs[i];
            }
            py::dict out;
            out["count"] = count;      // raw count, may exceed the cap
            out["indices"] = aidx;
            out["z_min"] = az;
            return out;
        }, py::arg("tile_idx"), py::arg("max_entries") = 0)
        .def("debug_draw_ranges", [](CameraRenderer& cr) {
            const auto& pr = cr.phys_renderer;
            py::dict out;
            py::list groups;
            for (int g = 0; g < 3; g++) {
                py::dict d;
                d["offset"] = pr.group_offset(g);
                d["count"]  = pr.group_count(g);
                d["visible"] = cr.geomgroup[g];
                groups.append(d);
            }
            out["groups"] = groups;
            out["transparent_offset"] = pr.transparent_offset();
            out["transparent_count"]  = pr.transparent_count();
            out["device_site_vertices"] = pr.site_vertices();
            out["vis_transparent"] = cr.vis_transparent;
            out["transparency_sort_metadata"] = !cr.transparency_geom_pos.empty();
            return out;
        })
        .def("set_geomgroup", [](CameraRenderer& cr, int group, bool visible) {
            if (group >= 0 && group < 6) cr.geomgroup[group] = visible;
        }, py::arg("group"), py::arg("visible"),
           "Set visibility for a geom group (0-5). Default: groups 0,1,2 visible.")
        .def("set_gouraud_shading", [](CameraRenderer& cr, bool enabled) {
            cr.gouraud_shading = enabled;
        }, py::arg("enabled"),
           "Enable/disable per-vertex Gouraud shading (default: true, matches MuJoCo). "
           "When disabled, uses per-fragment Blinn-Phong.")
        .def("set_reflection", [](CameraRenderer& cr, bool enabled) {
            cr.reflection_enabled = enabled;
        }, py::arg("enabled"),
           "Enable/disable the planar reflection pass (default: true, matches MuJoCo).")
        .def("set_shadow", [](CameraRenderer& cr, bool enabled) {
            cr.shadow_enabled = enabled;
        }, py::arg("enabled"),
           "Enable/disable headlight shadow mapping (default: false).")
        // Debug: the exact view/projection/MVP this renderer would use, built by
        // the SAME build_camera_state() the render path calls. Recomputing these
        // in Python would only prove Python agrees with itself; this returns what
        // the rasteriser actually consumes.
        .def("debug_camera_matrices", [](
                CameraRenderer& cr, PhysicsEngine& engine,
                int camera_idx, int env_idx) -> py::dict {
            if (!cr.initialized) throw std::runtime_error("CameraRenderer not initialized");
            engine.sync();
            const PhysicsModelDesc& desc = engine.model()->desc();
            if (camera_idx < 0 || camera_idx >= desc.ncamera)
                throw std::runtime_error("Invalid camera index");
            const CameraDesc& cam = desc.cameras[camera_idx];
            int nb = desc.nbody;
            int nenvs = engine.data().num_envs();
            std::vector<float> h_xpos(nenvs * nb * 3), h_xquat(nenvs * nb * 4);
            engine.data().read_xpos(h_xpos.data());
            engine.data().read_xquat(h_xquat.data());
            RenderState st = build_camera_state(
                cam.pos, cam.quat, cam.fovy, cr.width, cr.height,
                h_xpos.data() + env_idx * nb * 3,
                h_xquat.data() + env_idx * nb * 4,
                cam.body, desc, cr.bg_color,
                cr.gradient_bg, cr.gradient_top, cr.gradient_bot);
            auto mk = [](const float* m) {
                py::array_t<float> a({4, 4});
                memcpy(a.mutable_data(), m, 16 * sizeof(float));
                return a;
            };
            py::dict out;
            out["model_view"] = mk(st.model_view);
            out["projection"] = mk(st.projection);
            out["mvp"] = mk(st.mvp);
            out["camera_world_pos"] = py::make_tuple(
                st.camera_world_pos[0], st.camera_world_pos[1], st.camera_world_pos[2]);
            out["camera_forward"] = py::make_tuple(
                st.camera_forward[0], st.camera_forward[1], st.camera_forward[2]);
            out["fovy"] = cam.fovy;
            out["viewport"] = py::make_tuple(st.viewport_w, st.viewport_h);
            out["stat_extent"] = desc.stat_extent;
            out["znear"] = desc.map_znear * desc.stat_extent;
            out["zfar"] = desc.map_zfar * desc.stat_extent;
            return out;
        }, py::arg("engine"), py::arg("camera_idx"), py::arg("env_idx") = 0,
           "Debug: view/projection/MVP actually used for this camera.")
        .def("read_shadow_depth", [](CameraRenderer& cr) -> py::array_t<float> {
            if (!cr.shadow_fb) throw std::runtime_error("Shadow FB not initialized");
            int sz = cr.shadow_size;
            py::array_t<float> depth({sz, sz});
            cr.shadow_fb->read_depth(depth.mutable_data());
            return depth;
        }, "Read shadow depth buffer (debug)")
        .def("set_gradient_bg", [](CameraRenderer& cr,
                float bot_r, float bot_g, float bot_b,
                float top_r, float top_g, float top_b) {
            cr.gradient_bg = true;
            cr.gradient_bot[0] = bot_r; cr.gradient_bot[1] = bot_g; cr.gradient_bot[2] = bot_b;
            cr.gradient_top[0] = top_r; cr.gradient_top[1] = top_g; cr.gradient_top[2] = top_b;
        }, py::arg("bot_r"), py::arg("bot_g"), py::arg("bot_b"),
           py::arg("top_r"), py::arg("top_g"), py::arg("top_b"))
        .def("set_num_streams", [](CameraRenderer& cr, int n) {
            cr.num_workers_ = std::max(1, std::min(n, 32));
            // Workers are lazily created on next render_gpu_batch call
        }, py::arg("n") = 4,
           "Set number of parallel GPU streams for render_gpu_batch (default 4, max 16)")
        .def("render", [do_render_common](CameraRenderer& cr, PhysicsEngine& engine,
                          int camera_idx, int env_idx) -> py::array_t<uint8_t> {
            do_render_common(cr, engine, camera_idx, env_idx);

            py::array_t<uint8_t> img({cr.out_height, cr.out_width, 3});
            std::vector<uint8_t> rgba((size_t)cr.width * cr.height * 4);
            cr.fb->read_color_rgba8(rgba.data());

            auto ptr = img.mutable_data();
            const int ss = cr.ssaa;
            if (ss <= 1) {
                for (int i = 0; i < cr.width * cr.height; i++) {
                    ptr[i*3]   = rgba[i*4];
                    ptr[i*3+1] = rgba[i*4+1];
                    ptr[i*3+2] = rgba[i*4+2];
                }
            } else {
                // Box-average ss x ss, same filter as the GPU resolve. Rounding
                // matches (+0.5 before truncation) so this path and render_gpu*
                // agree to the last bit.
                const int inv_n = ss * ss;
                for (int oy = 0; oy < cr.out_height; oy++) {
                    for (int ox = 0; ox < cr.out_width; ox++) {
                        int acc[3] = {0, 0, 0};
                        for (int dy = 0; dy < ss; dy++) {
                            const int sy = oy * ss + dy;
                            for (int dx = 0; dx < ss; dx++) {
                                const size_t si = ((size_t)sy * cr.width + (size_t)ox * ss + dx) * 4;
                                acc[0] += rgba[si]; acc[1] += rgba[si+1]; acc[2] += rgba[si+2];
                            }
                        }
                        const size_t o = ((size_t)oy * cr.out_width + ox) * 3;
                        for (int c = 0; c < 3; c++)
                            ptr[o+c] = (uint8_t)((acc[c] + inv_n / 2) / inv_n);
                    }
                }
            }
            return img;
        }, py::arg("engine"), py::arg("camera_idx"), py::arg("env_idx") = 0,
           "Render a camera view, returns (H, W, 3) uint8 numpy array")
        .def("render_depth", [do_render_common](CameraRenderer& cr, PhysicsEngine& engine,
                          int camera_idx, int env_idx) -> py::array_t<float> {
            do_render_common(cr, engine, camera_idx, env_idx);
            // NOTE depth is returned at the RENDER resolution, not the ssaa
            // output size: averaging depth across a supersample block is not
            // meaningful (it would invent surfaces between the near and far
            // sample). Callers using ssaa>1 with depth must handle the larger
            // buffer themselves.
            py::array_t<float> depth({cr.height, cr.width});
            cr.fb->read_depth(depth.mutable_data());
            return depth;
        }, py::arg("engine"), py::arg("camera_idx"), py::arg("env_idx") = 0,
           "Render and return the depth buffer as (H, W) float32 array")
        .def("render_gpu", [do_render_common](CameraRenderer& cr, PhysicsEngine& engine,
                              int camera_idx, int env_idx,
                              bool flip_v, bool flip_h) -> GpuImage {
            do_render_common(cr, engine, camera_idx, env_idx);

            DeviceFramebuffer dfb = cr.fb->device_fb();
            hipgl::kernels::launch_rgba_f32_to_rgb_u8_ss(
                dfb.color_buffer, cr.d_rgb,
                cr.out_width, cr.out_height, cr.ssaa,
                flip_v, flip_h, nullptr);
            hipDeviceSynchronize();

            return GpuImage{cr.d_rgb, cr.out_height, cr.out_width};
        }, py::arg("engine"), py::arg("camera_idx"), py::arg("env_idx") = 0,
           py::arg("flip_v") = false, py::arg("flip_h") = false,
           "Render a camera view, returns GpuImage with __cuda_array_interface__ for zero-copy PyTorch interop")
        .def("render_gpu_batch", [do_render_common](CameraRenderer& cr, PhysicsEngine& engine,
                              int camera_idx, int num_envs,
                              bool flip_v, bool flip_h, py::object active) -> GpuImageBatch {
            if (!cr.initialized) throw std::runtime_error("CameraRenderer not initialized");
            if (num_envs <= 0) throw std::runtime_error("num_envs must be positive");
            std::vector<unsigned char> act_mask((size_t)num_envs, 1);
            int n_active = num_envs;
            if (!active.is_none()) {
                std::fill(act_mask.begin(), act_mask.end(), (unsigned char)0);
                n_active = 0;
                for (auto h : active) {
                    const int e = h.cast<int>();
                    if (e < 0 || e >= num_envs)
                        throw std::runtime_error("render_gpu_batch: active world outside the batch");
                    if (!act_mask[e]) { act_mask[e] = 1; n_active++; }
                }
                if (n_active == 0)
                    throw std::runtime_error("render_gpu_batch: empty active world set");
            }
            const bool masked = n_active < num_envs;
            const int hp_trace = hp_host_trace();
            double t_entry = hp_trace ? hp_now_ms() : 0.0;
            double t_pose = 0.0, t_cam = 0.0, t_issue = 0.0;
            const auto site_rgba = cr.site_snapshot(engine, 0, num_envs);
            hipStream_t s = engine.stream();
            const PhysicsModelDesc& desc = engine.model()->desc();
            if (camera_idx < 0 || camera_idx >= desc.ncamera)
                throw std::runtime_error("Invalid camera index");

            cr.ensure_batch_buffer(num_envs);
            size_t img_bytes = (size_t)cr.out_width * cr.out_height * 3;
            const CameraDesc& cam = desc.cameras[camera_idx];
            int nb = desc.nbody;
            // Every consumer uses the same copied per-world appearance. A site
            // hidden in XML may be visible now, including in just one worker.
            bool need_site_poses = false;
            for (int ei = 0; ei < num_envs && !need_site_poses; ++ei) {
                const float* rgba = site_rgba.empty() ? nullptr
                    : site_rgba.data() + (size_t)ei * desc.nsite * 4;
                need_site_poses = any_site_overlay_visible(desc, rgba);
            }
            const size_t body_count = (size_t)num_envs * nb;
            const size_t site_count = need_site_poses ? (size_t)num_envs * desc.nsite : 0;
            cr.ensure_batch_poses(body_count * 7 + site_count * 12);
            float* h_xpos = cr.h_batch_poses_;
            float* h_xquat = h_xpos + body_count * 3;
            float* h_site_xpos = site_count ? h_xquat + body_count * 4 : nullptr;
            float* h_site_xmat = site_count ? h_site_xpos + site_count * 3 : nullptr;
            const auto& dd = engine.data().device_data();
            // The engine stream orders all copies after current forward/step
            // work. Wait once for BOTH host poses and device geometry readiness.
            // Drain even an incomplete submission before storage can be reused.
            auto readback = [s](float* dst, const float* src, size_t count) {
                hipError_t err = hipMemcpyAsync(dst, src, count * sizeof(float),
                                               hipMemcpyDeviceToHost, s);
                if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
            };
            try {
                // xpos/xquat are two separate allocations. Even when the pool
                // hands them back adjacent, one copy spanning both is rejected
                // by HIP ("invalid argument"; seen at >=512 worlds), so always
                // issue one copy per allocation.
                readback(h_xpos, dd.xpos, body_count * 3);
                readback(h_xquat, dd.xquat, body_count * 4);
                if (site_count) {
                    readback(h_site_xpos, dd.site_xpos, site_count * 3);
                    readback(h_site_xmat, dd.site_xmat, site_count * 9);
                }
            } catch (...) {
                hipStreamSynchronize(s);
                throw;
            }
            const int want_workers = hp_desired_workers(cr.width, cr.num_workers_);
            int actual_workers = std::min(want_workers, num_envs);
            if (actual_workers > 1) {
                cr.ensure_workers(desc, num_envs, want_workers);
                actual_workers = std::min(actual_workers, (int)cr.workers_.size());
            }
            const DevicePhysicsModel& device_model = engine.model()->device_model();
            hipStream_t solo_rs = (actual_workers <= 1 && hp_solo_stream())
                                ? cr.ensure_solo_stream() : nullptr;
            std::unique_ptr<py::gil_scoped_release> solo_nogil;
            if (solo_rs) solo_nogil.reset(new py::gil_scoped_release());
            // Number of LEADING worlds whose geometry build is already on the
            // GPU when the host finishes waiting for the poses.
            int prefetched_geom = 0;
            if (cr.pose_evt_ && hipEventRecord(cr.pose_evt_, s) == hipSuccess) {
                unsigned gm = 0;
                for (int g = 0; g < 3; g++) if (cr.geomgroup[g]) gm |= (1u << g);
                if (actual_workers <= 1) {
                    hipStream_t prs = solo_rs ? solo_rs : s;
                    if (prs != s) hipStreamWaitEvent(prs, cr.pose_evt_, 0);
                    cr.phys_renderer.set_active_groups(gm, cr.vis_transparent);
                    cr.phys_renderer.build_geometry(
                        device_model, engine.data().device_data(), 0, prs);
                    prefetched_geom = 1;
                }
            }
            hipError_t pose_status = prefetched_geom
                ? hipEventSynchronize(cr.pose_evt_)
                : hipStreamSynchronize(s);
            if (pose_status != hipSuccess)
                throw std::runtime_error(hipGetErrorString(pose_status));
            if (hp_trace) t_pose = hp_now_ms();

            TextureState texture_state = {
                device_model.tex_data,
                device_model.tex_width,
                device_model.tex_height,
                device_model.tex_offset,
                device_model.ntexture,
                device_model.tex_iscube,
                device_model.tex_mip
            };


            using namespace hipgl::physics::kernels;

            std::vector<RenderState>& batch_states = cr.batch_states_;
            std::vector<PlanarReflectionPlan>& reflection_plans = cr.reflection_plans_;
            if ((int)batch_states.size() < num_envs) batch_states.resize((size_t)num_envs);
            if ((int)reflection_plans.size() < num_envs) reflection_plans.resize((size_t)num_envs);
            std::vector<std::vector<TransparentGeomSpan>> transparent_orders((size_t)num_envs);
            bool any_transparent_geom = false;
            bool any_reflective_geom = false;
            cr.ensure_batch_matrices(num_envs);
            for (int ei = 0; ei < num_envs; ei++) {
                float* e_xpos = h_xpos + ei * nb * 3;
                float* e_xquat = h_xquat + ei * nb * 4;
                // All other camera/render inputs are common to this invocation
                // (the binding holds the GIL). Reuse only within this batch and
                // only for bit-identical, freshly read parent transforms.
                const bool same_parent = ei > 0 &&
                    memcmp(e_xpos + cam.body * 3, e_xpos - nb * 3 + cam.body * 3,
                           3 * sizeof(float)) == 0 &&
                    memcmp(e_xquat + cam.body * 4, e_xquat - nb * 4 + cam.body * 4,
                           4 * sizeof(float)) == 0;
                RenderState st = same_parent ? batch_states[ei - 1] : build_camera_state(
                    cam.pos, cam.quat, cam.fovy,
                    cr.width, cr.height,
                    e_xpos, e_xquat, cam.body, desc, cr.bg_color,
                    cr.gradient_bg, cr.gradient_top, cr.gradient_bot,
                    cr.znear, cr.zfar);
                st.texture_2d_enabled = (texture_state.tex_data && texture_state.ntexture > 0);
                st.fog_start = cr.fog_start;
                st.fog_end = cr.fog_end;
                st.fog_color[0] = cr.fog_color[0];
                st.fog_color[1] = cr.fog_color[1];
                st.fog_color[2] = cr.fog_color[2];
                st.linear_output = cr.linear_output ? 1 : 0;
                st.output_exposure = cr.output_exposure;
                st.output_inv_gamma = cr.output_inv_gamma;
                st.gouraud_shading = cr.gouraud_shading ? 1 : 0;
                float* hm = cr.h_batch_matrices_
                          + (size_t)ei * HP_MAT_STRIDE;
                memcpy(hm,      st.mvp,           16 * sizeof(float));
                memcpy(hm + 16, st.model_view,    16 * sizeof(float));
                memcpy(hm + 32, st.normal_matrix,  9 * sizeof(float));
                batch_states[ei] = st;
                reflection_plans[ei] = make_planar_reflection_plan(
                    desc, st, e_xpos, e_xquat, cr.geomgroup, cr.reflection_enabled);
                hp_scrub_padding(batch_states[ei]);
                hp_scrub_padding(reflection_plans[ei]);
                if (reflection_plans[ei].render_scene)
                    memcpy(hm + 41, reflection_plans[ei].matrices.data(), 41 * sizeof(float));
                else
                    memset(hm + 41, 0, 41 * sizeof(float));
                any_reflective_geom |= reflection_plans[ei].geom_id >= 0;
                transparent_orders[ei] = make_transparent_draw_order(
                    desc, st, e_xpos, e_xquat, cr.geomgroup,
                    cr.transparency_geom_pos.empty() ? nullptr : cr.transparency_geom_pos.data(),
                    cr.transparency_rbound.empty() ? nullptr : cr.transparency_rbound.data());
                any_transparent_geom |= !transparent_orders[ei].empty();
            }
            const bool any_site_overlay = need_site_poses;
            int site_slot_verts = 0;
            bool site_stage_ok = false;
            if (need_site_poses && actual_workers > 1 && !cr.site_stage_broken_ &&
                !site_rgba.empty() && desc.nsite > 0) {
                const int* pv = cr.site_vert_counts(desc);
                int maxv = 0;
                for (int ei = 0; ei < num_envs && pv; ++ei) {
                    const float* rgba = site_rgba.data() + (size_t)ei * desc.nsite * 4;
                    int v = 0;
                    for (int si = 0; si < desc.nsite; ++si)
                        if (site_overlay_visible(desc.sites[si], rgba + (size_t)si * 4))
                            v += pv[si];
                    if (v > maxv) maxv = v;
                }
                site_slot_verts = maxv;
                site_stage_ok = maxv > 0 && cr.ensure_site_stage(num_envs * 2, maxv)
                                         && cr.ensure_site_dev(actual_workers, maxv);
            }
            const bool defer_bg =
                !cr.gradient_bg && !any_site_overlay && !any_transparent_geom &&
                !cr.shadow_enabled && !any_reflective_geom &&
                cr.ssaa >= 2 && (cr.out_width & 3) == 0;
            const bool use_batch_mat = (cr.d_batch_matrices_ != nullptr &&
                                        cr.h_batch_matrices_ != nullptr);
            bool batch_mat_evt_valid = false;
            if (use_batch_mat) {
                const size_t mat_bytes =
                    (size_t)num_envs * HP_MAT_STRIDE * sizeof(float);
                const bool mat_resident =
                    cr.batch_mat_resident_ && cr.h_batch_mat_shadow_ &&
                    cr.batch_mat_shadow_envs_ >= num_envs &&
                    memcmp(cr.h_batch_mat_shadow_, cr.h_batch_matrices_, mat_bytes) == 0;
                if (!mat_resident) {
                    hipMemcpyAsync(cr.d_batch_matrices_, cr.h_batch_matrices_,
                                   mat_bytes, hipMemcpyHostToDevice, s);
                    if (cr.batch_mat_evt_ && hipEventRecord(cr.batch_mat_evt_, s) == hipSuccess)
                        batch_mat_evt_valid = true;
                    else
                        hipStreamSynchronize(s);   // fall back to the old guarantee
                    if (cr.h_batch_mat_shadow_ && cr.batch_mat_shadow_envs_ >= num_envs) {
                        memcpy(cr.h_batch_mat_shadow_, cr.h_batch_matrices_, mat_bytes);
                        cr.batch_mat_resident_ = true;
                    }
                }
            }

            if (hp_trace) t_cam = hp_now_ms();
            if (actual_workers <= 1) {
                cr.pipeline.set_defer_pool_sizing(true);
            } else {
                for (int wi = 0; wi < actual_workers; wi++)
                    cr.workers_[wi]->pipeline->set_defer_pool_sizing(true);
            }
            for (int attempt = 0; attempt < 2; ++attempt) {
            if (actual_workers <= 1) {
                // Single-stream path: use original resources (backward compatible)
                hipStream_t old_stream = cr.pipeline.stream();
                hipStream_t rs = solo_rs ? solo_rs : s;
                if (rs != s) {
                    // Order the private stream behind the engine: the poses are
                    // already host-synced, but the batched MVP upload is issued
                    // on s AFTER that wait and the draws read it.
                    if (batch_mat_evt_valid)
                        hipStreamWaitEvent(rs, cr.batch_mat_evt_, 0);
                    else if (attempt == 0 && !prefetched_geom)
                        hipStreamWaitEvent(rs, cr.pose_evt_, 0);
                }
                cr.pipeline.set_stream(rs);

                for (int ei = 0; ei < num_envs; ei++) {
                    if (!act_mask[ei]) continue;
                    float* env_site_xpos = !h_site_xpos ? nullptr : h_site_xpos + ei * desc.nsite * 3;
                    float* env_site_xmat = !h_site_xmat ? nullptr : h_site_xmat + ei * desc.nsite * 9;
                    const float* env_site_rgba = site_rgba.empty() ? nullptr
                        : site_rgba.data() + (size_t)ei * desc.nsite * 4;

                    const RenderState& state = batch_states[ei];

                    if (ei < prefetched_geom) {
                        // already enqueued ahead of the pose wait
                    } else {
                        { unsigned m = 0; for (int g = 0; g < 3; g++) if (cr.geomgroup[g]) m |= (1u << g);
                          cr.phys_renderer.set_active_groups(m, cr.vis_transparent); }
                        cr.phys_renderer.build_geometry(
                            device_model, engine.data().device_data(), ei, rs);
                    }

                    const float* matrices = use_batch_mat
                        ? cr.d_batch_matrices_ + (size_t)ei * HP_MAT_STRIDE : nullptr;
                    const float* refl_matrices = matrices ? matrices + 41 : nullptr;
                    cr.pipeline.set_external_matrices(matrices);
                    cr.pipeline.set_deferred_bg(defer_bg);
                    cr.pipeline.set_render_state(state);
                    cr.pipeline.clear_deferred(true, true);
                    draw_opaque_with_planar_reflection(
                        cr.pipeline, *cr.fb, cr.phys_renderer, state, texture_state,
                        cr.geomgroup, transparent_orders[ei], reflection_plans[ei],
                        cr.reflection_resources, matrices, refl_matrices, [&](const RenderState& reflected) {
                            draw_site_overlays(cr.pipeline, reflected, desc, env_site_xpos, env_site_xmat, env_site_rgba);
                        });

                    draw_intrinsic_transparency(cr.pipeline, cr.phys_renderer, state, texture_state, transparent_orders[ei]);

                    draw_site_overlays(cr.pipeline, state, desc, env_site_xpos, env_site_xmat, env_site_rgba);
                    cr.pipeline.set_external_matrices(nullptr);

                    cr.pipeline.resolve();
                    DeviceFramebuffer dfb = cr.fb->device_fb();
                    uint8_t* dst = cr.d_rgb_batch + ei * img_bytes;
                    bool cvt_rides = false;
                    if (HP_DEFER_CONVERT && !cr.pipeline.deferred_bg() &&
                        cr.ssaa == 1 && !flip_v && !flip_h)
                        cvt_rides = cr.pipeline.convert_deferred(
                            dfb.color_buffer, dst, cr.out_width * cr.out_height);
                    if (!cvt_rides) {
                    if (!(cr.pipeline.deferred_bg() &&
                          hipgl::kernels::launch_rgba_f32_to_rgb_u8_ss_bg(
                              dfb.color_buffer, dfb.depth_buffer, dst,
                              cr.out_width, cr.out_height, cr.ssaa, flip_v, flip_h,
                              cr.bg_color[0], cr.bg_color[1], cr.bg_color[2],
                              state.clear_depth, rs)))
                        hipgl::kernels::launch_rgba_f32_to_rgb_u8_ss(
                            dfb.color_buffer, dst, cr.out_width, cr.out_height, cr.ssaa,
                            flip_v, flip_h, rs);
                    }
                }
                prefetched_geom = 0;   // a retry must rebuild every world
                cr.pipeline.flush_pending_convert();
                cr.pipeline.queue_overflow_read();
                hipStreamSynchronize(rs);
                cr.pipeline.set_stream(old_stream);
            } else {
                // Multi-stream path: distribute environments across workers
                if (batch_mat_evt_valid)
                    for (int wi = 0; wi < actual_workers; wi++)
                        hipStreamWaitEvent(cr.workers_[wi]->stream, cr.batch_mat_evt_, 0);
                static const int hp_skip_rebal = []{
                    const char* e = getenv("HP_SKIP_REBAL");
                    return e ? atoi(e) : 1;
                }();
                std::vector<std::vector<int>> wlist((size_t)actual_workers);
                {
                    int rank = 0;
                    for (int ei = 0; ei < num_envs; ei++) {
                        if (!act_mask[ei]) continue;
                        const int wi = (masked && hp_skip_rebal) ? (rank % actual_workers)
                                                                  : (ei % actual_workers);
                        wlist[(size_t)wi].push_back(ei);
                        rank++;
                    }
                }
                auto issue_world = [&](int ei, int owner, bool geom_batched) {
                    auto& w = *cr.workers_[owner];
                    hipStream_t ws = w.stream;

                    float* env_site_xpos = !h_site_xpos ? nullptr : h_site_xpos + ei * desc.nsite * 3;
                    float* env_site_xmat = !h_site_xmat ? nullptr : h_site_xmat + ei * desc.nsite * 9;
                    const float* env_site_rgba = site_rgba.empty() ? nullptr
                        : site_rgba.data() + (size_t)ei * desc.nsite * 4;

                    const RenderState& state = batch_states[ei];

                    double q0 = hp_trace >= 2 ? hp_now_ms() : 0.0;
                    if (geom_batched) {
                        w.phys_renderer->arm_pose(engine.data().device_data(), ei);
                    } else if (ei < prefetched_geom) {
                        // already enqueued ahead of the pose wait
                    } else {
                        { unsigned m = 0; for (int g = 0; g < 3; g++) if (cr.geomgroup[g]) m |= (1u << g);
                          w.phys_renderer->set_active_groups(m, cr.vis_transparent); }
                        w.phys_renderer->build_geometry(
                            device_model, engine.data().device_data(), ei, ws);
                    }
                    double q1 = hp_trace >= 2 ? hp_now_ms() : 0.0;

                    const float* matrices = use_batch_mat
                        ? cr.d_batch_matrices_ + (size_t)ei * HP_MAT_STRIDE : nullptr;
                    const float* refl_matrices = matrices ? matrices + 41 : nullptr;
                    w.pipeline->set_external_matrices(matrices);
                    w.pipeline->set_deferred_bg(defer_bg);
                    w.pipeline->set_render_state(state);
                    double q2 = hp_trace >= 2 ? hp_now_ms() : 0.0;
                    w.pipeline->clear_deferred(true, true);
                    double q3 = hp_trace >= 2 ? hp_now_ms() : 0.0;
                    draw_opaque_with_planar_reflection(
                        *w.pipeline, *w.fb, *w.phys_renderer, state, texture_state,
                        cr.geomgroup, transparent_orders[ei], reflection_plans[ei],
                        w.reflection_resources, matrices, refl_matrices, [&](const RenderState& reflected) {
                            draw_site_overlays(*w.pipeline, reflected, desc, env_site_xpos, env_site_xmat, env_site_rgba);
                        });
                    double q4 = hp_trace >= 2 ? hp_now_ms() : 0.0;

                    draw_intrinsic_transparency(*w.pipeline, *w.phys_renderer, state, texture_state, transparent_orders[ei]);

                    draw_site_overlays(*w.pipeline, state, desc,
                                       env_site_xpos, env_site_xmat, env_site_rgba,
                                       site_stage_ok ? cr.site_slot(ei * 2 + 1) : nullptr,
                                       site_slot_verts,
                                       site_stage_ok ? &cr.site_slot_ctx_[ei * 2 + 1] : nullptr,
                                       site_stage_ok ? cr.d_site_dev_[owner] : nullptr);
                    w.pipeline->set_external_matrices(nullptr);

                    w.pipeline->resolve();
                    DeviceFramebuffer dfb = w.fb->device_fb();
                    uint8_t* dst = cr.d_rgb_batch + ei * img_bytes;
                    bool cvt_rides = false;
                    if (HP_DEFER_CONVERT && !w.pipeline->deferred_bg() &&
                        cr.ssaa == 1 && !flip_v && !flip_h)
                        cvt_rides = w.pipeline->convert_deferred(
                            dfb.color_buffer, dst, cr.out_width * cr.out_height);
                    if (!cvt_rides) {
                    if (!(w.pipeline->deferred_bg() &&
                          hipgl::kernels::launch_rgba_f32_to_rgb_u8_ss_bg(
                              dfb.color_buffer, dfb.depth_buffer, dst,
                              cr.out_width, cr.out_height, cr.ssaa, flip_v, flip_h,
                              cr.bg_color[0], cr.bg_color[1], cr.bg_color[2],
                              state.clear_depth, ws)))
                        hipgl::kernels::launch_rgba_f32_to_rgb_u8_ss(
                            dfb.color_buffer, dst, cr.out_width, cr.out_height, cr.ssaa,
                            flip_v, flip_h, ws);
                    }
                    if (hp_trace >= 2) {
                        const double q5 = hp_now_ms();
                        static thread_local double b_geo=0,b_st=0,b_clr=0,b_draw=0,b_tail=0;
                        static thread_local int bn=0;
                        b_geo += q1-q0; b_st += q2-q1; b_clr += q3-q2;
                        b_draw += q4-q3; b_tail += q5-q4;
                        if (++bn % 512 == 0) {
                            fprintf(stderr, "[worldtrace] per-world us: geom=%.2f state=%.2f "
                                    "clear=%.2f drawopaque=%.2f tail(sites+resolve+rgb)=%.2f sum=%.2f\n",
                                    b_geo*1e3/512, b_st*1e3/512, b_clr*1e3/512,
                                    b_draw*1e3/512, b_tail*1e3/512,
                                    (b_geo+b_st+b_clr+b_draw+b_tail)*1e3/512);
                            fflush(stderr);
                            b_geo=b_st=b_clr=b_draw=b_tail=0;
                        }
                    }
                };
                static const int hp_geom_batch = []{
                    const char* e = getenv("HP_GEOM_BATCH");
                    int v = e ? atoi(e) : 2;
                    if (v < 1) v = 1;
                    if (v > 64) v = 64;
                    return v;
                }();
                auto issue_worker = [&](int wi) {
                    auto& w = *cr.workers_[wi];
                    hipStream_t ws = w.stream;
                    int kmax = hp_geom_batch;
                    if (kmax > 1 && prefetched_geom == 0) {
                        unsigned m = 0;
                        for (int g = 0; g < 3; g++) if (cr.geomgroup[g]) m |= (1u << g);
                        w.phys_renderer->set_active_groups(m, cr.vis_transparent);
                    } else {
                        kmax = 1;
                    }
                    const std::vector<int>& L = wlist[(size_t)wi];
                    for (size_t p = 0; p < L.size(); ) {
                        int kc = 1;
                        int stride = actual_workers;
                        if (kmax > 1 && p + 1 < L.size()) {
                            stride = L[p + 1] - L[p];
                            while (kc < kmax && p + kc < L.size() &&
                                   L[p + kc] - L[p + kc - 1] == stride)
                                kc++;
                        }
                        bool batched = false;
                        if (kc > 1)
                            batched = w.phys_renderer->build_geometry_batch(
                                device_model, engine.data().device_data(),
                                L[p], stride, kc, ws);
                        for (int j = 0; j < kc; j++) {
                            w.phys_renderer->set_position_slice(batched ? j : 0);
                            issue_world(L[p + j], wi, batched);
                        }
                        p += (size_t)kc;
                    }
                    w.phys_renderer->set_position_slice(0);
                };

                const long long par_tiles =
                    (long long)((cr.width + 15) / 16) * ((cr.height + 15) / 16);
                const bool par_ok = hp_par_issue() && actual_workers > 1 &&
                    num_envs >= hp_par_issue_min_envs() &&
                    par_tiles <= hp_par_issue_max_tiles();

                for (int wi = 0; wi < actual_workers; wi++) {
                    cr.workers_[wi]->pipeline->reset_bin_rotation();
                    cr.workers_[wi]->phys_renderer->set_position_slice(0);
                }

                bool graph_hit = false;
                const bool graph_ok = hp_render_graph() && hp_par_issue() &&
                                      actual_workers > 1 &&
                                      num_envs >= hp_par_issue_min_envs() &&
                                      par_tiles <= hp_graph_max_tiles() &&
                                      attempt == 0 &&
                                      (!need_site_poses || site_stage_ok) &&
                                      prefetched_geom == 0;
                {
                    static const int gdbg0 = getenv("HP_GRAPH_DBG") ? atoi(getenv("HP_GRAPH_DBG")) : 0;
                    if (gdbg0 && !graph_ok)
                        fprintf(stderr, "[graph] skip par_ok=%d attempt=%d sites=%d pref=%d\n",
                                (int)par_ok, attempt,
                                (int)(need_site_poses && !site_stage_ok), prefetched_geom);
                }
                if (!graph_ok) {
                    if (cr.graph_live_) cr.graph_drop();
                    cr.graph_sig_valid_ = false;
                    cr.graph_failed_ = false;
                } else {
                    std::vector<unsigned char>& sg = cr.graph_sig_new_;
                    sg.clear();
                    auto put = [&sg](const void* p, size_t n) {
                        const unsigned char* b = static_cast<const unsigned char*>(p);
                        sg.insert(sg.end(), b, b + n);
                    };
                    const int hdr[] = {
                        num_envs, actual_workers, cr.width, cr.height,
                        cr.out_width, cr.out_height, cr.ssaa, cr.msaa, camera_idx,
                        (int)flip_v, (int)flip_h, (int)defer_bg,
                        (int)cr.vis_transparent, (int)cr.geomgroup[0],
                        (int)cr.geomgroup[1], (int)cr.geomgroup[2],
                        (int)use_batch_mat, (int)cr.gradient_bg,
                        (int)cr.shadow_enabled, (int)cr.reflection_enabled,
                        (int)cr.gouraud_shading, (int)img_bytes };
                    put(hdr, sizeof(hdr));
                    { const int mh[] = { (int)masked, n_active, hp_skip_rebal };
                      put(mh, sizeof(mh)); put(act_mask.data(), act_mask.size()); }
                    put(cr.bg_color, sizeof(cr.bg_color));
                    for (size_t di = 0; di < cr.d_site_dev_.size(); ++di)
                    { const void* dp = (const void*)cr.d_site_dev_[di]; put(&dp, sizeof(dp)); }
                    { const void* sp = (const void*)cr.h_site_stage_;
                      const int si2[] = { site_slot_verts, (int)site_stage_ok,
                                          cr.site_stage_verts_, cr.site_stage_slots_ };
                      put(&sp, sizeof(sp)); put(si2, sizeof(si2)); }
                    { const void* pp[] = { (const void*)cr.d_rgb_batch,
                                           (const void*)cr.d_batch_matrices_ };
                      put(pp, sizeof(pp)); }
                    put(&texture_state, sizeof(texture_state));
                    put(&device_model, sizeof(device_model));
                    { const auto& dd_sig = engine.data().device_data();
                      put(&dd_sig, sizeof(dd_sig)); }
                    const size_t off_workers = sg.size();
                    for (int wi = 0; wi < actual_workers; wi++) {
                        auto& w = *cr.workers_[wi];
                        DeviceFramebuffer dfb_sig = w.fb->device_fb();
                        put(&dfb_sig, sizeof(dfb_sig));
                        const void* wp[] = {
                            (const void*)w.phys_renderer->device_positions(),
                            (const void*)w.phys_renderer->device_normals(),
                            (const void*)w.phys_renderer->device_texcoords(),
                            (const void*)w.phys_renderer->device_texids(),
                            (const void*)w.phys_renderer->device_colors(),
                            (const void*)w.phys_renderer->device_specular(),
                            (const void*)w.phys_renderer->device_shininess(),
                            (const void*)w.phys_renderer->device_emission(),
                            (const void*)w.reflection_resources.mask(),
                            (const void*)w.reflection_resources.matrices() };
                        put(wp, sizeof(wp));
                        const int wi2[] = { w.phys_renderer->transparent_offset(),
                                            w.phys_renderer->transparent_count() };
                        put(wi2, sizeof(wi2));
                        const unsigned long long fp =
                            w.pipeline->resource_fingerprint();
                        put(&fp, sizeof(fp));
                    }
                    const size_t off_states = sg.size();
                    // Per-world site appearance: no overlay is VISIBLE (that is
                    // gated above), but the colours still select that, so they
                    // are part of the image the replay is keyed on.
                    auto same_span = [](const void* a2, const void* b2, size_t n) {
                        return n == 0 || memcmp(a2, b2, n) == 0;
                    };
                    const float* sig_site = site_rgba.data();
                    const RenderState* sig_states = batch_states.data();
                    const PlanarReflectionPlan* sig_plans = reflection_plans.data();
                    std::vector<float> m_site;
                    std::vector<RenderState> m_states;
                    std::vector<PlanarReflectionPlan> m_plans;
                    if (masked) {
                        m_site.assign(site_rgba.begin(), site_rgba.end());
                        m_states.assign(batch_states.begin(), batch_states.begin() + num_envs);
                        m_plans.assign(reflection_plans.begin(), reflection_plans.begin() + num_envs);
                        const size_t per_site = (size_t)desc.nsite * 4;
                        for (int ei = 0; ei < num_envs; ei++) {
                            if (act_mask[ei]) continue;
                            if (!m_site.empty())
                                std::fill(m_site.begin() + (size_t)ei * per_site,
                                          m_site.begin() + (size_t)(ei + 1) * per_site, 0.0f);
                            memset(static_cast<void*>(&m_states[ei]), 0, sizeof(RenderState));
                            memset(static_cast<void*>(&m_plans[ei]), 0, sizeof(PlanarReflectionPlan));
                        }
                        sig_site = m_site.data();
                        sig_states = m_states.data();
                        sig_plans = m_plans.data();
                    }
                    const bool big_same = cr.graph_sig_valid_ &&
                        cr.graph_site_shadow_.size() == site_rgba.size() &&
                        (int)cr.graph_state_shadow_.size() == num_envs &&
                        (int)cr.graph_plan_shadow_.size() == num_envs &&
                        same_span(cr.graph_site_shadow_.data(), sig_site,
                                  site_rgba.size() * sizeof(float)) &&
                        same_span(cr.graph_state_shadow_.data(), sig_states,
                                  (size_t)num_envs * sizeof(RenderState)) &&
                        same_span(cr.graph_plan_shadow_.data(), sig_plans,
                                  (size_t)num_envs * sizeof(PlanarReflectionPlan));
                    const bool same = big_same &&
                        cr.graph_sig_.size() == sg.size() &&
                        memcmp(cr.graph_sig_.data(), sg.data(), sg.size()) == 0;
                    static const int gdbg = getenv("HP_GRAPH_DBG") ? atoi(getenv("HP_GRAPH_DBG")) : 0;
                    if (gdbg && !same) {
                        const bool s_site = cr.graph_site_shadow_.size() == site_rgba.size() &&
                            same_span(cr.graph_site_shadow_.data(), site_rgba.data(), site_rgba.size()*sizeof(float));
                        const bool s_state = (int)cr.graph_state_shadow_.size() == num_envs &&
                            same_span(cr.graph_state_shadow_.data(), batch_states.data(), (size_t)num_envs*sizeof(RenderState));
                        const bool s_plan = (int)cr.graph_plan_shadow_.size() == num_envs &&
                            same_span(cr.graph_plan_shadow_.data(), reflection_plans.data(), (size_t)num_envs*sizeof(PlanarReflectionPlan));
                        size_t sd = (size_t)-1;
                        if ((int)cr.graph_state_shadow_.size() == num_envs) {
                            const unsigned char* a3 = (const unsigned char*)cr.graph_state_shadow_.data();
                            const unsigned char* b3 = (const unsigned char*)batch_states.data();
                            for (size_t i = 0; i < (size_t)num_envs*sizeof(RenderState); ++i)
                                if (a3[i] != b3[i]) { sd = i; break; }
                        }
                        size_t pd = (size_t)-1;
                        if ((int)cr.graph_plan_shadow_.size() == num_envs) {
                            const unsigned char* a3 = (const unsigned char*)cr.graph_plan_shadow_.data();
                            const unsigned char* b3 = (const unsigned char*)reflection_plans.data();
                            for (size_t i = 0; i < (size_t)num_envs*sizeof(PlanarReflectionPlan); ++i)
                                if (a3[i] != b3[i]) { pd = i; break; }
                        }
                        fprintf(stderr, "[graphbig] site=%d state=%d plan=%d statediff=%zd(sz=%zu) plandiff=%zd(sz=%zu)\n",
                                (int)s_site,(int)s_state,(int)s_plan,(ssize_t)sd,sizeof(RenderState),
                                (ssize_t)pd,sizeof(PlanarReflectionPlan));
                    }
                    if (gdbg) {
                        size_t first = (size_t)-1;
                        if (cr.graph_sig_valid_ && cr.graph_sig_.size() == sg.size())
                            for (size_t i = 0; i < sg.size(); ++i)
                                if (cr.graph_sig_[i] != sg[i]) { first = i; break; }
                        fprintf(stderr, "[graph] sig=%zu valid=%d same=%d live=%d failed=%d firstdiff=%zd "
                                "off_workers=%zu off_states=%zu wstride=%zu\n",
                                sg.size(), (int)cr.graph_sig_valid_, (int)same,
                                (int)cr.graph_live_, (int)cr.graph_failed_, (ssize_t)first,
                                off_workers, off_states,
                                (off_states - off_workers) / (size_t)actual_workers);
                        fflush(stderr);
                    }
                    if (!same) {
                        if (cr.graph_live_) cr.graph_drop();
                        cr.graph_failed_ = false;
                        cr.graph_sig_.assign(sg.begin(), sg.end());
                        cr.graph_site_shadow_.assign(sig_site, sig_site + site_rgba.size());
                        cr.graph_state_shadow_.assign(sig_states, sig_states + num_envs);
                        cr.graph_plan_shadow_.assign(sig_plans, sig_plans + num_envs);
                        cr.graph_sig_valid_ = true;
                        cr.gstat_sigchg_++;
                    }

                    if (same && cr.graph_live_) {
                        if (site_stage_ok) {
                            auto refresh_worker = [&](int wi) {
                                for (int ei : wlist[(size_t)wi]) {
                                    const float* rg = site_rgba.data()
                                        + (size_t)ei * desc.nsite * 4;
                                    const float* xp = h_site_xpos
                                        + (size_t)ei * desc.nsite * 3;
                                    const float* xm = h_site_xmat
                                        + (size_t)ei * desc.nsite * 9;
                                    for (int pass = 0; pass < 2; ++pass)
                                        refresh_site_overlay_slot(
                                            desc, xp, xm, rg,
                                            cr.site_slot_ctx_[ei * 2 + pass],
                                            cr.site_slot(ei * 2 + pass),
                                            site_slot_verts);
                                }
                            };
                            if (par_ok) cr.issue_pool_.run(actual_workers, refresh_worker);
                            else for (int wi = 0; wi < actual_workers; wi++) refresh_worker(wi);
                        }
                        bool all = true;
                        for (int wi = 0; wi < actual_workers; wi++) {
                            auto& w = *cr.workers_[wi];
                            if (wlist[(size_t)wi].empty()) continue;
                            if (!w.graph_exec ||
                                hipGraphLaunch(w.graph_exec, w.stream) != hipSuccess) {
                                all = false; break;
                            }
                        }
                        if (all) { graph_hit = true; cr.gstat_hit_++; }
                        else {
                            cr.graph_drop();
                            cr.graph_failed_ = true;
                        }
                    } else if (same && !cr.graph_failed_) {
                        // SECOND sighting of this signature: capture + launch.
                        auto capture_worker = [&](int wi) {
                            auto& w = *cr.workers_[wi];
                            if (wlist[(size_t)wi].empty()) return;
                            w.pipeline->reset_nc_rotation();
                            if (hipStreamBeginCapture(
                                    w.stream, hipStreamCaptureModeThreadLocal)
                                == hipSuccess) {
                                issue_worker(wi);
                                // The trailing deferred convert must be INSIDE
                                // the graph: on a replay there is no host-side
                                // pending state for the join block to flush.
                                w.pipeline->flush_pending_convert();
                                hipGraph_t g = nullptr;
                                hipError_t e = hipStreamEndCapture(w.stream, &g);
                                if (gdbg && e != hipSuccess)
                                    fprintf(stderr, "[graph] capture failed wi=%d: %s\n",
                                            wi, hipGetErrorString(e));
                                if (e == hipSuccess && g) {
                                    hipGraphExec_t ge = nullptr;
                                    if (hipGraphInstantiate(&ge, g, nullptr,
                                                            nullptr, 0) == hipSuccess)
                                        w.graph_exec = ge;
                                    hipGraphDestroy(g);
                                }
                                if (w.graph_exec) {
                                    if (hipGraphLaunch(w.graph_exec, w.stream)
                                        == hipSuccess)
                                        return;
                                    hipGraphExecDestroy(w.graph_exec);
                                    w.graph_exec = nullptr;
                                }
                                // Capture was invalidated (an illegal call
                                // inside it) -- nothing was enqueued, so the
                                // eager issue below is the whole of this
                                // worker's work.
                            }
                            issue_worker(wi);
                        };
                        if (par_ok)
                            cr.issue_pool_.run(actual_workers, capture_worker);
                        else
                            for (int wi = 0; wi < actual_workers; wi++) capture_worker(wi);
                        bool all = true;
                        for (int wi = 0; wi < actual_workers; wi++)
                            if (!wlist[(size_t)wi].empty() && !cr.workers_[wi]->graph_exec) all = false;
                        cr.gstat_capture_++;
                        if (all) cr.graph_live_ = true;
                        else { cr.graph_drop(); cr.graph_failed_ = true; }
                        graph_hit = true;   // the work is already enqueued
                    }
                }

                if (!graph_hit) {
                cr.gstat_eager_++;
                if (par_ok) {
                    cr.issue_pool_.run(actual_workers, issue_worker);
                } else {
                    for (int wi = 0; wi < actual_workers; wi++) issue_worker(wi);
                }
                }
                if (site_stage_ok) {
                    for (int si2 = 0; si2 < num_envs * 2; ++si2)
                        if (cr.site_slot_ctx_[si2].miss) { cr.site_stage_broken_ = true; break; }
                    if (cr.site_stage_broken_ && cr.graph_live_) {
                        cr.graph_drop();
                        cr.graph_sig_valid_ = false;
                    }
                }
                // Sync all worker streams
                if (hp_trace) t_issue = hp_now_ms();
                prefetched_geom = 0;   // a retry must rebuild every world
                for (int wi = 0; wi < actual_workers; wi++)
                    cr.workers_[wi]->pipeline->flush_pending_convert();
                for (int wi = 0; wi < actual_workers; wi++)
                    cr.workers_[wi]->pipeline->queue_overflow_read();
                for (int wi = 0; wi < actual_workers; wi++) {
                    hipStreamSynchronize(cr.workers_[wi]->stream);
                }
            }

            // Streams are joined, so the sticky overflow words are final.
            // Always drain every pipeline: take_overflow() also clears the word,
            // and a short-circuit would leave a stale 1 to fire on the next batch.
            bool overflowed = false;
            if (actual_workers <= 1) {
                overflowed = cr.pipeline.take_overflow();
            } else {
                for (int wi = 0; wi < actual_workers; wi++)
                    if (cr.workers_[wi]->pipeline->take_overflow()) overflowed = true;
            }
            if (!overflowed) break;
            if (attempt == 0) {
                if (actual_workers <= 1) {
                    cr.pipeline.set_force_exact(true);
                } else {
                    for (int wi = 0; wi < actual_workers; wi++)
                        cr.workers_[wi]->pipeline->set_force_exact(true);
                }
            } else {
                // Exact sizing still overflowed. That means the clamp is not a
                // sizing miss, so say so rather than returning a quietly
                // incomplete frame.
                fprintf(stderr, "[instinct_sim] render_gpu_batch: tile pool overflowed "
                                "even with exact sizing; frame may be missing geometry\n");
            }
            }
            // Leave exact sizing off for the next batch; the pool only grows, so
            // the capacity learned by the retry is retained and reused.
            if (actual_workers <= 1) {
                cr.pipeline.set_force_exact(false);
            } else {
                for (int wi = 0; wi < actual_workers; wi++)
                    cr.workers_[wi]->pipeline->set_force_exact(false);
            }

            if (hp_trace) {
                const double t_end = hp_now_ms();
                static double a_pose = 0, a_cam = 0, a_issue = 0, a_join = 0, a_tot = 0;
                static int n = 0;
                a_pose  += t_pose  - t_entry;
                a_cam   += t_cam   - t_pose;
                a_issue += (t_issue > 0 ? t_issue : t_cam) - t_cam;
                a_join  += t_end   - (t_issue > 0 ? t_issue : t_cam);
                a_tot   += t_end   - t_entry;
                if (++n % 8 == 0) {
                    fprintf(stderr,
                        "[hosttrace] w=%d prologue(pose)=%.3f cam+H2D=%.3f "
                        "issue=%.3f join=%.3f total=%.3f ms\n",
                        num_envs, a_pose/8, a_cam/8, a_issue/8, a_join/8, a_tot/8);
                    fflush(stderr);
                    a_pose = a_cam = a_issue = a_join = a_tot = 0;
                }
            }
            cr.gstat_calls_++;
            cr.gstat_worlds_ += n_active;
            if (masked) cr.gstat_masked_++;
            {
                static const int gstats = getenv("HP_RENDER_GRAPH_STATS")
                    ? atoi(getenv("HP_RENDER_GRAPH_STATS")) : 0;
                if (gstats && cr.gstat_calls_ % (gstats > 1 ? gstats : 1000) == 0) {
                    fprintf(stderr, "[rgraph] cam=%d calls=%lld masked=%lld avg_worlds=%.2f "
                            "hit=%lld capture=%lld eager=%lld sigchg=%lld\n",
                            camera_idx, cr.gstat_calls_, cr.gstat_masked_,
                            (double)cr.gstat_worlds_ / (double)cr.gstat_calls_,
                            cr.gstat_hit_, cr.gstat_capture_, cr.gstat_eager_,
                            cr.gstat_sigchg_);
                    fflush(stderr);
                }
            }
            return GpuImageBatch{cr.d_rgb_batch, num_envs, cr.out_height, cr.out_width};
        }, py::arg("engine"), py::arg("camera_idx"), py::arg("num_envs"),
           py::arg("flip_v") = false, py::arg("flip_h") = false,
           py::arg("active") = py::none(),
           "Render all envs for one camera, returns GpuImageBatch (num_envs, H, W, 3) on GPU");

    // ── Convenience functions ────────────────────────────────────────────────
    m.def("create_engine", [](const std::string& xml, int num_envs, int gpu_id,
                              py::object body_invweight0, py::object dof_invweight0,
                              const std::string& meshdir, bool render_only) {
        auto model = new PhysicsModel();
        if (!model->load_from_xml(xml, meshdir)) {
            delete model;
            throw std::runtime_error("Failed to parse MJCF XML");
        }

        if (!body_invweight0.is_none() && !dof_invweight0.is_none()) {
            // Accept either float32 or float64 arrays for invweight
            int nb = model->desc().nbody;
            int nv = model->desc().nv;
            auto& desc = model->desc_mut();
            try {
                auto biw = body_invweight0.cast<py::array_t<double>>();
                auto diw = dof_invweight0.cast<py::array_t<double>>();
                auto biw_buf = biw.request();
                auto diw_buf = diw.request();
                if ((int)biw_buf.size != nb * 2)
                    throw std::runtime_error("body_invweight0 must have shape (nbody*2,)");
                if ((int)diw_buf.size != nv)
                    throw std::runtime_error("dof_invweight0 must have shape (nv,)");
                for (int i = 0; i < nb * 2; i++) {
                    desc.body_invweight0[i] = (float)((double*)biw_buf.ptr)[i];
                    desc.body_invweight0_d[i] = ((double*)biw_buf.ptr)[i];
                }
                for (int i = 0; i < nv; i++) {
                    desc.dof_invweight0[i] = (float)((double*)diw_buf.ptr)[i];
                    desc.dof_invweight0_d[i] = ((double*)diw_buf.ptr)[i];
                }
            } catch (py::cast_error&) {
                auto biw = body_invweight0.cast<py::array_t<float>>();
                auto diw = dof_invweight0.cast<py::array_t<float>>();
                auto biw_buf = biw.request();
                auto diw_buf = diw.request();
                if ((int)biw_buf.size != nb * 2)
                    throw std::runtime_error("body_invweight0 must have shape (nbody*2,)");
                if ((int)diw_buf.size != nv)
                    throw std::runtime_error("dof_invweight0 must have shape (nv,)");
                memcpy(desc.body_invweight0, (float*)biw_buf.ptr, nb * 2 * sizeof(float));
                memcpy(desc.dof_invweight0, (float*)diw_buf.ptr, nv * sizeof(float));
                for (int i = 0; i < nb * 2; i++) desc.body_invweight0_d[i] = (double)desc.body_invweight0[i];
                for (int i = 0; i < nv; i++) desc.dof_invweight0_d[i] = (double)desc.dof_invweight0[i];
            }
            desc.has_invweight0 = true;
            if (!model->upload_invweight()) {
                delete model;
                throw std::runtime_error("Failed to upload invweight arrays to GPU");
            }
        }

        auto engine = new PhysicsEngine();
        engine->set_render_only(render_only);
        if (!engine->init(model, num_envs, gpu_id)) {
            delete engine; delete model;
            throw std::runtime_error("Failed to initialize physics engine");
        }
        engine->options().impratio = model->desc().impratio;
        return py::make_tuple(
            py::cast(model, py::return_value_policy::take_ownership),
            py::cast(engine, py::return_value_policy::take_ownership)
        );
    }, py::arg("xml"), py::arg("num_envs") = 1, py::arg("gpu_id") = 0,
       py::arg("body_invweight0") = py::none(), py::arg("dof_invweight0") = py::none(),
       py::arg("meshdir") = "", py::arg("render_only") = false);

    py::class_<GpuArray>(m, "GpuArray")
        .def_property_readonly("shape", &GpuArray::shape_tuple)
        .def_property_readonly("typestr", [](const GpuArray& a) { return a.typestr; })
        .def_property_readonly("ptr", [](const GpuArray& a) { return reinterpret_cast<uintptr_t>(a.ptr); })
        .def_property_readonly("__cuda_array_interface__", &GpuArray::cuda_array_interface)
        .def("numpy", &GpuArray::numpy, "Copy to a new host numpy array");

    // Mesh soup accessor (verts in the hip geom frame, one triangle per three
    // consecutive corners) for the BLAS tests. Added to the existing class.
    py::reinterpret_borrow<py::class_<PhysicsModel>>(m.attr("PhysicsModel"))
        .def("mesh_soup", [](const PhysicsModel& pm) {
            const PhysicsModelDesc& d = pm.desc();
            const int nv = d.mesh_total_verts, nm = d.nmesh;
            py::array_t<float> verts({(ssize_t)nv, (ssize_t)3});
            py::array_t<int> adr((ssize_t)nm), num((ssize_t)nm);
            if (nv > 0) {
                if (d.mesh_vert)
                    memcpy(verts.mutable_data(), d.mesh_vert, (size_t)nv * 3 * sizeof(float));
                else if (pm.device_model().mesh_vert) {
                    hipError_t err = hipMemcpy(verts.mutable_data(), pm.device_model().mesh_vert,
                                               (size_t)nv * 3 * sizeof(float), hipMemcpyDeviceToHost);
                    if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
                }
            }
            for (int i = 0; i < nm; ++i) {
                adr.mutable_data()[i] = d.meshes[i].vert_adr;
                num.mutable_data()[i] = d.meshes[i].vert_num;
            }
            return py::make_tuple(verts, adr, num);
        }, "(verts float32 (nverts,3), vert_adr int32[nmesh], vert_num int32[nmesh]) mesh soup");

    py::class_<RayTracerPy>(m, "RayTracer")
        .def(py::init<>())
        .def("init", [](RayTracerPy& self, PhysicsModel& model, int width, int height, int ssaa) {
            self.rt.init(model.desc(), model.device_model(), width, height, ssaa);
            self.model = &model;
            self.model_owner = py::cast(&model, py::return_value_policy::reference);
        }, py::arg("model"), py::arg("width") = 256, py::arg("height") = 256, py::arg("ssaa") = 1,
           "Build the RT scene (render-geom list, mesh BLAS, lights). Call set_textures() first.")
        .def_property_readonly("width",  [](const RayTracerPy& s) { return s.rt.width(); })
        .def_property_readonly("height", [](const RayTracerPy& s) { return s.rt.height(); })
        .def_property_readonly("ssaa",   [](const RayTracerPy& s) { return s.rt.ssaa(); })
        .def("set_geomgroup", [](RayTracerPy& s, int group, bool visible) {
            s.rt.set_geomgroup(group, visible);
        }, py::arg("group"), py::arg("visible"),
           "Geom-group visibility (0-5, default 0,1,2 on); groups >= 3 never render.")
        .def("set_sitegroup", [](RayTracerPy& s, int group, bool visible) {
            s.rt.set_sitegroup(group, visible);
        }, py::arg("group"), py::arg("visible"), "Site-group visibility (0-5, default 0,1,2 on).")
        .def("set_site_rgba", [](RayTracerPy& s, PhysicsEngine& engine,
                                 py::array_t<float, py::array::c_style | py::array::forcecast> rgba, int env_idx) {
            s.check(engine);
            const int ne = engine.data().num_envs();
            if (rgba.ndim() != 2 || rgba.shape(0) != s.rt.nsite() || rgba.shape(1) != 4)
                throw std::runtime_error("set_site_rgba: expected (nsite, 4) float32");
            if (env_idx < 0 || env_idx >= ne) throw std::runtime_error("set_site_rgba: env_idx out of range");
            s.rt.set_site_rgba(ne, rgba.data(), env_idx);
        }, py::arg("engine"), py::arg("rgba"), py::arg("env_idx") = 0,
           "Per-world (nsite, 4) site RGBA (CameraRenderer semantics; signed alpha preserved).")
        .def("set_site_rgba_batch", [](RayTracerPy& s, PhysicsEngine& engine,
                                       py::array_t<float, py::array::c_style | py::array::forcecast> rgba) {
            s.check(engine);
            const int ne = engine.data().num_envs();
            if (rgba.ndim() != 3 || rgba.shape(0) != ne || rgba.shape(1) != s.rt.nsite() || rgba.shape(2) != 4)
                throw std::runtime_error("set_site_rgba_batch: expected (num_envs, nsite, 4) float32");
            s.rt.set_site_rgba(ne, rgba.data(), -1);
        }, py::arg("engine"), py::arg("rgba"))
        .def("get_site_rgba", [](RayTracerPy& s, PhysicsEngine& engine, int env_idx) {
            s.check(engine);
            if (env_idx < 0 || env_idx >= engine.data().num_envs())
                throw std::runtime_error("get_site_rgba: env_idx out of range");
            py::array_t<float> out({(ssize_t)s.rt.nsite(), (ssize_t)4});
            s.rt.get_site_rgba(engine.data().num_envs(), out.mutable_data(), env_idx);
            return out;
        }, py::arg("engine"), py::arg("env_idx") = 0)
        .def("get_site_rgba_batch", [](RayTracerPy& s, PhysicsEngine& engine) {
            s.check(engine);
            const int ne = engine.data().num_envs();
            py::array_t<float> out({(ssize_t)ne, (ssize_t)s.rt.nsite(), (ssize_t)4});
            s.rt.get_site_rgba(ne, out.mutable_data(), -1);
            return out;
        }, py::arg("engine"))
        .def("clear_site_rgba", [](RayTracerPy& s, PhysicsEngine& engine, py::object env_idx) {
            s.check(engine);
            s.rt.clear_site_rgba(env_idx.is_none() ? -1 : env_idx.cast<int>());
        }, py::arg("engine"), py::arg("env_idx") = py::none(),
           "Restore model site RGBA for one world, or all worlds when env_idx is None.")
        .def("set_texgen_planes", [](RayTracerPy& s, int gid, py::object planes) {
            if (planes.is_none()) { s.rt.set_texgen_planes(gid, nullptr); return; }
            auto a = planes.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
            if (a.size() != 12) throw std::runtime_error("set_texgen_planes: expected (3, 4) planes");
            s.rt.set_texgen_planes(gid, a.data());
        }, py::arg("gid"), py::arg("planes"),
           "GL_OBJECT_LINEAR texgen planes (s, t, r rows of (a, b, c, d)) over geom gid's hip-local "
           "position; used with set_shading(texgen_mj=True). None clears.")
        .def("set_texture_srgb", [](RayTracerPy& s, std::vector<int> srgb) {
            std::vector<int> f(srgb.size());
            for (size_t i = 0; i < srgb.size(); ++i) f[i] = srgb[i] ? RT_TEX_SRGB : 0;
            s.rt.set_texture_flags(f.data(), (int)f.size());
        }, py::arg("srgb"),
           "Per-texture flags: true where MuJoCo's texture is sRGB (mjModel.tex_colorspace == 2); "
           "those use llvmpipe's float sampler under set_shading(mesa_tex=True).")
        .def("set_texture_srgb_data", [](RayTracerPy& s, int texid,
                                         py::array_t<uint8_t, py::array::c_style | py::array::forcecast> rgb) {
            if (!s.model) throw std::runtime_error("RayTracer not initialized");
            if (rgb.ndim() != 3 || rgb.shape(2) != 3)
                throw std::runtime_error("set_texture_srgb_data: expected (h, w, 3) uint8");
            s.rt.set_texture_srgb_data(texid, s.model->device_model().ntexture, rgb.data(),
                                       (int)rgb.shape(1), (int)rgb.shape(0));
        }, py::arg("texid"), py::arg("rgb"),
           "sRGB-encoded level-0 texels of 2D texture texid (MuJoCo tex_data); the RT builds and "
           "samples llvmpipe's sRGB pyramid under set_shading(mesa_tex=True).")
        .def("set_mesh_normals", [](RayTracerPy& s, int mesh_id,
                                    py::array_t<float, py::array::c_style | py::array::forcecast> nrm) {
            if (nrm.ndim() != 2 || nrm.shape(1) != 3) throw std::runtime_error("set_mesh_normals: expected (n, 3)");
            s.rt.set_mesh_normals(mesh_id, nrm.data(), (int)nrm.shape(0));
        }, py::arg("mesh_id"), py::arg("normals"),
           "Per-corner normals (hip soup order/frame) for one mesh, e.g. MuJoCo's mesh normals.")
        .def("set_lights", [](RayTracerPy& s, py::list lights) {
            std::vector<RtLight> L;
            for (auto item : lights) {
                py::dict d = item.cast<py::dict>();
                RtLight l;
                std::memset(&l, 0, sizeof(l));
                auto v3 = [&](const char* k, float* dst, float def0, float def1, float def2) {
                    dst[0] = def0; dst[1] = def1; dst[2] = def2;
                    if (d.contains(k)) { auto v = d[k].cast<std::vector<float>>(); for (int i = 0; i < 3 && i < (int)v.size(); ++i) dst[i] = v[i]; }
                };
                const std::string ty = d.contains("type") ? d["type"].cast<std::string>() : "directional";
                l.type = ty == "directional" ? 0 : ty == "spot" ? 1 : ty == "point" ? 2 : -1;
                if (l.type < 0) throw std::runtime_error("set_lights: type must be 'directional', 'spot' or 'point'");
                l.active = d.contains("active") ? (d["active"].cast<bool>() ? 1 : 0) : 1;
                l.castshadow = d.contains("castshadow") ? (d["castshadow"].cast<bool>() ? 1 : 0) : 0;
                v3("pos", l.pos, 0.f, 0.f, 0.f); v3("dir", l.dir, 0.f, 0.f, -1.f);
                v3("ambient", l.amb, 0.f, 0.f, 0.f); v3("diffuse", l.dif, 0.f, 0.f, 0.f);
                v3("specular", l.spc, 0.f, 0.f, 0.f); v3("attenuation", l.att, 1.f, 0.f, 0.f);
                const float dl = std::sqrt(l.dir[0] * l.dir[0] + l.dir[1] * l.dir[1] + l.dir[2] * l.dir[2]);
                if (dl > 1e-8f) for (int k = 0; k < 3; ++k) l.dir[k] /= dl;
                const float cut = d.contains("cutoff") ? d["cutoff"].cast<float>() : 45.0f;
                l.cos_cut = l.type == 1 ? std::cos(cut * 3.14159265358979f / 180.0f) : -1.0f;
                l.expo = l.type == 1 && d.contains("exponent") ? d["exponent"].cast<float>() : 0.0f;
                l.att_default = (l.att[0] == 1.0f && l.att[1] == 0.0f && l.att[2] == 0.0f) ? 1 : 0;
                L.push_back(l);
            }
            if (L.empty() || L[0].type != 0) throw std::runtime_error("set_lights: entry 0 must be the (directional) headlight");
            s.rt.set_lights(L.data(), (int)L.size());
        }, py::arg("lights"),
           "Replace the light table: a list of dicts (type 'directional'/'spot'/'point', active, castshadow, "
           "pos, dir, ambient, diffuse, specular, attenuation, cutoff [deg], exponent); entry 0 is the "
           "headlight. Only the first 8 active lights shine (Mesa); none active: GL global ambient 0.3.")
        .def("set_geom_ellipsoid", [](RayTracerPy& s, int gid, std::vector<float> size) {
            if (size.size() != 3) throw std::runtime_error("set_geom_ellipsoid: expected 3 semi-axes");
            return s.rt.set_geom_ellipsoid(gid, size.data());
        }, py::arg("gid"), py::arg("size"),
           "Draw render geom gid as an ellipsoid (hip parses MJCF ellipsoids as boxes). False if not applicable.")
        .def("nlights_active", [](RayTracerPy& s) { return s.rt.nlights_active(); })
        .def("set_geom_rgba", [](RayTracerPy& s, int gid, std::vector<float> rgba) {
            if (rgba.size() != 4) throw std::runtime_error("set_geom_rgba: expected 4 values");
            return s.rt.set_geom_rgba(gid, rgba.data());
        }, py::arg("gid"), py::arg("rgba"),
           "Override render geom gid's RGBA (e.g. MuJoCo's resolved material/geom colour). False if gid is "
           "not rendered.")
        .def("set_mode", [](RayTracerPy& s, const std::string& mode) {
            if (mode == "grey") s.rt.set_mode(RT_MODE_GREY);
            else if (mode == "hipgl") s.rt.set_mode(RT_MODE_HIPGL);
            else if (mode == "mjw") s.rt.set_mode(RT_MODE_MJW);
            else throw std::runtime_error("set_mode: expected 'grey', 'hipgl' or 'mjw'");
        }, py::arg("mode"),
           "Shading mode: 'hipgl' (default; GL fixed-function model of the raster / MuJoCo, see "
           "set_shading), 'mjw' (mujoco_warp's per-pixel model, validation) or 'grey' (unlit |N.V|).")
        .def("set_cull", [](RayTracerPy& s, const std::string& c) {
            if (c == "per_tri") s.rt.set_cull(RT_CULL_PER_TRI);
            else if (c == "mjw_mesh") s.rt.set_cull(RT_CULL_MJW_MESH);
            else if (c == "none") s.rt.set_cull(RT_CULL_NONE);
            else throw std::runtime_error("set_cull: expected 'per_tri', 'mjw_mesh' or 'none'");
        }, py::arg("cull"))
        .def("set_traversal", [](RayTracerPy& s, const std::string& t) {
            if (t == "skip") s.rt.set_traversal(RT_TRAV_SKIP);
            else if (t == "ordered") s.rt.set_traversal(RT_TRAV_ORDERED);
            else throw std::runtime_error("set_traversal: expected 'skip' or 'ordered'");
        }, py::arg("traversal"))
        .def("set_clip", [](RayTracerPy& s, bool on) { s.rt.set_clip(on); }, py::arg("enabled"),
           "Clip hits to [znear, zfar] planar distance (default off in M1).")
        .def("set_clip_planes", [](RayTracerPy& s, float zn, float zf) {
            if (!(zn > 0.0f) || !(zf > zn)) throw std::runtime_error("set_clip_planes: need 0 < znear < zfar");
            s.rt.set_clip_planes(zn, zf);
        }, py::arg("znear"), py::arg("zfar"))
        .def("set_bg_color", [](RayTracerPy& s, float r, float g, float b, float a) {
            (void)a; s.rt.set_bg_color(r, g, b);
        }, py::arg("r"), py::arg("g"), py::arg("b"), py::arg("a") = 1.0f)
        .def("set_stats", [](RayTracerPy& s, bool on) { s.rt.set_stats(on); }, py::arg("enabled"),
           "Collect traversal counters (slower stats kernel variant).")
        .def("set_tile_cull", [](RayTracerPy& s, bool on) { s.rt.set_tile_cull(on); }, py::arg("enabled"),
           "Per-wave tile frustum cull + sorted candidates with early break (default on). Off = the M1 "
           "linear candidate loop. Results are identical either way.")
        .def("set_swizzle", [](RayTracerPy& s, bool on) { s.rt.set_swizzle(on); }, py::arg("enabled"),
           "XCD workgroup swizzle (default off; A/B only: with the worlds-fastest block order it is slower). Results are identical.")
        .def("set_shade_split", [](RayTracerPy& s, py::object mode) {
            if (py::isinstance<py::str>(mode)) {
                const std::string m = mode.cast<std::string>();
                if (m != "auto") throw std::runtime_error("set_shade_split: expected True, False or 'auto'");
                s.rt.set_shade_split(2);
            } else {
                s.rt.set_shade_split(mode.cast<bool>() ? 1 : 0);
            }
        }, py::arg("mode"),
           "Full shading as a trace pass + a separate shade kernel over compact hit records (True), one fused "
           "kernel (False), or 'auto' (default: fused for small latency-bound launches, else split). "
           "Images are identical in every mode.")
        .def("set_record_cap", [](RayTracerPy& s, size_t bytes) { s.rt.set_record_cap(bytes); }, py::arg("bytes"),
           "Byte cap of the deferred-shading record buffer (default 4 GiB); larger batches are chunked.")
        .def("set_adaptive_tiles", [](RayTracerPy& s, bool on) { s.rt.set_adaptive_tiles(on); }, py::arg("enabled"),
           "Small-batch tiles (default on): single-sample passes with few workgroups use 4x4 or 2x2-pixel tiles "
           "per wave (lower latency at small N). Results are identical.")
        .def("set_gradient_bg", [](RayTracerPy& s, float bot_r, float bot_g, float bot_b,
                                   float top_r, float top_g, float top_b) {
            const float bot[3] = {bot_r, bot_g, bot_b}, top[3] = {top_r, top_g, top_b};
            s.rt.set_gradient_bg(bot, top);
        }, py::arg("bot_r"), py::arg("bot_g"), py::arg("bot_b"),
           py::arg("top_r"), py::arg("top_g"), py::arg("top_b"))
        .def("set_fog", [](RayTracerPy& s, float start, float end, float r, float g, float b) {
            s.rt.set_fog(start, end, r, g, b);
        }, py::arg("start"), py::arg("end"), py::arg("r") = 0.0f, py::arg("g") = 0.0f, py::arg("b") = 0.0f,
           "Linear fog on the world-space planar distance (end <= start disables).")
        .def("set_linear_output", [](RayTracerPy& s, bool enabled, float exposure, float gamma) {
            s.rt.set_linear_output(enabled, exposure, gamma);
        }, py::arg("enabled"), py::arg("exposure") = 1.0f, py::arg("gamma") = 2.2f)
        .def("set_gouraud_shading", [](RayTracerPy& s, bool on) { s.rt.set_shade_flag(RT_SH_GOURAUD, on); },
           py::arg("enabled"), "Meshes: light the triangle corners and blend (default) or light each hit.")
        .def("set_reflection", [](RayTracerPy& s, bool on) { s.rt.set_shade_flag(RT_SH_REFLECT, on); },
           py::arg("enabled"), "Planar mirror on the single reflective plane/box (default on).")
        .def("set_shadow", [](RayTracerPy& s, bool on) { s.rt.set_shade_flag(RT_SH_SHADOW, on); },
           py::arg("enabled"),
           "EXPERIMENTAL hard shadows (default off, in no preset): one shadow ray per scene light "
           "(directional / spot, not the headlight) from the first opaque surface; an occluded light "
           "keeps its ambient term. Forces the fused shading kernel.")
        .def("set_num_streams", [](RayTracerPy&, int) {}, py::arg("n") = 4, "No-op (one launch per call).")
        // RT-specific shading controls.
        .def("set_shading", [](RayTracerPy& s, py::kwargs kw) {
            static const std::pair<const char*, int> kBits[] = {
                {"gouraud", RT_SH_GOURAUD}, {"vtess", RT_SH_VTESS}, {"texgen_mj", RT_SH_TEXGEN_MJ},
                {"reflection", RT_SH_REFLECT}, {"quad_alt", RT_SH_QUAD_ALT},
                {"mj_mesh_normals", RT_SH_MJ_MESHNRM}, {"lod_pixel", RT_SH_LOD_PIXEL},
                {"tex_rgba", RT_SH_TEX_RGBA}, {"msaa_sample", RT_SH_MSAA_SAMPLE}, {"mesa_ff", RT_SH_MESA_FF | RT_SH_MESA_TEX},
                {"mesa_spec", RT_SH_MESA_FF}, {"mesa_tex", RT_SH_MESA_TEX}, {"hw_tex", RT_SH_HW_TEX}, {"hw_mips", RT_SH_HW_MIPS},
                {"shadow", RT_SH_SHADOW}};
            for (auto item : kw) {
                const std::string k = py::str(item.first);
                bool found = false;
                for (const auto& b : kBits)
                    if (k == b.first) { s.rt.set_shade_flag(b.second, item.second.cast<bool>()); found = true; }
                if (!found) throw std::runtime_error("set_shading: unknown option '" + k + "'");
            }
        }, "Shading options as keyword booleans: gouraud, vtess (Gouraud on MuJoCo's primitive "
           "tessellation), texgen_mj (MuJoCo texgen tables), reflection, quad_alt (other quad "
           "diagonal), mj_mesh_normals (MuJoCo face-normal rule for UV-less meshes), lod_pixel "
           "(texture LOD from the output-pixel footprint under SSAA), tex_rgba (textured geoms keep "
           "rgba), msaa_sample (MSAA: shade at the first covered sample, not the pixel centre), mesa_spec "
           "(Mesa's shininess-0 specular quirk), mesa_tex (llvmpipe's 8-bit AoS / sRGB sampler instead "
           "of hardware-GL float filtering), mesa_ff (both), hw_tex (hardware-GL sampler: float filtering, "
           "exact LOD; used when mesa_tex is off), hw_mips (with hw_tex: RT-owned glGenerateMipmap box "
           "pyramid instead of the model's llvmpipe one).")
        .def("set_shading_preset", [](RayTracerPy& s, const std::string& p) {
            if (p == "raster") s.rt.set_shade_flags(RT_SH_GOURAUD | RT_SH_REFLECT);
            else if (p == "mujoco") s.rt.set_shade_flags(RT_SH_PRESET_MUJOCO);
            else throw std::runtime_error("set_shading_preset: expected 'raster' or 'mujoco'");
        }, py::arg("preset"),
           "'mujoco' (default): MuJoCo GL emulation (gouraud, reflection, vtess, texgen_mj, "
           "mj_mesh_normals, mesa_spec, hw_tex, lod_pixel, tex_rgba); 'raster': the rasterizer's model "
           "(gouraud, reflection). Individual flags via set_shading(...).")
        .def("shading", [](RayTracerPy& s) {
            const int f = s.rt.shade_flags();
            py::dict d;
            d["gouraud"] = bool(f & RT_SH_GOURAUD); d["vtess"] = bool(f & RT_SH_VTESS);
            d["texgen_mj"] = bool(f & RT_SH_TEXGEN_MJ); d["reflection"] = bool(f & RT_SH_REFLECT);
            d["quad_alt"] = bool(f & RT_SH_QUAD_ALT); d["mj_mesh_normals"] = bool(f & RT_SH_MJ_MESHNRM);
            d["lod_pixel"] = bool(f & RT_SH_LOD_PIXEL); d["tex_rgba"] = bool(f & RT_SH_TEX_RGBA);
            d["msaa_sample"] = bool(f & RT_SH_MSAA_SAMPLE); d["mesa_spec"] = bool(f & RT_SH_MESA_FF); d["mesa_tex"] = bool(f & RT_SH_MESA_TEX);
            d["mesa_ff"] = (f & (RT_SH_MESA_FF | RT_SH_MESA_TEX)) == (RT_SH_MESA_FF | RT_SH_MESA_TEX);
            d["hw_tex"] = bool(f & RT_SH_HW_TEX); d["hw_mips"] = bool(f & RT_SH_HW_MIPS);
            d["shadow"] = bool(f & RT_SH_SHADOW);
            d["aa"] = s.rt.aa() == RT_AA_MSAA4 ? "msaa4" : "ssaa";
            d["lod_bias"] = s.rt.lod_bias();
            d["mirror_li"] = s.rt.mirror_li();
            return d;
        })
        .def("set_aa", [](RayTracerPy& s, const std::string& aa) {
            if (aa == "ssaa") s.rt.set_aa(RT_AA_SSAA);
            else if (aa == "msaa4") {
                if (s.rt.ssaa() != 1) throw std::runtime_error("set_aa('msaa4') needs init(..., ssaa=1)");
                s.rt.set_aa(RT_AA_MSAA4);
            } else throw std::runtime_error("set_aa: expected 'ssaa' or 'msaa4'");
        }, py::arg("aa"),
           "'ssaa' (init's ssaa x ssaa grid, every sample shaded) or 'msaa4' (4 GL-pattern coverage "
           "samples, one shade per covered surface; requires ssaa=1).")
        .def("set_lod_bias", [](RayTracerPy& s, float b) { s.rt.set_lod_bias(b); }, py::arg("bias"),
           "Added to the texture LOD (0 = raster).")
        .def("settings", [](RayTracerPy& s) {
            py::dict d;
            d["mode"] = s.rt.mode(); d["cull"] = s.rt.cull(); d["traversal"] = s.rt.traversal();
            d["clip"] = s.rt.clip(); d["znear"] = s.rt.znear(); d["zfar"] = s.rt.zfar();
            d["tile_cull"] = s.rt.tile_cull(); d["swizzle"] = s.rt.swizzle();
            d["adaptive_tiles"] = s.rt.adaptive_tiles();
            d["shade_split"] = s.rt.shade_split() == 2 ? py::object(py::str("auto")) : py::object(py::bool_(s.rt.shade_split() == 1));
            py::list g; for (int i = 0; i < 6; ++i) g.append(s.rt.geomgroup(i));
            d["geomgroup"] = g;
            return d;
        })
        .def("render", [](RayTracerPy& self, PhysicsEngine& engine, int camera_idx, int env_idx) {
            const int row0 = 0;
            self.run(engine, {camera_idx}, {env_idx}, &row0, 1, RT_OUT_RGB, false, false);
            const int H = self.rt.height(), W = self.rt.width();
            py::array_t<uint8_t> img({H, W, 3});
            hipError_t err = hipMemcpy(img.mutable_data(), self.rt.d_rgb(), (size_t)H * W * 3, hipMemcpyDeviceToHost);
            if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
            return img;
        }, py::arg("engine"), py::arg("camera_idx"), py::arg("env_idx") = 0,
           "Ray trace one view -> (H, W, 3) uint8, same row order as CameraRenderer.render (row 0 = top).")
        .def("render_depth", [](RayTracerPy& self, PhysicsEngine& engine, int camera_idx, int env_idx) {
            const int row0 = 0;
            self.run(engine, {camera_idx}, {env_idx}, &row0, 1, RT_OUT_DEPTH, false, false);
            const int H = self.rt.height(), W = self.rt.width();
            py::array_t<float> d({H, W});
            hipError_t err = hipMemcpy(d.mutable_data(), self.rt.d_depth(), (size_t)H * W * sizeof(float), hipMemcpyDeviceToHost);
            if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
            return d;
        }, py::arg("engine"), py::arg("camera_idx"), py::arg("env_idx") = 0,
           "(H, W) float32 planar depth in metres along camera -Z, 0 on a miss.")
        .def("render_seg", [](RayTracerPy& self, PhysicsEngine& engine, int camera_idx, int env_idx) {
            const int row0 = 0;
            self.run(engine, {camera_idx}, {env_idx}, &row0, 1, RT_OUT_SEG, false, false);
            const int H = self.rt.height(), W = self.rt.width();
            py::array_t<int> sg({H, W});
            hipError_t err = hipMemcpy(sg.mutable_data(), self.rt.d_seg(), (size_t)H * W * sizeof(int), hipMemcpyDeviceToHost);
            if (err != hipSuccess) throw std::runtime_error(hipGetErrorString(err));
            return sg;
        }, py::arg("engine"), py::arg("camera_idx"), py::arg("env_idx") = 0,
           "(H, W) int32 hip geom id (== MuJoCo geom id), -1 on a miss.")
        .def("render_gpu_batch", [](RayTracerPy& self, PhysicsEngine& engine, int camera_idx, int num_envs,
                                    bool flip_v, bool flip_h, py::object active) -> GpuImageBatch {
            if (num_envs <= 0) throw std::runtime_error("num_envs must be positive");
            const auto worlds = RayTracerPy::slots(active, num_envs);
            self.run(engine, {camera_idx}, worlds, nullptr, num_envs, RT_OUT_RGB, flip_v, flip_h);
            return GpuImageBatch{self.rt.d_rgb(), num_envs, self.rt.height(), self.rt.width(),
                                 self.rt.rgb_buffer()};
        }, py::arg("engine"), py::arg("camera_idx"), py::arg("num_envs"),
           py::arg("flip_v") = false, py::arg("flip_h") = false, py::arg("active") = py::none(),
           "(N, H, W, 3) u8 on device. Rows not in `active` keep the previous call's frame when that "
           "call used the same camera and num_envs, else they are zero. The view owns its memory, but "
           "the next RT render call that writes RGB may overwrite its content.")
        .def("render_all_gpu_batch", [](RayTracerPy& self, PhysicsEngine& engine, std::vector<int> camera_ids,
                                        int num_envs, py::object active, const std::string& outputs) {
            if (num_envs <= 0) throw std::runtime_error("num_envs must be positive");
            int mask = 0;
            {
                std::string tok;
                for (size_t i = 0; i <= outputs.size(); ++i) {
                    if (i == outputs.size() || outputs[i] == ',') {
                        if (tok == "rgb") mask |= RT_OUT_RGB;
                        else if (tok == "depth") mask |= RT_OUT_DEPTH;
                        else if (tok == "seg") mask |= RT_OUT_SEG;
                        else if (!tok.empty()) throw std::runtime_error("render_all_gpu_batch: outputs are rgb, depth, seg");
                        tok.clear();
                    } else tok += outputs[i];
                }
                if (!mask) throw std::runtime_error("render_all_gpu_batch: no outputs requested");
            }
            const auto worlds = RayTracerPy::slots(active, num_envs);
            self.run(engine, camera_ids, worlds, nullptr, num_envs, mask, false, false);
            py::object owner = py::cast(&self, py::return_value_policy::reference);
            const ssize_t C = (ssize_t)camera_ids.size(), N = num_envs;
            const ssize_t H = self.rt.height(), W = self.rt.width();
            py::object rgb = py::none(), dep = py::none(), seg = py::none();
            if (mask & RT_OUT_RGB)   rgb = py::cast(GpuArray{self.rt.d_rgb(), {C, N, H, W, 3}, "|u1", 1, owner, self.rt.rgb_buffer()});
            if (mask & RT_OUT_DEPTH) dep = py::cast(GpuArray{self.rt.d_depth(), {C, N, H, W}, "<f4", 4, owner, self.rt.depth_buffer()});
            if (mask & RT_OUT_SEG)   seg = py::cast(GpuArray{self.rt.d_seg(), {C, N, H, W}, "<i4", 4, owner, self.rt.seg_buffer()});
            return py::make_tuple(rgb, dep, seg);
        }, py::arg("engine"), py::arg("camera_ids"), py::arg("num_envs"), py::arg("active") = py::none(),
           py::arg("outputs") = "rgb,depth,seg",
           "One launch for all cameras -> (rgb u8 (C,N,H,W,3), depth f32 (C,N,H,W), seg i32 (C,N,H,W)); "
           "outputs (e.g. 'rgb') skips the others (None; MSAA4 / even SSAA then skip the pixel-centre "
           "depth/seg pass). "
           "Rows not in `active` keep the previous call's frame when that call used the same camera list "
           "and num_envs, else they are cleared (rgb 0, depth 0, seg -1). The arrays own their memory, "
           "but a later RT render call may overwrite their content.")
        .def("ray_batch", [](RayTracerPy& self, PhysicsEngine& engine, int env_idx,
                             py::array_t<float, py::array::c_style | py::array::forcecast> origins,
                             py::array_t<float, py::array::c_style | py::array::forcecast> dirs) {
            self.check(engine);
            if (origins.ndim() != 2 || origins.shape(1) != 3 || dirs.ndim() != 2 || dirs.shape(1) != 3 ||
                origins.shape(0) != dirs.shape(0))
                throw std::runtime_error("ray_batch: expected origins (N,3) and dirs (N,3)");
            if (origins.shape(0) > (ssize_t)INT_MAX)
                throw std::runtime_error("ray_batch: at most 2^31-1 rays per call");
            const int n = (int)origins.shape(0);
            py::array_t<int> gid((ssize_t)n);
            py::array_t<float> t((ssize_t)n);
            py::array_t<float> nrm({(ssize_t)n, (ssize_t)3});
            if (n > 0)
                self.rt.ray_batch(self.model->device_model(), engine.data().device_data(), engine.stream(),
                                  env_idx, origins.data(), dirs.data(), n,
                                  gid.mutable_data(), t.mutable_data(), nrm.mutable_data());
            return py::make_tuple(gid, t, nrm);
        }, py::arg("engine"), py::arg("env_idx"), py::arg("origins"), py::arg("dirs"),
           "World rays -> (gid int32[N] (-1 miss), t float32[N] (-1 miss), unit normal float32[N,3]).")
        .def("debug_rays", [](RayTracerPy& self, PhysicsEngine& engine, int camera_idx, int env_idx) {
            self.check(engine);
            const int H = self.rt.height(), W = self.rt.width();
            py::array_t<float> eye((ssize_t)3), Rc((ssize_t)9);
            py::array_t<float> dirs({(ssize_t)H, (ssize_t)W, (ssize_t)3});
            self.rt.debug_rays(self.model->device_model(), engine.data().device_data(), engine.stream(),
                               camera_idx, env_idx, eye.mutable_data(), Rc.mutable_data(), dirs.mutable_data());
            return py::make_tuple(eye, Rc, dirs);
        }, py::arg("engine"), py::arg("camera_idx"), py::arg("env_idx") = 0,
           "(eye[3], Rc[9] row-major, dirs (H,W,3) unit world dirs of pixel centres, row 0 = top).")
        .def("stats", [](RayTracerPy& self) {
            const auto st = self.rt.stats();
            py::dict d;
            d["nrg"] = st.nrg; d["ntri"] = st.ntri; d["nsite_items"] = st.nsite_items;
            d["nblas_nodes"] = st.nblas_nodes; d["nblas_nodes64"] = st.nblas_nodes64;
            d["last_ms"] = st.last_ms;
            if (st.counters_valid) {
                d["cand_per_ray"] = st.cand_per_ray;
                d["cand_per_tile"] = st.cand_per_tile;
                d["nodes_per_ray"] = st.nodes_per_ray;
                d["tris_per_ray"] = st.tris_per_ray;
                d["list_per_tile"] = st.list_per_tile;
                d["visited_per_tile"] = st.visited_per_tile;
            } else {
                d["cand_per_ray"] = py::none(); d["cand_per_tile"] = py::none();
                d["nodes_per_ray"] = py::none(); d["tris_per_ray"] = py::none();
            }
            return d;
        }, "Scene sizes, last K1+K2 ms, and (with set_stats(True)) traversal counters: cand_per_ray = "
           "instance AABBs accepted per ray; cand_per_tile = instances accepted by >= 1 lane of an 8x8 "
           "tile, per tile and SSAA sample; nodes_per_ray / tris_per_ray = BLAS work per ray.");
}
