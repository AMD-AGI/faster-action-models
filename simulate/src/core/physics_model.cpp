// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// PhysicsModel: MJCF XML parser + GPU upload.
//
// Parses MJCF sufficient for LIBERO/Franka environments:
//   - <compiler> for angle units, coordinate conventions
//   - <default> classes for attribute inheritance
//   - <option> for timestep, gravity
//   - <worldbody> tree of <body> with <joint>, <geom>, <inertial>
//   - <actuator> with <motor>/<position>/<velocity>/<general>
//   - <contact> with <exclude>
//   - <equality> with <weld>/<joint>
//   - <sensor>/<camera>
//
// After parsing, builds BFS tree levels and uploads all arrays to GPU.

#include <atomic>
#include <sstream>
#include "hipgl/physics/physics_model.h"
#include "hipgl/physics/mesh_loader.h"
#include "hipgl/physics/convex_hull.h"
#include "hipgl/texture_mip_builder.h"
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <cstring>
#include <strings.h>   // strcasecmp
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cmath>
#include <queue>
#include <vector>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>

#include "tinyxml2.h"

using namespace tinyxml2;

namespace hipgl {
namespace physics {

// ── Helpers ─────────────────────────────────────────────────────────────────

static void parse_doubles(const char* str, double* out, int n) {
    if (!str) return;
    for (int i = 0; i < n; i++) {
        while (*str == ' ' || *str == '\t' || *str == '\n' || *str == '\r') str++;
        char* end; out[i] = strtod(str, &end); str = end;
    }
}

// Body/inertial quaternions are physics inputs, not arbitrary four-vectors.
// Keep this validation scoped: other MJCF numeric/orientation fields retain
// their existing parsing and conversion conventions.
static std::invalid_argument body_quaternion_error(
    const char* body_name, const char* field, const char* reason) {
    return std::invalid_argument(std::string("instinct_sim: body '") +
        (body_name && *body_name ? body_name : "<unnamed>") + "' " + field + ": " + reason);
}

static void parse_body_quaternion(const char* text, double* out,
                                  const char* body_name, const char* field) {
    double parsed[4];
    const char* cursor = text;
    for (int i = 0; i < 4; ++i) {
        while (std::isspace(static_cast<unsigned char>(*cursor))) ++cursor;
        char* end = nullptr;
        parsed[i] = std::strtod(cursor, &end);
        if (end == cursor || (*end && !std::isspace(static_cast<unsigned char>(*end))))
            throw body_quaternion_error(body_name, field, "expected exactly four numeric tokens (wxyz)");
        if (!std::isfinite(parsed[i]))
            throw body_quaternion_error(body_name, field, "components must be finite");
        cursor = end;
    }
    while (std::isspace(static_cast<unsigned char>(*cursor))) ++cursor;
    if (*cursor)
        throw body_quaternion_error(body_name, field, "expected exactly four numeric tokens (wxyz)");
    std::copy(parsed, parsed + 4, out);
}

static void normalize_body_quaternion(double* quat, float* mirror,
                                      const char* body_name, const char* field) {
    double scale = 0;
    for (int i = 0; i < 4; ++i) {
        if (!std::isfinite(quat[i]))
            throw body_quaternion_error(body_name, field, "components must be finite");
        scale = std::max(scale, std::abs(quat[i]));
    }
    if (scale == 0)
        throw body_quaternion_error(body_name, field, "zero quaternion is not allowed");
    // Scaling first avoids norm overflow/underflow for finite nonzero input.
    // Division by positive values preserves the input sign, including -q.
    double scaled[4], squared = 0;
    for (int i = 0; i < 4; ++i) {
        scaled[i] = quat[i] / scale;
        squared += scaled[i] * scaled[i];
    }
    const double length = std::sqrt(squared);
    for (int i = 0; i < 4; ++i) {
        quat[i] = scaled[i] / length;
        mirror[i] = static_cast<float>(quat[i]);
    }
}

static double attr_double(XMLElement* el, const char* name, double def) {
    if (!el) return def;
    const char* s = el->Attribute(name);
    if (!s) return def;
    return strtod(s, nullptr);
}

static void parse_floats(const char* str, float* out, int n) {
    if (!str) return;
    for (int i = 0; i < n; i++) {
        while (*str == ' ' || *str == '\t' || *str == '\n' || *str == '\r') str++;
        if (*str == '\0') break;
        out[i] = strtof(str, const_cast<char**>(&str));
    }
}

// Convert MuJoCo xyaxes (6 floats: x-axis, y-axis of camera frame) to quaternion.
// MuJoCo convention: x = normalize(x_raw), z = normalize(cross(x, y_raw)), y = cross(z, x)
static void xyaxes_to_quat(const float* xyaxes, float* quat) {
    float x[3] = {xyaxes[0], xyaxes[1], xyaxes[2]};
    float y_raw[3] = {xyaxes[3], xyaxes[4], xyaxes[5]};

    // Normalize x
    float xlen = sqrtf(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
    if (xlen > 1e-8f) { x[0] /= xlen; x[1] /= xlen; x[2] /= xlen; }

    // z = normalize(cross(x, y_raw))
    float z[3] = {
        x[1]*y_raw[2] - x[2]*y_raw[1],
        x[2]*y_raw[0] - x[0]*y_raw[2],
        x[0]*y_raw[1] - x[1]*y_raw[0]
    };
    float zlen = sqrtf(z[0]*z[0] + z[1]*z[1] + z[2]*z[2]);
    if (zlen > 1e-8f) { z[0] /= zlen; z[1] /= zlen; z[2] /= zlen; }

    // y = cross(z, x)
    float y[3] = {
        z[1]*x[2] - z[2]*x[1],
        z[2]*x[0] - z[0]*x[2],
        z[0]*x[1] - z[1]*x[0]
    };

    // Rotation matrix [x y z] columns → quaternion (w, x, y, z)
    float R00 = x[0], R10 = x[1], R20 = x[2];
    float R01 = y[0], R11 = y[1], R21 = y[2];
    float R02 = z[0], R12 = z[1], R22 = z[2];

    float trace = R00 + R11 + R22;
    if (trace > 0.0f) {
        float s = 0.5f / sqrtf(trace + 1.0f);
        quat[0] = 0.25f / s;
        quat[1] = (R21 - R12) * s;
        quat[2] = (R02 - R20) * s;
        quat[3] = (R10 - R01) * s;
    } else if (R00 > R11 && R00 > R22) {
        float s = 2.0f * sqrtf(1.0f + R00 - R11 - R22);
        quat[0] = (R21 - R12) / s;
        quat[1] = 0.25f * s;
        quat[2] = (R01 + R10) / s;
        quat[3] = (R02 + R20) / s;
    } else if (R11 > R22) {
        float s = 2.0f * sqrtf(1.0f + R11 - R00 - R22);
        quat[0] = (R02 - R20) / s;
        quat[1] = (R01 + R10) / s;
        quat[2] = 0.25f * s;
        quat[3] = (R12 + R21) / s;
    } else {
        float s = 2.0f * sqrtf(1.0f + R22 - R00 - R11);
        quat[0] = (R10 - R01) / s;
        quat[1] = (R02 + R20) / s;
        quat[2] = (R12 + R21) / s;
        quat[3] = 0.25f * s;
    }
    // Normalize quaternion
    float qlen = sqrtf(quat[0]*quat[0]+quat[1]*quat[1]+quat[2]*quat[2]+quat[3]*quat[3]);
    if (qlen > 1e-8f) { quat[0]/=qlen; quat[1]/=qlen; quat[2]/=qlen; quat[3]/=qlen; }
    // Ensure w > 0 (canonical form)
    if (quat[0] < 0.0f) { quat[0]=-quat[0]; quat[1]=-quat[1]; quat[2]=-quat[2]; quat[3]=-quat[3]; }
}

static float attr_float(XMLElement* el, const char* name, float def) {
    if (!el) return def;
    const char* s = el->Attribute(name);
    if (!s) return def;
    return strtof(s, nullptr);
}

static int attr_int(XMLElement* el, const char* name, int def) {
    if (!el) return def;
    const char* s = el->Attribute(name);
    if (!s) return def;
    return (int)strtol(s, nullptr, 10);
}

static const char* attr_str(XMLElement* el, const char* name) {
    if (!el) return nullptr;
    return el->Attribute(name);
}

static void safe_strncpy(char* dst, const char* src, int maxlen) {
    if (!src) { dst[0] = '\0'; return; }
    strncpy(dst, src, maxlen - 1);
    dst[maxlen - 1] = '\0';
}

static constexpr float DEG2RAD = 3.14159265358979f / 180.0f;

// MJCF euler -> quaternion (w,x,y,z), angles already in radians. Follows
// MuJoCo's <compiler eulerseq> (default "xyz"): character i names the axis of
// rotation i; a lowercase letter rotates about the moving (intrinsic) axis,
// q = q * q_i, an uppercase one about the fixed (extrinsic) axis, q = q_i * q.
// With the default this is q = qx * qy * qz.
static void euler_to_quat(const double e[3], const std::string& seq, double q[4]) {
    q[0] = 1.0; q[1] = q[2] = q[3] = 0.0;
    for (int i = 0; i < 3; ++i) {
        const char c = i < (int)seq.size() ? seq[i] : "xyz"[i];
        double r[4] = {cos(0.5 * e[i]), 0.0, 0.0, 0.0};
        const int ax = (c == 'x' || c == 'X') ? 1 : (c == 'y' || c == 'Y') ? 2 : 3;
        r[ax] = sin(0.5 * e[i]);
        const bool intrinsic = (c >= 'a' && c <= 'z');
        const double* a = intrinsic ? q : r;
        const double* b = intrinsic ? r : q;
        const double o[4] = {
            a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
            a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
            a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
            a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]};
        for (int k = 0; k < 4; ++k) q[k] = o[k];
    }
}

static int parse_site_type(const char* type_str) {
    if (!type_str || strcmp(type_str, "sphere") == 0) return SITE_TYPE_SPHERE;
    if (strcmp(type_str, "capsule") == 0) return SITE_TYPE_CAPSULE;
    if (strcmp(type_str, "ellipsoid") == 0) return SITE_TYPE_ELLIPSOID;
    if (strcmp(type_str, "cylinder") == 0) return SITE_TYPE_CYLINDER;
    if (strcmp(type_str, "box") == 0) return SITE_TYPE_BOX;
    return SITE_TYPE_SPHERE;
}

static void init_site_desc(SiteDesc& s, int body) {
    memset(&s, 0, sizeof(s));
    s.body = body;
    s.type = SITE_TYPE_SPHERE;
    s.group = 0;
    s.size[0] = 0.005f;
    s.size[1] = 0.005f;
    s.size[2] = 0.005f;
    s.rgba[0] = 0.5f;
    s.rgba[1] = 0.5f;
    s.rgba[2] = 0.5f;
    s.rgba[3] = 1.0f;
    s.quat[0] = 1.0f;
    s.quat_d[0] = 1.0;
}

static void parse_site_fields(XMLElement* el, SiteDesc& s, bool angle_radian, const std::string& eulerseq) {
    safe_strncpy(s.name, el->Attribute("name"), MAX_NAME_LEN);
    parse_floats(el->Attribute("pos"), s.pos, 3);
    parse_doubles(el->Attribute("pos"), s.pos_d, 3);
    parse_floats(el->Attribute("size"), s.size, 3);
    parse_floats(el->Attribute("rgba"), s.rgba, 4);
    s.group = attr_int(el, "group", s.group);
    s.type = parse_site_type(el->Attribute("type"));
    const char* squat = el->Attribute("quat");
    if (squat) parse_floats(squat, s.quat, 4);
    if (squat) parse_doubles(squat, s.quat_d, 4);
    const char* seuler = el->Attribute("euler");
    if (seuler && !squat) {
        double e[3] = {0, 0, 0};
        parse_doubles(seuler, e, 3);
        if (!angle_radian) { e[0] *= DEG2RAD; e[1] *= DEG2RAD; e[2] *= DEG2RAD; }
        euler_to_quat(e, eulerseq, s.quat_d);
        for (int k = 0; k < 4; ++k) s.quat[k] = (float)s.quat_d[k];
    }
}

// ── Default class system ────────────────────────────────────────────────────
// Stores default attribute values per class for joints, geoms, and actuators.

struct JointDefaults {
    float damping       = 0.0f;
    float stiffness     = 0.0f;
    float armature      = 0.0f;
    float frictionloss  = 0.0f;
    float solref_lim[2] = {0.02f, 1.0f};
    float solimp_lim[5] = {0.9f, 0.95f, 0.001f, 0.5f, 2.0f};
    float axis[3]       = {0, 0, 1};
    float range[2]      = {-3.14159f, 3.14159f};
    int   limited       = -1;  // -1 = unset
    bool  has_damping   = false;
    bool  has_stiff     = false;
    bool  has_armature  = false;
    bool  has_fricloss  = false;
    bool  has_solref_lim = false;
    bool  has_solimp_lim = false;
    bool  has_axis      = false;
    bool  has_range     = false;
};

struct GeomDefaults {
    float friction[3] = {1.0f, 0.005f, 0.0001f};
    float solref[2]   = {0.02f, 1.0f};
    float solimp[5]   = {0.9f, 0.95f, 0.001f, 0.5f, 2.0f};
    float margin      = 0.0f;
    float gap         = 0.0f;
    int   contype     = 1;
    int   conaffinity = 1;
    int   condim      = 3;
    int   type        = -1;  // -1 = unset
    bool  has_friction = false;
    bool  has_solref   = false;
    bool  has_solimp   = false;
    bool  has_margin   = false;
    bool  has_gap      = false;
    bool  has_contype  = false;
    bool  has_conaff   = false;
    bool  has_condim   = false;
};

struct DefaultClass {
    std::string name;
    JointDefaults jnt;
    GeomDefaults  geom;
};

struct ParserState {
    PhysicsModelDesc& desc;
    bool angle_radian = false;  // false = degree (MuJoCo default)
    bool autolimits = true;     // infer joint limits from explicit or inherited ranges
    std::string eulerseq = "xyz";  // <compiler eulerseq> (MuJoCo default)
    double settotalmass = -1.0;  // <compiler settotalmass="X">: rescale all body masses to sum to X
    std::unordered_map<std::string, DefaultClass> defaults;
    DefaultClass global_defaults;  // unnamed top-level default
    std::unordered_map<std::string, int> joint_name_map;
    std::unordered_map<std::string, int> body_name_map;
    std::unordered_map<std::string, int> mesh_name_map;  // mesh name → mesh index
    std::string meshdir;  // base directory for mesh file resolution

    struct MatInfo { float specular; float shininess; float emission; float reflectance; float rgba[4];
                     float texrepeat[2]; int texuniform; };
    std::unordered_map<std::string, MatInfo> material_map;
    // geom index -> material name, recorded in BOTH geom parse paths (body
    // geoms and worldbody geoms) so the deferred texgen pass can resolve
    // texrepeat/texuniform regardless of where the geom was declared.
    std::unordered_map<int, std::string> geom_material;
};

// ── Default parsing ─────────────────────────────────────────────────────────

static void parse_default_joint(XMLElement* el, JointDefaults& jd) {
    const char* s;
    s = el->Attribute("damping");
    if (s) { jd.damping = strtof(s, nullptr); jd.has_damping = true; }
    s = el->Attribute("stiffness");
    if (s) { jd.stiffness = strtof(s, nullptr); jd.has_stiff = true; }
    s = el->Attribute("armature");
    if (s) { jd.armature = strtof(s, nullptr); jd.has_armature = true; }
    s = el->Attribute("frictionloss");
    if (s) { jd.frictionloss = strtof(s, nullptr); jd.has_fricloss = true; }
    s = el->Attribute("solreflimit");
    if (s) { parse_floats(s, jd.solref_lim, 2); jd.has_solref_lim = true; }
    s = el->Attribute("solimplimit");
    if (s) { parse_floats(s, jd.solimp_lim, 5); jd.has_solimp_lim = true; }
    s = el->Attribute("axis");
    if (s) { parse_floats(s, jd.axis, 3); jd.has_axis = true; }
    s = el->Attribute("range");
    if (s) { parse_floats(s, jd.range, 2); jd.has_range = true; }
    s = el->Attribute("limited");
    if (s) {
        if (strcmp(s, "true") == 0 || strcmp(s, "1") == 0)
            jd.limited = 1;
        else if (strcmp(s, "auto") == 0)
            jd.limited = 2;  // auto: limited if range is set
        else
            jd.limited = 0;
    }
}

static void parse_default_geom(XMLElement* el, GeomDefaults& gd) {
    const char* s;
    s = el->Attribute("friction");
    if (s) { parse_floats(s, gd.friction, 3); gd.has_friction = true; }
    s = el->Attribute("solref");
    if (s) { parse_floats(s, gd.solref, 2); gd.has_solref = true; }
    s = el->Attribute("solimp");
    if (s) { parse_floats(s, gd.solimp, 5); gd.has_solimp = true; }
    s = el->Attribute("margin");
    if (s) { gd.margin = strtof(s, nullptr); gd.has_margin = true; }
    s = el->Attribute("gap");
    if (s) { gd.gap = strtof(s, nullptr); gd.has_gap = true; }
    s = el->Attribute("contype");
    if (s) { gd.contype = (int)strtol(s, nullptr, 10); gd.has_contype = true; }
    s = el->Attribute("conaffinity");
    if (s) { gd.conaffinity = (int)strtol(s, nullptr, 10); gd.has_conaff = true; }
    s = el->Attribute("condim");
    if (s) { gd.condim = (int)strtol(s, nullptr, 10); gd.has_condim = true; }
    s = el->Attribute("type");
    if (s) {
        if (strcmp(s, "sphere") == 0) gd.type = GEOM_SPHERE;
        else if (strcmp(s, "box") == 0) gd.type = GEOM_BOX;
        else if (strcmp(s, "capsule") == 0) gd.type = GEOM_CAPSULE;
        else if (strcmp(s, "cylinder") == 0) gd.type = GEOM_CAPSULE;
    }
}

static void parse_defaults_recursive(XMLElement* el, ParserState& state, DefaultClass parent) {
    const char* class_name = el->Attribute("class");
    DefaultClass dc = parent;
    if (class_name) dc.name = class_name;

    XMLElement* jel = el->FirstChildElement("joint");
    if (jel) parse_default_joint(jel, dc.jnt);

    XMLElement* gel = el->FirstChildElement("geom");
    if (gel) parse_default_geom(gel, dc.geom);

    if (class_name) {
        state.defaults[class_name] = dc;
    } else {
        state.global_defaults = dc;
    }

    for (XMLElement* child = el->FirstChildElement("default"); child;
         child = child->NextSiblingElement("default")) {
        parse_defaults_recursive(child, state, dc);
    }
}

// ── Apply defaults to a joint ───────────────────────────────────────────────

static void apply_joint_defaults(JointDesc& j, const JointDefaults& jd) {
    if (jd.has_damping) { j.damping = jd.damping; j.damping_d = (double)jd.damping; }
    if (jd.has_stiff)   j.stiffness = jd.stiffness;
    if (jd.has_armature) { j.armature = jd.armature; j.armature_d = (double)jd.armature; }
    if (jd.has_fricloss) j.frictionloss = jd.frictionloss;
    if (jd.has_solref_lim) { j.solref_lim[0] = jd.solref_lim[0]; j.solref_lim[1] = jd.solref_lim[1]; }
    if (jd.has_solimp_lim) { memcpy(j.solimp_lim, jd.solimp_lim, 5 * sizeof(float)); }
    if (jd.has_axis) {
        j.axis[0] = jd.axis[0]; j.axis[1] = jd.axis[1]; j.axis[2] = jd.axis[2];
    }
    if (jd.has_range) {
        j.range[0] = jd.range[0]; j.range[1] = jd.range[1];
    }
    if (jd.limited >= 0) j.limited = jd.limited;
}

static void apply_geom_defaults(GeomDesc& g, const GeomDefaults& gd) {
    if (gd.has_friction) {
        g.friction[0] = gd.friction[0]; g.friction[1] = gd.friction[1]; g.friction[2] = gd.friction[2];
    }
    if (gd.has_solref) { g.solref[0] = gd.solref[0]; g.solref[1] = gd.solref[1]; }
    if (gd.has_solimp) { memcpy(g.solimp, gd.solimp, 5 * sizeof(float)); }
    if (gd.has_margin) g.margin = gd.margin;
    if (gd.has_gap)    g.gap = gd.gap;
    if (gd.has_contype) g.contype = gd.contype;
    if (gd.has_conaff)  g.conaffinity = gd.conaffinity;
    if (gd.has_condim)  g.condim = gd.condim;
    if (gd.type >= 0)   g.type = (GeomType)gd.type;
}

// ── Look up defaults for a class name ───────────────────────────────────────

static const DefaultClass& get_defaults(ParserState& state, const char* class_name) {
    if (class_name) {
        auto it = state.defaults.find(class_name);
        if (it != state.defaults.end()) return it->second;
    }
    return state.global_defaults;
}

// ── Body parser ─────────────────────────────────────────────────────────────


// Camera attributes beyond pos/quat/xyaxes/fovy: euler is converted like a
// body euler; everything else the renderers cannot honour is flagged.
static void parse_camera_extras(tinyxml2::XMLElement* cel, CameraDesc& c, const ParserState& state) {
    const char* ceuler = cel->Attribute("euler");
    if (ceuler && !cel->Attribute("quat") && !cel->Attribute("xyaxes")) {
        double e[3] = {0, 0, 0};
        parse_doubles(ceuler, e, 3);
        if (!state.angle_radian) { e[0] *= DEG2RAD; e[1] *= DEG2RAD; e[2] *= DEG2RAD; }
        double q[4];
        euler_to_quat(e, state.eulerseq, q);
        for (int k = 0; k < 4; ++k) c.quat[k] = (float)q[k];
    }
    const char* mode = cel->Attribute("mode");
    if (mode && strcmp(mode, "fixed") != 0) c.unsupported |= CAM_UNSUP_MODE;
    if (cel->Attribute("target")) c.unsupported |= CAM_UNSUP_MODE;
    if (cel->Attribute("sensorsize") || cel->Attribute("focal") || cel->Attribute("focalpixel") ||
        cel->Attribute("principal") || cel->Attribute("principalpixel"))
        c.unsupported |= CAM_UNSUP_INTRINSICS;
    const char* ortho = cel->Attribute("orthographic");
    if (ortho && strcmp(ortho, "true") == 0) c.unsupported |= CAM_UNSUP_ORTHO;
    if (!cel->Attribute("quat") && !cel->Attribute("xyaxes") && !ceuler &&
        (cel->Attribute("axisangle") || cel->Attribute("zaxis")))
        c.unsupported |= CAM_UNSUP_ORIENTATION;
}

static void parse_body(XMLElement* body_el, int parent_idx, ParserState& state) {
    PhysicsModelDesc& desc = state.desc;
    if (desc.nbody >= MAX_BODIES)
        throw std::runtime_error("PhysicsModel: more than MAX_BODIES (" + std::to_string(MAX_BODIES) + ") bodies");
    int idx = desc.nbody++;
    BodyDesc& b = desc.bodies[idx];
    memset(&b, 0, sizeof(b));
    b.parent = parent_idx;
    b.quat[0] = 1.0f;  b.quat_d[0] = 1.0;
    b.iquat[0] = 1.0f; b.iquat_d[0] = 1.0;
    // An empty / site-only body is a massless frame, not a small rigid body.
    // Keep the zero mass and inertia from memset unless <inertial> or geoms
    // supply them. Artificial defaults otherwise contribute to every ancestor
    // joint's composite inertia and gravity bias (e.g. an end-effector site).
    b.jnt_adr = desc.njnt;
    b.jnt_num = 0;

    safe_strncpy(b.name, body_el->Attribute("name"), MAX_NAME_LEN);
    if (b.name[0]) state.body_name_map[b.name] = idx;

    // Determine which default class to use for children
    const char* childclass = body_el->Attribute("childclass");
    const DefaultClass& body_dc = get_defaults(state, childclass);

    parse_floats(body_el->Attribute("pos"), b.pos, 3);
    parse_doubles(body_el->Attribute("pos"), b.pos_d, 3);
    const char* quat_str = body_el->Attribute("quat");
    if (quat_str) {
        parse_body_quaternion(quat_str, b.quat_d, body_el->Attribute("name"), "body quat");
    } else {
        const char* euler_str = body_el->Attribute("euler");
        if (euler_str) {
            double euler_d[3] = {0, 0, 0};
            parse_doubles(euler_str, euler_d, 3);
            if (!state.angle_radian) {
                euler_d[0] *= DEG2RAD; euler_d[1] *= DEG2RAD; euler_d[2] *= DEG2RAD;
            }
            euler_to_quat(euler_d, state.eulerseq, b.quat_d);
            for (int k = 0; k < 4; ++k) b.quat[k] = (float)b.quat_d[k];
        }
        const char* body_aa_str = body_el->Attribute("axisangle");
        if (body_aa_str && !quat_str && !euler_str) {
            float aa[4] = {0, 0, 1, 0};
            parse_floats(body_aa_str, aa, 4);
            float ax = aa[0], ay = aa[1], az = aa[2], angle = aa[3];
            if (!state.angle_radian) angle *= DEG2RAD;
            float n = sqrtf(ax*ax + ay*ay + az*az);
            if (n > 1e-8f) { ax /= n; ay /= n; az /= n; }
            float ha = angle * 0.5f;
            float sha = sinf(ha);
            b.quat[0] = cosf(ha); b.quat[1] = ax*sha; b.quat[2] = ay*sha; b.quat[3] = az*sha;
            double aa_d[4] = {0, 0, 1, 0};
            parse_doubles(body_aa_str, aa_d, 4);
            double axd = aa_d[0], ayd = aa_d[1], azd = aa_d[2], angled = aa_d[3];
            if (!state.angle_radian) angled *= DEG2RAD;
            double nd = sqrt(axd*axd + ayd*ayd + azd*azd);
            if (nd > 1e-15) { axd /= nd; ayd /= nd; azd /= nd; }
            double had = angled * 0.5;
            double shad = sin(had);
            b.quat_d[0] = cos(had); b.quat_d[1] = axd*shad; b.quat_d[2] = ayd*shad; b.quat_d[3] = azd*shad;
        }
    }
    normalize_body_quaternion(b.quat_d, b.quat, body_el->Attribute("name"), "body quat");
    b.gravcomp = attr_double(body_el, "gravcomp", 0.0);

    // Parse <inertial>
    XMLElement* inertial = body_el->FirstChildElement("inertial");
    if (inertial) {
        b.mass = attr_float(inertial, "mass", b.mass);
        b.mass_d = attr_double(inertial, "mass", b.mass_d);
        parse_floats(inertial->Attribute("pos"), b.ipos, 3);
        parse_doubles(inertial->Attribute("pos"), b.ipos_d, 3);
        const char* iq = inertial->Attribute("quat");
        if (iq) {
            parse_body_quaternion(iq, b.iquat_d, body_el->Attribute("name"), "inertial quat");
            normalize_body_quaternion(b.iquat_d, b.iquat, body_el->Attribute("name"), "inertial quat");
        }
        const char* diag = inertial->Attribute("diaginertia");
        if (diag) {
            parse_floats(diag, b.inertia, 3);
            parse_doubles(diag, b.inertia_d, 3);
        } else {
            const char* full = inertial->Attribute("fullinertia");
            if (full) {
                double fi_d[6];
                parse_doubles(full, fi_d, 6);
                // fi_d = [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
                // Build symmetric 3x3 inertia matrix and diagonalize
                double A[3][3] = {
                    {fi_d[0], fi_d[3], fi_d[4]},
                    {fi_d[3], fi_d[1], fi_d[5]},
                    {fi_d[4], fi_d[5], fi_d[2]}
                };
                // Check if off-diagonal elements are significant
                double off_diag = fabs(fi_d[3]) + fabs(fi_d[4]) + fabs(fi_d[5]);
                if (off_diag > 1e-15 * (fabs(fi_d[0]) + fabs(fi_d[1]) + fabs(fi_d[2]))) {
                    // Jacobi eigendecomposition for symmetric 3x3
                    double V[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
                    for (int iter = 0; iter < 50; iter++) {
                        // Find largest off-diagonal element
                        int p = 0, q = 1;
                        double maxval = fabs(A[0][1]);
                        if (fabs(A[0][2]) > maxval) { p=0; q=2; maxval=fabs(A[0][2]); }
                        if (fabs(A[1][2]) > maxval) { p=1; q=2; maxval=fabs(A[1][2]); }
                        if (maxval < 1e-15) break;
                        double app = A[p][p], aqq = A[q][q], apq = A[p][q];
                        double theta = 0.5 * atan2(2.0*apq, app - aqq);
                        double c = cos(theta), s = sin(theta);
                        // Rotate A = G^T * A * G
                        double newA[3][3];
                        for (int i = 0; i < 3; i++)
                            for (int j = 0; j < 3; j++)
                                newA[i][j] = A[i][j];
                        for (int i = 0; i < 3; i++) {
                            newA[i][p] = c*A[i][p] + s*A[i][q];
                            newA[i][q] = -s*A[i][p] + c*A[i][q];
                        }
                        for (int j = 0; j < 3; j++) {
                            A[p][j] = c*newA[p][j] + s*newA[q][j];
                            A[q][j] = -s*newA[p][j] + c*newA[q][j];
                        }
                        for (int i = 0; i < 3; i++) {
                            if (i != p && i != q) {
                                A[i][p] = A[p][i];
                                A[i][q] = A[q][i];
                            }
                        }
                        A[p][q] = 0; A[q][p] = 0;
                        // Update eigenvectors V = V * G
                        for (int i = 0; i < 3; i++) {
                            double vp = V[i][p], vq = V[i][q];
                            V[i][p] = c*vp + s*vq;
                            V[i][q] = -s*vp + c*vq;
                        }
                    }
                    // Eigenvalues are diagonal of A, eigenvectors are columns of V
                    b.inertia_d[0] = A[0][0];
                    b.inertia_d[1] = A[1][1];
                    b.inertia_d[2] = A[2][2];
                    // Convert rotation matrix V to quaternion (iquat)
                    // Ensure proper rotation (det = +1)
                    double det = V[0][0]*(V[1][1]*V[2][2]-V[1][2]*V[2][1])
                               - V[0][1]*(V[1][0]*V[2][2]-V[1][2]*V[2][0])
                               + V[0][2]*(V[1][0]*V[2][1]-V[1][1]*V[2][0]);
                    if (det < 0) {
                        for (int i = 0; i < 3; i++) V[i][2] = -V[i][2];
                    }
                    // Rotation matrix to quaternion
                    double tr = V[0][0] + V[1][1] + V[2][2];
                    double qw, qx, qy, qz;
                    if (tr > 0) {
                        double s2 = sqrt(tr + 1.0) * 2.0;
                        qw = 0.25 * s2;
                        qx = (V[2][1] - V[1][2]) / s2;
                        qy = (V[0][2] - V[2][0]) / s2;
                        qz = (V[1][0] - V[0][1]) / s2;
                    } else if (V[0][0] > V[1][1] && V[0][0] > V[2][2]) {
                        double s2 = sqrt(1.0 + V[0][0] - V[1][1] - V[2][2]) * 2.0;
                        qw = (V[2][1] - V[1][2]) / s2;
                        qx = 0.25 * s2;
                        qy = (V[0][1] + V[1][0]) / s2;
                        qz = (V[0][2] + V[2][0]) / s2;
                    } else if (V[1][1] > V[2][2]) {
                        double s2 = sqrt(1.0 + V[1][1] - V[0][0] - V[2][2]) * 2.0;
                        qw = (V[0][2] - V[2][0]) / s2;
                        qx = (V[0][1] + V[1][0]) / s2;
                        qy = 0.25 * s2;
                        qz = (V[1][2] + V[2][1]) / s2;
                    } else {
                        double s2 = sqrt(1.0 + V[2][2] - V[0][0] - V[1][1]) * 2.0;
                        qw = (V[1][0] - V[0][1]) / s2;
                        qx = (V[0][2] + V[2][0]) / s2;
                        qy = (V[1][2] + V[2][1]) / s2;
                        qz = 0.25 * s2;
                    }
                    // Normalize and ensure w >= 0
                    double qnorm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
                    if (qnorm > 1e-15) { qw/=qnorm; qx/=qnorm; qy/=qnorm; qz/=qnorm; }
                    if (qw < 0) { qw=-qw; qx=-qx; qy=-qy; qz=-qz; }
                    b.iquat_d[0] = qw; b.iquat_d[1] = qx;
                    b.iquat_d[2] = qy; b.iquat_d[3] = qz;
                    b.iquat[0] = (float)qw; b.iquat[1] = (float)qx;
                    b.iquat[2] = (float)qy; b.iquat[3] = (float)qz;
                } else {
                    b.inertia_d[0] = fi_d[0];
                    b.inertia_d[1] = fi_d[1];
                    b.inertia_d[2] = fi_d[2];
                }
                b.inertia[0] = (float)b.inertia_d[0];
                b.inertia[1] = (float)b.inertia_d[1];
                b.inertia[2] = (float)b.inertia_d[2];
            }
        }
    }

    for (XMLElement* jel = body_el->FirstChildElement(); jel; jel = jel->NextSiblingElement()) {
        const bool is_freejoint = strcmp(jel->Name(), "freejoint") == 0;
        if (!is_freejoint && strcmp(jel->Name(), "joint") != 0) continue;
        if (desc.njnt >= MAX_JOINTS)
            throw std::runtime_error("PhysicsModel: more than MAX_JOINTS (" + std::to_string(MAX_JOINTS) + ") joints");
        int jidx = desc.njnt++;
        JointDesc& j = desc.joints[jidx];
        memset(&j, 0, sizeof(j));
        j.body = idx;
        j.axis[0] = 0; j.axis[1] = 0; j.axis[2] = 1;
        j.range[0] = -3.14159f; j.range[1] = 3.14159f;
        j.limited = 2;  // MuJoCo default: auto, resolved after class/element attributes

        safe_strncpy(j.name, jel->Attribute("name"), MAX_NAME_LEN);
        if (j.name[0]) state.joint_name_map[j.name] = jidx;

        // Apply class defaults, then element-specific defaults, then explicit attrs.
        // <freejoint> takes no class: MuJoCo applies no defaults to it.
        static const DefaultClass kNoDefaults{};
        const char* jclass = is_freejoint ? nullptr : jel->Attribute("class");
        const DefaultClass& jdc = is_freejoint ? kNoDefaults : jclass ? get_defaults(state, jclass) : body_dc;
        apply_joint_defaults(j, jdc.jnt);

        const char* type_str = is_freejoint ? "free" : jel->Attribute("type");
        if (!type_str || strcmp(type_str, "hinge") == 0)
            j.type = JOINT_HINGE;
        else if (strcmp(type_str, "slide") == 0)
            j.type = JOINT_SLIDE;
        else if (strcmp(type_str, "ball") == 0)
            j.type = JOINT_BALL;
        else if (strcmp(type_str, "free") == 0)
            j.type = JOINT_FREE;
        else
            j.type = JOINT_HINGE;

        // Explicit attributes override defaults
        const char* ax_str = jel->Attribute("axis");
        if (ax_str) { parse_floats(ax_str, j.axis, 3); parse_doubles(ax_str, j.axis_d, 3); }

        parse_floats(jel->Attribute("pos"), j.pos, 3);
        parse_doubles(jel->Attribute("pos"), j.pos_d, 3);

        const char* range_str = jel->Attribute("range");
        if (range_str) {
            parse_floats(range_str, j.range, 2);
            if (!state.angle_radian && (j.type == JOINT_HINGE || j.type == JOINT_BALL)) {
                j.range[0] *= DEG2RAD;
                j.range[1] *= DEG2RAD;
            }
        }

        const char* limited_str = jel->Attribute("limited");
        if (limited_str) {
            if (strcmp(limited_str, "true") == 0 || strcmp(limited_str, "1") == 0)
                j.limited = 1;
            else if (strcmp(limited_str, "auto") == 0)
                j.limited = 2;
            else
                j.limited = 0;
        }
        // A range inherited from a default class is still a specified range.
        if (j.limited == 2) {
            const bool has_range = range_str != nullptr || jdc.jnt.has_range;
            if (has_range && !state.autolimits)
                throw std::invalid_argument("joint range requires explicit limited when compiler autolimits=false");
            j.limited = has_range ? 1 : 0;
        }

        const char* stiff_str = jel->Attribute("stiffness");
        if (stiff_str) j.stiffness = strtof(stiff_str, nullptr);
        const char* damp_str = jel->Attribute("damping");
        if (damp_str) { j.damping = strtof(damp_str, nullptr); j.damping_d = strtod(damp_str, nullptr); }
        const char* arm_str = jel->Attribute("armature");
        if (arm_str) { j.armature = strtof(arm_str, nullptr); j.armature_d = strtod(arm_str, nullptr); }
        const char* fric_str = jel->Attribute("frictionloss");
        if (fric_str) j.frictionloss = strtof(fric_str, nullptr);
        // Default solref/solimp for limits (MuJoCo defaults)
        j.solref_lim[0] = 0.02f; j.solref_lim[1] = 1.0f;
        j.solimp_lim[0] = 0.9f; j.solimp_lim[1] = 0.95f; j.solimp_lim[2] = 0.001f;
        j.solimp_lim[3] = 0.5f; j.solimp_lim[4] = 2.0f;
        const char* sr_lim = jel->Attribute("solreflimit");
        if (sr_lim) parse_floats(sr_lim, j.solref_lim, 2);
        const char* si_lim = jel->Attribute("solimplimit");
        if (si_lim) parse_floats(si_lim, j.solimp_lim, 5);

        // Ensure axis_d is synced and normalized
        if (!ax_str) {
            // axis came from defaults (float only) — sync to double
            j.axis_d[0] = (double)j.axis[0]; j.axis_d[1] = (double)j.axis[1]; j.axis_d[2] = (double)j.axis[2];
        }
        {
            double nrm = sqrt(j.axis_d[0]*j.axis_d[0] + j.axis_d[1]*j.axis_d[1] + j.axis_d[2]*j.axis_d[2]);
            if (nrm > 1e-12 && fabs(nrm - 1.0) > 1e-12) {
                j.axis_d[0] /= nrm; j.axis_d[1] /= nrm; j.axis_d[2] /= nrm;
                j.axis[0] = (float)j.axis_d[0]; j.axis[1] = (float)j.axis_d[1]; j.axis[2] = (float)j.axis_d[2];
            }
        }

        b.jnt_num++;
    }

    // Parse <geom> children
    int gidx_start = desc.ngeom;
    for (XMLElement* gel = body_el->FirstChildElement("geom"); gel;
         gel = gel->NextSiblingElement("geom")) {
        if (desc.ngeom >= MAX_GEOMS)        // a dropped geom is silently missing physics
            throw std::runtime_error("PhysicsModel: more than MAX_GEOMS (" + std::to_string(MAX_GEOMS) + ") geoms");
        int gidx = desc.ngeom++;
        GeomDesc& g = desc.geoms[gidx];
        memset(&g, 0, sizeof(g));
        g.body = idx;
        g.quat[0] = 1.0f;
        g.contype = 1;
        g.conaffinity = 1;
        g.condim = 3;
        g.group = 0;
        g.friction[0] = 1.0f; g.friction[1] = 0.005f; g.friction[2] = 0.0001f;
        g.rgba[0] = 0.7f; g.rgba[1] = 0.7f; g.rgba[2] = 0.8f; g.rgba[3] = 1.0f;
        g.solref[0] = 0.02f; g.solref[1] = 1.0f;
        g.solimp[0] = 0.9f; g.solimp[1] = 0.95f; g.solimp[2] = 0.001f; g.solimp[3] = 0.5f; g.solimp[4] = 2.0f;
        safe_strncpy(g.name, gel->Attribute("name"), MAX_NAME_LEN);

        // Apply class defaults
        const char* gclass = gel->Attribute("class");
        const DefaultClass& gdc = gclass ? get_defaults(state, gclass) : body_dc;
        apply_geom_defaults(g, gdc.geom);

        g.mesh_id = -1;
        const char* group_str = gel->Attribute("group");
        if (group_str) g.group = (int)strtol(group_str, nullptr, 10);
        const char* type_str = gel->Attribute("type");
        if (type_str) {
            if (strcmp(type_str, "sphere") == 0)         g.type = GEOM_SPHERE;
            else if (strcmp(type_str, "box") == 0)       g.type = GEOM_BOX;
            else if (strcmp(type_str, "capsule") == 0)   g.type = GEOM_CAPSULE;
            else if (strcmp(type_str, "cylinder") == 0)  g.type = GEOM_CYLINDER;
            else if (strcmp(type_str, "plane") == 0)     g.type = GEOM_PLANE;
            else if (strcmp(type_str, "mesh") == 0) {
                const char* mesh_ref = gel->Attribute("mesh");
                if (mesh_ref) {
                    auto it = state.mesh_name_map.find(mesh_ref);
                    if (it != state.mesh_name_map.end()) {
                        g.type = GEOM_MESH;
                        g.mesh_id = it->second;
                    } else {
                        g.type = GEOM_BOX;
                        g.contype = 0; g.conaffinity = 0;
                    }
                } else {
                    g.type = GEOM_BOX;
                    g.contype = 0; g.conaffinity = 0;
                }
            }
            else g.type = GEOM_BOX;
        }

        parse_floats(gel->Attribute("pos"), g.pos, 3);
        const char* gquat = gel->Attribute("quat");
        if (gquat) parse_floats(gquat, g.quat, 4);
        const char* geuler = gel->Attribute("euler");
        if (geuler && !gquat) {
            // MuJoCo eulerseq semantics (see euler_to_quat).
            double euler_d[3] = {0, 0, 0};
            parse_doubles(geuler, euler_d, 3);
            if (!state.angle_radian) {
                euler_d[0] *= DEG2RAD; euler_d[1] *= DEG2RAD; euler_d[2] *= DEG2RAD;
            }
            double qd[4];
            euler_to_quat(euler_d, state.eulerseq, qd);
            for (int k = 0; k < 4; ++k) g.quat[k] = (float)qd[k];
        }
        const char* gaxisangle = gel->Attribute("axisangle");
        if (gaxisangle && !gquat && !geuler) {
            float aa[4] = {0, 0, 1, 0};
            parse_floats(gaxisangle, aa, 4);
            float ax = aa[0], ay = aa[1], az = aa[2];
            float angle = aa[3];
            if (!state.angle_radian) angle *= DEG2RAD;
            float n = sqrtf(ax*ax + ay*ay + az*az);
            if (n > 1e-8f) { ax /= n; ay /= n; az /= n; }
            float ha = angle * 0.5f;
            float sha = sinf(ha);
            g.quat[0] = cosf(ha);
            g.quat[1] = ax * sha;
            g.quat[2] = ay * sha;
            g.quat[3] = az * sha;
        }

        g.size[0] = 0.01f;
        parse_floats(gel->Attribute("size"), g.size, 3);

        // Handle fromto for capsule/cylinder: compute pos (midpoint),
        // quat (rotation aligning Z-axis with fromto direction), and
        // size[1] (half-length). Must run AFTER parse_floats(size) since
        // fromto overrides pos/quat/size[1] computed from the endpoints.
        const char* fromto_str = gel->Attribute("fromto");
        if (fromto_str && (g.type == GEOM_CAPSULE || g.type == GEOM_CYLINDER)) {
            float ft[6] = {0};
            parse_floats(fromto_str, ft, 6);
            g.pos[0] = 0.5f * (ft[0] + ft[3]);
            g.pos[1] = 0.5f * (ft[1] + ft[4]);
            g.pos[2] = 0.5f * (ft[2] + ft[5]);
            float dx = ft[3] - ft[0], dy = ft[4] - ft[1], dz = ft[5] - ft[2];
            float len = sqrtf(dx*dx + dy*dy + dz*dz);
            if (len > 1e-8f) {
                g.size[1] = len * 0.5f;
                float ax = dx / len, ay = dy / len, az = dz / len;
                float dot_z = az;
                if (dot_z > 0.9999f) {
                    g.quat[0] = 1; g.quat[1] = 0; g.quat[2] = 0; g.quat[3] = 0;
                } else if (dot_z < -0.9999f) {
                    g.quat[0] = 0; g.quat[1] = 1; g.quat[2] = 0; g.quat[3] = 0;
                } else {
                    float cx = -ay, cy = ax, cz = 0;
                    float cn = sqrtf(cx*cx + cy*cy + cz*cz);
                    cx /= cn; cy /= cn; cz /= cn;
                    float half_angle = acosf(fmaxf(-1.0f, fminf(1.0f, dot_z))) * 0.5f;
                    float sh = sinf(half_angle);
                    g.quat[0] = cosf(half_angle);
                    g.quat[1] = cx * sh;
                    g.quat[2] = cy * sh;
                    g.quat[3] = cz * sh;
                }
            }
        }
        const char* fr = gel->Attribute("friction");
        if (fr) parse_floats(fr, g.friction, 3);
        const char* rgba_str = gel->Attribute("rgba");
        if (rgba_str) parse_floats(rgba_str, g.rgba, 4);
        const char* ct = gel->Attribute("contype");
        if (ct) g.contype = (int)strtol(ct, nullptr, 10);
        const char* ca = gel->Attribute("conaffinity");
        if (ca) g.conaffinity = (int)strtol(ca, nullptr, 10);
        const char* cd = gel->Attribute("condim");
        if (cd) g.condim = (int)strtol(cd, nullptr, 10);
        const char* sr_str = gel->Attribute("solref");
        if (sr_str) parse_floats(sr_str, g.solref, 2);
        const char* si_str = gel->Attribute("solimp");
        if (si_str) parse_floats(si_str, g.solimp, 5);
        const char* margin_str = gel->Attribute("margin");
        if (margin_str) g.margin = strtof(margin_str, nullptr);
        const char* gap_str = gel->Attribute("gap");
        if (gap_str) g.gap = strtof(gap_str, nullptr);

        g.density = 1000.0f;
        const char* dens_str = gel->Attribute("density");
        if (dens_str) g.density = strtof(dens_str, nullptr);
        const char* gmass_str = gel->Attribute("mass");
        if (gmass_str) {
            float gmass = strtof(gmass_str, nullptr);
            if (gmass > 0) g.density = -gmass;  // negative signals explicit mass
        }

        // Resolve material properties for per-geom specular/shininess/emission.
        // MuJoCo convention: material rgba always overrides geom rgba for rendering.
        const char* mat_name = gel->Attribute("material");
        if (mat_name) {
            auto it = state.material_map.find(mat_name);
            if (it != state.material_map.end()) {
                desc.geom_specular[gidx] = it->second.specular;
                desc.geom_shininess[gidx] = it->second.shininess;
                desc.geom_emission[gidx] = it->second.emission;
                desc.geom_reflectance[gidx] = it->second.reflectance;
                state.geom_material[gidx] = mat_name;
                g.rgba[0] = it->second.rgba[0];
                g.rgba[1] = it->second.rgba[1];
                g.rgba[2] = it->second.rgba[2];
                g.rgba[3] = it->second.rgba[3];
            }
        }
    }

    // Auto-compute body mass/inertia from geoms if no <inertial> tag
    if (!inertial) {
        float total_mass = 0;
        float com[3] = {0, 0, 0};
        int geom_start = gidx_start;
        int geom_end = desc.ngeom;

        for (int gi = geom_start; gi < geom_end; gi++) {
            const GeomDesc& gg = desc.geoms[gi];
            if (gg.body != idx) continue;
            float vol = 0;
            const float* s = gg.size;
            switch (gg.type) {
                case GEOM_BOX:
                    vol = 8.0f * s[0] * s[1] * s[2];
                    break;
                case GEOM_SPHERE:
                    vol = (4.0f / 3.0f) * 3.14159265f * s[0] * s[0] * s[0];
                    break;
                case GEOM_CAPSULE: {
                    float r = s[0], h = s[1];
                    vol = 3.14159265f * r * r * 2.0f * h + (4.0f/3.0f) * 3.14159265f * r * r * r;
                    break;
                }
                default:
                    vol = 8.0f * s[0] * s[1] * s[2];
                    break;
            }
            float gm = (gg.density < 0) ? -gg.density : gg.density * vol;
            total_mass += gm;
            com[0] += gm * gg.pos[0];
            com[1] += gm * gg.pos[1];
            com[2] += gm * gg.pos[2];
        }

        if (total_mass > 1e-12f) {
            b.mass = total_mass;
            b.ipos[0] = com[0] / total_mass;
            b.ipos[1] = com[1] / total_mass;
            b.ipos[2] = com[2] / total_mass;

            // Full 3x3 symmetric inertia tensor: Ixx, Iyy, Izz, Ixy, Ixz, Iyz
            double I[6] = {0, 0, 0, 0, 0, 0};
            for (int gi = geom_start; gi < geom_end; gi++) {
                const GeomDesc& gg = desc.geoms[gi];
                if (gg.body != idx) continue;
                const float* s = gg.size;
                float vol = 0;
                double gI[3] = {0, 0, 0};  // geom-local diagonal inertia
                switch (gg.type) {
                    case GEOM_BOX:
                        vol = 8.0f * s[0] * s[1] * s[2];
                        { double gm = (gg.density < 0) ? (double)(-gg.density) : (double)(gg.density * vol);
                          double sx2 = s[0]*s[0], sy2 = s[1]*s[1], sz2 = s[2]*s[2];
                          gI[0] = gm / 3.0 * (sy2 + sz2);
                          gI[1] = gm / 3.0 * (sx2 + sz2);
                          gI[2] = gm / 3.0 * (sx2 + sy2);
                        }
                        break;
                    case GEOM_SPHERE:
                        vol = (4.0f/3.0f) * 3.14159265f * s[0]*s[0]*s[0];
                        { double gm = (gg.density < 0) ? (double)(-gg.density) : (double)(gg.density * vol);
                          gI[0] = gI[1] = gI[2] = 0.4 * gm * s[0] * s[0];
                        }
                        break;
                    case GEOM_CAPSULE: {
                        float r = s[0], h = s[1];
                        vol = 3.14159265f * r*r * 2.0f*h + (4.0f/3.0f)*3.14159265f*r*r*r;
                        double gm = (gg.density < 0) ? (double)(-gg.density) : (double)(gg.density * vol);
                        double mc = 3.14159265 * r*r * 2.0*h * ((gg.density<0)?((-gg.density)/vol):gg.density);
                        double ms = gm - mc;
                        gI[0] = gI[1] = mc*(3*r*r + 4*h*h)/12.0 + ms*(2*r*r/5.0 + h*h + 0.75*h*r);
                        gI[2] = mc*r*r/2.0 + ms*2*r*r/5.0;
                        break;
                    }
                    default: {
                        vol = 8.0f * s[0]*s[1]*s[2];
                        double gm = (gg.density < 0) ? (double)(-gg.density) : (double)(gg.density * vol);
                        double sx2 = s[0]*s[0], sy2 = s[1]*s[1], sz2 = s[2]*s[2];
                        gI[0] = gm/3.0*(sy2+sz2); gI[1] = gm/3.0*(sx2+sz2); gI[2] = gm/3.0*(sx2+sy2);
                        break;
                    }
                }
                double gm = (gg.density < 0) ? (double)(-gg.density) : (double)(gg.density * vol);

                // Rotate geom-local diagonal inertia by geom quaternion:
                // I_rotated = R * diag(gI) * R^T
                double qw = gg.quat[0], qx = gg.quat[1], qy = gg.quat[2], qz = gg.quat[3];
                // Rotation matrix from quaternion (rows of R)
                double R[3][3] = {
                    {1-2*(qy*qy+qz*qz), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)},
                    {2*(qx*qy+qw*qz), 1-2*(qx*qx+qz*qz), 2*(qy*qz-qw*qx)},
                    {2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx*qx+qy*qy)}
                };
                // I_rot[i][j] = sum_k R[i][k] * gI[k] * R[j][k]
                double Ir[3][3];
                for (int i = 0; i < 3; i++)
                    for (int j = 0; j < 3; j++)
                        Ir[i][j] = R[i][0]*gI[0]*R[j][0] + R[i][1]*gI[1]*R[j][1] + R[i][2]*gI[2]*R[j][2];

                // Parallel axis theorem (with off-diagonal terms)
                double dx = gg.pos[0] - b.ipos[0], dy = gg.pos[1] - b.ipos[1], dz = gg.pos[2] - b.ipos[2];
                I[0] += Ir[0][0] + gm*(dy*dy + dz*dz);  // Ixx
                I[1] += Ir[1][1] + gm*(dx*dx + dz*dz);  // Iyy
                I[2] += Ir[2][2] + gm*(dx*dx + dy*dy);  // Izz
                I[3] += Ir[0][1] - gm*dx*dy;             // Ixy
                I[4] += Ir[0][2] - gm*dx*dz;             // Ixz
                I[5] += Ir[1][2] - gm*dy*dz;             // Iyz
            }

            // Diagonalize full 3x3 inertia tensor to get principal moments + iquat
            double off_diag = fabs(I[3]) + fabs(I[4]) + fabs(I[5]);
            if (off_diag > 1e-15 * (fabs(I[0]) + fabs(I[1]) + fabs(I[2]))) {
                double A[3][3] = {
                    {I[0], I[3], I[4]},
                    {I[3], I[1], I[5]},
                    {I[4], I[5], I[2]}
                };
                double V[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
                for (int iter = 0; iter < 50; iter++) {
                    int p = 0, q = 1;
                    double maxval = fabs(A[0][1]);
                    if (fabs(A[0][2]) > maxval) { p=0; q=2; maxval=fabs(A[0][2]); }
                    if (fabs(A[1][2]) > maxval) { p=1; q=2; maxval=fabs(A[1][2]); }
                    if (maxval < 1e-15) break;
                    double app = A[p][p], aqq = A[q][q], apq = A[p][q];
                    double theta = 0.5 * atan2(2.0*apq, app - aqq);
                    double cs = cos(theta), sn = sin(theta);
                    double newA[3][3];
                    for (int i = 0; i < 3; i++)
                        for (int j = 0; j < 3; j++)
                            newA[i][j] = A[i][j];
                    for (int i = 0; i < 3; i++) {
                        newA[i][p] = cs*A[i][p] + sn*A[i][q];
                        newA[i][q] = -sn*A[i][p] + cs*A[i][q];
                    }
                    for (int j = 0; j < 3; j++) {
                        A[p][j] = cs*newA[p][j] + sn*newA[q][j];
                        A[q][j] = -sn*newA[p][j] + cs*newA[q][j];
                    }
                    for (int i = 0; i < 3; i++) {
                        if (i != p && i != q) {
                            A[i][p] = A[p][i];
                            A[i][q] = A[q][i];
                        }
                    }
                    A[p][q] = 0; A[q][p] = 0;
                    for (int i = 0; i < 3; i++) {
                        double vp = V[i][p], vq = V[i][q];
                        V[i][p] = cs*vp + sn*vq;
                        V[i][q] = -sn*vp + cs*vq;
                    }
                }
                b.inertia_d[0] = A[0][0];
                b.inertia_d[1] = A[1][1];
                b.inertia_d[2] = A[2][2];
                // Convert rotation matrix V to quaternion (iquat)
                double det = V[0][0]*(V[1][1]*V[2][2]-V[1][2]*V[2][1])
                           - V[0][1]*(V[1][0]*V[2][2]-V[1][2]*V[2][0])
                           + V[0][2]*(V[1][0]*V[2][1]-V[1][1]*V[2][0]);
                if (det < 0) {
                    for (int i = 0; i < 3; i++) V[i][2] = -V[i][2];
                }
                double tr = V[0][0] + V[1][1] + V[2][2];
                double qw, qx, qy, qz;
                if (tr > 0) {
                    double s2 = sqrt(tr + 1.0) * 2.0;
                    qw = 0.25 * s2;
                    qx = (V[2][1] - V[1][2]) / s2;
                    qy = (V[0][2] - V[2][0]) / s2;
                    qz = (V[1][0] - V[0][1]) / s2;
                } else if (V[0][0] > V[1][1] && V[0][0] > V[2][2]) {
                    double s2 = sqrt(1.0 + V[0][0] - V[1][1] - V[2][2]) * 2.0;
                    qw = (V[2][1] - V[1][2]) / s2;
                    qx = 0.25 * s2;
                    qy = (V[0][1] + V[1][0]) / s2;
                    qz = (V[0][2] + V[2][0]) / s2;
                } else if (V[1][1] > V[2][2]) {
                    double s2 = sqrt(1.0 + V[1][1] - V[0][0] - V[2][2]) * 2.0;
                    qw = (V[0][2] - V[2][0]) / s2;
                    qx = (V[0][1] + V[1][0]) / s2;
                    qy = 0.25 * s2;
                    qz = (V[1][2] + V[2][1]) / s2;
                } else {
                    double s2 = sqrt(1.0 + V[2][2] - V[0][0] - V[1][1]) * 2.0;
                    qw = (V[1][0] - V[0][1]) / s2;
                    qx = (V[0][2] + V[2][0]) / s2;
                    qy = (V[1][2] + V[2][1]) / s2;
                    qz = 0.25 * s2;
                }
                double qnorm = sqrt(qw*qw + qx*qx + qy*qy + qz*qz);
                if (qnorm > 1e-15) { qw/=qnorm; qx/=qnorm; qy/=qnorm; qz/=qnorm; }
                if (qw < 0) { qw=-qw; qx=-qx; qy=-qy; qz=-qz; }
                b.iquat_d[0] = qw; b.iquat_d[1] = qx;
                b.iquat_d[2] = qy; b.iquat_d[3] = qz;
                b.iquat[0] = (float)qw; b.iquat[1] = (float)qx;
                b.iquat[2] = (float)qy; b.iquat[3] = (float)qz;
            } else {
                b.inertia_d[0] = I[0];
                b.inertia_d[1] = I[1];
                b.inertia_d[2] = I[2];
            }
            b.inertia[0] = (float)b.inertia_d[0];
            b.inertia[1] = (float)b.inertia_d[1];
            b.inertia[2] = (float)b.inertia_d[2];

            b.mass_d = (double)b.mass;
            b.ipos_d[0] = (double)b.ipos[0];
            b.ipos_d[1] = (double)b.ipos[1];
            b.ipos_d[2] = (double)b.ipos[2];
        }
    }

    // Full/inferred inertia may replace iquat with a principal-axis rotation.
    // Preserve its eigenvector/sign convention, then derive the float mirror
    // from the final normalized double orientation.
    normalize_body_quaternion(b.iquat_d, b.iquat, body_el->Attribute("name"), "inertial quat");

    // Parse <camera> children
    for (XMLElement* cel = body_el->FirstChildElement("camera"); cel;
         cel = cel->NextSiblingElement("camera")) {
        if (desc.ncamera >= MAX_CAMERAS) break;
        int cidx = desc.ncamera++;
        CameraDesc& c = desc.cameras[cidx];
        memset(&c, 0, sizeof(c));
        c.body = idx;
        c.quat[0] = 1.0f;
        c.fovy = 45.0f * DEG2RAD;

        safe_strncpy(c.name, cel->Attribute("name"), MAX_NAME_LEN);
        parse_floats(cel->Attribute("pos"), c.pos, 3);
        const char* cquat = cel->Attribute("quat");
        if (cquat) {
            parse_floats(cquat, c.quat, 4);
        } else {
            const char* xyaxes_str = cel->Attribute("xyaxes");
            if (xyaxes_str) {
                float xy[6];
                parse_floats(xyaxes_str, xy, 6);
                xyaxes_to_quat(xy, c.quat);
            }
        }
        const char* fovy_str = cel->Attribute("fovy");
        if (fovy_str) {
            c.fovy = strtof(fovy_str, nullptr) * DEG2RAD;
        }
        parse_camera_extras(cel, c, state);
    }

    // Parse <site> children
    for (XMLElement* sel = body_el->FirstChildElement("site"); sel;
         sel = sel->NextSiblingElement("site")) {
        if (desc.nsite >= MAX_SITES) break;
        int sidx = desc.nsite++;
        SiteDesc& s = desc.sites[sidx];
        init_site_desc(s, idx);
        parse_site_fields(sel, s, state.angle_radian, state.eulerseq);
    }

    // Recurse into child <body> elements
    for (XMLElement* child = body_el->FirstChildElement("body"); child;
         child = child->NextSiblingElement("body")) {
        parse_body(child, idx, state);
    }
}

// ── Main loader ─────────────────────────────────────────────────────────────

PhysicsModel::PhysicsModel() : uploaded_(false) {
    memset(&desc_, 0, sizeof(desc_));
    desc_.gravity[0] = 0.0f;
    desc_.gravity[1] = 0.0f;
    desc_.gravity[2] = -9.81f;
    desc_.gravity_d[0] = 0.0;
    desc_.gravity_d[1] = 0.0;
    desc_.gravity_d[2] = -9.81;
    desc_.timestep = 0.002f;
}

PhysicsModel::~PhysicsModel() {
    free_device();
    delete[] desc_.mesh_vert;
    delete[] desc_.mesh_normal;
    delete[] desc_.mesh_texcoord;
    delete[] desc_.texture_data;
    delete[] desc_.convex_hull_vert;
    delete[] desc_.convex_hull_face;
    delete[] desc_.convex_hull_face_normal;
    delete[] desc_.convex_hull_edge;
    desc_.mesh_vert = nullptr;
    desc_.mesh_normal = nullptr;
    desc_.mesh_texcoord = nullptr;
    desc_.texture_data = nullptr;
    desc_.convex_hull_vert = nullptr;
    desc_.convex_hull_face = nullptr;
    desc_.convex_hull_face_normal = nullptr;
    desc_.convex_hull_edge = nullptr;
}

bool PhysicsModel::load_from_xml(const std::string& xml_string, const std::string& meshdir_override) {
    free_device();
    delete[] desc_.mesh_vert;
    delete[] desc_.mesh_normal;
    delete[] desc_.mesh_texcoord;
    delete[] desc_.texture_data;
    delete[] desc_.convex_hull_vert;
    delete[] desc_.convex_hull_face;
    delete[] desc_.convex_hull_face_normal;
    delete[] desc_.convex_hull_edge;
    memset(&desc_, 0, sizeof(desc_));
    desc_.gravity[0] = 0.0f;
    desc_.gravity[1] = 0.0f;
    desc_.gravity[2] = -9.81f;
    desc_.gravity_d[0] = 0.0;
    desc_.gravity_d[1] = 0.0;
    desc_.gravity_d[2] = -9.81;
    desc_.timestep = 0.002f;

    XMLDocument doc;
    if (doc.Parse(xml_string.c_str(), xml_string.size()) != XML_SUCCESS) {
        fprintf(stderr, "instinct_sim: failed to parse MJCF XML\n");
        return false;
    }

    XMLElement* mujoco = doc.FirstChildElement("mujoco");
    if (!mujoco) {
        fprintf(stderr, "instinct_sim: missing <mujoco> root element\n");
        return false;
    }

    ParserState state{desc_};

    // ── Parse <compiler> ────────────────────────────────────────────────
    XMLElement* compiler = mujoco->FirstChildElement("compiler");
    if (compiler) {
        const char* al = compiler->Attribute("autolimits");
        if (al) state.autolimits = strcmp(al, "false") != 0 && strcmp(al, "0") != 0;
        const char* angle = compiler->Attribute("angle");
        if (angle && strcmp(angle, "radian") == 0)
            state.angle_radian = true;
        const char* eseq = compiler->Attribute("eulerseq");
        if (eseq && strlen(eseq) == 3) state.eulerseq = eseq;
        const char* mdir = compiler->Attribute("meshdir");
        if (mdir) state.meshdir = mdir;
        const char* stm = compiler->Attribute("settotalmass");
        if (stm) state.settotalmass = strtod(stm, nullptr);
    }
    if (!meshdir_override.empty()) state.meshdir = meshdir_override;

    // ── Parse <asset><mesh> elements ────────────────────────────────────
    desc_.nmesh = 0;
    desc_.mesh_total_verts = 0;
    desc_.mesh_vert = nullptr;
    desc_.mesh_normal = nullptr;
    desc_.mesh_texcoord = nullptr;
    desc_.ntexture = 0;
    desc_.texture_data_size = 0;
    desc_.texture_data = nullptr;
    memset(desc_.geom_texid, -1, sizeof(desc_.geom_texid));

    // GPU extensions, spelled as MuJoCo <custom><numeric> so the file still loads in
    // MuJoCo (converted PhysX / SAPIEN scenes set both; every other model keeps the defaults):
    //   instinct_sim_friction_combine  pair friction: 0 = max (MuJoCo), 1 = mean (PhysX default)
    //   instinct_sim_max_depenetration_velocity  cap (m/s) on how fast a contact pushes a
    //                                penetration out (PhysX maxDepenetrationVelocity-like), 0 = off
    //   <text name="instinct_sim_onesided_geoms" data="g1 g2 ...">  static boxes that, like a
    //                                PhysX triangle mesh (back faces culled), give no contact to a
    //                                mesh hull whose center is below their top face
    //   instinct_sim_obj_texcoords     1 = use OBJ vt texcoords (else OBJ meshes get texgen, the
    //                                behaviour the LIBERO renderer was validated with)
    desc_.friction_combine = 0;
    desc_.max_depen_vel = 0.0f;
    bool obj_texcoords = false;
    std::vector<std::string> onesided_names;
    for (XMLElement* cu = mujoco->FirstChildElement("custom"); cu; cu = cu->NextSiblingElement("custom"))
        for (XMLElement* tx = cu->FirstChildElement("text"); tx; tx = tx->NextSiblingElement("text")) {
            const char* nm = tx->Attribute("name");
            const char* dt = tx->Attribute("data");
            if (nm && dt && strcmp(nm, "instinct_sim_onesided_geoms") == 0) {
                std::istringstream ss(dt);
                std::string w;
                while (ss >> w) onesided_names.push_back(w);
            }
        }
    for (XMLElement* cu = mujoco->FirstChildElement("custom"); cu; cu = cu->NextSiblingElement("custom"))
        for (XMLElement* nu = cu->FirstChildElement("numeric"); nu; nu = nu->NextSiblingElement("numeric")) {
            const char* nm = nu->Attribute("name");
            if (nm && strcmp(nm, "instinct_sim_friction_combine") == 0)
                desc_.friction_combine = (int)attr_double(nu, "data", 0.0);
            if (nm && strcmp(nm, "instinct_sim_max_depenetration_velocity") == 0)
                desc_.max_depen_vel = (float)attr_double(nu, "data", 0.0);
            if (nm && strcmp(nm, "instinct_sim_obj_texcoords") == 0)
                obj_texcoords = attr_double(nu, "data", 0.0) != 0.0;
        }

    XMLElement* asset = mujoco->FirstChildElement("asset");
    if (asset) {
        // First pass: load all meshes into a temporary buffer to compute total size
        struct TempMesh { MeshData data; std::string name; float scale[3]; };
        std::vector<TempMesh> temp_meshes;
        int total_verts = 0;

        for (XMLElement* mel = asset->FirstChildElement("mesh"); mel;
             mel = mel->NextSiblingElement("mesh")) {
            if ((int)temp_meshes.size() >= MAX_MESHES) break;
            const char* mname = mel->Attribute("name");
            const char* mfile = mel->Attribute("file");
            if (!mfile) continue;

            std::string fpath = mfile;
            if (!state.meshdir.empty() && fpath[0] != '/') {
                std::string dir = state.meshdir;
                if (!dir.empty() && dir.back() != '/') dir += '/';
                fpath = dir + fpath;
            }

            TempMesh tm;
            if (!load_mesh(fpath, tm.data)) {
                fprintf(stderr, "instinct_sim: failed to load mesh '%s'\n", fpath.c_str());
                continue;
            }
            if (!obj_texcoords && tm.data.has_texcoords && fpath.size() >= 4 &&
                strcasecmp(fpath.c_str() + fpath.size() - 4, ".obj") == 0) {
                tm.data.has_texcoords = false;      // see instinct_sim_obj_texcoords above
                tm.data.texcoords.clear();
            }

            // MuJoCo mjCMesh::ApplyTransformations, before scaling: vertices -= refpos, then
            // v <- R(refquat)^T v (normals too). robocasa's objaverse meshes all carry a refquat
            // (90 deg about z / axis cycling); ignoring it rotated their hulls and AABBs, so the
            // broadphase culled touching pairs (objects sank through pans).
            {
                double rp[3] = {0, 0, 0}, rq[4] = {1, 0, 0, 0};
                if (mel->Attribute("refpos")) parse_doubles(mel->Attribute("refpos"), rp, 3);
                if (mel->Attribute("refquat")) parse_doubles(mel->Attribute("refquat"), rq, 4);
                const bool has_p = rp[0] != 0 || rp[1] != 0 || rp[2] != 0;
                const bool has_q = rq[0] != 1 || rq[1] != 0 || rq[2] != 0 || rq[3] != 0;
                if (has_p || has_q) {
                    const double qn = sqrt(rq[0]*rq[0] + rq[1]*rq[1] + rq[2]*rq[2] + rq[3]*rq[3]);
                    const double w = rq[0]/qn, x = rq[1]/qn, y = rq[2]/qn, z = rq[3]/qn;
                    const double R[9] = {1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
                                         2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
                                         2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)};
                    auto rotT = [&](float* v) {   // v <- R^T v
                        const double a = v[0], b = v[1], c = v[2];
                        v[0] = (float)(R[0]*a + R[3]*b + R[6]*c);
                        v[1] = (float)(R[1]*a + R[4]*b + R[7]*c);
                        v[2] = (float)(R[2]*a + R[5]*b + R[8]*c);
                    };
                    for (int i = 0; i < tm.data.nvertices; i++) {
                        float* v = &tm.data.vertices[i*3];
                        v[0] = (float)(v[0] - rp[0]); v[1] = (float)(v[1] - rp[1]); v[2] = (float)(v[2] - rp[2]);
                        if (has_q) rotT(v);
                    }
                    if (has_q && (int)tm.data.normals.size() >= tm.data.nvertices * 3)
                        for (int i = 0; i < tm.data.nvertices; i++) rotT(&tm.data.normals[i*3]);
                }
            }

            tm.scale[0] = tm.scale[1] = tm.scale[2] = 1.0f;
            const char* scale_str = mel->Attribute("scale");
            if (scale_str) parse_floats(scale_str, tm.scale, 3);
            if (tm.scale[0] != 1 || tm.scale[1] != 1 || tm.scale[2] != 1) {
                fprintf(stderr, "[MESH] scaling '%s' by (%.4f, %.4f, %.4f) nverts=%d\n",
                        fpath.c_str(), tm.scale[0], tm.scale[1], tm.scale[2], tm.data.nvertices);
                scale_mesh(tm.data, tm.scale[0], tm.scale[1], tm.scale[2]);
            }

            if (!tm.data.has_texcoords && tm.data.nvertices > 0) {
                tm.data.has_texcoords = true;
                tm.data.generated_texcoords = true;   // scale applied after geoms resolve
                tm.data.texcoords.resize(tm.data.nvertices * MESH_UV_COMPONENTS);
                for (int i = 0; i < tm.data.nvertices; i++) {
                    tm.data.texcoords[i*MESH_UV_COMPONENTS+0] = tm.data.vertices[i*3+0];
                    tm.data.texcoords[i*MESH_UV_COMPONENTS+1] = tm.data.vertices[i*3+1];
                }
            }

            tm.name = mname ? mname : mfile;
            total_verts += tm.data.nvertices;

            if (total_verts > MAX_MESH_TOTAL_VERTS) {
                fprintf(stderr, "instinct_sim: mesh vertex limit exceeded (%d > %d), skipping '%s'\n",
                        total_verts, MAX_MESH_TOTAL_VERTS, fpath.c_str());
                total_verts -= tm.data.nvertices;
                continue;
            }

            temp_meshes.push_back(std::move(tm));
        }

        if (!temp_meshes.empty()) {
            desc_.mesh_vert = new float[total_verts * 3];
            desc_.mesh_normal = new float[total_verts * 3];
            desc_.mesh_texcoord = new float[total_verts * MESH_UV_COMPONENTS];
            memset(desc_.mesh_texcoord, 0, total_verts * MESH_UV_COMPONENTS * sizeof(float));
            int offset = 0;

            for (auto& tm : temp_meshes) {
                int idx = desc_.nmesh++;
                auto& mi = desc_.meshes[idx];
                mi.vert_adr = offset;
                mi.vert_num = tm.data.nvertices;
                mi.scale[0] = tm.scale[0]; mi.scale[1] = tm.scale[1]; mi.scale[2] = tm.scale[2];
                mi.generated_texcoords = tm.data.generated_texcoords;
                safe_strncpy(mi.name, tm.name.c_str(), MAX_NAME_LEN);

                memcpy(&desc_.mesh_vert[offset * 3],
                       tm.data.vertices.data(), tm.data.nvertices * 3 * sizeof(float));
                memcpy(&desc_.mesh_normal[offset * 3],
                       tm.data.normals.data(), tm.data.nvertices * 3 * sizeof(float));
                if (tm.data.has_texcoords) {
                    memcpy(&desc_.mesh_texcoord[offset * MESH_UV_COMPONENTS],
                           tm.data.texcoords.data(), tm.data.nvertices * MESH_UV_COMPONENTS * sizeof(float));
                }

                offset += tm.data.nvertices;
                state.mesh_name_map[tm.name] = idx;
            }
            desc_.mesh_total_verts = offset;
        }
    }
    if (desc_.nmesh > 0) {
        fprintf(stderr, "instinct_sim: loaded %d meshes (%d total vertices)\n",
                desc_.nmesh, desc_.mesh_total_verts);
    }

    // Parse <asset><material> elements for per-geom specular/shininess/emission
    if (asset) {
        for (XMLElement* mel = asset->FirstChildElement("material"); mel;
             mel = mel->NextSiblingElement("material")) {
            const char* mname = mel->Attribute("name");
            if (!mname) continue;
            ParserState::MatInfo mi;
            mi.specular = 0.5f;
            mi.shininess = 0.5f;
            mi.emission = 0.0f;
            mi.reflectance = 0.0f;   // MuJoCo default: 0 disables reflection
            mi.rgba[0] = mi.rgba[1] = mi.rgba[2] = mi.rgba[3] = 1.0f;
            mi.texrepeat[0] = mi.texrepeat[1] = 1.0f;   // MuJoCo defaults
            mi.texuniform = 0;
            const char* s;
            s = mel->Attribute("specular");  if (s) mi.specular = strtof(s, nullptr);
            s = mel->Attribute("shininess"); if (s) mi.shininess = strtof(s, nullptr);
            s = mel->Attribute("emission");  if (s) mi.emission = strtof(s, nullptr);
            s = mel->Attribute("reflectance"); if (s) mi.reflectance = strtof(s, nullptr);
            s = mel->Attribute("rgba");      if (s) parse_floats(s, mi.rgba, 4);
            s = mel->Attribute("texrepeat"); if (s) parse_floats(s, mi.texrepeat, 2);
            s = mel->Attribute("texuniform");
            if (s) mi.texuniform = (strcmp(s, "true") == 0 || strcmp(s, "1") == 0) ? 1 : 0;
            state.material_map[mname] = mi;
        }
        if (!state.material_map.empty()) {
            fprintf(stderr, "instinct_sim: parsed %d materials\n", (int)state.material_map.size());
        }
    }

    // Initialize per-geom material arrays to MuJoCo defaults
    for (int i = 0; i < MAX_GEOMS; i++) {
        desc_.geom_specular[i] = 0.5f;
        desc_.geom_shininess[i] = 0.5f;
        desc_.geom_emission[i] = 0.0f;
        desc_.geom_reflectance[i] = 0.0f;
    }

    // ── Parse <default> ─────────────────────────────────────────────────
    XMLElement* def = mujoco->FirstChildElement("default");
    if (def) {
        parse_defaults_recursive(def, state, DefaultClass{});
    }

    // ── Parse <option> ──────────────────────────────────────────────────
    desc_.impratio = 1.0f;
    desc_.cone_type = 1;  // default: elliptic (matches MuJoCo default for condim>=3)
    desc_.opt_density = 0.0f;
    desc_.opt_viscosity = 0.0f;
    desc_.o_margin = 0.0f;
    XMLElement* option = mujoco->FirstChildElement("option");
    if (option) {
        desc_.timestep = attr_float(option, "timestep", desc_.timestep);
        desc_.impratio = attr_float(option, "impratio", desc_.impratio);
        desc_.opt_density = attr_float(option, "density", desc_.opt_density);
        desc_.opt_viscosity = attr_float(option, "viscosity", desc_.opt_viscosity);
        desc_.o_margin = attr_float(option, "o_margin", desc_.o_margin);
        const char* cone_str = option->Attribute("cone");
        if (cone_str) {
            if (std::strcmp(cone_str, "pyramidal") == 0) desc_.cone_type = 0;
            else if (std::strcmp(cone_str, "elliptic") == 0) desc_.cone_type = 1;
        }
        const char* grav = option->Attribute("gravity");
        if (grav) { parse_floats(grav, desc_.gravity, 3); parse_doubles(grav, desc_.gravity_d, 3); }
    }

    // ── Parse <visual><headlight> ──────────────────────────────────────
    // MuJoCo defaults
    desc_.headlight_ambient[0] = desc_.headlight_ambient[1] = desc_.headlight_ambient[2] = 0.1f;
    desc_.headlight_diffuse[0] = desc_.headlight_diffuse[1] = desc_.headlight_diffuse[2] = 0.4f;
    desc_.headlight_specular[0] = desc_.headlight_specular[1] = desc_.headlight_specular[2] = 0.5f;
    desc_.headlight_active = 1;
    desc_.map_znear = 0.01f;
    desc_.map_zfar  = 50.0f;
    XMLElement* visual = mujoco->FirstChildElement("visual");
    if (visual) {
        XMLElement* headlight = visual->FirstChildElement("headlight");
        if (headlight) {
            parse_floats(headlight->Attribute("ambient"), desc_.headlight_ambient, 3);
            parse_floats(headlight->Attribute("diffuse"), desc_.headlight_diffuse, 3);
            parse_floats(headlight->Attribute("specular"), desc_.headlight_specular, 3);
            const char* act = headlight->Attribute("active");
            if (act && std::strcmp(act, "0") == 0) desc_.headlight_active = 0;
        }
        XMLElement* map_el = visual->FirstChildElement("map");
        if (map_el) {
            const char* zn = map_el->Attribute("znear");
            if (zn) desc_.map_znear = strtof(zn, nullptr);
            const char* zf = map_el->Attribute("zfar");
            if (zf) desc_.map_zfar = strtof(zf, nullptr);
        }
    }

    // ── Parse <worldbody> ───────────────────────────────────────────────
    XMLElement* worldbody = mujoco->FirstChildElement("worldbody");
    if (!worldbody) {
        fprintf(stderr, "instinct_sim: missing <worldbody>\n");
        return false;
    }

    desc_.nbody = 1;
    BodyDesc& world = desc_.bodies[0];
    memset(&world, 0, sizeof(world));
    world.parent = -1;
    world.quat[0] = 1.0f;
    safe_strncpy(world.name, "world", MAX_NAME_LEN);
    state.body_name_map["world"] = 0;

    // Parse worldbody geoms
    for (XMLElement* gel = worldbody->FirstChildElement("geom"); gel;
         gel = gel->NextSiblingElement("geom")) {
        if (desc_.ngeom >= MAX_GEOMS)        // a dropped geom is silently missing physics
            throw std::runtime_error("PhysicsModel: more than MAX_GEOMS (" + std::to_string(MAX_GEOMS) + ") geoms");
        int gidx = desc_.ngeom++;
        GeomDesc& g = desc_.geoms[gidx];
        memset(&g, 0, sizeof(g));
        g.body = 0;
        g.quat[0] = 1.0f;
        g.contype = 1;
        g.conaffinity = 1;
        g.condim = 3;
        g.group = 0;
        g.friction[0] = 1.0f; g.friction[1] = 0.005f; g.friction[2] = 0.0001f;
        g.rgba[0] = 0.7f; g.rgba[1] = 0.7f; g.rgba[2] = 0.8f; g.rgba[3] = 1.0f;
        g.solref[0] = 0.02f; g.solref[1] = 1.0f;
        g.solimp[0] = 0.9f; g.solimp[1] = 0.95f; g.solimp[2] = 0.001f; g.solimp[3] = 0.5f; g.solimp[4] = 2.0f;
        safe_strncpy(g.name, gel->Attribute("name"), MAX_NAME_LEN);

        g.mesh_id = -1;
        const char* group_str = gel->Attribute("group");
        if (group_str) g.group = (int)strtol(group_str, nullptr, 10);
        apply_geom_defaults(g, state.global_defaults.geom);

        const char* type_str = gel->Attribute("type");
        if (!type_str || strcmp(type_str, "plane") == 0) {
            g.type = GEOM_PLANE;
        } else if (strcmp(type_str, "box") == 0) {
            g.type = GEOM_BOX;
        } else if (strcmp(type_str, "sphere") == 0) {
            g.type = GEOM_SPHERE;
        } else if (strcmp(type_str, "capsule") == 0) {
            g.type = GEOM_CAPSULE;
        } else if (strcmp(type_str, "cylinder") == 0) {
            g.type = GEOM_CYLINDER;
        } else if (strcmp(type_str, "mesh") == 0) {
            const char* mesh_ref = gel->Attribute("mesh");
            if (mesh_ref) {
                auto it = state.mesh_name_map.find(mesh_ref);
                if (it != state.mesh_name_map.end()) {
                    g.type = GEOM_MESH;
                    g.mesh_id = it->second;
                } else {
                    g.type = GEOM_BOX;
                    g.contype = 0; g.conaffinity = 0;
                }
            } else {
                g.type = GEOM_BOX;
            }
        } else {
            g.type = GEOM_BOX;
        }

        parse_floats(gel->Attribute("pos"), g.pos, 3);
        const char* gquat = gel->Attribute("quat");
        if (gquat) parse_floats(gquat, g.quat, 4);
        const char* geuler = gel->Attribute("euler");
        if (geuler && !gquat) {
            double e[3] = {0, 0, 0};
            parse_doubles(geuler, e, 3);
            if (!state.angle_radian) { e[0] *= DEG2RAD; e[1] *= DEG2RAD; e[2] *= DEG2RAD; }
            double qd[4];
            euler_to_quat(e, state.eulerseq, qd);
            for (int k = 0; k < 4; ++k) g.quat[k] = (float)qd[k];
        }
        const char* gaxisangle = gel->Attribute("axisangle");
        if (gaxisangle && !gquat && !geuler) {
            float aa[4] = {0, 0, 1, 0};
            parse_floats(gaxisangle, aa, 4);
            float ax = aa[0], ay = aa[1], az = aa[2], angle = aa[3];
            if (!state.angle_radian) angle *= DEG2RAD;
            const float n = sqrtf(ax*ax + ay*ay + az*az);
            if (n > 1e-8f) { ax /= n; ay /= n; az /= n; }
            const float ha = angle * 0.5f, sha = sinf(ha);
            g.quat[0] = cosf(ha); g.quat[1] = ax * sha; g.quat[2] = ay * sha; g.quat[3] = az * sha;
        }
        parse_floats(gel->Attribute("size"), g.size, 3);
        const char* fromto_str2 = gel->Attribute("fromto");
        if (fromto_str2 && (g.type == GEOM_CAPSULE || g.type == GEOM_CYLINDER)) {
            float ft[6] = {0};
            parse_floats(fromto_str2, ft, 6);
            g.pos[0] = 0.5f*(ft[0]+ft[3]); g.pos[1] = 0.5f*(ft[1]+ft[4]); g.pos[2] = 0.5f*(ft[2]+ft[5]);
            float dx=ft[3]-ft[0], dy=ft[4]-ft[1], dz=ft[5]-ft[2];
            float len = sqrtf(dx*dx+dy*dy+dz*dz);
            if (len > 1e-8f) {
                g.size[1] = len * 0.5f;
                float az_ = dz/len;
                if (az_ > 0.9999f) { g.quat[0]=1; g.quat[1]=0; g.quat[2]=0; g.quat[3]=0; }
                else if (az_ < -0.9999f) { g.quat[0]=0; g.quat[1]=1; g.quat[2]=0; g.quat[3]=0; }
                else {
                    float cx=-dy/len, cy=dx/len;
                    float cn=sqrtf(cx*cx+cy*cy);
                    cx/=cn; cy/=cn;
                    float ha=acosf(fmaxf(-1.0f,fminf(1.0f,az_)))*0.5f, sh=sinf(ha);
                    g.quat[0]=cosf(ha); g.quat[1]=cx*sh; g.quat[2]=cy*sh; g.quat[3]=0;
                }
            }
        }
        const char* fr = gel->Attribute("friction");
        if (fr) parse_floats(fr, g.friction, 3);
        const char* rgba_str = gel->Attribute("rgba");
        if (rgba_str) parse_floats(rgba_str, g.rgba, 4);
        const char* ct = gel->Attribute("contype");
        if (ct) g.contype = (int)strtol(ct, nullptr, 10);
        const char* ca = gel->Attribute("conaffinity");
        if (ca) g.conaffinity = (int)strtol(ca, nullptr, 10);
        const char* cd = gel->Attribute("condim");
        if (cd) g.condim = (int)strtol(cd, nullptr, 10);
        const char* sr_str = gel->Attribute("solref");
        if (sr_str) parse_floats(sr_str, g.solref, 2);
        const char* si_str = gel->Attribute("solimp");
        if (si_str) parse_floats(si_str, g.solimp, 5);
        const char* margin_str = gel->Attribute("margin");
        if (margin_str) g.margin = strtof(margin_str, nullptr);
        const char* gap_str = gel->Attribute("gap");
        if (gap_str) g.gap = strtof(gap_str, nullptr);

        g.density = 1000.0f;
        const char* dens_str = gel->Attribute("density");
        if (dens_str) g.density = strtof(dens_str, nullptr);
        const char* gmass_str = gel->Attribute("mass");
        if (gmass_str) {
            float gmass = strtof(gmass_str, nullptr);
            if (gmass > 0) g.density = -gmass;
        }

        // Resolve material properties
        const char* mat_name = gel->Attribute("material");
        if (mat_name) {
            auto it = state.material_map.find(mat_name);
            if (it != state.material_map.end()) {
                desc_.geom_specular[gidx] = it->second.specular;
                desc_.geom_shininess[gidx] = it->second.shininess;
                desc_.geom_emission[gidx] = it->second.emission;
                desc_.geom_reflectance[gidx] = it->second.reflectance;
                state.geom_material[gidx] = mat_name;
                g.rgba[0] = it->second.rgba[0];
                g.rgba[1] = it->second.rgba[1];
                g.rgba[2] = it->second.rgba[2];
                g.rgba[3] = it->second.rgba[3];
            }
        }
    }

    // Parse worldbody lights
    desc_.nscene_light = 0;
    for (XMLElement* lel = worldbody->FirstChildElement("light"); lel;
         lel = lel->NextSiblingElement("light")) {
        if (desc_.nscene_light >= MAX_SCENE_LIGHTS) break;
        int lidx = desc_.nscene_light++;
        SceneLightDesc& sl = desc_.scene_lights[lidx];
        memset(&sl, 0, sizeof(sl));
        sl.diffuse[0] = sl.diffuse[1] = sl.diffuse[2] = 0.7f;
        sl.specular[0] = sl.specular[1] = sl.specular[2] = 0.3f;
        sl.dir[2] = -1.0f;
        sl.cutoff = 45.0f;
        sl.exponent = 10.0f;
        sl.attenuation[0] = 1.0f;
        safe_strncpy(sl.name, lel->Attribute("name"), MAX_NAME_LEN);
        parse_floats(lel->Attribute("pos"), sl.pos, 3);
        parse_floats(lel->Attribute("dir"), sl.dir, 3);
        parse_floats(lel->Attribute("diffuse"), sl.diffuse, 3);
        parse_floats(lel->Attribute("specular"), sl.specular, 3);
        parse_floats(lel->Attribute("ambient"), sl.ambient, 3);
        const char* cut = lel->Attribute("cutoff");
        if (cut) sl.cutoff = strtof(cut, nullptr);
        const char* exp_str = lel->Attribute("exponent");
        if (exp_str) sl.exponent = strtof(exp_str, nullptr);
        parse_floats(lel->Attribute("attenuation"), sl.attenuation, 3);
        // MJCF spells this two ways and LIBERO uses BOTH. The legacy form is
        // directional="true"; current MuJoCo uses type="directional" (mjLIGHT_
        // DIRECTIONAL, which shows up as mjModel.light_type == 1). Accepting
        // only the legacy attribute silently demoted every type="directional"
        // light to a POINT light with distance attenuation.
        //
        // Concretely: libero_10 task 8 (KITCHEN_SCENE8, "put both moka pots on
        // the stove") declares <light name="light_kitchen" type="directional"
        // pos="-1.75 0 1.6" .../>. MuJoCo reports light_type=[1,0,0]; the GPU simulator
        // parsed [0,0,0]. The scene rendered ~20 luminance levels too dark,
        // achromatic and concentrated in flat regions (agentview MAE 21.1 vs
        // ~5.0 on scenes with no directional light), and the task scored 0/2
        // against MuJoCo's 2/2.
        const char* dirattr = lel->Attribute("directional");
        if (dirattr && std::strcmp(dirattr, "true") == 0) sl.directional = 1;
        const char* typeattr = lel->Attribute("type");
        if (typeattr && std::strcmp(typeattr, "directional") == 0) sl.directional = 1;
    }

    // Parse worldbody cameras
    for (XMLElement* cel = worldbody->FirstChildElement("camera"); cel;
         cel = cel->NextSiblingElement("camera")) {
        if (desc_.ncamera >= MAX_CAMERAS) break;
        int cidx = desc_.ncamera++;
        CameraDesc& c = desc_.cameras[cidx];
        memset(&c, 0, sizeof(c));
        c.body = 0;
        c.quat[0] = 1.0f;
        c.fovy = 45.0f * DEG2RAD;
        safe_strncpy(c.name, cel->Attribute("name"), MAX_NAME_LEN);
        parse_floats(cel->Attribute("pos"), c.pos, 3);
        const char* cquat = cel->Attribute("quat");
        if (cquat) {
            parse_floats(cquat, c.quat, 4);
        } else {
            const char* xyaxes_str = cel->Attribute("xyaxes");
            if (xyaxes_str) {
                float xy[6];
                parse_floats(xyaxes_str, xy, 6);
                xyaxes_to_quat(xy, c.quat);
            }
        }
        const char* fovy_str = cel->Attribute("fovy");
        if (fovy_str) {
            c.fovy = strtof(fovy_str, nullptr) * DEG2RAD;
        }
        parse_camera_extras(cel, c, state);
    }

    // Parse worldbody sites
    for (XMLElement* sel = worldbody->FirstChildElement("site"); sel;
         sel = sel->NextSiblingElement("site")) {
        if (desc_.nsite >= MAX_SITES) break;
        int sidx = desc_.nsite++;
        SiteDesc& s = desc_.sites[sidx];
        init_site_desc(s, 0);
        parse_site_fields(sel, s, state.angle_radian, state.eulerseq);
    }

    // Parse child <body> elements
    try {
        for (XMLElement* child = worldbody->FirstChildElement("body"); child;
             child = child->NextSiblingElement("body")) {
            parse_body(child, 0, state);
        }
    } catch (const std::invalid_argument& error) {
        // Preserve load_from_xml's false/error-report convention and reject
        // invalid orientations before any device upload is attempted.
        fprintf(stderr, "%s\n", error.what());
        return false;
    }

    // ── Deferred texgen for meshes that carried no texcoords ────────────────
    // MuJoCo's generated UVs are  u = 0.5*scl0*lx - 0.5,  v = -0.5*scl1*ly - 0.5
    // with scl_k = texuniform ? texrepeat_k : texrepeat_k/size_k. size is
    // geom_size, so this cannot run at mesh-parse time -- geoms do not exist
    // yet. The raw local x,y stashed there are rescaled in place here.
    //
    // The cache is per MESH but the scale is per GEOM. Across all 40 LIBERO
    // tasks every texcoord-less mesh has exactly one (size, texrepeat,
    // texuniform) combination, so this is exact; the warning below fires if a
    // future asset breaks that assumption, in which case the scale has to move
    // into the shader rather than be baked here.
    if (!(getenv("HIPGL_MESH_TEXGEN") && strcmp(getenv("HIPGL_MESH_TEXGEN"), "0") == 0)) {
        std::vector<int> texgen_geom(desc_.nmesh, -1);
        for (int i = 0; i < desc_.ngeom; i++) {
            const GeomDesc& g = desc_.geoms[i];
            if (g.type != GEOM_MESH || g.mesh_id < 0 || g.mesh_id >= desc_.nmesh) continue;
            if (!desc_.meshes[g.mesh_id].generated_texcoords) continue;
            const int prev = texgen_geom[g.mesh_id];
            if (prev < 0) { texgen_geom[g.mesh_id] = i; continue; }
            const GeomDesc& p = desc_.geoms[prev];
            const bool same =
                fabsf(p.size[0] - g.size[0]) < 1e-9f &&
                fabsf(p.size[1] - g.size[1]) < 1e-9f &&
                state.geom_material[prev] == state.geom_material[i];
            if (!same) {
                fprintf(stderr,
                        "instinct_sim: WARNING mesh '%s' needs generated UVs but is shared by "
                        "geoms with different size/texrepeat ('%s' vs '%s'); baking the first. "
                        "Per-geom texgen would need to move into the shader.\n",
                        desc_.meshes[g.mesh_id].name, p.name, g.name);
            }
        }
        int ngen = 0;
        for (int mid = 0; mid < desc_.nmesh; mid++) {
            const int gi = texgen_geom[mid];
            if (gi < 0) continue;                    // unused mesh: leave raw
            const GeomDesc& g = desc_.geoms[gi];
            // texrepeat/texuniform come from the geom's MATERIAL. Note
            // desc_.geom_texrepeat is NOT usable here: this parser never fills
            // it (Python's set_textures does, later), so it would read as the
            // default 1.0 and silently drop any texrepeat != 1 -- which is
            // exactly the microwave case (texrepeat 3).
            float rep0 = 1.0f, rep1 = 1.0f;
            int uniform = 0;
            {
                auto gm = state.geom_material.find(gi);
                if (gm != state.geom_material.end()) {
                    auto mt = state.material_map.find(gm->second);
                    if (mt != state.material_map.end()) {
                        rep0 = mt->second.texrepeat[0];
                        rep1 = mt->second.texrepeat[1];
                        uniform = mt->second.texuniform;
                    }
                }
            }
            float scl0 = rep0, scl1 = rep1;
            if (!uniform) {
                if (g.size[0] > 0.0f) scl0 = rep0 / g.size[0];
                if (g.size[1] > 0.0f) scl1 = rep1 / g.size[1];
            }
            const auto& mi = desc_.meshes[mid];
            for (int v = 0; v < mi.vert_num; v++) {
                const int o = (mi.vert_adr + v) * MESH_UV_COMPONENTS;
                const float lx = desc_.mesh_texcoord[o+0];   // raw local x
                const float ly = desc_.mesh_texcoord[o+1];   // raw local y
                desc_.mesh_texcoord[o+0] =  0.5f * scl0 * lx - 0.5f;
                desc_.mesh_texcoord[o+1] = -0.5f * scl1 * ly - 0.5f;
            }
            ngen++;
        }
        if (ngen > 0) {
            fprintf(stderr, "instinct_sim: applied MuJoCo texgen to %d mesh(es) without texcoords\n",
                    ngen);
        }
    }


    // ── Apply settotalmass rescaling ─────────────────────────────────────
    if (state.settotalmass > 0) {
        double raw_total = 0;
        for (int i = 1; i < desc_.nbody; i++)
            raw_total += (double)desc_.bodies[i].mass;
        if (raw_total > 1e-12) {
            double scale = state.settotalmass / raw_total;
            for (int i = 1; i < desc_.nbody; i++) {
                auto& b = desc_.bodies[i];
                b.mass = (float)(b.mass * scale);
                b.mass_d = (double)b.mass;
                b.inertia[0] = (float)(b.inertia[0] * scale);
                b.inertia[1] = (float)(b.inertia[1] * scale);
                b.inertia[2] = (float)(b.inertia[2] * scale);
                b.inertia_d[0] = (double)b.inertia[0];
                b.inertia_d[1] = (double)b.inertia[1];
                b.inertia_d[2] = (double)b.inertia[2];
            }
        }
    }

    for (const std::string& n : onesided_names) {
        bool found = false;
        for (auto& g : desc_.geoms)
            if (n == g.name) { g.onesided = (g.type == GEOM_BOX) ? 1 : 0; found = true; }
        if (!found) fprintf(stderr, "instinct_sim: instinct_sim_onesided_geoms: no geom named %s\n", n.c_str());
    }

    // ── Compute stat_extent (matching MuJoCo's setStat algorithm) ──
    // MuJoCo: bbox of body positions + joint anchors + sites + geom positions
    //   - bodies (skip world body 0): xpos as point (radius=0)
    //   - geoms: rbound for regular geoms, 0.1*max(size[0],size[1]) for planes
    //   - extent = max axis span of the bounding box
    //   - extent = max(extent, 2*meansize) as a floor
    {
        float bmin[3] = {1e10f, 1e10f, 1e10f};
        float bmax[3] = {-1e10f, -1e10f, -1e10f};

        // Include body positions (skip world body 0)
        for (int i = 1; i < desc_.nbody; i++) {
            float* p = desc_.bodies[i].pos;
            for (int k = 0; k < 3; k++) {
                if (p[k] < bmin[k]) bmin[k] = p[k];
                if (p[k] > bmax[k]) bmax[k] = p[k];
            }
        }

        // Include geoms with appropriate rbound
        float meansize_sum = 0.0f;
        int meansize_count = 0;
        for (int i = 0; i < desc_.ngeom; i++) {
            const GeomDesc& g = desc_.geoms[i];
            float rbound = 0.0f;
            if (g.type == GEOM_PLANE) {
                // Planes: use 0.1 * max(size[0], size[1])
                float s0 = g.size[0], s1 = g.size[1];
                if (s0 > 0 || s1 > 0) {
                    rbound = (s0 > s1 ? s0 : s1) * 0.1f;
                } else {
                    rbound = 1.0f;  // infinite plane
                }
            } else {
                // Regular geoms: rbound = circumscribing sphere radius
                switch (g.type) {
                    case GEOM_SPHERE:  rbound = g.size[0]; break;
                    case GEOM_BOX:     rbound = sqrtf(g.size[0]*g.size[0]+g.size[1]*g.size[1]+g.size[2]*g.size[2]); break;
                    case GEOM_CAPSULE:  rbound = g.size[0] + g.size[1]; break;
                    case GEOM_CYLINDER: rbound = sqrtf(g.size[0]*g.size[0] + g.size[1]*g.size[1]); break;
                    default:            rbound = g.size[0]; break;
                }
            }

            // Accumulate meansize from non-plane geom rbounds
            if (g.type != GEOM_PLANE && rbound > 0) {
                meansize_sum += rbound;
                meansize_count++;
            }

            // Approximate world position
            int bi = g.body;
            float wx = desc_.bodies[bi].pos[0] + g.pos[0];
            float wy = desc_.bodies[bi].pos[1] + g.pos[1];
            float wz = desc_.bodies[bi].pos[2] + g.pos[2];
            float pos[3] = {wx, wy, wz};
            for (int k = 0; k < 3; k++) {
                if (pos[k] - rbound < bmin[k]) bmin[k] = pos[k] - rbound;
                if (pos[k] + rbound > bmax[k]) bmax[k] = pos[k] + rbound;
            }
        }

        if (bmax[0] > bmin[0]) {
            desc_.stat_center[0] = 0.5f*(bmin[0]+bmax[0]);
            desc_.stat_center[1] = 0.5f*(bmin[1]+bmax[1]);
            desc_.stat_center[2] = 0.5f*(bmin[2]+bmax[2]);
            float dx = bmax[0]-bmin[0], dy = bmax[1]-bmin[1], dz = bmax[2]-bmin[2];
            // MuJoCo: extent = max axis span (not diagonal)
            desc_.stat_extent = dx;
            if (dy > desc_.stat_extent) desc_.stat_extent = dy;
            if (dz > desc_.stat_extent) desc_.stat_extent = dz;
            if (desc_.stat_extent < 1e-5f) desc_.stat_extent = 1e-5f;

            // MuJoCo: extent = max(extent, 2*meansize)
            if (meansize_count > 0) {
                float meansize = meansize_sum / meansize_count;
                if (2.0f * meansize > desc_.stat_extent) {
                    desc_.stat_extent = 2.0f * meansize;
                }
            }
        } else {
            desc_.stat_center[0] = desc_.stat_center[1] = desc_.stat_center[2] = 0.0f;
            desc_.stat_extent = 2.0f;
        }
        // fprintf(stderr, "instinct_sim: stat_extent=%.4f\n", desc_.stat_extent);
    }

    // ── Compute qpos_adr and dof_adr ────────────────────────────────────
    int qpos_offset = 0;
    int dof_offset = 0;
    for (int i = 0; i < desc_.njnt; i++) {
        JointDesc& j = desc_.joints[i];
        j.qpos_adr = qpos_offset;
        j.dof_adr = dof_offset;
        switch (j.type) {
            case JOINT_FREE:  qpos_offset += 7; dof_offset += 6; break;
            case JOINT_BALL:  qpos_offset += 4; dof_offset += 3; break;
            case JOINT_HINGE: qpos_offset += 1; dof_offset += 1; break;
            case JOINT_SLIDE: qpos_offset += 1; dof_offset += 1; break;
        }
    }
    // Bail BEFORE anything indexes qpos0/qpos0_d by qpos_adr. Both are declared
    // [MAX_DOF], but qpos_adr scales with nq, not nv: a free joint contributes
    // 7 qpos to 6 qvel, so a model can satisfy nv <= MAX_DOF and still overrun
    // qpos0 (nv=114 -> nq=133 writes 5 floats past the end). That is a host heap
    // overwrite during parsing, which crashes long before PhysicsEngine::init()
    // gets a chance to range-check the model.
    if (qpos_offset > MAX_DOF || dof_offset > MAX_DOF) {
        fprintf(stderr,
                "instinct_sim: model needs nq=%d nv=%d but qpos0/qvel storage is "
                "MAX_DOF=%d — refusing to parse (raise MAX_DOF in "
                "include/hipgl/physics/physics_types.h and rebuild)\n",
                qpos_offset, dof_offset, MAX_DOF);
        return false;
    }
    desc_.nq = qpos_offset;
    desc_.nv = dof_offset;

    // ── Default qpos ────────────────────────────────────────────────────
    memset(desc_.qpos0, 0, sizeof(desc_.qpos0));
    for (int i = 0; i < desc_.njnt; i++) {
        JointDesc& j = desc_.joints[i];
        if (j.type == JOINT_FREE) {
            const BodyDesc& b = desc_.bodies[j.body];
            float wx = 0, wy = 0, wz = 0;
            int cur = j.body;
            std::vector<int> chain;
            while (cur > 0) { chain.push_back(cur); cur = desc_.bodies[cur].parent; }
            for (int ci = (int)chain.size() - 1; ci >= 0; ci--) {
                const BodyDesc& cb = desc_.bodies[chain[ci]];
                wx += cb.pos[0]; wy += cb.pos[1]; wz += cb.pos[2];
            }
            desc_.qpos0[j.qpos_adr + 0] = wx;
            desc_.qpos0[j.qpos_adr + 1] = wy;
            desc_.qpos0[j.qpos_adr + 2] = wz;
            desc_.qpos0[j.qpos_adr + 3] = b.quat[0];
            desc_.qpos0[j.qpos_adr + 4] = b.quat[1];
            desc_.qpos0[j.qpos_adr + 5] = b.quat[2];
            desc_.qpos0[j.qpos_adr + 6] = b.quat[3];
        } else if (j.type == JOINT_BALL) {
            desc_.qpos0[j.qpos_adr] = 1.0f;
        }
    }

    // ── Parse <actuator> (name-based joint mapping) ─────────────────────
    XMLElement* actuator = mujoco->FirstChildElement("actuator");
    if (actuator) {
        for (XMLElement* act = actuator->FirstChildElement(); act;
             act = act->NextSiblingElement()) {
            if (desc_.nu >= MAX_ACTUATORS)
                throw std::runtime_error("PhysicsModel: more than MAX_ACTUATORS (" +
                                         std::to_string(MAX_ACTUATORS) + ") actuators");
            const char* joint_name = act->Attribute("joint");
            if (!joint_name) continue;

            // Resolve joint by name
            int jnt_idx = -1;
            auto it = state.joint_name_map.find(joint_name);
            if (it != state.joint_name_map.end()) {
                jnt_idx = it->second;
            } else {
                fprintf(stderr, "instinct_sim: actuator references unknown joint '%s'\n", joint_name);
                continue;
            }

            int aidx = desc_.nu++;
            ActuatorDesc& a = desc_.actuators[aidx];
            memset(&a, 0, sizeof(a));
            a.jnt = jnt_idx;

            // Determine actuator type from element name
            const char* tag = act->Name();
            if (strcmp(tag, "position") == 0) {
                a.type = ACTUATOR_POSITION;
                a.gain = attr_float(act, "kp", 1.0f);
                a.kv = attr_float(act, "kv", 0.0f);
                if (a.kv == 0.0f) a.kv = attr_float(act, "dampratio", 0.0f);
            } else if (strcmp(tag, "velocity") == 0) {
                a.type = ACTUATOR_VELOCITY;
                a.gain = attr_float(act, "kv", 1.0f);
            } else if (strcmp(tag, "motor") == 0) {
                a.type = ACTUATOR_MOTOR;
                a.gain = attr_float(act, "gear", 1.0f);
            } else {
                // <general> or unknown: detect type from attributes
                const char* biastype_str = act->Attribute("biastype");
                const char* gainprm_str = act->Attribute("gainprm");
                const char* biasprm_str = act->Attribute("biasprm");

                if (biastype_str && strcmp(biastype_str, "affine") == 0) {
                    a.type = ACTUATOR_GENERAL;
                    float gp[3] = {1.0f, 0.0f, 0.0f};
                    if (gainprm_str) parse_floats(gainprm_str, gp, 3);
                    a.gain = gp[0];
                    float bp[3] = {0.0f, 0.0f, 0.0f};
                    if (biasprm_str) parse_floats(biasprm_str, bp, 3);
                    a.biasprm[0] = bp[0]; a.biasprm[1] = bp[1]; a.biasprm[2] = bp[2];
                } else {
                    a.type = ACTUATOR_MOTOR;
                    a.gain = attr_float(act, "kp", 0.0f);
                    if (a.gain > 0.0f) {
                        a.type = ACTUATOR_POSITION;
                        a.kv = attr_float(act, "kv", 0.0f);
                    } else {
                        float gp[3] = {0.0f, 0.0f, 0.0f};
                        if (gainprm_str) parse_floats(gainprm_str, gp, 3);
                        if (gp[0] > 0.0f) {
                            a.gain = gp[0];
                        } else {
                            a.gain = attr_float(act, "gear", 1.0f);
                        }
                    }
                }
            }

            a.forcerange[0] = -1e6f; a.forcerange[1] = 1e6f;
            a.ctrlrange[0] = -1e6f; a.ctrlrange[1] = 1e6f;
            const char* fr = act->Attribute("forcerange");
            if (fr) parse_floats(fr, a.forcerange, 2);
            const char* cr = act->Attribute("ctrlrange");
            if (cr) parse_floats(cr, a.ctrlrange, 2);

            // Fill double-precision fields from the same XML
            a.gain_d = (double)a.gain;
            a.kv_d = (double)a.kv;
            for (int k = 0; k < 3; k++) a.biasprm_d[k] = (double)a.biasprm[k];
            for (int k = 0; k < 2; k++) a.forcerange_d[k] = (double)a.forcerange[k];
            for (int k = 0; k < 2; k++) a.ctrlrange_d[k] = (double)a.ctrlrange[k];
            // Re-parse as double for attributes that were parsed as float
            if (strcmp(tag, "position") == 0) {
                a.gain_d = attr_double(act, "kp", 1.0);
                a.kv_d = attr_double(act, "kv", 0.0);
                if (a.kv_d == 0.0) a.kv_d = attr_double(act, "dampratio", 0.0);
            } else if (strcmp(tag, "velocity") == 0) {
                a.gain_d = attr_double(act, "kv", 1.0);
            } else if (strcmp(tag, "motor") == 0) {
                a.gain_d = attr_double(act, "gear", 1.0);
            } else {
                const char* gps = act->Attribute("gainprm");
                const char* bps = act->Attribute("biasprm");
                if (gps) { double gp_d[3] = {1,0,0}; parse_doubles(gps, gp_d, 3); a.gain_d = gp_d[0]; }
                if (bps) { parse_doubles(bps, a.biasprm_d, 3); }
                if (a.type == ACTUATOR_MOTOR && a.gain_d == 0.0) {
                    a.gain_d = attr_double(act, "kp", 0.0);
                    if (a.gain_d > 0.0) a.kv_d = attr_double(act, "kv", 0.0);
                    else a.gain_d = attr_double(act, "gear", 1.0);
                }
            }
            if (fr) parse_doubles(fr, a.forcerange_d, 2);
            if (cr) parse_doubles(cr, a.ctrlrange_d, 2);
        }
    }

    // ── Parse <contact><exclude> ────────────────────────────────────────
    XMLElement* contact = mujoco->FirstChildElement("contact");
    if (contact) {
        for (XMLElement* excl = contact->FirstChildElement("exclude"); excl;
             excl = excl->NextSiblingElement("exclude")) {
            if (desc_.npair_filter >= MAX_PAIR_FILTERS) break;
            const char* b1name = excl->Attribute("body1");
            const char* b2name = excl->Attribute("body2");
            if (!b1name || !b2name) continue;

            auto it1 = state.body_name_map.find(b1name);
            auto it2 = state.body_name_map.find(b2name);
            if (it1 == state.body_name_map.end() || it2 == state.body_name_map.end()) continue;

            int b1 = it1->second, b2 = it2->second;
            if (b1 > b2) std::swap(b1, b2);
            desc_.pair_filter[desc_.npair_filter++] = (b1 << 16) | b2;
        }
        // Sort for binary search in collision kernel
        std::sort(desc_.pair_filter, desc_.pair_filter + desc_.npair_filter);
    }

    // ── Parse <equality> ────────────────────────────────────────────────
    XMLElement* equality = mujoco->FirstChildElement("equality");
    if (equality) {
        for (XMLElement* eq = equality->FirstChildElement(); eq;
             eq = eq->NextSiblingElement()) {
            if (desc_.nequality >= MAX_EQUALITY) break;
            const char* tag = eq->Name();
            EqualityDesc& e = desc_.equalities[desc_.nequality];
            memset(&e, 0, sizeof(e));

            if (strcmp(tag, "weld") == 0) {
                e.type = EQUALITY_WELD;
                const char* b1 = eq->Attribute("body1");
                const char* b2 = eq->Attribute("body2");
                if (!b1 || !b2) continue;
                auto it1 = state.body_name_map.find(b1);
                auto it2 = state.body_name_map.find(b2);
                if (it1 == state.body_name_map.end() || it2 == state.body_name_map.end()) continue;
                e.body1 = it1->second;
                e.body2 = it2->second;
                desc_.nequality++;
            } else if (strcmp(tag, "joint") == 0) {
                e.type = EQUALITY_JOINT;
                const char* j1 = eq->Attribute("joint1");
                const char* j2 = eq->Attribute("joint2");
                if (!j1) continue;
                auto it1 = state.joint_name_map.find(j1);
                if (it1 == state.joint_name_map.end()) continue;
                e.body1 = it1->second;  // joint1 index stored in body1
                if (j2) {
                    auto it2 = state.joint_name_map.find(j2);
                    if (it2 != state.joint_name_map.end())
                        e.body2 = it2->second;
                }
                float polycoef[5] = {0, 1, 0, 0, 0};
                parse_floats(eq->Attribute("polycoef"), polycoef, 5);
                memcpy(e.data, polycoef, 5 * sizeof(float));
                desc_.nequality++;
            }
        }
    }

    // ── Extract convex hull collision data for all meshes ────────────────
    // MuJoCo uses convex hulls only for collision detection, not rendering.
    // Geoms render their original mesh vertices; hull data goes into separate
    // collision-specific arrays (convex_hull_vert/face/edge).
    {
        delete[] desc_.convex_hull_vert;
        delete[] desc_.convex_hull_face;
        delete[] desc_.convex_hull_face_normal;
        delete[] desc_.convex_hull_edge;
        desc_.convex_hull_vert = nullptr;
        desc_.convex_hull_face = nullptr;
        desc_.convex_hull_face_normal = nullptr;
        desc_.convex_hull_edge = nullptr;
        desc_.convex_total_verts = 0;
        desc_.convex_total_faces = 0;
        desc_.convex_total_edges = 0;

        // Collect mesh IDs actually used by collision geoms (contype|conaffinity > 0)
        std::unordered_set<int> collision_mesh_ids;
        for (int gi = 0; gi < desc_.ngeom; gi++) {
            auto& g = desc_.geoms[gi];
            if (g.type == GEOM_MESH && g.mesh_id >= 0 &&
                (g.contype != 0 || g.conaffinity != 0))
                collision_mesh_ids.insert(g.mesh_id);
        }

        struct TempHull { ConvexHullCollisionData data; int mesh_id; };
        std::vector<TempHull> temp_hulls;
        int total_v = 0, total_f = 0, total_e = 0;
        int n_cvx_hulled = 0;

        for (int mi = 0; mi < desc_.nmesh; mi++) {
            if (collision_mesh_ids.find(mi) == collision_mesh_ids.end()) {
                desc_.mesh_convex[mi] = {0, 0, 0, 0, 0, 0};
                continue;
            }

            auto& mesh_info = desc_.meshes[mi];
            const float* src = &desc_.mesh_vert[mesh_info.vert_adr * 3];
            int src_n = mesh_info.vert_num;

            // SAT collision requires convex shapes.  Compute the actual convex
            // hull so that concave meshes produce correct contacts and normals.
            // For large meshes, subsample vertices first to keep hull computation
            // fast (the hull of a subsample closely approximates the full hull).
            ConvexHullCollisionData hull;
            constexpr int MAX_HULL_INPUT = 3000;
            const float* hull_src = src;
            int hull_src_n = src_n;
            std::vector<float> subsampled;

            if (src_n > MAX_HULL_INPUT) {
                int stride = (src_n + MAX_HULL_INPUT - 1) / MAX_HULL_INPUT;
                // Always keep stride as multiple of 3 to preserve triangle structure
                stride = ((stride + 2) / 3) * 3;
                for (int vi = 0; vi < src_n; vi += stride) {
                    int end = std::min(vi + 3, src_n);
                    for (int k = vi; k < end; k++) {
                        subsampled.push_back(src[k * 3 + 0]);
                        subsampled.push_back(src[k * 3 + 1]);
                        subsampled.push_back(src[k * 3 + 2]);
                    }
                }
                hull_src = subsampled.data();
                hull_src_n = (int)(subsampled.size() / 3);
            }

            MeshData cvx = compute_convex_hull(hull_src, hull_src_n);
            if (cvx.nvertices >= 4) {
                hull = extract_collision_hull(cvx.vertices.data(), cvx.nvertices);
                n_cvx_hulled++;
            } else {
                hull = extract_collision_hull(src, src_n);
            }
            if (hull.nvert < 4 || hull.nface < 4) {
                desc_.mesh_convex[mi] = {0, 0, 0, 0, 0, 0};
                continue;
            }

            desc_.mesh_convex[mi].vert_adr = total_v;
            desc_.mesh_convex[mi].vert_num = hull.nvert;
            desc_.mesh_convex[mi].face_adr = total_f;
            desc_.mesh_convex[mi].face_num = hull.nface;
            desc_.mesh_convex[mi].edge_adr = total_e;
            desc_.mesh_convex[mi].edge_num = hull.nedge;

            total_v += hull.nvert;
            total_f += hull.nface;
            total_e += hull.nedge;
            temp_hulls.push_back({std::move(hull), mi});
        }

        if (total_v > 0) {
            desc_.convex_total_verts = total_v;
            desc_.convex_total_faces = total_f;
            desc_.convex_total_edges = total_e;
            desc_.convex_hull_vert = new float[total_v * 3];
            desc_.convex_hull_face = new int[total_f * 3];
            desc_.convex_hull_face_normal = new float[total_f * 3];
            desc_.convex_hull_edge = new int[total_e * 2];

            for (auto& th : temp_hulls) {
                auto& ch = desc_.mesh_convex[th.mesh_id];
                memcpy(&desc_.convex_hull_vert[ch.vert_adr * 3],
                       th.data.verts.data(), th.data.nvert * 3 * sizeof(float));
                memcpy(&desc_.convex_hull_face[ch.face_adr * 3],
                       th.data.faces.data(), th.data.nface * 3 * sizeof(int));
                memcpy(&desc_.convex_hull_face_normal[ch.face_adr * 3],
                       th.data.face_normals.data(), th.data.nface * 3 * sizeof(float));
                memcpy(&desc_.convex_hull_edge[ch.edge_adr * 2],
                       th.data.edges.data(), th.data.nedge * 2 * sizeof(int));
            }

            fprintf(stderr, "instinct_sim: built collision hulls for %d meshes "
                    "(%d convex-hulled, %d verts, %d faces, %d edges)\n",
                    (int)temp_hulls.size(), n_cvx_hulled, total_v, total_f, total_e);
        }
    }

    build_tree_levels();

    if (!upload_to_device()) {
        fprintf(stderr, "instinct_sim: failed to upload model to GPU\n");
        return false;
    }

    uploaded_ = true;
    return true;
}

void PhysicsModel::build_tree_levels() {
    std::vector<int> level_of(desc_.nbody, -1);
    std::queue<int> q;
    q.push(0);
    level_of[0] = 0;
    int max_depth = 0;

    while (!q.empty()) {
        int b = q.front(); q.pop();
        for (int c = 1; c < desc_.nbody; c++) {
            if (desc_.bodies[c].parent == b && level_of[c] == -1) {
                level_of[c] = level_of[b] + 1;
                if (level_of[c] > max_depth) max_depth = level_of[c];
                q.push(c);
            }
        }
    }

    desc_.tree_depth = max_depth;

    std::vector<std::vector<int>> levels(max_depth + 1);
    for (int i = 0; i < desc_.nbody; i++) {
        if (level_of[i] >= 0)
            levels[level_of[i]].push_back(i);
    }

    int pos = 0;
    for (int d = 0; d <= max_depth; d++) {
        desc_.level_start[d] = pos;
        for (int b : levels[d]) {
            desc_.level_body_ids[pos++] = b;
        }
    }
    desc_.level_start[max_depth + 1] = pos;
}

void PhysicsModel::set_fixed_body_poses(const std::vector<int>& body_ids,
                                       const std::vector<double>& positions,
                                       const std::vector<double>& quaternions) {
    apply_fixed_body_poses(0, -1, body_ids, positions, quaternions);
}

void PhysicsModel::set_fixed_body_poses_env(int num_envs, int env,
                                           const std::vector<int>& body_ids,
                                           const std::vector<double>& positions,
                                           const std::vector<double>& quaternions) {
    if (num_envs <= 0 || env < 0 || env >= num_envs) {
        throw std::invalid_argument("set_fixed_body_poses_env: env out of range");
    }
    if (uploaded_ && d_model_.body_pose_nenv != 0 && d_model_.body_pose_nenv != num_envs) {
        throw std::invalid_argument("set_fixed_body_poses_env: num_envs differs from the allocated per-world poses");
    }
    apply_fixed_body_poses(num_envs, env, body_ids, positions, quaternions);
}

void PhysicsModel::apply_fixed_body_poses(int num_envs, int env,
                                          const std::vector<int>& body_ids,
                                          const std::vector<double>& positions,
                                          const std::vector<double>& quaternions) {
    const size_t count = body_ids.size();
    if (positions.size() != count * 3 || quaternions.size() != count * 4) {
        throw std::invalid_argument("set_fixed_body_poses: expected N ids, N*3 positions, N*4 quaternions");
    }
    std::unordered_set<int> seen;
    std::vector<float> positions_f(count * 3), quaternions_f(count * 4);
    // Validate the ENTIRE batch before changing any host or device value.
    for (size_t row = 0; row < count; ++row) {
        const int id = body_ids[row];
        if (id <= 0 || id >= desc_.nbody || !seen.insert(id).second) {
            throw std::invalid_argument("set_fixed_body_poses: invalid, world, or duplicate body id");
        }
        int ancestor = id, depth = 0;
        while (ancestor > 0) {
            if (ancestor >= desc_.nbody || ++depth > desc_.nbody ||
                desc_.bodies[ancestor].jnt_num != 0) {
                throw std::invalid_argument("set_fixed_body_poses: body is not fixed to the world");
            }
            ancestor = desc_.bodies[ancestor].parent;
        }
        for (int axis = 0; axis < 3; ++axis) {
            const double value = positions[row * 3 + axis];
            const float narrowed = static_cast<float>(value);
            if (!std::isfinite(value) || !std::isfinite(narrowed)) {
                throw std::invalid_argument("set_fixed_body_poses: positions must be finite in float and double");
            }
            positions_f[row * 3 + axis] = narrowed;
        }
        double norm2 = 0.0;
        for (int axis = 0; axis < 4; ++axis) {
            const double value = quaternions[row * 4 + axis];
            if (!std::isfinite(value)) {
                throw std::invalid_argument("set_fixed_body_poses: quaternions must be finite");
            }
            norm2 += value * value;
            quaternions_f[row * 4 + axis] = static_cast<float>(value);
        }
        if (!std::isfinite(norm2) || std::abs(norm2 - 1.0) > 1e-6) {
            throw std::invalid_argument("set_fixed_body_poses: quaternions must have unit length (wxyz)");
        }
    }
    if (count == 0) return;
    if (!uploaded_ || !d_model_.body_pos || !d_model_.body_quat ||
        !d_model_.body_pos_d || !d_model_.body_quat_d) {
        throw std::runtime_error("set_fixed_body_poses: model pose arrays are not uploaded");
    }
    const auto checked = [](hipError_t error, const char* operation) {
        if (error != hipSuccess) {
            throw std::runtime_error(std::string("set_fixed_body_poses: ") +
                                     operation + ": " + hipGetErrorString(error));
        }
    };
    // The model may live on a different device than the caller's current one.
    // Synchronize its allocation's device, and restore the caller's selection.
    hipPointerAttribute_t attributes{};
    checked(hipPointerGetAttributes(&attributes, d_model_.body_pos_d), "query pose allocation");
    int previous_device = 0;
    checked(hipGetDevice(&previous_device), "get current device");
    const bool switch_device = previous_device != attributes.device;
    if (switch_device) checked(hipSetDevice(attributes.device), "select model device");
    try {
        checked(hipDeviceSynchronize(), "synchronize before pose update");
        DevicePhysicsModel& dm = d_model_;
        const size_t nb = (size_t)desc_.nbody;
        if (env >= 0 && dm.body_pos_env_d == nullptr) {
            // First per-world request: every world starts at the model-wide pose.
            const size_t nw = (size_t)num_envs;
            checked(hipMalloc(&dm.body_pos_env,    nw * nb * 3 * sizeof(float)),  "alloc per-world float positions");
            checked(hipMalloc(&dm.body_quat_env,   nw * nb * 4 * sizeof(float)),  "alloc per-world float quaternions");
            checked(hipMalloc(&dm.body_pos_env_d,  nw * nb * 3 * sizeof(double)), "alloc per-world double positions");
            checked(hipMalloc(&dm.body_quat_env_d, nw * nb * 4 * sizeof(double)), "alloc per-world double quaternions");
            for (size_t w = 0; w < nw; ++w) {
                checked(hipMemcpy(dm.body_pos_env + w * nb * 3, dm.body_pos, nb * 3 * sizeof(float),
                                  hipMemcpyDeviceToDevice), "seed per-world float positions");
                checked(hipMemcpy(dm.body_quat_env + w * nb * 4, dm.body_quat, nb * 4 * sizeof(float),
                                  hipMemcpyDeviceToDevice), "seed per-world float quaternions");
                checked(hipMemcpy(dm.body_pos_env_d + w * nb * 3, dm.body_pos_d, nb * 3 * sizeof(double),
                                  hipMemcpyDeviceToDevice), "seed per-world double positions");
                checked(hipMemcpy(dm.body_quat_env_d + w * nb * 4, dm.body_quat_d, nb * 4 * sizeof(double),
                                  hipMemcpyDeviceToDevice), "seed per-world double quaternions");
            }
            dm.body_pose_nenv = num_envs;
        }
        const auto upload = [&](float* pf, float* qf, double* pd, double* qd) {
            for (size_t row = 0; row < count; ++row) {
                const int id = body_ids[row];
                checked(hipMemcpy(pf + id * 3, positions_f.data() + row * 3,
                                  3 * sizeof(float), hipMemcpyHostToDevice), "upload float positions");
                checked(hipMemcpy(qf + id * 4, quaternions_f.data() + row * 4,
                                  4 * sizeof(float), hipMemcpyHostToDevice), "upload float quaternions");
                checked(hipMemcpy(pd + id * 3, positions.data() + row * 3,
                                  3 * sizeof(double), hipMemcpyHostToDevice), "upload double positions");
                checked(hipMemcpy(qd + id * 4, quaternions.data() + row * 4,
                                  4 * sizeof(double), hipMemcpyHostToDevice), "upload double quaternions");
            }
        };
        if (env >= 0) {
            const size_t w = (size_t)env;
            upload(dm.body_pos_env + w * nb * 3, dm.body_quat_env + w * nb * 4,
                   dm.body_pos_env_d + w * nb * 3, dm.body_quat_env_d + w * nb * 4);
            // The host descriptor keeps the model-wide pose.
            if (switch_device) checked(hipSetDevice(previous_device), "restore caller device");
            return;
        }
        upload(dm.body_pos, dm.body_quat, dm.body_pos_d, dm.body_quat_d);
        for (int w = 0; w < dm.body_pose_nenv; ++w) {
            upload(dm.body_pos_env + (size_t)w * nb * 3, dm.body_quat_env + (size_t)w * nb * 4,
                   dm.body_pos_env_d + (size_t)w * nb * 3, dm.body_quat_env_d + (size_t)w * nb * 4);
        }
        for (size_t row = 0; row < count; ++row) {
            BodyDesc& body = desc_.bodies[body_ids[row]];
            memcpy(body.pos, positions_f.data() + row * 3, 3 * sizeof(float));
            memcpy(body.quat, quaternions_f.data() + row * 4, 4 * sizeof(float));
            memcpy(body.pos_d, positions.data() + row * 3, 3 * sizeof(double));
            memcpy(body.quat_d, quaternions.data() + row * 4, 4 * sizeof(double));
        }
    } catch (...) {
        if (switch_device) (void)hipSetDevice(previous_device);
        throw;
    }
    if (switch_device) checked(hipSetDevice(previous_device), "restore caller device");
}

bool PhysicsModel::upload_invweight() {
    if (!desc_.has_invweight0) return true;
    DevicePhysicsModel& dm = d_model_;
    int nb = desc_.nbody;
    int nv = desc_.nv;

    if (dm.body_invweight0)   { hipFree(dm.body_invweight0);   dm.body_invweight0 = nullptr; }
    if (dm.body_invweight0_d) { hipFree(dm.body_invweight0_d); dm.body_invweight0_d = nullptr; }
    if (dm.dof_invweight0)    { hipFree(dm.dof_invweight0);    dm.dof_invweight0 = nullptr; }
    if (dm.dof_invweight0_d)  { hipFree(dm.dof_invweight0_d);  dm.dof_invweight0_d = nullptr; }

    hipError_t err;
    err = hipMalloc(&dm.body_invweight0, nb * 2 * sizeof(float));
    if (err != hipSuccess) return false;
    err = hipMemcpy(dm.body_invweight0, desc_.body_invweight0,
                    nb * 2 * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) return false;

    err = hipMalloc(&dm.body_invweight0_d, nb * 2 * sizeof(double));
    if (err != hipSuccess) return false;
    err = hipMemcpy(dm.body_invweight0_d, desc_.body_invweight0_d,
                    nb * 2 * sizeof(double), hipMemcpyHostToDevice);
    if (err != hipSuccess) return false;

    err = hipMalloc(&dm.dof_invweight0, nv * sizeof(float));
    if (err != hipSuccess) return false;
    err = hipMemcpy(dm.dof_invweight0, desc_.dof_invweight0,
                    nv * sizeof(float), hipMemcpyHostToDevice);
    if (err != hipSuccess) return false;

    err = hipMalloc(&dm.dof_invweight0_d, nv * sizeof(double));
    if (err != hipSuccess) return false;
    err = hipMemcpy(dm.dof_invweight0_d, desc_.dof_invweight0_d,
                    nv * sizeof(double), hipMemcpyHostToDevice);
    if (err != hipSuccess) return false;

    return true;
}

bool PhysicsModel::upload_textures() {
    DevicePhysicsModel& dm = d_model_;
    if (desc_.ntexture <= 0 || !desc_.texture_data) return true;

    const int nt = desc_.ntexture;
    const int ng = desc_.ngeom;
    if (nt > PhysicsModelDesc::MAX_TEXTURES || desc_.texture_data_size <= 0) {
        fprintf(stderr, "[upload_textures] invalid texture count or RGB data size\n");
        return false;
    }
    std::vector<int> h_tw(nt), h_th(nt), h_to(nt), h_tc(nt);
    std::vector<TextureMipLevel> h_base(nt);
    for (int i = 0; i < nt; i++) {
        h_tw[i] = desc_.textures[i].width;
        h_th[i] = desc_.textures[i].height;
        h_to[i] = desc_.textures[i].data_offset;
        h_tc[i] = desc_.textures[i].is_cube;
        h_base[i] = {h_to[i], h_tw[i], h_th[i]};
    }
    // Read once when textures are uploaded, never in the render loop. The
    // level-zero bytes and old sampler remain available for controlled A/Bs.
    const char* mip_env = std::getenv("HIPGL_MIPMAPS");
    const bool mip_enabled = !mip_env || std::atoi(mip_env) != 0;
    TextureMipPyramid pyramid;
    if (mip_enabled) {
        try {
            pyramid = build_texture_mip_pyramid(desc_.texture_data,
                size_t(desc_.texture_data_size), h_base.data(), h_tc.data(), nt);
        } catch (const std::exception& e) {
            fprintf(stderr, "[upload_textures] mip generation failed: %s\n", e.what());
            return false;
        }
    }
    const unsigned char* rgb = mip_enabled ? pyramid.data.data() : desc_.texture_data;
    const size_t rgb_bytes = mip_enabled ? pyramid.data.size() : size_t(desc_.texture_data_size);

    // Free old texture GPU data
    if (dm.tex_width)  { hipFree(dm.tex_width);  dm.tex_width = nullptr; }
    if (dm.tex_height) { hipFree(dm.tex_height); dm.tex_height = nullptr; }
    if (dm.tex_offset) { hipFree(dm.tex_offset); dm.tex_offset = nullptr; }
    if (dm.tex_iscube) { hipFree(dm.tex_iscube); dm.tex_iscube = nullptr; }
    if (dm.tex_mip)   { hipFree(dm.tex_mip);   dm.tex_mip = nullptr; }
    if (dm.tex_data)   { hipFree(dm.tex_data);   dm.tex_data = nullptr; }
    if (dm.geom_texid) { hipFree(dm.geom_texid); dm.geom_texid = nullptr; }

    dm.ntexture = nt;

    hipError_t err;
    err = hipMalloc(&dm.tex_width, nt * sizeof(int));
    if (err != hipSuccess) return false;
    hipMemcpy(dm.tex_width, h_tw.data(), nt * sizeof(int), hipMemcpyHostToDevice);

    err = hipMalloc(&dm.tex_height, nt * sizeof(int));
    if (err != hipSuccess) return false;
    hipMemcpy(dm.tex_height, h_th.data(), nt * sizeof(int), hipMemcpyHostToDevice);

    err = hipMalloc(&dm.tex_offset, nt * sizeof(int));
    if (err != hipSuccess) return false;
    hipMemcpy(dm.tex_offset, h_to.data(), nt * sizeof(int), hipMemcpyHostToDevice);

    err = hipMalloc(&dm.tex_iscube, nt * sizeof(int));
    if (err != hipSuccess) return false;
    hipMemcpy(dm.tex_iscube, h_tc.data(), nt * sizeof(int), hipMemcpyHostToDevice);

    if (mip_enabled) {
        err = hipMalloc(&dm.tex_mip, pyramid.levels.size() * sizeof(TextureMipLevel));
        if (err != hipSuccess) return false;
        err = hipMemcpy(dm.tex_mip, pyramid.levels.data(),
                        pyramid.levels.size() * sizeof(TextureMipLevel), hipMemcpyHostToDevice);
        if (err != hipSuccess) return false;
    }

    err = hipMalloc(&dm.tex_data, rgb_bytes);
    if (err != hipSuccess) {
        fprintf(stderr, "[upload_textures] FAILED: hipMalloc tex_data (%zu bytes): %s\n",
                rgb_bytes, hipGetErrorString(err));
        return false;
    }
    err = hipMemcpy(dm.tex_data, rgb, rgb_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "[upload_textures] FAILED: hipMemcpy tex_data (%zu bytes): %s\n",
                rgb_bytes, hipGetErrorString(err));
        return false;
    }
    err = hipMalloc(&dm.geom_texid, ng * sizeof(int));
    if (err != hipSuccess) return false;
    hipMemcpy(dm.geom_texid, desc_.geom_texid, ng * sizeof(int), hipMemcpyHostToDevice);

    // Upload per-geom texrepeat (S, T tiling factors)
    if (dm.geom_texrepeat) { hipFree(dm.geom_texrepeat); dm.geom_texrepeat = nullptr; }
    err = hipMalloc(&dm.geom_texrepeat, ng * 2 * sizeof(float));
    if (err != hipSuccess) return false;
    hipMemcpy(dm.geom_texrepeat, desc_.geom_texrepeat, ng * 2 * sizeof(float), hipMemcpyHostToDevice);

    static std::atomic<int> s_tex_generation{0};
    dm.tex_generation = ++s_tex_generation;
    return true;
}

// ── GPU upload ──────────────────────────────────────────────────────────────

template<typename T>
static bool upload_array(T** d_ptr, const T* h_data, int count, const char* name = "") {
    if (count <= 0) { *d_ptr = nullptr; return true; }
    hipError_t err = hipMalloc(d_ptr, count * sizeof(T));
    if (err != hipSuccess) {
        fprintf(stderr, "  [upload_array] FAILED hipMalloc for '%s' (%d x %zu = %zu bytes): %s\n",
                name, count, sizeof(T), (size_t)count * sizeof(T), hipGetErrorString(err));
        return false;
    }
    err = hipMemcpy(*d_ptr, h_data, count * sizeof(T), hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        fprintf(stderr, "  [upload_array] FAILED hipMemcpy for '%s': %s\n", name, hipGetErrorString(err));
        return false;
    }
    return true;
}

bool PhysicsModel::upload_to_device() {
    DevicePhysicsModel& dm = d_model_;
    memset(&dm, 0, sizeof(dm));
    dm.nbody = desc_.nbody;
    dm.njnt = desc_.njnt;
    dm.ngeom = desc_.ngeom;
    dm.nq = desc_.nq;
    dm.nv = desc_.nv;
    dm.nu = desc_.nu;
    dm.timestep = desc_.timestep;
    dm.gravity[0] = desc_.gravity[0];
    dm.gravity[1] = desc_.gravity[1];
    dm.gravity[2] = desc_.gravity[2];
    dm.gravity_d[0] = desc_.gravity_d[0];
    dm.gravity_d[1] = desc_.gravity_d[1];
    dm.gravity_d[2] = desc_.gravity_d[2];
    dm.cone_type = desc_.cone_type;
    dm.friction_combine = desc_.friction_combine;
    dm.max_depen_vel = desc_.max_depen_vel;

    dm.opt_density = desc_.opt_density;
    dm.opt_viscosity = desc_.opt_viscosity;
    dm.tree_depth = desc_.tree_depth;
    dm.npair_filter = desc_.npair_filter;
    dm.nequality = desc_.nequality;
    dm.ncamera = desc_.ncamera;

    int nb = desc_.nbody, nj = desc_.njnt, ng = desc_.ngeom, na = desc_.nu;

    fprintf(stderr, "  [upload] Starting upload: nb=%d nj=%d ng=%d na=%d\n", nb, nj, ng, na);
    // ── Body arrays ─────────────────────────────────────────────────────
    std::vector<int>   h_bp(nb), h_bja(nb), h_bjn(nb);
    std::vector<float> h_bpos(nb*3), h_bq(nb*4), h_bm(nb), h_bi(nb*3), h_bip(nb*3), h_biq(nb*4);
    for (int i = 0; i < nb; i++) {
        const BodyDesc& b = desc_.bodies[i];
        h_bp[i] = b.parent; h_bja[i] = b.jnt_adr; h_bjn[i] = b.jnt_num;
        memcpy(&h_bpos[i*3], b.pos, 12); memcpy(&h_bq[i*4], b.quat, 16);
        h_bm[i] = b.mass; memcpy(&h_bi[i*3], b.inertia, 12); memcpy(&h_bip[i*3], b.ipos, 12);
        memcpy(&h_biq[i*4], b.iquat, 16);
    }
    if (!upload_array(&dm.body_parent, h_bp.data(), nb)) return false;

    // Compute body_weldid: walk up parent chain until body with joints, matching MuJoCo
    std::vector<int> h_bweld(nb);
    for (int i = 0; i < nb; i++) {
        int j = i;
        while (j > 0 && desc_.bodies[j].jnt_num == 0) {
            j = desc_.bodies[j].parent;
        }
        h_bweld[i] = j;
    }
    if (!upload_array(&dm.body_weldid, h_bweld.data(), nb)) return false;

    if (!upload_array(&dm.body_jnt_adr, h_bja.data(), nb)) return false;
    if (!upload_array(&dm.body_jnt_num, h_bjn.data(), nb)) return false;

    // ── Precompute body-to-root chain for constraint Jacobian ──────────
    // For each body, walk up to root collecting (joint_idx, body, parent) tuples.
    // This eliminates pointer-chasing in fill_chain_jacobian_dir_d at runtime.
    {
        std::vector<int> chain_start(nb), chain_len(nb);
        std::vector<int> chain_jnt_idx, chain_jnt_body, chain_parent;
        for (int b = 0; b < nb; b++) {
            chain_start[b] = (int)chain_jnt_idx.size();
            int cur = b;
            int max_walk = 32;
            while (cur > 0 && max_walk-- > 0) {
                int ja = desc_.bodies[cur].jnt_adr;
                int jn = desc_.bodies[cur].jnt_num;
                int par = desc_.bodies[cur].parent;
                for (int k = 0; k < jn; k++) {
                    chain_jnt_idx.push_back(ja + k);
                    chain_jnt_body.push_back(cur);
                    chain_parent.push_back(par);
                }
                cur = par;
            }
            chain_len[b] = (int)chain_jnt_idx.size() - chain_start[b];
        }
        int total = (int)chain_jnt_idx.size();
        dm.body_chain_total = total;
        if (!upload_array(&dm.body_chain_start, chain_start.data(), nb)) return false;
        if (!upload_array(&dm.body_chain_len, chain_len.data(), nb)) return false;
        if (total > 0) {
            if (!upload_array(&dm.body_chain_jnt_idx, chain_jnt_idx.data(), total)) return false;
            if (!upload_array(&dm.body_chain_jnt_body, chain_jnt_body.data(), total)) return false;
            if (!upload_array(&dm.body_chain_parent, chain_parent.data(), total)) return false;
        } else {
            dm.body_chain_jnt_idx = nullptr;
            dm.body_chain_jnt_body = nullptr;
            dm.body_chain_parent = nullptr;
        }
    }
    if (!upload_array(&dm.body_pos, h_bpos.data(), nb*3)) return false;
    if (!upload_array(&dm.body_quat, h_bq.data(), nb*4)) return false;
    if (!upload_array(&dm.body_mass, h_bm.data(), nb)) return false;
    if (!upload_array(&dm.body_inertia, h_bi.data(), nb*3)) return false;
    if (!upload_array(&dm.body_ipos, h_bip.data(), nb*3)) return false;
    if (!upload_array(&dm.body_iquat, h_biq.data(), nb*4)) return false;

    // Double-precision shadow uploads (parsed directly from XML as double)
    {
        std::vector<double> d_bpos(nb*3), d_bq(nb*4), d_bm(nb), d_bi(nb*3), d_bip(nb*3), d_biq(nb*4);
        for (int i = 0; i < nb; i++) {
            memcpy(&d_bpos[i*3], desc_.bodies[i].pos_d, 3*sizeof(double));
            memcpy(&d_bq[i*4], desc_.bodies[i].quat_d, 4*sizeof(double));
            d_bm[i] = desc_.bodies[i].mass_d;
            memcpy(&d_bi[i*3], desc_.bodies[i].inertia_d, 3*sizeof(double));
            memcpy(&d_bip[i*3], desc_.bodies[i].ipos_d, 3*sizeof(double));
            memcpy(&d_biq[i*4], desc_.bodies[i].iquat_d, 4*sizeof(double));
        }
        if (!upload_array(&dm.body_pos_d, d_bpos.data(), nb*3)) return false;
        if (!upload_array(&dm.body_quat_d, d_bq.data(), nb*4)) return false;
        if (!upload_array(&dm.body_mass_d, d_bm.data(), nb)) return false;
        if (!upload_array(&dm.body_inertia_d, d_bi.data(), nb*3)) return false;
        if (!upload_array(&dm.body_ipos_d, d_bip.data(), nb*3)) return false;
        if (!upload_array(&dm.body_iquat_d, d_biq.data(), nb*4)) return false;
    }

    // Gravity compensation: only the bodies that ask for it.
    {
        std::vector<int> gc_body;
        std::vector<double> gc_scale;
        for (int i = 1; i < nb; i++) {
            if (desc_.bodies[i].gravcomp != 0.0) {
                gc_body.push_back(i);
                gc_scale.push_back(desc_.bodies[i].gravcomp);
            }
        }
        dm.n_gravcomp = (int)gc_body.size();
        dm.gravcomp_body = nullptr;
        dm.gravcomp_scale_d = nullptr;
        if (dm.n_gravcomp > 0) {
            if (!upload_array(&dm.gravcomp_body, gc_body.data(), dm.n_gravcomp)) return false;
            if (!upload_array(&dm.gravcomp_scale_d, gc_scale.data(), dm.n_gravcomp)) return false;
        }
    }

    // Upload MuJoCo-precomputed body_invweight0 and dof_invweight0
    if (desc_.has_invweight0) {
        if (!upload_array(&dm.body_invweight0, desc_.body_invweight0, nb * 2)) return false;
        if (!upload_array(&dm.body_invweight0_d, desc_.body_invweight0_d, nb * 2)) return false;
        if (!upload_array(&dm.dof_invweight0, desc_.dof_invweight0, desc_.nv)) return false;
        if (!upload_array(&dm.dof_invweight0_d, desc_.dof_invweight0_d, desc_.nv)) return false;
    } else {
        dm.body_invweight0 = nullptr;
        dm.body_invweight0_d = nullptr;
        dm.dof_invweight0 = nullptr;
        dm.dof_invweight0_d = nullptr;
    }

    // Compute equivalent inertia box FULL dimensions for fluid drag (MuJoCo inertia-based model)
    // MuJoCo: box[i] = sqrt(max(MINVAL, 6*(Ij+Ik-Ii)/mass))  — full dimension
    if (desc_.opt_density > 0.0f || desc_.opt_viscosity > 0.0f) {
        std::vector<float> h_ibox(nb * 3);
        for (int i = 0; i < nb; i++) {
            float mass = desc_.bodies[i].mass;
            float Ixx = desc_.bodies[i].inertia[0];
            float Iyy = desc_.bodies[i].inertia[1];
            float Izz = desc_.bodies[i].inertia[2];
            if (mass > 1e-10f) {
                float a2 = 6.0f * (-Ixx + Iyy + Izz) / mass;
                float b2 = 6.0f * ( Ixx - Iyy + Izz) / mass;
                float c2 = 6.0f * ( Ixx + Iyy - Izz) / mass;
                h_ibox[i*3+0] = (a2 > 1e-10f) ? sqrtf(a2) : 1e-10f;
                h_ibox[i*3+1] = (b2 > 1e-10f) ? sqrtf(b2) : 1e-10f;
                h_ibox[i*3+2] = (c2 > 1e-10f) ? sqrtf(c2) : 1e-10f;
            } else {
                h_ibox[i*3+0] = h_ibox[i*3+1] = h_ibox[i*3+2] = 1e-10f;
            }
        }
        if (!upload_array(&dm.body_inertia_box, h_ibox.data(), nb * 3)) return false;
    } else {
        dm.body_inertia_box = nullptr;
    }

    if (!upload_array(&dm.level_start, desc_.level_start, desc_.tree_depth + 2)) return false;
    if (!upload_array(&dm.level_body_ids, desc_.level_body_ids, nb)) return false;

    // ── Joint arrays ────────────────────────────────────────────────────
    std::vector<int>   h_jt(nj), h_jb(nj), h_jqa(nj), h_jda(nj), h_jlim(nj);
    std::vector<float> h_jax(nj*3), h_jp(nj*3), h_jr(nj*2), h_js(nj), h_jd(nj), h_ja(nj), h_jfl(nj);
    std::vector<float> h_jsr_lim(nj*2), h_jsi_lim(nj*5);
    for (int i = 0; i < nj; i++) {
        const JointDesc& j = desc_.joints[i];
        h_jt[i] = j.type; h_jb[i] = j.body; h_jqa[i] = j.qpos_adr; h_jda[i] = j.dof_adr;
        h_jlim[i] = j.limited;
        memcpy(&h_jax[i*3], j.axis, 12); memcpy(&h_jp[i*3], j.pos, 12);
        memcpy(&h_jr[i*2], j.range, 8);
        h_js[i] = j.stiffness; h_jd[i] = j.damping;
        h_ja[i] = j.armature; h_jfl[i] = j.frictionloss;
        memcpy(&h_jsr_lim[i*2], j.solref_lim, 8);
        memcpy(&h_jsi_lim[i*5], j.solimp_lim, 20);
    }
    if (!upload_array(&dm.jnt_type, h_jt.data(), nj)) return false;
    if (!upload_array(&dm.jnt_body, h_jb.data(), nj)) return false;
    if (!upload_array(&dm.jnt_qpos_adr, h_jqa.data(), nj)) return false;
    if (!upload_array(&dm.jnt_dof_adr, h_jda.data(), nj)) return false;
    if (!upload_array(&dm.jnt_axis, h_jax.data(), nj*3)) return false;
    if (!upload_array(&dm.jnt_pos, h_jp.data(), nj*3)) return false;
    {
        std::vector<double> d_jax(nj*3), d_jp(nj*3), d_jd(nj), d_ja(nj);
        for (int i = 0; i < nj; i++) {
            memcpy(&d_jax[i*3], desc_.joints[i].axis_d, 3*sizeof(double));
            memcpy(&d_jp[i*3], desc_.joints[i].pos_d, 3*sizeof(double));
            d_jd[i] = desc_.joints[i].damping_d;
            d_ja[i] = desc_.joints[i].armature_d;
        }
        if (!upload_array(&dm.jnt_axis_d, d_jax.data(), nj*3)) return false;
        if (!upload_array(&dm.jnt_pos_d, d_jp.data(), nj*3)) return false;
        if (!upload_array(&dm.jnt_damping_d, d_jd.data(), nj)) return false;
        if (!upload_array(&dm.jnt_armature_d, d_ja.data(), nj)) return false;
    }
    if (!upload_array(&dm.jnt_range, h_jr.data(), nj*2)) return false;
    if (!upload_array(&dm.jnt_stiffness, h_js.data(), nj)) return false;
    {
        std::vector<double> d_js(nj);
        for (int i = 0; i < nj; i++) d_js[i] = (double)h_js[i];
        if (!upload_array(&dm.jnt_stiffness_d, d_js.data(), nj)) return false;
    }
    if (!upload_array(&dm.jnt_damping, h_jd.data(), nj)) return false;
    if (!upload_array(&dm.jnt_armature, h_ja.data(), nj)) return false;
    if (!upload_array(&dm.jnt_frictionloss, h_jfl.data(), nj)) return false;
    if (!upload_array(&dm.jnt_solref_lim, h_jsr_lim.data(), nj*2)) return false;
    if (!upload_array(&dm.jnt_solimp_lim, h_jsi_lim.data(), nj*5)) return false;
    {
        std::vector<double> d_jr(nj*2), d_jfl(nj), d_jsr_lim(nj*2), d_jsi_lim(nj*5);
        for (int i = 0; i < nj*2; i++) d_jr[i] = (double)h_jr[i];
        for (int i = 0; i < nj; i++) d_jfl[i] = (double)h_jfl[i];
        for (int i = 0; i < nj*2; i++) d_jsr_lim[i] = (double)h_jsr_lim[i];
        for (int i = 0; i < nj*5; i++) d_jsi_lim[i] = (double)h_jsi_lim[i];
        if (!upload_array(&dm.jnt_range_d, d_jr.data(), nj*2)) return false;
        if (!upload_array(&dm.jnt_frictionloss_d, d_jfl.data(), nj)) return false;
        if (!upload_array(&dm.jnt_solref_lim_d, d_jsr_lim.data(), nj*2)) return false;
        if (!upload_array(&dm.jnt_solimp_lim_d, d_jsi_lim.data(), nj*5)) return false;
    }
    if (!upload_array(&dm.jnt_limited, h_jlim.data(), nj)) return false;
    if (!upload_array(&dm.qpos0, desc_.qpos0, desc_.nq)) return false;
    {
        int nq_val = desc_.nq;
        std::vector<double> d_qpos0(nq_val);
        for (int i = 0; i < nq_val; i++) d_qpos0[i] = (double)desc_.qpos0[i];
        if (!upload_array(&dm.qpos0_d, d_qpos0.data(), nq_val)) return false;
    }

    // ── Geom arrays ─────────────────────────────────────────────────────
    std::vector<int>   h_gt(ng), h_gb(ng), h_gct(ng), h_gca(ng), h_gcd(ng), h_gos(ng, 0);
    bool any_onesided = false;
    std::vector<float> h_gp(ng*3), h_gq(ng*4), h_gs(ng*3), h_gf(ng*3), h_gc(ng*4);
    std::vector<float> h_gsr(ng*2), h_gsi(ng*5), h_gmargin(ng, 0.0f), h_ggap(ng, 0.0f);
    for (int i = 0; i < ng; i++) {
        const GeomDesc& g = desc_.geoms[i];
        h_gt[i] = g.type; h_gb[i] = g.body;
        memcpy(&h_gp[i*3], g.pos, 12); memcpy(&h_gq[i*4], g.quat, 16);
        memcpy(&h_gs[i*3], g.size, 12); memcpy(&h_gf[i*3], g.friction, 12);
        memcpy(&h_gc[i*4], g.rgba, 16);
        h_gct[i] = g.contype; h_gca[i] = g.conaffinity; h_gcd[i] = g.condim;
        memcpy(&h_gsr[i*2], g.solref, 8);
        memcpy(&h_gsi[i*5], g.solimp, 20);
        h_gmargin[i] = g.margin;
        h_ggap[i] = g.gap;
        h_gos[i] = g.onesided;
        any_onesided |= g.onesided != 0;
    }
    dm.geom_onesided = nullptr;
    if (any_onesided && !upload_array(&dm.geom_onesided, h_gos.data(), ng)) return false;
    if (!upload_array(&dm.geom_type, h_gt.data(), ng)) return false;
    if (!upload_array(&dm.geom_body, h_gb.data(), ng)) return false;
    if (!upload_array(&dm.geom_pos, h_gp.data(), ng*3)) return false;
    if (!upload_array(&dm.geom_quat, h_gq.data(), ng*4)) return false;
    if (!upload_array(&dm.geom_size, h_gs.data(), ng*3)) return false;
    if (!upload_array(&dm.geom_friction, h_gf.data(), ng*3)) return false;
    if (!upload_array(&dm.geom_rgba, h_gc.data(), ng*4)) return false;
    if (!upload_array(&dm.geom_contype, h_gct.data(), ng)) return false;
    if (!upload_array(&dm.geom_conaffinity, h_gca.data(), ng)) return false;
    if (!upload_array(&dm.geom_condim, h_gcd.data(), ng)) return false;
    dm.max_condim = 1;
    for (int i = 0; i < ng; i++) dm.max_condim = std::max(dm.max_condim, (int)h_gcd[i]);
    if (!upload_array(&dm.geom_solref, h_gsr.data(), ng*2)) return false;
    if (!upload_array(&dm.geom_solimp, h_gsi.data(), ng*5)) return false;
    if (!upload_array(&dm.geom_margin, h_gmargin.data(), ng)) return false;
    if (!upload_array(&dm.geom_gap, h_ggap.data(), ng)) return false;
    {
        std::vector<double> d_gp(ng*3), d_gq(ng*4), d_gs(ng*3), d_gf(ng*3);
        std::vector<double> d_gsr(ng*2), d_gsi(ng*5);
        for (int i = 0; i < ng*3; i++) d_gp[i] = (double)h_gp[i];
        for (int i = 0; i < ng*4; i++) d_gq[i] = (double)h_gq[i];
        for (int i = 0; i < ng*3; i++) d_gs[i] = (double)h_gs[i];
        for (int i = 0; i < ng*3; i++) d_gf[i] = (double)h_gf[i];
        for (int i = 0; i < ng*2; i++) d_gsr[i] = (double)h_gsr[i];
        for (int i = 0; i < ng*5; i++) d_gsi[i] = (double)h_gsi[i];
        if (!upload_array(&dm.geom_pos_d, d_gp.data(), ng*3)) return false;
        if (!upload_array(&dm.geom_quat_d, d_gq.data(), ng*4)) return false;
        if (!upload_array(&dm.geom_size_d, d_gs.data(), ng*3)) return false;
        if (!upload_array(&dm.geom_friction_d, d_gf.data(), ng*3)) return false;
        if (!upload_array(&dm.geom_solref_d, d_gsr.data(), ng*2)) return false;
        if (!upload_array(&dm.geom_solimp_d, d_gsi.data(), ng*5)) return false;
    }

    // Per-geom material properties (specular, shininess, emission)
    if (!upload_array(&dm.geom_specular, desc_.geom_specular, ng)) return false;
    if (!upload_array(&dm.geom_shininess, desc_.geom_shininess, ng)) return false;
    if (!upload_array(&dm.geom_emission, desc_.geom_emission, ng)) return false;
    {
        // The legacy optional device-side approximation retains its first
        // reflector gate. Preserve host material values so the real mirror
        // pass can choose the first reflector visible under geom-group flags.
        std::vector<float> device_reflectance(desc_.geom_reflectance, desc_.geom_reflectance + ng);
        int kept = -1;
        for (int i = 0; i < ng; i++) {
            const GeomDesc& gd = desc_.geoms[i];
            bool eligible = (gd.type == GEOM_PLANE || gd.type == GEOM_BOX) &&
                            (gd.rgba[3] >= 1.0f) &&
                            (desc_.geom_reflectance[i] > 0.0f);
            if (eligible && kept < 0) {
                kept = i;                      // first one wins
            } else {
                device_reflectance[i] = 0.0f;
            }
        }
        if (kept >= 0) {
            fprintf(stderr, "instinct_sim: reflective geom %d '%s' reflectance %.3f\n",
                    kept, desc_.geoms[kept].name, desc_.geom_reflectance[kept]);
        }
        if (!upload_array(&dm.geom_reflectance, device_reflectance.data(), ng)) return false;
    }

    // ── Actuator arrays ─────────────────────────────────────────────────
    std::vector<int>   h_aj(na), h_at(na);
    std::vector<float> h_ag(na), h_akv(na), h_abp(na*3), h_afr(na*2), h_acr(na*2);
    for (int i = 0; i < na; i++) {
        const ActuatorDesc& a = desc_.actuators[i];
        h_aj[i] = a.jnt; h_at[i] = a.type;
        h_ag[i] = a.gain; h_akv[i] = a.kv;
        memcpy(&h_abp[i*3], a.biasprm, 12);
        memcpy(&h_afr[i*2], a.forcerange, 8);
        memcpy(&h_acr[i*2], a.ctrlrange, 8);
    }
    if (!upload_array(&dm.actuator_jnt, h_aj.data(), na)) return false;
    if (!upload_array(&dm.actuator_type, h_at.data(), na)) return false;
    if (!upload_array(&dm.actuator_gain, h_ag.data(), na)) return false;
    if (!upload_array(&dm.actuator_kv, h_akv.data(), na)) return false;
    if (!upload_array(&dm.actuator_biasprm, h_abp.data(), na*3)) return false;
    if (!upload_array(&dm.actuator_forcerange, h_afr.data(), na*2)) return false;
    if (!upload_array(&dm.actuator_ctrlrange, h_acr.data(), na*2)) return false;
    {
        std::vector<double> d_ag(na), d_akv(na), d_abp(na*3), d_afr(na*2), d_acr(na*2);
        for (int i = 0; i < na; i++) {
            const ActuatorDesc& a = desc_.actuators[i];
            d_ag[i] = a.gain_d; d_akv[i] = a.kv_d;
            memcpy(&d_abp[i*3], a.biasprm_d, 3*sizeof(double));
            memcpy(&d_afr[i*2], a.forcerange_d, 2*sizeof(double));
            memcpy(&d_acr[i*2], a.ctrlrange_d, 2*sizeof(double));
        }
        if (!upload_array(&dm.actuator_gain_d, d_ag.data(), na)) return false;
        if (!upload_array(&dm.actuator_kv_d, d_akv.data(), na)) return false;
        if (!upload_array(&dm.actuator_biasprm_d, d_abp.data(), na*3)) return false;
        if (!upload_array(&dm.actuator_forcerange_d, d_afr.data(), na*2)) return false;
        if (!upload_array(&dm.actuator_ctrlrange_d, d_acr.data(), na*2)) return false;
    }

    // ── Equality constraint arrays ──────────────────────────────────────
    int neq = desc_.nequality;
    if (neq > 0) {
        std::vector<int>   h_et(neq), h_eb1(neq), h_eb2(neq);
        std::vector<float> h_ed(neq * 6);
        for (int i = 0; i < neq; i++) {
            const EqualityDesc& e = desc_.equalities[i];
            h_et[i] = e.type; h_eb1[i] = e.body1; h_eb2[i] = e.body2;
            memcpy(&h_ed[i*6], e.data, 24);
        }
        if (!upload_array(&dm.eq_type, h_et.data(), neq)) return false;
        if (!upload_array(&dm.eq_body1, h_eb1.data(), neq)) return false;
        if (!upload_array(&dm.eq_body2, h_eb2.data(), neq)) return false;
        if (!upload_array(&dm.eq_data, h_ed.data(), neq*6)) return false;
    }

    // ── Pair filters ────────────────────────────────────────────────────
    if (desc_.npair_filter > 0) {
        if (!upload_array(&dm.pair_filter, desc_.pair_filter, desc_.npair_filter)) return false;
    }

    // ── Camera arrays ───────────────────────────────────────────────────
    int nc = desc_.ncamera;
    if (nc > 0) {
        std::vector<int>   h_cb(nc);
        std::vector<float> h_cp(nc*3), h_cq(nc*4), h_cf(nc);
        for (int i = 0; i < nc; i++) {
            const CameraDesc& c = desc_.cameras[i];
            h_cb[i] = c.body;
            memcpy(&h_cp[i*3], c.pos, 12); memcpy(&h_cq[i*4], c.quat, 16);
            h_cf[i] = c.fovy;
        }
        if (!upload_array(&dm.cam_body, h_cb.data(), nc)) return false;
        if (!upload_array(&dm.cam_pos, h_cp.data(), nc*3)) return false;
        if (!upload_array(&dm.cam_quat, h_cq.data(), nc*4)) return false;
        if (!upload_array(&dm.cam_fovy, h_cf.data(), nc)) return false;
    }

    // ── Site arrays ────────────────────────────────────────────────────
    dm.nsite = desc_.nsite;
    int ns = desc_.nsite;
    if (ns > 0) {
        std::vector<int>   h_sb(ns), h_st(ns), h_sg(ns);
        std::vector<float> h_sp(ns*3), h_ss(ns*3), h_sr(ns*4), h_sq(ns*4);
        std::vector<double> h_sp_d(ns*3), h_sq_d(ns*4);
        for (int i = 0; i < ns; i++) {
            const SiteDesc& s = desc_.sites[i];
            h_sb[i] = s.body;
            h_st[i] = s.type;
            h_sg[i] = s.group;
            memcpy(&h_sp[i*3], s.pos, 12);
            memcpy(&h_ss[i*3], s.size, 12);
            memcpy(&h_sr[i*4], s.rgba, 16);
            memcpy(&h_sq[i*4], s.quat, 16);
            std::copy(s.pos_d, s.pos_d + 3, h_sp_d.begin() + i*3);
            std::copy(s.quat_d, s.quat_d + 4, h_sq_d.begin() + i*4);
        }
        if (!upload_array(&dm.site_body, h_sb.data(), ns)) return false;
        if (!upload_array(&dm.site_type, h_st.data(), ns)) return false;
        if (!upload_array(&dm.site_group, h_sg.data(), ns)) return false;
        if (!upload_array(&dm.site_pos, h_sp.data(), ns*3)) return false;
        if (!upload_array(&dm.site_size, h_ss.data(), ns*3)) return false;
        if (!upload_array(&dm.site_rgba, h_sr.data(), ns*4)) return false;
        if (!upload_array(&dm.site_quat, h_sq.data(), ns*4)) return false;
        if (!upload_array(&dm.site_pos_d, h_sp_d.data(), ns*3)) return false;
        if (!upload_array(&dm.site_quat_d, h_sq_d.data(), ns*4)) return false;
    }

    // ── Mesh arrays (for rendering) ─────────────────────────────────────
    dm.nmesh = desc_.nmesh;
    dm.mesh_total_verts = desc_.mesh_total_verts;
    if (desc_.nmesh > 0 && desc_.mesh_total_verts > 0) {
        std::vector<int> h_mva(desc_.nmesh), h_mvn(desc_.nmesh);
        for (int i = 0; i < desc_.nmesh; i++) {
            h_mva[i] = desc_.meshes[i].vert_adr;
            h_mvn[i] = desc_.meshes[i].vert_num;
        }
        if (!upload_array(&dm.mesh_vert_adr, h_mva.data(), desc_.nmesh)) return false;
        if (!upload_array(&dm.mesh_vert_num, h_mvn.data(), desc_.nmesh)) return false;
        if (!upload_array(&dm.mesh_vert, desc_.mesh_vert, desc_.mesh_total_verts * 3)) return false;
        if (!upload_array(&dm.mesh_normal, desc_.mesh_normal, desc_.mesh_total_verts * 3)) return false;
        if (desc_.mesh_texcoord) {
            if (!upload_array(&dm.mesh_texcoord, desc_.mesh_texcoord, desc_.mesh_total_verts * MESH_UV_COMPONENTS)) return false;
        }
    }

    // ── Convex hull collision arrays ─────────────────────────────────────
    dm.convex_total_verts = desc_.convex_total_verts;
    dm.convex_total_faces = desc_.convex_total_faces;
    dm.convex_total_edges = desc_.convex_total_edges;
    if (desc_.nmesh > 0 && desc_.convex_total_verts > 0) {
        int nm = desc_.nmesh;
        std::vector<int> h_cva(nm), h_cvn(nm), h_cfa(nm), h_cfn(nm), h_cea(nm), h_cen(nm);
        for (int i = 0; i < nm; i++) {
            h_cva[i] = desc_.mesh_convex[i].vert_adr;
            h_cvn[i] = desc_.mesh_convex[i].vert_num;
            h_cfa[i] = desc_.mesh_convex[i].face_adr;
            h_cfn[i] = desc_.mesh_convex[i].face_num;
            h_cea[i] = desc_.mesh_convex[i].edge_adr;
            h_cen[i] = desc_.mesh_convex[i].edge_num;
        }
        if (!upload_array(&dm.mesh_convex_vert_adr, h_cva.data(), nm)) return false;
        if (!upload_array(&dm.mesh_convex_vert_num, h_cvn.data(), nm)) return false;
        if (!upload_array(&dm.mesh_convex_vert,     desc_.convex_hull_vert,        desc_.convex_total_verts * 3)) return false;
        if (!upload_array(&dm.mesh_convex_face_adr, h_cfa.data(), nm)) return false;
        if (!upload_array(&dm.mesh_convex_face_num, h_cfn.data(), nm)) return false;
        if (!upload_array(&dm.mesh_convex_face,     desc_.convex_hull_face,        desc_.convex_total_faces * 3)) return false;
        if (!upload_array(&dm.mesh_convex_face_normal, desc_.convex_hull_face_normal, desc_.convex_total_faces * 3)) return false;
        if (!upload_array(&dm.mesh_convex_edge_adr, h_cea.data(), nm)) return false;
        if (!upload_array(&dm.mesh_convex_edge_num, h_cen.data(), nm)) return false;
        if (!upload_array(&dm.mesh_convex_edge,     desc_.convex_hull_edge,        desc_.convex_total_edges * 2)) return false;
    }

    // Use the same mip/cube metadata path for initial model uploads and later
    // Python set_textures() calls.
    dm.ntexture = desc_.ntexture;
    if (desc_.ntexture > 0 && desc_.texture_data && desc_.texture_data_size > 0) {
        if (!upload_textures()) return false;
    } else {
        if (!upload_array(&dm.geom_texid, desc_.geom_texid, ng)) return false;
    }

    // Per-geom mesh id and group
    std::vector<int> h_gmi(ng), h_ggrp(ng);
    for (int i = 0; i < ng; i++) {
        h_gmi[i] = desc_.geoms[i].mesh_id;
        h_ggrp[i] = desc_.geoms[i].group;
    }
    if (!upload_array(&dm.geom_meshid, h_gmi.data(), ng)) return false;
    if (!upload_array(&dm.geom_group, h_ggrp.data(), ng)) return false;

    return true;
}

static void safe_free(void* ptr) {
    if (ptr) hipFree(ptr);
}

void PhysicsModel::free_device() {
    if (!uploaded_) return;
    DevicePhysicsModel& dm = d_model_;
    safe_free(dm.body_parent); safe_free(dm.body_weldid);
    safe_free(dm.body_jnt_adr); safe_free(dm.body_jnt_num);
    safe_free(dm.body_pos); safe_free(dm.body_quat); safe_free(dm.body_mass);
    safe_free(dm.body_inertia); safe_free(dm.body_ipos); safe_free(dm.body_iquat);
    safe_free(dm.body_mass_d); safe_free(dm.body_inertia_d); safe_free(dm.body_ipos_d);
    safe_free(dm.body_iquat_d); safe_free(dm.body_pos_d); safe_free(dm.body_quat_d);
    safe_free(dm.body_pos_env); safe_free(dm.body_quat_env);
    safe_free(dm.body_pos_env_d); safe_free(dm.body_quat_env_d);
    dm.body_pose_nenv = 0;
    safe_free(dm.body_invweight0); safe_free(dm.body_invweight0_d);
    safe_free(dm.gravcomp_body); safe_free(dm.gravcomp_scale_d);
    dm.n_gravcomp = 0;
    safe_free(dm.dof_invweight0); safe_free(dm.dof_invweight0_d);
    safe_free(dm.level_start); safe_free(dm.level_body_ids);
    safe_free(dm.body_chain_start); safe_free(dm.body_chain_len);
    safe_free(dm.body_chain_jnt_idx); safe_free(dm.body_chain_jnt_body);
    safe_free(dm.body_chain_parent);
    safe_free(dm.jnt_type); safe_free(dm.jnt_body); safe_free(dm.jnt_qpos_adr);
    safe_free(dm.jnt_dof_adr); safe_free(dm.jnt_axis); safe_free(dm.jnt_pos);
    safe_free(dm.jnt_axis_d); safe_free(dm.jnt_pos_d);
    safe_free(dm.jnt_damping_d); safe_free(dm.jnt_armature_d);
    safe_free(dm.jnt_range); safe_free(dm.jnt_range_d);
    safe_free(dm.jnt_stiffness); safe_free(dm.jnt_stiffness_d); safe_free(dm.jnt_damping);
    safe_free(dm.jnt_armature); safe_free(dm.jnt_frictionloss); safe_free(dm.jnt_frictionloss_d);
    safe_free(dm.jnt_solref_lim); safe_free(dm.jnt_solref_lim_d);
    safe_free(dm.jnt_solimp_lim); safe_free(dm.jnt_solimp_lim_d);
    safe_free(dm.jnt_limited);
    safe_free(dm.qpos0); safe_free(dm.qpos0_d);
    safe_free(dm.geom_type); safe_free(dm.geom_body); safe_free(dm.geom_pos);
    safe_free(dm.geom_quat); safe_free(dm.geom_size); safe_free(dm.geom_friction);
    safe_free(dm.geom_rgba);
    safe_free(dm.geom_pos_d); safe_free(dm.geom_quat_d);
    safe_free(dm.geom_size_d); safe_free(dm.geom_friction_d);
    safe_free(dm.geom_contype); safe_free(dm.geom_conaffinity); safe_free(dm.geom_condim);
    safe_free(dm.geom_solref); safe_free(dm.geom_solref_d);
    safe_free(dm.geom_solimp); safe_free(dm.geom_solimp_d);
    safe_free(dm.geom_margin); safe_free(dm.geom_gap); safe_free(dm.geom_onesided);
    safe_free(dm.actuator_jnt); safe_free(dm.actuator_type);
    safe_free(dm.actuator_gain); safe_free(dm.actuator_kv);
    safe_free(dm.actuator_biasprm);
    safe_free(dm.actuator_forcerange); safe_free(dm.actuator_ctrlrange);
    safe_free(dm.actuator_gain_d); safe_free(dm.actuator_kv_d);
    safe_free(dm.actuator_biasprm_d);
    safe_free(dm.actuator_forcerange_d); safe_free(dm.actuator_ctrlrange_d);
    safe_free(dm.eq_type); safe_free(dm.eq_body1); safe_free(dm.eq_body2); safe_free(dm.eq_data);
    safe_free(dm.pair_filter);
    safe_free(dm.cam_body); safe_free(dm.cam_pos); safe_free(dm.cam_quat); safe_free(dm.cam_fovy);
    safe_free(dm.site_body); safe_free(dm.site_type); safe_free(dm.site_group);
    safe_free(dm.site_pos); safe_free(dm.site_size); safe_free(dm.site_rgba); safe_free(dm.site_quat);
    safe_free(dm.site_pos_d); safe_free(dm.site_quat_d);
    safe_free(dm.mesh_vert_adr); safe_free(dm.mesh_vert_num);
    safe_free(dm.mesh_vert); safe_free(dm.mesh_normal); safe_free(dm.mesh_texcoord);
    safe_free(dm.mesh_convex_vert_adr); safe_free(dm.mesh_convex_vert_num);
    safe_free(dm.mesh_convex_vert);
    safe_free(dm.mesh_convex_face_adr); safe_free(dm.mesh_convex_face_num);
    safe_free(dm.mesh_convex_face); safe_free(dm.mesh_convex_face_normal);
    safe_free(dm.mesh_convex_edge_adr); safe_free(dm.mesh_convex_edge_num);
    safe_free(dm.mesh_convex_edge);
    safe_free(dm.geom_meshid); safe_free(dm.geom_group);
    safe_free(dm.tex_width); safe_free(dm.tex_height); safe_free(dm.tex_offset);
    safe_free(dm.tex_iscube); safe_free(dm.tex_mip); safe_free(dm.geom_texrepeat);
    safe_free(dm.tex_data); safe_free(dm.geom_texid);
    memset(&dm, 0, sizeof(dm));
    uploaded_ = false;
}

} // namespace physics
} // namespace hipgl
