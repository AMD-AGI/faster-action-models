// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// mesh_loader.cpp - STL/OBJ/MSH file parsers for GPU mesh rendering.
//
// Supports:
//   STL binary - Robot collision meshes
//   OBJ text   - Robot visual meshes
//   MSH binary - MuJoCo compiled mesh format (LIBERO objects)

#include "hipgl/physics/mesh_loader.h"
#include "hipgl/types.h"   // MESH_UV_COMPONENTS
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace hipgl {
namespace physics {

static void compute_face_normal(const float* v0, const float* v1, const float* v2, float* n) {
    float e1[3] = {v1[0]-v0[0], v1[1]-v0[1], v1[2]-v0[2]};
    float e2[3] = {v2[0]-v0[0], v2[1]-v0[1], v2[2]-v0[2]};
    n[0] = e1[1]*e2[2] - e1[2]*e2[1];
    n[1] = e1[2]*e2[0] - e1[0]*e2[2];
    n[2] = e1[0]*e2[1] - e1[1]*e2[0];
    float len = sqrtf(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
    if (len > 1e-12f) { n[0]/=len; n[1]/=len; n[2]/=len; }
}

static void smooth_stl_normals(MeshData& out) {
    const size_t nv = out.nvertices;
    const size_t nf = out.nfaces;
    if (nv == 0 || nf == 0) return;

    {
        const char* off = getenv("HIPGL_STL_SMOOTH");
        if (off && strcmp(off, "0") == 0) return;
    }

    // Weld by quantised position. 1e-6 m is well below any LIBERO mesh feature
    // and well above float round-trip error on these coordinates.
    struct Key {
        int64_t x, y, z;
        bool operator==(const Key& o) const { return x==o.x && y==o.y && z==o.z; }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const {
            size_t h = (size_t)(k.x * 73856093) ^ (size_t)(k.y * 19349663)
                     ^ (size_t)(k.z * 83492791);
            return h;
        }
    };
    const double QUANT = 1e6;   // 1e-6 m buckets
    std::unordered_map<Key, int, KeyHash> weld;
    weld.reserve(nv * 2);
    std::vector<int> vmap(nv);
    int nweld = 0;
    for (size_t v = 0; v < nv; v++) {
        Key k{ (int64_t)llround((double)out.vertices[v*3+0] * QUANT),
               (int64_t)llround((double)out.vertices[v*3+1] * QUANT),
               (int64_t)llround((double)out.vertices[v*3+2] * QUANT) };
        auto it = weld.find(k);
        if (it == weld.end()) { weld.emplace(k, nweld); vmap[v] = nweld++; }
        else                  { vmap[v] = it->second; }
    }

    // Per-face area-weighted normal. The cross product's magnitude IS twice the
    // triangle area, so leaving it unnormalised applies the area weight for free.
    std::vector<double> fnrm(nf * 3, 0.0);
    for (size_t i = 0; i < nf; i++) {
        const int i0 = out.faces[i*3+0], i1 = out.faces[i*3+1], i2 = out.faces[i*3+2];
        if (i0 < 0 || i1 < 0 || i2 < 0) continue;
        const float* a = &out.vertices[(size_t)i0*3];
        const float* b = &out.vertices[(size_t)i1*3];
        const float* c = &out.vertices[(size_t)i2*3];
        const double e1[3] = {(double)b[0]-a[0], (double)b[1]-a[1], (double)b[2]-a[2]};
        const double e2[3] = {(double)c[0]-a[0], (double)c[1]-a[1], (double)c[2]-a[2]};
        fnrm[i*3+0] = e1[1]*e2[2] - e1[2]*e2[1];
        fnrm[i*3+1] = e1[2]*e2[0] - e1[0]*e2[2];
        fnrm[i*3+2] = e1[0]*e2[1] - e1[1]*e2[0];
    }

    // PASS 1: unrestricted area-weighted sum per welded vertex. This is only a
    // REFERENCE direction for the crease test, not the output.
    std::vector<double> ref((size_t)nweld * 3, 0.0);
    for (size_t i = 0; i < nf; i++) {
        for (int j = 0; j < 3; j++) {
            const int w = vmap[out.faces[i*3+j]];
            ref[(size_t)w*3+0] += fnrm[i*3+0];
            ref[(size_t)w*3+1] += fnrm[i*3+1];
            ref[(size_t)w*3+2] += fnrm[i*3+2];
        }
    }
    for (int w = 0; w < nweld; w++) {
        double* r = &ref[(size_t)w*3];
        const double l = sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
        if (l > 1e-20) { r[0]/=l; r[1]/=l; r[2]/=l; }
    }

    // PASS 2: re-accumulate, but only faces whose normal is within CREASE_DEG of
    // the reference. Averaging blindly across a hard edge destroys it, which is
    // why plain area weighting fit hand_vis (31% hard-edge verts) far better
    // than finger_vis (89%). Scored against mjModel.mesh_normal, mean deg:
    //
    //   mesh                    flat   area   cr30   CR45   cr60
    //   gripper0_hand_vis      19.94   5.13   3.18   2.44   3.59
    //   gripper0_finger_vis    46.51  18.35  11.54   8.30  11.12
    //   flat_stove_burnerplate 40.93  13.84   3.96   1.02   9.61
    //   robot0_link0 (OBJ ctl)  0.07   0.06   0.06   0.06   0.06
    //
    // 45 deg is the minimum for all three STL meshes simultaneously, and the
    // curve is shallow either side, so it is a basin rather than a fitted point.
    const double CREASE_DEG = 45.0;
    const double cos_thr = cos(CREASE_DEG * 3.14159265358979323846 / 180.0);

    // GATE: bail out on predominantly hard-edged meshes (see header comment).
    // A vertex is "hard" when any incident face deviates more than 60 deg from
    // the vertex reference; if most of the mesh is hard, Gouraud interpolation
    // of the smoothed normals does more damage than flat shading. Computed from
    // the mesh alone -- no name lists, so a new asset is classified on its own
    // geometry.
    {
        long hard = 0, counted = 0;
        std::vector<int> incident((size_t)nweld, 0);
        std::vector<unsigned char> is_hard((size_t)nweld, 0);
        const double cos_hard = cos(60.0 * 3.14159265358979323846 / 180.0);
        for (size_t i = 0; i < nf; i++) {
            const double* fnv = &fnrm[i*3];
            const double fl = sqrt(fnv[0]*fnv[0] + fnv[1]*fnv[1] + fnv[2]*fnv[2]);
            if (fl <= 1e-20) continue;
            for (int j = 0; j < 3; j++) {
                const int w = vmap[out.faces[i*3+j]];
                incident[w]++;
                const double* r = &ref[(size_t)w*3];
                if ((fnv[0]*r[0] + fnv[1]*r[1] + fnv[2]*r[2]) / fl < cos_hard)
                    is_hard[w] = 1;
            }
        }
        for (int w = 0; w < nweld; w++) {
            if (incident[w] < 2) continue;      // boundary/isolated: not informative
            counted++;
            if (is_hard[w]) hard++;
        }
        if (counted > 0 && (double)hard / (double)counted > 0.60) return;
    }
    std::vector<double> acc((size_t)nweld * 3, 0.0);
    for (size_t i = 0; i < nf; i++) {
        const double* fnv = &fnrm[i*3];
        const double fl = sqrt(fnv[0]*fnv[0] + fnv[1]*fnv[1] + fnv[2]*fnv[2]);
        if (fl <= 1e-20) continue;                       // zero-area: no weight
        for (int j = 0; j < 3; j++) {
            const int w = vmap[out.faces[i*3+j]];
            const double* r = &ref[(size_t)w*3];
            // unit face normal . unit reference
            if ((fnv[0]*r[0] + fnv[1]*r[1] + fnv[2]*r[2]) / fl < cos_thr) continue;
            acc[(size_t)w*3+0] += fnv[0];
            acc[(size_t)w*3+1] += fnv[1];
            acc[(size_t)w*3+2] += fnv[2];
        }
    }

    for (size_t v = 0; v < nv; v++) {
        const int w = vmap[v];
        const double x = acc[(size_t)w*3+0], y = acc[(size_t)w*3+1], z = acc[(size_t)w*3+2];
        const double len = sqrt(x*x + y*y + z*z);
        if (len <= 1e-20) continue;   // cancelled: keep the original facet normal
        out.normals[v*3+0] = (float)(x / len);
        out.normals[v*3+1] = (float)(y / len);
        out.normals[v*3+2] = (float)(z / len);
    }
}

// ── STL binary parser ─────────────────────────────────────────────────────

bool load_stl(const std::string& path, MeshData& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "mesh_loader: cannot open STL '%s'\n", path.c_str());
        return false;
    }

    char header[80];
    if (fread(header, 1, 80, f) != 80) { fclose(f); return false; }

    uint32_t ntri;
    if (fread(&ntri, sizeof(uint32_t), 1, f) != 1) { fclose(f); return false; }

    out.vertices.resize(ntri * 9);
    out.normals.resize(ntri * 9);
    out.faces.resize(ntri * 3);
    out.nvertices = ntri * 3;
    out.nfaces = ntri;

    for (uint32_t i = 0; i < ntri; i++) {
        float normal[3], v[3][3];
        uint16_t attr;

        if (fread(normal, sizeof(float), 3, f) != 3) { fclose(f); return false; }
        for (int j = 0; j < 3; j++) {
            if (fread(v[j], sizeof(float), 3, f) != 3) { fclose(f); return false; }
        }
        if (fread(&attr, sizeof(uint16_t), 1, f) != 1) { fclose(f); return false; }

        for (int j = 0; j < 3; j++) {
            int vi = i * 3 + j;
            out.vertices[vi*3+0] = v[j][0];
            out.vertices[vi*3+1] = v[j][1];
            out.vertices[vi*3+2] = v[j][2];
            out.normals[vi*3+0] = normal[0];
            out.normals[vi*3+1] = normal[1];
            out.normals[vi*3+2] = normal[2];
            out.faces[i*3+j] = vi;
        }
    }

    fclose(f);

    smooth_stl_normals(out);

    return true;
}

// ── OBJ text parser ───────────────────────────────────────────────────────

bool load_obj(const std::string& path, MeshData& out) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        fprintf(stderr, "mesh_loader: cannot open OBJ '%s'\n", path.c_str());
        return false;
    }

    std::vector<float> positions;
    std::vector<float> file_normals;
    std::vector<float> file_uvs;
    std::vector<int> face_pos_idx;
    std::vector<int> face_nrm_idx;
    std::vector<int> face_uv_idx;

    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string token;
        ss >> token;

        if (token == "v") {
            float x, y, z;
            ss >> x >> y >> z;
            positions.push_back(x);
            positions.push_back(y);
            positions.push_back(z);
        }
        else if (token == "vt") {
            // MuJoCo's OBJ decoder (plugin/obj_decoder) stores v' = 1 - v; textures reach
            // the renderer from MuJoCo's own tex_data, so follow the same convention.
            float u = 0.f, v = 0.f;
            ss >> u >> v;
            file_uvs.push_back(u);
            file_uvs.push_back(1.f - v);
        }
        else if (token == "vn") {
            float nx, ny, nz;
            ss >> nx >> ny >> nz;
            file_normals.push_back(nx);
            file_normals.push_back(ny);
            file_normals.push_back(nz);
        }
        else if (token == "f") {
            std::vector<int> fpi, fni, fti;
            std::string vert;
            while (ss >> vert) {
                int pi = 0, ti = 0, ni = 0;
                if (sscanf(vert.c_str(), "%d/%d/%d", &pi, &ti, &ni) >= 1 ||
                    sscanf(vert.c_str(), "%d//%d", &pi, &ni) >= 1 ||
                    sscanf(vert.c_str(), "%d", &pi) >= 1) {
                    fpi.push_back(pi > 0 ? pi - 1 : (int)(positions.size()/3) + pi);
                    fni.push_back(ni > 0 ? ni - 1 : -1);
                    fti.push_back(ti > 0 ? ti - 1 : (ti < 0 ? (int)(file_uvs.size()/2) + ti : -1));
                }
            }
            for (int t = 1; t + 1 < (int)fpi.size(); t++) {
                face_pos_idx.push_back(fpi[0]);
                face_pos_idx.push_back(fpi[t]);
                face_pos_idx.push_back(fpi[t+1]);
                face_nrm_idx.push_back(fni[0]);
                face_nrm_idx.push_back(fni[t]);
                face_nrm_idx.push_back(fni[t+1]);
                face_uv_idx.push_back(fti[0]);
                face_uv_idx.push_back(fti[t]);
                face_uv_idx.push_back(fti[t+1]);
            }
        }
    }

    int nvert = (int)face_pos_idx.size();
    int nface = nvert / 3;
    out.vertices.resize(nvert * 3);
    out.normals.resize(nvert * 3);
    out.faces.resize(nvert);
    out.nvertices = nvert;
    out.nfaces = nface;
    // Real texcoords only when every face corner references one.
    bool uv_ok = !file_uvs.empty() && nvert > 0;
    for (int i = 0; uv_ok && i < nvert; i++)
        uv_ok = face_uv_idx[i] >= 0 && face_uv_idx[i] * 2 + 1 < (int)file_uvs.size();
    if (uv_ok) {
        out.has_texcoords = true;
        out.texcoords.resize((size_t)nvert * MESH_UV_COMPONENTS);
        for (int i = 0; i < nvert; i++) {
            out.texcoords[i*MESH_UV_COMPONENTS+0] = file_uvs[face_uv_idx[i]*2+0];
            out.texcoords[i*MESH_UV_COMPONENTS+1] = file_uvs[face_uv_idx[i]*2+1];
        }
    }

    for (int i = 0; i < nvert; i++) {
        int pi = face_pos_idx[i];
        out.vertices[i*3+0] = positions[pi*3+0];
        out.vertices[i*3+1] = positions[pi*3+1];
        out.vertices[i*3+2] = positions[pi*3+2];

        int ni = face_nrm_idx[i];
        if (ni >= 0 && ni*3+2 < (int)file_normals.size()) {
            out.normals[i*3+0] = file_normals[ni*3+0];
            out.normals[i*3+1] = file_normals[ni*3+1];
            out.normals[i*3+2] = file_normals[ni*3+2];
        } else {
            int fi = i / 3;
            float* v0 = &out.vertices[(fi*3+0)*3];
            float* v1 = &out.vertices[(fi*3+1)*3];
            float* v2 = &out.vertices[(fi*3+2)*3];
            float n[3];
            compute_face_normal(v0, v1, v2, n);
            out.normals[i*3+0] = n[0];
            out.normals[i*3+1] = n[1];
            out.normals[i*3+2] = n[2];
        }

        out.faces[i] = i;
    }

    return true;
}

// ── MuJoCo MSH binary parser ──────────────────────────────────────────────
// Format: [nvertex(i32), nnormal(i32), ntexcoord(i32), nface(i32),
//          verts(3*nv floats), normals(3*nn floats),
//          texcoords(2*nt floats), faces(3*nf ints)]

bool load_msh(const std::string& path, MeshData& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "mesh_loader: cannot open MSH '%s'\n", path.c_str());
        return false;
    }

    int32_t nvertex, nnormal, ntexcoord, nface;
    if (fread(&nvertex, sizeof(int32_t), 1, f) != 1 ||
        fread(&nnormal, sizeof(int32_t), 1, f) != 1 ||
        fread(&ntexcoord, sizeof(int32_t), 1, f) != 1 ||
        fread(&nface, sizeof(int32_t), 1, f) != 1) {
        fclose(f);
        return false;
    }

    std::vector<float> raw_verts(nvertex * 3);
    std::vector<float> raw_normals(nnormal * 3);
    std::vector<float> raw_texcoords(ntexcoord * MESH_UV_COMPONENTS);
    std::vector<int32_t> raw_faces(nface * 3);

    if ((int)fread(raw_verts.data(), sizeof(float), nvertex*3, f) != nvertex*3) { fclose(f); return false; }
    if (nnormal > 0) {
        if ((int)fread(raw_normals.data(), sizeof(float), nnormal*3, f) != nnormal*3) { fclose(f); return false; }
    }
    if (ntexcoord > 0) {
        if ((int)fread(raw_texcoords.data(), sizeof(float), ntexcoord*MESH_UV_COMPONENTS, f) != ntexcoord*MESH_UV_COMPONENTS) { fclose(f); return false; }
    }
    if ((int)fread(raw_faces.data(), sizeof(int32_t), nface*3, f) != nface*3) { fclose(f); return false; }

    fclose(f);

    // MSH files are MuJoCo's compiled format -- vertices are already centered.
    // Do NOT apply additional centering.

    int total_verts = nface * 3;
    out.vertices.resize(total_verts * 3);
    out.normals.resize(total_verts * 3);
    out.faces.resize(total_verts);
    out.nvertices = total_verts;
    out.nfaces = nface;
    out.has_texcoords = (ntexcoord > 0);
    if (out.has_texcoords) out.texcoords.resize(total_verts * MESH_UV_COMPONENTS);

    for (int fi = 0; fi < nface; fi++) {
        for (int j = 0; j < 3; j++) {
            int vi_out = fi * 3 + j;
            int vi_in = raw_faces[fi * 3 + j];
            if (vi_in < 0 || vi_in >= nvertex) vi_in = 0;

            out.vertices[vi_out*3+0] = raw_verts[vi_in*3+0];
            out.vertices[vi_out*3+1] = raw_verts[vi_in*3+1];
            out.vertices[vi_out*3+2] = raw_verts[vi_in*3+2];

            if (nnormal > 0 && vi_in < nnormal) {
                out.normals[vi_out*3+0] = raw_normals[vi_in*3+0];
                out.normals[vi_out*3+1] = raw_normals[vi_in*3+1];
                out.normals[vi_out*3+2] = raw_normals[vi_in*3+2];
            } else {
                float* v0 = &out.vertices[(fi*3+0)*3];
                float* v1 = &out.vertices[(fi*3+1)*3];
                float* v2 = &out.vertices[(fi*3+2)*3];
                float n[3];
                compute_face_normal(v0, v1, v2, n);
                out.normals[vi_out*3+0] = n[0];
                out.normals[vi_out*3+1] = n[1];
                out.normals[vi_out*3+2] = n[2];
            }

            if (ntexcoord > 0 && vi_in < ntexcoord) {
                out.texcoords[vi_out*MESH_UV_COMPONENTS+0] = raw_texcoords[vi_in*MESH_UV_COMPONENTS+0];
                out.texcoords[vi_out*MESH_UV_COMPONENTS+1] = raw_texcoords[vi_in*MESH_UV_COMPONENTS+1];
            }

            out.faces[vi_out] = vi_out;
        }
    }

    return true;
}

// ── Generic loader (dispatches by extension) ──────────────────────────────

bool load_mesh(const std::string& path, MeshData& out) {
    size_t dot = path.rfind('.');
    if (dot == std::string::npos) {
        fprintf(stderr, "mesh_loader: no extension in '%s'\n", path.c_str());
        return false;
    }

    std::string ext = path.substr(dot + 1);
    for (auto& c : ext) c = (char)tolower(c);

    if (ext == "stl") return load_stl(path, out);
    if (ext == "obj") return load_obj(path, out);
    if (ext == "msh") return load_msh(path, out);

    fprintf(stderr, "mesh_loader: unsupported extension '%s'\n", ext.c_str());
    return false;
}

// ── Apply scale to mesh vertices and normals ─────────────────────────────

void scale_mesh(MeshData& mesh, float sx, float sy, float sz) {
    for (int i = 0; i < mesh.nvertices; i++) {
        mesh.vertices[i*3+0] *= sx;
        mesh.vertices[i*3+1] *= sy;
        mesh.vertices[i*3+2] *= sz;
    }

    // Normals transform by inverse-transpose of scale: n' = normalize(S^{-T} * n)
    float inv_sx = (fabsf(sx) > 1e-12f) ? 1.0f / sx : 0.0f;
    float inv_sy = (fabsf(sy) > 1e-12f) ? 1.0f / sy : 0.0f;
    float inv_sz = (fabsf(sz) > 1e-12f) ? 1.0f / sz : 0.0f;

    for (int i = 0; i < mesh.nvertices; i++) {
        float nx = mesh.normals[i*3+0] * inv_sx;
        float ny = mesh.normals[i*3+1] * inv_sy;
        float nz = mesh.normals[i*3+2] * inv_sz;
        float len = sqrtf(nx*nx + ny*ny + nz*nz);
        if (len > 1e-12f) { nx /= len; ny /= len; nz /= len; }
        mesh.normals[i*3+0] = nx;
        mesh.normals[i*3+1] = ny;
        mesh.normals[i*3+2] = nz;
    }
}

} // namespace physics
} // namespace hipgl
