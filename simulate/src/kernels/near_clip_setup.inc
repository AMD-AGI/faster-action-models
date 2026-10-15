// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Included by fused_vertex_tri.hip after the vertex transform and lighting
// helpers. Clipping is deliberately kept off the ordinary triangle path.

struct NearClipVertex {
    float clip[4];
    TransformedVertex vertex;
};

// OpenGL clips the already shaded vertex outputs. In particular, lighting at
// an intersection vertex must be interpolated from the original endpoints,
// not evaluated again at the intersection position.
HIPGL_DEVICE inline NearClipVertex near_clip_intersection(
    const NearClipVertex& va, const NearClipVertex& vb, float da, float db)
{
    // Always evaluate from the inside endpoint. Adjacent triangles traverse
    // their shared edge in opposite directions; canonical evaluation keeps
    // the generated position identical after the subpixel snap.
    const bool a_inside = da >= 0.f;
    const NearClipVertex& a = a_inside ? va : vb;
    const NearClipVertex& b = a_inside ? vb : va;
    const float inside_distance = a_inside ? da : db;
    const float outside_distance = a_inside ? db : da;
    const float t = inside_distance / (inside_distance - outside_distance);
    NearClipVertex out;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        out.clip[k] = a.clip[k] + t * (b.clip[k] - a.clip[k]);
        out.vertex.color[k] = a.vertex.color[k]
                            + t * (b.vertex.color[k] - a.vertex.color[k]);
        out.vertex.lit_color[k] = a.vertex.lit_color[k]
                                + t * (b.vertex.lit_color[k] - a.vertex.lit_color[k]);
    }
    // Avoid leaving an intersection a rounding ulp behind the clipping plane.
    out.clip[2] = -out.clip[3];
#pragma unroll
    for (int k = 0; k < 3; ++k) {
        out.vertex.world_pos[k] = a.vertex.world_pos[k]
                                + t * (b.vertex.world_pos[k] - a.vertex.world_pos[k]);
        out.vertex.world_normal[k] = a.vertex.world_normal[k]
                                   + t * (b.vertex.world_normal[k] - a.vertex.world_normal[k]);
        out.vertex.spec_color[k] = a.vertex.spec_color[k]
                                 + t * (b.vertex.spec_color[k] - a.vertex.spec_color[k]);
    }
#pragma unroll
    for (int k = 0; k < TEXCOORD_COMPONENTS; ++k)
        out.vertex.texcoord[k] = a.vertex.texcoord[k]
                               + t * (b.vertex.texcoord[k] - a.vertex.texcoord[k]);
    // Texture IDs are primitive attributes; all vertices of an HP primitive
    // carry the same ID. The numeric vertex outputs remain interpolated.
    out.vertex.texid = a.vertex.texid;
    out.vertex.mat_specular = a.vertex.mat_specular
                            + t * (b.vertex.mat_specular - a.vertex.mat_specular);
    out.vertex.mat_shininess = a.vertex.mat_shininess
                             + t * (b.vertex.mat_shininess - a.vertex.mat_shininess);
    out.vertex.mat_emission = a.vertex.mat_emission
                            + t * (b.vertex.mat_emission - a.vertex.mat_emission);
    return out;
}

HIPGL_DEVICE inline bool near_clip_project(
    NearClipVertex& vertex, int vp_x, int vp_y, int vp_w, int vp_h)
{
    const float w = vertex.clip[3];
    if (!(w > 0.f) || !isfinite(w)) return false;
    const float inverse_w = 1.f / w;
    const float x = vertex.clip[0] * inverse_w;
    const float y = vertex.clip[1] * inverse_w;
    const float z = vertex.clip[2] * inverse_w;
    if (!isfinite(x) || !isfinite(y) || !isfinite(z)) return false;
    vertex.vertex.inv_w = inverse_w;
    vertex.vertex.screen_z = z * .5f + .5f;
    vertex.vertex.screen_x = vp_x + (x * .5f + .5f) * static_cast<float>(vp_w);
    vertex.vertex.screen_y = vp_y + (1.f - (y * .5f + .5f)) * static_cast<float>(vp_h);
    vertex.vertex.clip_zw[0] = vertex.clip[2];
    vertex.vertex.clip_zw[1] = w;
    return true;
}

// The arithmetic and fill rule mirror the ordinary fused setup. This helper
// is used only after clipping, so the common fused kernel retains its existing
// register lifetime and floating-point expression graph.
HIPGL_DEVICE inline void near_clip_setup_triangle(
    const NearClipVertex& a, const NearClipVertex& b, const NearClipVertex& c,
    int first_vertex, int output_index, TriangleSetup& tri, int32_t* tri_stats)
{
    tri.first_vertex = first_vertex;
    tri.v[0] = a.vertex;
    tri.v[1] = b.vertex;
    tri.v[2] = c.vertex;
    constexpr float SUBPX = 256.f;
    constexpr float INV_SUBPX = 1.f / 256.f;
    const float x0 = roundf(a.vertex.screen_x * SUBPX) * INV_SUBPX;
    const float y0 = roundf(a.vertex.screen_y * SUBPX) * INV_SUBPX;
    const float x1 = roundf(b.vertex.screen_x * SUBPX) * INV_SUBPX;
    const float y1 = roundf(b.vertex.screen_y * SUBPX) * INV_SUBPX;
    const float x2 = roundf(c.vertex.screen_x * SUBPX) * INV_SUBPX;
    const float y2 = roundf(c.vertex.screen_y * SUBPX) * INV_SUBPX;
    const float e0x = x1 - x0, e0y = y1 - y0;
    const float e1x = x2 - x1, e1y = y2 - y1;
    const float e2x = x0 - x2, e2y = y0 - y2;
    const float area = e0x * e2y - e0y * e2x;
    hipgl_set_flags(tri, !isfinite(area) || area == 0.f || fabsf(area) < d_min_tri_area,
                         area < 0.f);
    if (tri_stats) {
        if (tri.degenerate)
            tri_stats[output_index * TRI_STAT_COUNT + TRI_STAT_DEGENERATE] = 1;
        if (tri.backfacing)
            tri_stats[output_index * TRI_STAT_COUNT + TRI_STAT_BACKFACE] = 1;
    }
    if (tri.degenerate) return;
    const float sign = area > 0.f ? -1.f : 1.f;
    const float A0 = sign * (-e0y), B0 = sign * e0x;
    const float A1 = sign * (-e1y), B1 = sign * e1x;
    const float A2 = sign * (-e2y), B2 = sign * e2x;
    tri.edge_eq[0][0] = A0;
    tri.edge_eq[0][1] = B0;
    tri.edge_eq[0][2] = (float)(-((double)A0 * (double)x0 + (double)B0 * (double)y0));
    tri.edge_eq[1][0] = A1;
    tri.edge_eq[1][1] = B1;
    tri.edge_eq[1][2] = (float)(-((double)A1 * (double)x1 + (double)B1 * (double)y1));
    tri.edge_eq[2][0] = A2;
    tri.edge_eq[2][1] = B2;
    tri.edge_eq[2][2] = (float)(-((double)A2 * (double)x2 + (double)B2 * (double)y2));
    constexpr float FILL_BIAS = -1.f / 65536.f;
    tri.edge_bias[0] = (A0 > 0.f || (A0 == 0.f && B0 > 0.f)) ? 0.f : FILL_BIAS;
    tri.edge_bias[1] = (A1 > 0.f || (A1 == 0.f && B1 > 0.f)) ? 0.f : FILL_BIAS;
    tri.edge_bias[2] = (A2 > 0.f || (A2 == 0.f && B2 > 0.f)) ? 0.f : FILL_BIAS;
    hipgl_set_bbox(tri, static_cast<int>(floorf(math::min3(x0, x1, x2))),
                        static_cast<int>(floorf(math::min3(y0, y1, y2))),
                        static_cast<int>(ceilf(math::max3(x0, x1, x2))),
                        static_cast<int>(ceilf(math::max3(y0, y1, y2))));
    tri.z_ord = hipgl_f2ord_setup(fminf(a.vertex.screen_z,
                               fminf(b.vertex.screen_z, c.vertex.screen_z)));
}

// Reconstruct only a queued straddler. The caller supplies both deterministic
// output slots, so primitive ordering/storage policy stays outside the clipper.
HIPGL_DEVICE inline void near_clip_emit_source_triangle(
    const float* positions, const float* normals,
    const float* texcoords, const int* texids, const float* colors,
    const float* spec_colors, const float* per_vert_specular,
    const float* per_vert_shininess, const float* per_vert_emission,
    const float* mvp, const float* model_view, const float* normal_mat,
    int vp_x, int vp_y, int vp_w, int vp_h, const RenderState& state,
    int source_index, int first_output, int second_output,
    TriangleSetup* triangles, int32_t* tri_stats,
    bool have_wp = false,
    float wp00 = 0.f, float wp01 = 0.f, float wp02 = 0.f,
    float wp10 = 0.f, float wp11 = 0.f, float wp12 = 0.f,
    float wp20 = 0.f, float wp21 = 0.f, float wp22 = 0.f,
    bool have_gq = false,
    float gq0 = 1.f, float gq1 = 0.f, float gq2 = 0.f, float gq3 = 0.f)
{
    hipgl_mark_degenerate(triangles[first_output]);
    hipgl_mark_degenerate(triangles[second_output]);
    triangles[first_output].first_vertex = source_index * 3;
    triangles[second_output].first_vertex = source_index * 3;
    const bool shade = state.gouraud_shading && state.lighting_enabled;
    const auto load_vertex = [&](int vertex_index, float wx, float wy, float wz) {
        NearClipVertex out;
        // ONE local position triple, ALWAYS the one handed downstream. When the
        // caller did not fold a pose, it is filled from exactly the global
        // address fused_transform_vertex_v would have read itself, so the bits
        // are the same either way and the `pos_in ? pos_in : global` select on
        // an alloca disappears.
        float posv[3];
        if (have_wp) {
            posv[0] = wx; posv[1] = wy; posv[2] = wz;
        } else {
            const float* gp = positions + vertex_index * 3;
            posv[0] = gp[0]; posv[1] = gp[1]; posv[2] = gp[2];
        }
        // Same form the main path uses (fused_vertex_tri.hip:1801): hp_qrot
        // into a local, then hand the THREE FLOATS over, never the pointer.
        float wn0 = 0.f, wn1 = 0.f, wn2 = 0.f;
        bool have_n = false;
        if (have_gq && normals) {
            const float gq[4] = { gq0, gq1, gq2, gq3 };
            const float* nm = normals + vertex_index * 3;
            const float ln[3] = { nm[0], nm[1], nm[2] };
            float wn[3];
            hp_qrot(gq, ln, wn);
            wn0 = wn[0]; wn1 = wn[1]; wn2 = wn[2];
            have_n = true;
        }
        const float x = posv[0], y = posv[1], z = posv[2];
        float* clip = out.clip;
        clip[0] = mvp[0]*x + mvp[4]*y + mvp[8]*z + mvp[12];
        clip[1] = mvp[1]*x + mvp[5]*y + mvp[9]*z + mvp[13];
        clip[2] = mvp[2]*x + mvp[6]*y + mvp[10]*z + mvp[14];
        clip[3] = mvp[3]*x + mvp[7]*y + mvp[11]*z + mvp[15];
        fused_transform_vertex_v(positions, normals, texcoords, texids, colors,
            spec_colors, per_vert_specular, per_vert_shininess, per_vert_emission,
            vertex_index, mvp, model_view, normal_mat,
            vp_x, vp_y, vp_w, vp_h, out.vertex, clip,
            posv, have_n, wn0, wn1, wn2);
        if (shade) compute_gouraud_lighting(out.vertex, state);
        return out;
    };
    NearClipVertex v0 = load_vertex(source_index * 3,     wp00, wp01, wp02);
    NearClipVertex v1 = load_vertex(source_index * 3 + 1, wp10, wp11, wp12);
    NearClipVertex v2 = load_vertex(source_index * 3 + 2, wp20, wp21, wp22);
    const float d0 = v0.clip[2] + v0.clip[3];
    const float d1 = v1.clip[2] + v1.clip[3];
    const float d2 = v2.clip[2] + v2.clip[3];
    const unsigned inside = (d0 >= 0.f ? 1u : 0u)
                          | (d1 >= 0.f ? 2u : 0u)
                          | (d2 >= 0.f ? 4u : 0u);

    const auto emit_triangle = [&](NearClipVertex& a, NearClipVertex& b,
                                   NearClipVertex& c) {
        if (!near_clip_project(a, vp_x, vp_y, vp_w, vp_h) ||
            !near_clip_project(b, vp_x, vp_y, vp_w, vp_h) ||
            !near_clip_project(c, vp_x, vp_y, vp_w, vp_h)) return;
        near_clip_setup_triangle(a, b, c, source_index * 3,
            first_output, triangles[first_output], tri_stats);
    };
    const auto emit_quad = [&](NearClipVertex& a, NearClipVertex& b,
                               NearClipVertex& c, NearClipVertex& d) {
        if (!near_clip_project(a, vp_x, vp_y, vp_w, vp_h) ||
            !near_clip_project(b, vp_x, vp_y, vp_w, vp_h) ||
            !near_clip_project(c, vp_x, vp_y, vp_w, vp_h) ||
            !near_clip_project(d, vp_x, vp_y, vp_w, vp_h)) return;
        near_clip_setup_triangle(a, b, c, source_index * 3,
            first_output, triangles[first_output], tri_stats);
        near_clip_setup_triangle(a, c, d, source_index * 3,
            second_output, triangles[second_output], tri_stats);
    };

    // These are the six mixed Sutherland-Hodgman results, traversing the
    // original edges 2->0, 0->1, 1->2. Keeping the vertices explicit preserves
    // the same fan/order while avoiding dynamically indexed local arrays:
    // those arrays caused per-dispatch GPU scratch allocation and reclamation.
    switch (inside) {
    case 1: {
        NearClipVertex i20 = near_clip_intersection(v2, v0, d2, d0);
        NearClipVertex i01 = near_clip_intersection(v0, v1, d0, d1);
        emit_triangle(i20, v0, i01);
        break;
    }
    case 2: {
        NearClipVertex i01 = near_clip_intersection(v0, v1, d0, d1);
        NearClipVertex i12 = near_clip_intersection(v1, v2, d1, d2);
        emit_triangle(i01, v1, i12);
        break;
    }
    case 4: {
        NearClipVertex i20 = near_clip_intersection(v2, v0, d2, d0);
        NearClipVertex i12 = near_clip_intersection(v1, v2, d1, d2);
        emit_triangle(i20, i12, v2);
        break;
    }
    case 3: {
        NearClipVertex i20 = near_clip_intersection(v2, v0, d2, d0);
        NearClipVertex i12 = near_clip_intersection(v1, v2, d1, d2);
        emit_quad(i20, v0, v1, i12);
        break;
    }
    case 5: {
        NearClipVertex i01 = near_clip_intersection(v0, v1, d0, d1);
        NearClipVertex i12 = near_clip_intersection(v1, v2, d1, d2);
        emit_quad(v0, i01, i12, v2);
        break;
    }
    case 6: {
        NearClipVertex i20 = near_clip_intersection(v2, v0, d2, d0);
        NearClipVertex i01 = near_clip_intersection(v0, v1, d0, d1);
        emit_quad(i20, i01, v1, v2);
        break;
    }
    case 7:
        emit_triangle(v0, v1, v2);
        break;
    default:
        break;
    }
}


static_assert(TEXCOORD_COMPONENTS == 3,
              "nc_g clip scratch assumes the 3-component texcoord layout");

#define NC_G_FENCE() __asm__ __volatile__("" ::: "memory")

// Slot s in [0,5]: 0..2 are triangles[first_output].v[0..2], 3..5 are
// triangles[second_output].v[0..2]. Slot s's clip[4] is parked in that same
// record's edge_eq[j][0..2] / edge_bias[j], which the triangle setup at the end
// of this function is going to overwrite with the real edge equations.
struct NcGSlots {
    TriangleSetup* T0;
    TriangleSetup* T1;
};

HIPGL_DEVICE __forceinline__ TransformedVertex* nc_g_vtx(const NcGSlots& S, int s)
{
    TriangleSetup* T = (s < 3) ? S.T0 : S.T1;
    return &T->v[(s < 3) ? s : (s - 3)];
}

HIPGL_DEVICE __forceinline__ void nc_g_copy(const NcGSlots& S, int dst, int src)
{
    static_assert(sizeof(TransformedVertex) == 128,
                  "nc_g_copy moves the vertex as 8 x float4");
    const float4* A = reinterpret_cast<const float4*>(nc_g_vtx(S, src));
    float4* O = reinterpret_cast<float4*>(nc_g_vtx(S, dst));
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        float4 t0 = A[h*4+0], t1 = A[h*4+1], t2 = A[h*4+2], t3 = A[h*4+3];
        O[h*4+0] = t0; O[h*4+1] = t1; O[h*4+2] = t2; O[h*4+3] = t3;
    }
}

// near_clip_setup_triangle's arithmetic, reading the three vertices that are
// already sitting in tri.v[0..2] instead of copying them in.
HIPGL_DEVICE __forceinline__ void nc_g_setup(TriangleSetup* T, int first_vertex,
                                             int output_index, int32_t* tri_stats)
{
    ;
    TriangleSetup& tri = *T;
    tri.first_vertex = first_vertex;
    constexpr float SUBPX = 256.f;
    constexpr float INV_SUBPX = 1.f / 256.f;
    const float x0 = roundf(tri.v[0].screen_x * SUBPX) * INV_SUBPX;
    const float y0 = roundf(tri.v[0].screen_y * SUBPX) * INV_SUBPX;
    const float x1 = roundf(tri.v[1].screen_x * SUBPX) * INV_SUBPX;
    const float y1 = roundf(tri.v[1].screen_y * SUBPX) * INV_SUBPX;
    const float x2 = roundf(tri.v[2].screen_x * SUBPX) * INV_SUBPX;
    const float y2 = roundf(tri.v[2].screen_y * SUBPX) * INV_SUBPX;
    const float e0x = x1 - x0, e0y = y1 - y0;
    const float e1x = x2 - x1, e1y = y2 - y1;
    const float e2x = x0 - x2, e2y = y0 - y2;
    const float area = e0x * e2y - e0y * e2x;
    const float zmin = fminf(tri.v[0].screen_z,
                       fminf(tri.v[1].screen_z, tri.v[2].screen_z));
    hipgl_set_flags(tri, !isfinite(area) || area == 0.f || fabsf(area) < d_min_tri_area,
                         area < 0.f);
    if (tri_stats) {
        if (tri.degenerate)
            tri_stats[output_index * TRI_STAT_COUNT + TRI_STAT_DEGENERATE] = 1;
        if (tri.backfacing)
            tri_stats[output_index * TRI_STAT_COUNT + TRI_STAT_BACKFACE] = 1;
    }
    if (tri.degenerate) return;
    const float sign = area > 0.f ? -1.f : 1.f;
    const float A0 = sign * (-e0y), B0 = sign * e0x;
    const float A1 = sign * (-e1y), B1 = sign * e1x;
    const float A2 = sign * (-e2y), B2 = sign * e2x;
    tri.edge_eq[0][0] = A0;
    tri.edge_eq[0][1] = B0;
    tri.edge_eq[0][2] = (float)(-((double)A0 * (double)x0 + (double)B0 * (double)y0));
    tri.edge_eq[1][0] = A1;
    tri.edge_eq[1][1] = B1;
    tri.edge_eq[1][2] = (float)(-((double)A1 * (double)x1 + (double)B1 * (double)y1));
    tri.edge_eq[2][0] = A2;
    tri.edge_eq[2][1] = B2;
    tri.edge_eq[2][2] = (float)(-((double)A2 * (double)x2 + (double)B2 * (double)y2));
    constexpr float FILL_BIAS = -1.f / 65536.f;
    tri.edge_bias[0] = (A0 > 0.f || (A0 == 0.f && B0 > 0.f)) ? 0.f : FILL_BIAS;
    tri.edge_bias[1] = (A1 > 0.f || (A1 == 0.f && B1 > 0.f)) ? 0.f : FILL_BIAS;
    tri.edge_bias[2] = (A2 > 0.f || (A2 == 0.f && B2 > 0.f)) ? 0.f : FILL_BIAS;
    hipgl_set_bbox(tri, static_cast<int>(floorf(math::min3(x0, x1, x2))),
                        static_cast<int>(floorf(math::min3(y0, y1, y2))),
                        static_cast<int>(ceilf(math::max3(x0, x1, x2))),
                        static_cast<int>(ceilf(math::max3(y0, y1, y2))));
    tri.z_ord = hipgl_f2ord_setup(zmin);
}

// Same entry point, same arguments, same output slots, same expression graph.
HIPGL_DEVICE __forceinline__ void near_clip_emit_source_triangle_g(
    const float* positions, const float* normals,
    const float* texcoords, const int* texids, const float* colors,
    const float* spec_colors, const float* per_vert_specular,
    const float* per_vert_shininess, const float* per_vert_emission,
    const float* mvp, const float* model_view, const float* normal_mat,
    int vp_x, int vp_y, int vp_w, int vp_h, const RenderState& state,
    int source_index, int first_output, int second_output,
    TriangleSetup* triangles, int32_t* tri_stats,
    bool have_wp,
    float wp00, float wp01, float wp02,
    float wp10, float wp11, float wp12,
    float wp20, float wp21, float wp22,
    bool have_gq, float gq0, float gq1, float gq2, float gq3)
{
    NcGSlots S;
    S.T0 = &triangles[first_output];
    S.T1 = &triangles[second_output];
    hipgl_mark_degenerate(*S.T0);
    hipgl_mark_degenerate(*S.T1);
    S.T0->first_vertex = source_index * 3;
    S.T1->first_vertex = source_index * 3;
    const bool shade = state.gouraud_shading && state.lighting_enabled;

    // ── STEP 1: DECIDE THE CASE BEFORE SHADING ANYTHING ───────────────────
    // The near-plane distance needs only clip[2] and clip[3], i.e. eight FMAs
    // against the same mvp rows the full transform uses -- so `inside` is known
    // for the price of 24 FMAs, BEFORE a single vertex is built. That is worth
    // paying for: knowing the case up front lets every vertex be written
    // STRAIGHT INTO the output slot it will be published from, which removes
    // the slot permutation entirely for the one-vertex-inside cases and cuts it
    // to two moves for the quads. Each removed move is a global store->load
    // round trip on the one lane the wave is waiting for, and those round trips
    // -- not the arithmetic and not the register count -- are what this fold
    // costs. Bit-exact by construction: load_vertex below recomputes clip[2]
    // and clip[3] from the identical expression on the identical inputs.
    const auto near_d = [&](int vertex_index, float wx, float wy,
                            float wz) -> float {
        float x, y, z;
        if (have_wp) {
            x = wx; y = wy; z = wz;
        } else {
            const float* gp = positions + vertex_index * 3;
            x = gp[0]; y = gp[1]; z = gp[2];
        }
        const float c2 = mvp[2]*x + mvp[6]*y + mvp[10]*z + mvp[14];
        const float c3 = mvp[3]*x + mvp[7]*y + mvp[11]*z + mvp[15];
        return c2 + c3;
    };
    const float d0 = near_d(source_index * 3,     wp00, wp01, wp02);
    const float d1 = near_d(source_index * 3 + 1, wp10, wp11, wp12);
    const float d2 = near_d(source_index * 3 + 2, wp20, wp21, wp22);
    const unsigned inside = (d0 >= 0.f ? 1u : 0u)
                          | (d1 >= 0.f ? 2u : 0u)
                          | (d2 >= 0.f ? 4u : 0u);

    // ── THE PLAN, AS DATA ─────────────────────────────────────────────────
    // Slots 0..2 are triangles[first_output].v[0..2] and 3..5 are
    // triangles[second_output].v[0..2]. dpack says where v0,v1,v2 are built;
    // ipack describes the two Sutherland-Hodgman cuts as (endpoint vertex a,
    // endpoint vertex b, destination slot) in the original edge order and with
    // the original argument order, so the inside-endpoint canonicalisation and
    // therefore the generated position are unchanged. With this placement the
    // published triangle is ALWAYS slots 0,1,2 and the quad's fourth vertex is
    // ALWAYS slot 5, so the projection/abort order and the two quad moves are
    // case-independent. The whole seven-case switch decides three integers --
    // writing the cases out longhand instead cost 25 KB of code object and 11%
    // of the wall.
    unsigned dpack, ipack;
    bool quad;
    switch (inside) {
    case 1:  dpack = 1u|(3u<<3)|(4u<<6);
             ipack = 2u|(0u<<3)|(0u<<6) | (0u<<9)|(1u<<12)|(2u<<15);
             quad = false; break;
    case 2:  dpack = 3u|(1u<<3)|(4u<<6);
             ipack = 0u|(1u<<3)|(0u<<6) | (1u<<9)|(2u<<12)|(2u<<15);
             quad = false; break;
    case 4:  dpack = 3u|(4u<<3)|(2u<<6);
             ipack = 2u|(0u<<3)|(0u<<6) | (1u<<9)|(2u<<12)|(1u<<15);
             quad = false; break;
    case 3:  dpack = 1u|(2u<<3)|(4u<<6);
             ipack = 2u|(0u<<3)|(0u<<6) | (1u<<9)|(2u<<12)|(5u<<15);
             quad = true;  break;
    case 5:  dpack = 0u|(4u<<3)|(5u<<6);
             ipack = 0u|(1u<<3)|(1u<<6) | (1u<<9)|(2u<<12)|(2u<<15);
             quad = true;  break;
    case 6:  dpack = 4u|(2u<<3)|(5u<<6);
             ipack = 2u|(0u<<3)|(0u<<6) | (0u<<9)|(1u<<12)|(1u<<15);
             quad = true;  break;
    case 7:  dpack = 0u|(1u<<3)|(2u<<6);
             ipack = 0u;  quad = false; break;
    default:
        return;   // nothing in front of the plane; both records stay degenerate
    }
    const int ds0 = (int)(dpack & 7u);
    const int ds1 = (int)((dpack >> 3) & 7u);
    const int ds2 = (int)((dpack >> 6) & 7u);

    // Projection lives with its producer: every clip[4] below is in registers a
    // few instructions before it is needed, so nothing round-trips through
    // scratch. Success is a per-slot bit, consulted afterwards in the original
    // a,b,c[,d] order so a straddler is rejected on exactly the same vertex.
    unsigned okmask = 0u;
    const auto project_c = [&](int sl, float c0, float c1, float c2,
                               float c3) -> bool {
        const float w = c3;
        if (!(w > 0.f) || !isfinite(w)) return false;
        const float inverse_w = 1.f / w;
        const float x = c0 * inverse_w;
        const float y = c1 * inverse_w;
        const float z = c2 * inverse_w;
        if (!isfinite(x) || !isfinite(y) || !isfinite(z)) return false;
        TransformedVertex* V = nc_g_vtx(S, sl);
        V->inv_w    = inverse_w;
        V->screen_z = z * .5f + .5f;
        V->screen_x = vp_x + (x * .5f + .5f) * static_cast<float>(vp_w);
        V->screen_y = vp_y + (1.f - (y * .5f + .5f)) * static_cast<float>(vp_h);
        V->clip_zw[0] = c2;
        V->clip_zw[1] = w;
        return true;
    };

    // ── STEP 2: BUILD THE THREE SOURCE VERTICES IN PLACE ──────────────────
    // Byte for byte the register clipper's load_vertex, except that
    // fused_transform_vertex_v is handed the STAGED slot as its destination.
    // That single change is what breaks the 128-VGPR wall: no 30-dword
    // TransformedVertex is ever assembled in registers at a program point where
    // the main path's own peak is already 113.
    float k00 = 0.f, k01 = 0.f, k03 = 0.f;
    float k10 = 0.f, k11 = 0.f, k13 = 0.f;
    float k20 = 0.f, k21 = 0.f, k23 = 0.f;
    const auto load_vertex = [&](int vertex_index, int slot, float& kx,
                                 float& ky, float& kw,
                                 float wx, float wy, float wz) {
        float posv[3];
        if (have_wp) {
            posv[0] = wx; posv[1] = wy; posv[2] = wz;
        } else {
            const float* gp = positions + vertex_index * 3;
            posv[0] = gp[0]; posv[1] = gp[1]; posv[2] = gp[2];
        }
        float wn0 = 0.f, wn1 = 0.f, wn2 = 0.f;
        bool have_n = false;
        if (have_gq && normals) {
            const float gq[4] = { gq0, gq1, gq2, gq3 };
            const float* nm = normals + vertex_index * 3;
            const float ln[3] = { nm[0], nm[1], nm[2] };
            float wn[3];
            hp_qrot(gq, ln, wn);
            wn0 = wn[0]; wn1 = wn[1]; wn2 = wn[2];
            have_n = true;
        }
        const float x = posv[0], y = posv[1], z = posv[2];
        float clip[4];
        clip[0] = mvp[0]*x + mvp[4]*y + mvp[8]*z + mvp[12];
        clip[1] = mvp[1]*x + mvp[5]*y + mvp[9]*z + mvp[13];
        clip[2] = mvp[2]*x + mvp[6]*y + mvp[10]*z + mvp[14];
        clip[3] = mvp[3]*x + mvp[7]*y + mvp[11]*z + mvp[15];
        TransformedVertex* dst = nc_g_vtx(S, slot);
        fused_transform_vertex_v(positions, normals, texcoords, texids, colors,
            spec_colors, per_vert_specular, per_vert_shininess, per_vert_emission,
            vertex_index, mvp, model_view, normal_mat,
            vp_x, vp_y, vp_w, vp_h, *dst, clip,
            posv, have_n, wn0, wn1, wn2);
        if (shade) compute_gouraud_lighting(*dst, state);
        if (project_c(slot, clip[0], clip[1], clip[2], clip[3]))
            okmask |= 1u << slot;
        // A cut interpolates clip[0], clip[1] and clip[3] of its two endpoints;
        // clip[2] is never read from them (it is forced to -clip[3]).
        kx = clip[0]; ky = clip[1]; kw = clip[3];
    };
    load_vertex(source_index * 3,     ds0, k00, k01, k03, wp00, wp01, wp02);
    load_vertex(source_index * 3 + 1, ds1, k10, k11, k13, wp10, wp11, wp12);
    load_vertex(source_index * 3 + 2, ds2, k20, k21, k23, wp20, wp21, wp22);

    // ── STEP 3: THE CUTS ──────────────────────────────────────────────────
    // near_clip_intersection, 16 B at a time. Same inside-endpoint
    // canonicalisation, same `a + t*(b - a)` per field, same
    // `out.clip[2] = -out.clip[3]` fix-up. Groups 0..6 span dwords 0..27 of
    // TransformedVertex: every dword in 2..25 is an interpolated output, dword
    // 13 is texid (a primitive attribute, carried from the inside endpoint as
    // the original does), and dwords 0,1 (screen_z, inv_w) and 26,27 (clip_zw)
    // are overwritten unconditionally by the projection that follows, so
    // interpolating them is dead arithmetic no consumer ever sees. Group 7
    // (screen_x, screen_y) is left to the projection for the same reason.
    // 14 loads + 7 stores instead of 90 scalar accesses.
    if (inside != 7u) {
#pragma clang loop unroll(disable)
        for (int i = 0; i < 2; ++i) {
            const unsigned f = ipack >> (9 * i);
            const int va = (int)(f & 7u);
            const int vb = (int)((f >> 3) & 7u);
            const int so = (int)((f >> 6) & 7u);
            const int sa = (va == 0) ? ds0 : ((va == 1) ? ds1 : ds2);
            const int sb = (vb == 0) ? ds0 : ((vb == 1) ? ds1 : ds2);
            const float da = (va == 0) ? d0 : ((va == 1) ? d1 : d2);
            const float db = (vb == 0) ? d0 : ((vb == 1) ? d1 : d2);
            const bool a_inside = da >= 0.f;
            const int ia = a_inside ? sa : sb;
            const int ib = a_inside ? sb : sa;
            const float inside_distance  = a_inside ? da : db;
            const float outside_distance = a_inside ? db : da;
            const float t = inside_distance / (inside_distance - outside_distance);
            const int ka = a_inside ? va : vb;
            const int kb = a_inside ? vb : va;
            const float pa0 = (ka == 0) ? k00 : ((ka == 1) ? k10 : k20);
            const float pa1 = (ka == 0) ? k01 : ((ka == 1) ? k11 : k21);
            const float pa3 = (ka == 0) ? k03 : ((ka == 1) ? k13 : k23);
            const float pb0 = (kb == 0) ? k00 : ((kb == 1) ? k10 : k20);
            const float pb1 = (kb == 0) ? k01 : ((kb == 1) ? k11 : k21);
            const float pb3 = (kb == 0) ? k03 : ((kb == 1) ? k13 : k23);
            const float oc0 = pa0 + t * (pb0 - pa0);
            const float oc1 = pa1 + t * (pb1 - pa1);
            const float oc3 = pa3 + t * (pb3 - pa3);
            const float4* A = reinterpret_cast<const float4*>(nc_g_vtx(S, ia));
            const float4* B = reinterpret_cast<const float4*>(nc_g_vtx(S, ib));
            float4* O = reinterpret_cast<float4*>(nc_g_vtx(S, so));
#ifndef NC_G_CHUNK
#define NC_G_CHUNK 4
#endif
#pragma unroll
            for (int base = 0; base < 7; base += NC_G_CHUNK) {
                float4 av[NC_G_CHUNK], bv[NC_G_CHUNK];
#pragma unroll
                for (int j = 0; j < NC_G_CHUNK; ++j) {
                    const int g = base + j;
                    if (g < 7) { av[j] = A[g]; bv[j] = B[g]; }
                }
#pragma unroll
                for (int j = 0; j < NC_G_CHUNK; ++j) {
                    const int g = base + j;
                    if (g < 7) {
                        const float4 a = av[j], b = bv[j];
                        float4 o;
                        o.x = a.x + t * (b.x - a.x);
                        o.y = a.y + t * (b.y - a.y);
                        o.z = a.z + t * (b.z - a.z);
                        o.w = a.w + t * (b.w - a.w);
                        if (g == 3) o.y = a.y;   // texid, from endpoint a
                        O[g] = o;
                    }
                }
            }
            // After the interpolation: groups 0 and 6 carry screen_z / inv_w /
            // clip_zw, which the projection owns.
            if (project_c(so, oc0, oc1, -oc3, oc3)) okmask |= 1u << so;
        }
    }

    // ── STEP 4: PUBLISH ───────────────────────────────────────────────────
    // The placement above makes the emitted triangle slots 0,1,2 in every case
    // and the quad's fourth vertex slot 5, so the abort order is fixed and the
    // only moves left are the two the quad's shared vertices need.
    if (!((okmask >> 0) & 1u)) return;
    if (!((okmask >> 1) & 1u)) return;
    if (!((okmask >> 2) & 1u)) return;
    if (quad) {
        if (!((okmask >> 5) & 1u)) return;
        nc_g_copy(S, 3, 0);   // second.v[0] = a
        nc_g_copy(S, 4, 2);   // second.v[1] = c
    }
    nc_g_setup(S.T0, source_index * 3, first_output, tri_stats);
    if (quad)
        nc_g_setup(S.T1, source_index * 3, second_output, tri_stats);
}
