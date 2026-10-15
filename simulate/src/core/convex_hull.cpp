// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "hipgl/physics/convex_hull.h"
#include <cmath>
#include <cstring>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <cstdio>
#include <array>
#include <cfloat>
#include <map>
#include <stdexcept>
#include <limits>

namespace hipgl {
namespace physics {

namespace {

static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559 && FLT_MANT_DIG==24,
              "Convex hull exact predicates require IEEE-754 binary32 input coordinates");

struct Vec3 {
    double x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator*(double s) const { return {x * s, y * s, z * s}; }
    double dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }
    double length() const { return sqrt(x * x + y * y + z * z); }
    Vec3 normalized() const {
        double l = length();
        return l > 0 ? Vec3(x / l, y / l, z / l) : Vec3(0, 0, 0);
    }
};

struct Triangle {
    int v[3];
    Vec3 normal;
    double dist; // signed distance from origin to plane (normal.dot(vertex))
    bool alive;
};

// Every coordinate reaching orientation() is an original finite float32.
// Multiplying by 2^149 turns it into an exact signed integer of at most 277
// bits. Differences need 278 bits; the six-term triple-product determinant
// needs at most 837. Thus 28 uint32 limbs (896 bits) is a proven bound, unrelated
// to the number of hull points/faces. This fallback runs only when the fast
// floating sign filter is uncertain; it is not a geometric epsilon.
struct ExactInteger {
    static constexpr int capacity=28;
    std::array<uint32_t,capacity> limb{};
    int used=0,sign=0;
    void trim() {while(used && limb[used-1]==0) used--;if(!used) sign=0;}
};

ExactInteger exact_coordinate(double value) {
    float f=(float)value;
    if((double)f!=value) throw std::logic_error("Exact hull predicate requires original float32 coordinates");
    uint32_t bits;std::memcpy(&bits,&f,sizeof(bits));
    uint32_t exponent=(bits>>23)&255,significand=bits&0x7fffff;
    if(exponent==255) throw std::invalid_argument("Nonfinite exact hull predicate input");
    if(exponent) significand|=0x800000;
    ExactInteger out;
    if(!significand) return out;
    unsigned shift=exponent?exponent-1:0;
    unsigned word=shift/32,offset=shift%32;
    uint64_t packed=(uint64_t)significand<<offset;
    out.limb[word]=(uint32_t)packed;out.limb[word+1]=(uint32_t)(packed>>32);
    out.used=word+2;out.sign=(bits>>31)?-1:1;out.trim();return out;
}

int exact_compare_magnitude(const ExactInteger& a,const ExactInteger& b) {
    if(a.used!=b.used) return a.used>b.used?1:-1;
    for(int i=a.used-1;i>=0;i--) if(a.limb[i]!=b.limb[i]) return a.limb[i]>b.limb[i]?1:-1;
    return 0;
}

ExactInteger exact_add(const ExactInteger& a,const ExactInteger& b) {
    if(!a.sign) return b;if(!b.sign) return a;
    ExactInteger out;
    if(a.sign==b.sign) {
        out.sign=a.sign;out.used=std::max(a.used,b.used);
        uint64_t carry=0;
        for(int i=0;i<out.used;i++) {
            uint64_t sum=(uint64_t)a.limb[i]+b.limb[i]+carry;
            out.limb[i]=(uint32_t)sum;carry=sum>>32;
        }
        if(carry) {
            if(out.used>=ExactInteger::capacity) throw std::logic_error("Exact hull integer bound exceeded");
            out.limb[out.used++]=(uint32_t)carry;
        }
    } else {
        int comparison=exact_compare_magnitude(a,b);
        if(!comparison) return out;
        const ExactInteger& high=comparison>0?a:b;
        const ExactInteger& low=comparison>0?b:a;
        out.sign=high.sign;out.used=high.used;
        uint64_t borrow=0;
        for(int i=0;i<out.used;i++) {
            uint64_t subtract=(uint64_t)low.limb[i]+borrow;
            uint64_t value=high.limb[i];
            out.limb[i]=(uint32_t)(value-subtract);borrow=value<subtract;
        }
    }
    out.trim();return out;
}

ExactInteger exact_subtract(const ExactInteger& a,ExactInteger b) {
    b.sign=-b.sign;return exact_add(a,b);
}

ExactInteger exact_multiply(const ExactInteger& a,const ExactInteger& b) {
    ExactInteger out;
    if(!a.sign || !b.sign) return out;
    if(a.used+b.used>ExactInteger::capacity) throw std::logic_error("Exact hull integer bound exceeded");
    out.sign=a.sign*b.sign;out.used=a.used+b.used;
    for(int i=0;i<a.used;i++) {
        uint64_t carry=0;
        for(int j=0;j<b.used;j++) {
            uint64_t value=(uint64_t)a.limb[i]*b.limb[j]+out.limb[i+j]+carry;
            out.limb[i+j]=(uint32_t)value;carry=value>>32;
        }
        out.limb[i+b.used]=(uint32_t)carry;
    }
    out.trim();return out;
}

int exact_orientation(const Vec3& a,const Vec3& b,const Vec3& c,const Vec3& p) {
    const double av[3]={a.x,a.y,a.z},bv[3]={b.x,b.y,b.z},cv[3]={c.x,c.y,c.z},pv[3]={p.x,p.y,p.z};
    ExactInteger u[3],v[3],w[3];
    for(int k=0;k<3;k++) {
        auto origin=exact_coordinate(av[k]);
        u[k]=exact_subtract(exact_coordinate(bv[k]),origin);
        v[k]=exact_subtract(exact_coordinate(cv[k]),origin);
        w[k]=exact_subtract(exact_coordinate(pv[k]),origin);
    }
    ExactInteger determinant;
    for(int k=0;k<3;k++) {
        int j=(k+1)%3,l=(k+2)%3;
        auto minor=exact_subtract(exact_multiply(v[j],w[l]),exact_multiply(v[l],w[j]));
        determinant=exact_add(determinant,exact_multiply(u[k],minor));
    }
    return determinant.sign;
}

// Sign of the oriented tetrahedral volume, evaluated relative to a vertex
// rather than subtracting two large world-origin plane offsets. The error
// bound is on arithmetic only: a fixed physical visibility threshold can omit
// genuinely visible neighboring faces and tear the incremental horizon.
inline int orientation(const Vec3& a,const Vec3& b,const Vec3& c,const Vec3& p) {
    long double bx=(long double)b.x-a.x,by=(long double)b.y-a.y,bz=(long double)b.z-a.z;
    long double cx=(long double)c.x-a.x,cy=(long double)c.y-a.y,cz=(long double)c.z-a.z;
    long double px=(long double)p.x-a.x,py=(long double)p.y-a.y,pz=(long double)p.z-a.z;
    long double t0=bx*cy*pz,t1=by*cz*px,t2=bz*cx*py;
    long double t3=bz*cy*px,t4=by*cx*pz,t5=bx*cz*py;
    long double det=(t0+t1+t2)-(t3+t4+t5);
    long double magnitude=fabsl(t0)+fabsl(t1)+fabsl(t2)+fabsl(t3)+fabsl(t4)+fabsl(t5);
    long double error=32*LDBL_EPSILON*magnitude;
    return det>error ? 1 : (det < -error ? -1 : exact_orientation(a,b,c,p));
}

// Pack an edge as (min, max) for use as hash key
inline uint64_t edge_key(int a, int b) {
    if (a > b) std::swap(a, b);
    return ((uint64_t)a << 32) | (uint64_t)b;
}

// Deduplicate vertices: expanded mesh has many duplicates
std::vector<int> deduplicate(const Vec3* pts, int n, std::vector<Vec3>& unique) {
    std::vector<int> remap(n);
    // Preserve the raw asset frame and every distinct input coordinate.
    // Approximate 0.1mm grid dedup erased thin features; packing signed cells
    // in overlapping bit lanes also conflated unrelated coordinates. Exact
    // finite coordinate tuples have equality independent of hash collisions.
    std::map<std::array<double,3>, int> grid;
    for (int i = 0; i < n; i++) {
        if(!std::isfinite(pts[i].x) || !std::isfinite(pts[i].y) || !std::isfinite(pts[i].z))
            throw std::invalid_argument("Convex hull input contains a nonfinite coordinate");
        std::array<double,3> key{{pts[i].x,pts[i].y,pts[i].z}};
        auto it = grid.find(key);
        if (it != grid.end()) {
            remap[i] = it->second;
        } else {
            int uid = (int)unique.size();
            unique.push_back(pts[i]);
            grid[key] = uid;
            remap[i] = uid;
        }
    }
    return remap;
}

} // anonymous namespace

MeshData compute_convex_hull(const float* verts, int nvert) {
    MeshData result;
    if (nvert < 4) return result;
    if (!verts) throw std::invalid_argument("Convex hull input pointer is null");

    std::vector<Vec3> all_pts(nvert);
    for (int i = 0; i < nvert; i++)
        all_pts[i] = {verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2]};

    // Deduplicate (expanded meshes have 3x duplicates)
    std::vector<Vec3> pts;
    deduplicate(all_pts.data(), nvert, pts);
    int n = (int)pts.size();
    if (n < 4) return result;

    // --- Find initial tetrahedron ---
    // Use the widest coordinate axis, avoiding a nearly coincident seed line.
    Vec3 low=pts[0],high=pts[0];
    for(const auto& p:pts) {
        low.x=std::min(low.x,p.x);low.y=std::min(low.y,p.y);low.z=std::min(low.z,p.z);
        high.x=std::max(high.x,p.x);high.y=std::max(high.y,p.y);high.z=std::max(high.z,p.z);
    }
    Vec3 extent=high-low;
    int seed_axis=extent.y>extent.x ? 1 : 0;
    if(extent.z>(seed_axis==0?extent.x:extent.y)) seed_axis=2;
    auto coordinate=[&](const Vec3& p) {return seed_axis==0?p.x:(seed_axis==1?p.y:p.z);};
    int minx = 0, maxx = 0;
    for (int i = 1; i < n; i++) {
        if (coordinate(pts[i]) < coordinate(pts[minx])) minx = i;
        if (coordinate(pts[i]) > coordinate(pts[maxx])) maxx = i;
    }
    int i0 = minx, i1 = maxx;
    if (i0 == i1) return result; // degenerate

    // Point furthest from line (i0, i1)
    Vec3 lineDir = (pts[i1] - pts[i0]).normalized();
    int i2 = -1;
    double best = -1;
    for (int i = 0; i < n; i++) {
        if (i == i0 || i == i1) continue;
        Vec3 d = pts[i] - pts[i0];
        Vec3 proj = lineDir * d.dot(lineDir);
        double dist2 = (d - proj).dot(d - proj);
        if (dist2 > best) { best = dist2; i2 = i; }
    }
    if (i2 < 0 || best == 0) return result;

    // Point furthest from plane (i0, i1, i2)
    Vec3 triNorm = (pts[i1] - pts[i0]).cross(pts[i2] - pts[i0]).normalized();
    int i3 = -1;
    best = -1;
    for (int i = 0; i < n; i++) {
        if (i == i0 || i == i1 || i == i2) continue;
        if (orientation(pts[i0],pts[i1],pts[i2],pts[i])==0) continue;
        double d = fabs(triNorm.dot(pts[i] - pts[i0]));
        if (d > best) { best = d; i3 = i; }
    }
    if (i3 < 0 || orientation(pts[i0],pts[i1],pts[i2],pts[i3]) == 0) return result;

    // Orient: i3 above plane (i0, i1, i2)
    if (orientation(pts[i0],pts[i1],pts[i2],pts[i3]) < 0)
        std::swap(i1, i2);

    // --- Simple face-list incremental convex hull ---
    // Faces with outward normals (CCW winding from outside)
    std::vector<Triangle> faces;
    auto add_face = [&](int a, int b, int c) {
        Triangle t;
        t.v[0] = a; t.v[1] = b; t.v[2] = c;
        t.normal = (pts[b] - pts[a]).cross(pts[c] - pts[a]).normalized();
        if(t.normal.dot(t.normal)==0)
            throw std::runtime_error("Convex hull construction produced a degenerate face");
        t.dist = t.normal.dot(pts[a]);
        t.alive = true;
        faces.push_back(t);
    };

    // Initial tetrahedron (4 faces, outward-facing)
    // Face opposite i3 (base): normal points away from i3
    add_face(i0, i2, i1); // reversed to face away from i3
    add_face(i0, i1, i3);
    add_face(i1, i2, i3);
    add_face(i2, i0, i3);

    // Verify initial tetrahedron: i3 should be inside all 4 faces
    // (face 0's normal points away from i3, so dist should be negative)

    std::vector<bool> in_hull(n, false);
    in_hull[i0] = in_hull[i1] = in_hull[i2] = in_hull[i3] = true;

    for (int pi = 0; pi < n; pi++) {
        if (in_hull[pi]) continue;
        const Vec3& p = pts[pi];

        // Find visible faces
        std::vector<int> visible;
        for (int fi = 0; fi < (int)faces.size(); fi++) {
            if (!faces[fi].alive) continue;
            const auto& f=faces[fi];
            if (orientation(pts[f.v[0]],pts[f.v[1]],pts[f.v[2]],p)>0)
                visible.push_back(fi);
        }
        if (visible.empty()) continue;

        in_hull[pi] = true;

        // Collect horizon edges: edges shared between visible and non-visible faces.
        // An edge (a,b) on a visible face is a horizon edge if its reverse (b,a) is on
        // a non-visible face.
        std::unordered_set<int> vis_set(visible.begin(), visible.end());

        // Collect all directed edges of visible faces
        // For each edge, count how many visible faces contain it (as undirected)
        std::unordered_map<uint64_t, int> edge_vis_count;
        struct DirEdge { int from, to; };
        std::vector<DirEdge> vis_edges;

        for (int fi : visible) {
            auto& f = faces[fi];
            for (int ei = 0; ei < 3; ei++) {
                int a = f.v[ei], b = f.v[(ei + 1) % 3];
                uint64_t k = edge_key(a, b);
                edge_vis_count[k]++;
                vis_edges.push_back({a, b});
            }
        }

        // Horizon edges: those with count == 1 (on boundary between visible and non-visible)
        std::vector<DirEdge> horizon;
        for (auto& de : vis_edges) {
            uint64_t k = edge_key(de.from, de.to);
            if (edge_vis_count[k] == 1)
                horizon.push_back(de);
        }

        // A visible patch of a convex closed surface has one simple directed
        // horizon cycle. Reject malformed updates instead of constructing a
        // plausible but non-convex triangle soup for downstream collision.
        std::unordered_map<int,int> outgoing,incoming;
        for(const auto& edge:horizon) {
            if(!outgoing.emplace(edge.from,edge.to).second ||
               !incoming.emplace(edge.to,edge.from).second)
                throw std::runtime_error("Convex hull horizon is not a simple cycle");
        }
        if(horizon.size()<3 || outgoing.size()!=incoming.size())
            throw std::runtime_error("Convex hull horizon is incomplete");
        int start=horizon[0].from,at=start;
        for(size_t k=0;k<horizon.size();k++) {
            auto next=outgoing.find(at);
            if(next==outgoing.end() || (k>0 && at==start))
                throw std::runtime_error("Convex hull horizon is disconnected");
            at=next->second;
        }
        if(at!=start) throw std::runtime_error("Convex hull horizon is open");

        // Kill visible faces
        for (int fi : visible) faces[fi].alive = false;

        // Create new faces connecting horizon edges to the new point
        for (auto& he : horizon) {
            add_face(he.from, he.to, pi);
        }
    }

    // --- Extract alive triangles ---
    int nfaces_out = 0;
    for (auto& f : faces)
        if (f.alive) nfaces_out++;

    // No invalid surface is returned or silently replaced with raw triangles.
    // Validate oriented closure, convex supporting planes, and containment of
    // every deduplicated input point. Dynamic counts are intentional.
    struct EdgeUse {int count=0,balance=0;};
    std::unordered_map<uint64_t,EdgeUse> edge_use;
    std::unordered_set<int> active_vertices;
    for(const auto& f:faces) if(f.alive) {
        for(int k=0;k<3;k++) {
            int a=f.v[k],b=f.v[(k+1)%3];
            auto& edge=edge_use[edge_key(a,b)];edge.count++;edge.balance+=(a<b?1:-1);
            active_vertices.insert(a);
        }
        for(const auto& p:pts)
            if(orientation(pts[f.v[0]],pts[f.v[1]],pts[f.v[2]],p)>0)
                throw std::runtime_error("Convex hull output has a non-supporting face");
    }
    for(const auto& edge:edge_use)
        if(edge.second.count!=2 || edge.second.balance!=0)
            throw std::runtime_error("Convex hull output is not closed and oriented");
    if((int)active_vertices.size()-(int)edge_use.size()+nfaces_out!=2)
        throw std::runtime_error("Convex hull output has invalid Euler characteristic");

    int nverts_out = nfaces_out * 3;
    result.nfaces = nfaces_out;
    result.nvertices = nverts_out;
    result.vertices.resize(nverts_out * 3);
    result.normals.resize(nverts_out * 3);
    result.faces.resize(nfaces_out * 3);
    result.has_texcoords = false;

    int out_idx = 0;
    for (auto& f : faces) {
        if (!f.alive) continue;
        Vec3 va = pts[f.v[0]], vb = pts[f.v[1]], vc = pts[f.v[2]];
        Vec3 n = (vb - va).cross(vc - va).normalized();
        Vec3 fv[3] = {va, vb, vc};
        for (int vi = 0; vi < 3; vi++) {
            int idx = out_idx * 3 + vi;
            result.vertices[idx * 3 + 0] = fv[vi].x;
            result.vertices[idx * 3 + 1] = fv[vi].y;
            result.vertices[idx * 3 + 2] = fv[vi].z;
            result.normals[idx * 3 + 0] = n.x;
            result.normals[idx * 3 + 1] = n.y;
            result.normals[idx * 3 + 2] = n.z;
            result.faces[idx] = idx;
        }
        out_idx++;
    }

    return result;
}

ConvexHullCollisionData extract_collision_hull(const float* verts, int nvert) {
    ConvexHullCollisionData result;
    if (nvert < 3) return result;

    // Step 1: Deduplicate exact coordinates without quantizing the surface.
    std::vector<Vec3> all_pts(nvert);
    for (int i = 0; i < nvert; i++)
        all_pts[i] = {verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2]};

    std::vector<Vec3> unique;
    std::vector<int> remap = deduplicate(all_pts.data(), nvert, unique);
    int n_unique = (int)unique.size();

    result.nvert = n_unique;
    result.verts.resize(n_unique * 3);
    for (int i = 0; i < n_unique; i++) {
        result.verts[i * 3 + 0] = unique[i].x;
        result.verts[i * 3 + 1] = unique[i].y;
        result.verts[i * 3 + 2] = unique[i].z;
    }

    // Step 2: Build indexed triangles from expanded vertices (groups of 3)
    int n_tris = nvert / 3;
    std::unordered_set<uint64_t> unique_edges_set;

    for (int t = 0; t < n_tris; t++) {
        int v0 = remap[t * 3 + 0];
        int v1 = remap[t * 3 + 1];
        int v2 = remap[t * 3 + 2];
        if (v0 == v1 || v1 == v2 || v0 == v2) continue;  // degenerate

        result.faces.push_back(v0);
        result.faces.push_back(v1);
        result.faces.push_back(v2);

        Vec3 a = unique[v0], b = unique[v1], c = unique[v2];
        Vec3 n = (b - a).cross(c - a).normalized();
        result.face_normals.push_back(n.x);
        result.face_normals.push_back(n.y);
        result.face_normals.push_back(n.z);
        result.nface++;

        // Collect unique undirected edges
        unique_edges_set.insert(edge_key(v0, v1));
        unique_edges_set.insert(edge_key(v1, v2));
        unique_edges_set.insert(edge_key(v2, v0));
    }

    // Step 3: Build edge list
    result.nedge = (int)unique_edges_set.size();
    result.edges.reserve(result.nedge * 2);
    for (uint64_t ek : unique_edges_set) {
        int a = (int)(ek >> 32);
        int b = (int)(ek & 0xFFFFFFFF);
        result.edges.push_back(a);
        result.edges.push_back(b);
    }

    return result;
}

} // namespace physics
} // namespace hipgl
