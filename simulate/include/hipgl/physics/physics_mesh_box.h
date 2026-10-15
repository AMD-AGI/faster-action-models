#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cmath>

// Complete convex-mesh / box contact geometry. Local storage is independent
// of hull size: scalar accumulators, eight box corners, endpoint witnesses,
// and a proven-bounded single-triangle clip buffer. No hull vertices, faces,
// axes, or witness stream is silently truncated to a fixed contact buffer.
// Input topology is the native convex-hull builder's closed, consistently
// oriented triangle surface: each undirected edge appears in both directions.
// face_normal() honors its outward normals; it does not repair open topology.
namespace hipgl { namespace physics { namespace mesh_box_detail {

struct V {
    double x, y, z;
    HIPGL_DEVICE V operator+(V b) const { return {x+b.x,y+b.y,z+b.z}; }
    HIPGL_DEVICE V operator-(V b) const { return {x-b.x,y-b.y,z-b.z}; }
    HIPGL_DEVICE V operator*(double s) const { return {x*s,y*s,z*s}; }
};

HIPGL_DEVICE inline double dot(V a,V b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
HIPGL_DEVICE inline double component(V a,int k) { return k==0?a.x:(k==1?a.y:a.z); }
HIPGL_DEVICE inline V cross(V a,V b) {
    return {a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};
}
HIPGL_DEVICE inline V point(const float* p,int i=0) {
    return {double(p[3*i]),double(p[3*i+1]),double(p[3*i+2])};
}
HIPGL_DEVICE inline V unit(V n) {
    double squared=dot(n,n);
    return squared>0 ? n*(1.0/sqrt(squared)) : V{0,0,0};
}
HIPGL_DEVICE inline V rotate(const float* q,V v,bool inverse=false) {
    double w=q[0],x=q[1],y=q[2],z=q[3];
    double norm=w*w+x*x+y*y+z*z;
    if(inverse) {x=-x;y=-y;z=-z;}
    V xyz{x,y,z};
    V t=cross(xyz,v);
    return v+(t*w+cross(xyz,t))*(2.0/norm);
}

struct Hull {
    const float* vertices;
    const int* faces;
    const float* normals;
    int nv,nf;
};

HIPGL_DEVICE inline V face_normal(const Hull& h,int f) {
    V a=point(h.vertices,h.faces[3*f]);
    V b=point(h.vertices,h.faces[3*f+1]);
    V c=point(h.vertices,h.faces[3*f+2]);
    V n=unit(cross(b-a,c-a));
    // The builder's outward orientation is authoritative; recomputing the
    // plane in double avoids amplifying float normal error at thin features.
    if(dot(n,point(h.normals,f))<0) n=n*(-1);
    return n;
}

HIPGL_DEVICE inline bool test_axis(V axis,const Hull& h,V center,
    const V* axes,const double* sizes,double margin,double& best,V& normal) {
    if(dot(axis,axis)==0) return true;
    V n=unit(axis);
    double radius=0;
    for(int k=0;k<3;k++) radius+=sizes[k]*fabs(dot(n,axes[k]));
    double bc=dot(n,center),box_low=bc-radius,box_high=bc+radius;
    double threshold=fmax(best,-margin);
    double low=dot(n,point(h.vertices)),high=low;
    // Partial hull projections only EXPAND as vertices are added. Once both
    // directional escape bounds already exceed the current best (and the
    // margin separation threshold), unseen vertices cannot make this axis
    // separating or preferable. This certified early exit is not truncation.
    if(high-box_low>=threshold && box_high-low>=threshold) return true;
    // Memory-level pipelining only: the hull vertices of a block are loaded
    // together (independent loads in flight at once instead of one
    // load-latency round trip per vertex), then folded in the SAME order with
    // the SAME per-vertex operations and per-vertex certified early exit as
    // the plain loop, so low/high/return are bit-identical to it.
    constexpr int kBlock=16;
    for(int i0=1;i0<h.nv;i0+=kBlock) {
        const int count=(h.nv-i0<kBlock)?(h.nv-i0):kBlock;
        const float* block=h.vertices+3*i0;
        float xyz[3*kBlock];
        // Unconditional, clamped (always in-bounds) loads: no per-load branch,
        // so no per-load wait. Slots past count are never read below.
#pragma unroll
        for(int b=0;b<kBlock;b++) {
            const int c=(b<count)?b:(count-1);
            xyz[3*b]=block[3*c];xyz[3*b+1]=block[3*c+1];xyz[3*b+2]=block[3*c+2];
        }
#pragma unroll
        for(int b=0;b<kBlock;b++) if(b<count) {
            double v=dot(n,point(xyz,b));
            low=fmin(low,v);high=fmax(high,v);
            if(high-box_low>=threshold && box_high-low>=threshold) return true;
        }
    }
    double toward=high-box_low,away=box_high-low;
    double depth=fmin(toward,away);
    if(depth+margin<0) return false;
    if(depth<best) {
        best=depth;
        normal=(toward<=away)?n:n*(-1);
    }
    return true;
}

HIPGL_DEVICE inline bool inside_hull(V p,const Hull& h,double tolerance) {
    for(int f=0;f<h.nf;f++) {
        V n=face_normal(h,f);
        if(dot(n,p-point(h.vertices,h.faces[3*f]))>tolerance) return false;
    }
    return true;
}
HIPGL_DEVICE inline bool inside_box(V p,V c,const V* axes,
    const double* sizes,double tolerance) {
    V d=p-c;
    for(int k=0;k<3;k++)
        if(fabs(dot(d,axes[k]))>sizes[k]+tolerance) return false;
    return true;
}

HIPGL_DEVICE inline bool clip_plane(V p,V direction,V normal,double offset,
    double tolerance,double& lo,double& hi) {
    double numerator=offset-dot(normal,p);
    double denominator=dot(normal,direction);
    if(denominator==0) return numerator>=-tolerance;
    double bound=numerator/denominator;
    if(denominator>0) hi=fmin(hi,bound);
    else lo=fmax(lo,bound);
    return lo<=hi+1e-10;
}
HIPGL_DEVICE inline bool clip_hull(V p,V direction,const Hull& h,
    double tolerance,double& lo,double& hi) {
    for(int f=0;f<h.nf;f++) {
        V n=face_normal(h,f);
        if(!clip_plane(p,direction,n,dot(n,point(h.vertices,h.faces[3*f])),
                       tolerance,lo,hi)) return false;
    }
    return true;
}
HIPGL_DEVICE inline bool clip_box(V p,V direction,V center,const V* axes,
    const double* sizes,double tolerance,double& lo,double& hi) {
    for(int k=0;k<3;k++) {
        if(!clip_plane(p,direction,axes[k],dot(axes[k],center)+sizes[k],
                       tolerance,lo,hi)) return false;
        V opposite=axes[k]*(-1);
        if(!clip_plane(p,direction,opposite,dot(opposite,center)+sizes[k],
                       tolerance,lo,hi)) return false;
    }
    return true;
}

struct Witnesses {
    V low[3],high[3];
    int count;
};

HIPGL_DEVICE inline void admit(V p,const Hull& h,V center,const V* axes,
    const double* sizes,V normal,double support,double tolerance,Witnesses& witnesses) {
    if(fabs(dot(normal,p)-support)>2*tolerance ||
       !inside_box(p,center,axes,sizes,tolerance) ||
       !inside_hull(p,h,tolerance)) return;
    for(int k=0;k<3;k++) {
        if(!witnesses.count || component(p,k)<component(witnesses.low[k],k)) witnesses.low[k]=p;
        if(!witnesses.count || component(p,k)>component(witnesses.high[k],k)) witnesses.high[k]=p;
    }
    witnesses.count++;
}

HIPGL_DEVICE inline double plane_distance(V p,V n,double offset,double tolerance) {
    double d=dot(n,p)-offset;
    // Tangent support planes can differ by double roundoff. Snap only this
    // classification, not a distance-expanded box that would create area at
    // an actually one-dimensional contact.
    return fabs(d)<=tolerance?0:d;
}

HIPGL_DEVICE inline bool append_polygon(V p,V* polygon,int& count,double tolerance) {
    if(count && dot(p-polygon[count-1],p-polygon[count-1])<=tolerance*tolerance*1e-6) return true;
    // A convex triangle clipped by six planes has at most 3+6=9 vertices.
    // The tenth slot permits a repeated closing vertex before deduplication.
    // Fail the contact on violated assumptions; never truncate a polygon.
    if(count>=10) return false;
    polygon[count++]=p;
    return true;
}

HIPGL_DEVICE inline bool triangle_moment(V a,V b,V c,V center,const V* axes,
    const double* sizes,V normal,double tolerance,double& area,V& moment) {
    V polygon[10],clipped[10];
    polygon[0]=a;polygon[1]=b;polygon[2]=c;
    int count=3;
    for(int k=0;k<3;k++) for(int side=-1;side<=1;side+=2) {
        V n=axes[k]*side;
        double offset=dot(n,center)+sizes[k];
        int next=0;
        V previous=polygon[count-1];
        double dp=plane_distance(previous,n,offset,tolerance);
        for(int i=0;i<count;i++) {
            V current=polygon[i];
            double dc=plane_distance(current,n,offset,tolerance);
            if((dp<=0)!=(dc<=0)) {
                V crossing=previous+(current-previous)*(dp/(dp-dc));
                if(!append_polygon(crossing,clipped,next,tolerance)) return false;
            }
            if(dc<=0 && !append_polygon(current,clipped,next,tolerance)) return false;
            previous=current;dp=dc;
        }
        if(next>1 && dot(clipped[0]-clipped[next-1],clipped[0]-clipped[next-1])<=tolerance*tolerance*1e-6) next--;
        count=next;
        if(count<3) return true;
        for(int i=0;i<count;i++) polygon[i]=clipped[i];
    }
    // Supporting hull triangles partition a planar patch. Integrate their
    // clipped areas and first moments, so retriangulation cannot bias torque.
    // This area centroid is not a claim of exact MuJoCo MPR witness equivalence
    // on a nonunique two-dimensional contact patch.
    for(int i=1;i+1<count;i++) {
        double twice_area=fabs(dot(normal,cross(polygon[i]-polygon[0],polygon[i+1]-polygon[0])));
        area+=twice_area;
        moment=moment+(polygon[0]+polygon[i]+polygon[i+1])*(twice_area/3.0);
    }
    return true;
}

// A small, conservative face-patch extension. It applies only when the SAT
// winner is an actual face normal and its opposing incident face is within
// five degrees. True edge winners and other unique features keep the original
// single extremal-feature contact below.
// Every emitted point has ITS OWN geometric depth d. With normal mesh->box,
// p+n*d/2 is on the mesh surface and p-n*d/2 is on the box surface; these need
// not both be global extremal support points when the faces are tilted.
struct FacePatch {
    const Hull* hull;
    V center;
    const V* axes;
    const double* sizes;
    bool mesh_reference;
    V rn,in,u,v;
    double rd,id,tolerance,margin,scale;
    int box_axis;
    double box_sign;
};

HIPGL_DEVICE inline bool lex_larger(V p,V q,const FacePatch& patch) {
    double du=dot(p-q,patch.u),dv=dot(p-q,patch.v);
    double tie=1e-12*patch.scale;
    return du>tie || (fabs(du)<=tie && dv>tie);
}

struct PatchSelection {
    V points[4];
    double depths[4];
    int phase;
    bool found;
    double score;
};

HIPGL_DEVICE inline void patch_consider(V incident,const FacePatch& patch,PatchSelection& selection) {
    double d=patch.rd-dot(patch.rn,incident);
    if(d+patch.margin < -patch.tolerance) return;
    V reference=incident+patch.rn*d;
    if(patch.mesh_reference) {
        if(!inside_hull(reference,*patch.hull,patch.tolerance) ||
           !inside_box(incident,patch.center,patch.axes,patch.sizes,patch.tolerance)) return;
    } else {
        if(!inside_box(reference,patch.center,patch.axes,patch.sizes,patch.tolerance) ||
           !inside_hull(incident,*patch.hull,patch.tolerance)) return;
    }
    // Geometric objectives, not stream order or duplicate multiplicity:
    // deepest point, farthest point, largest triangle, then largest convex
    // quadrilateral coverage. Scores are evaluated on the reference plane.
    double score=d;
    if(selection.phase==1) score=dot(reference-selection.points[0],reference-selection.points[0]);
    if(selection.phase==2) score=fabs(dot(patch.rn,cross(selection.points[1]-selection.points[0],reference-selection.points[0])));
    if(selection.phase==3) {
        score=0;
        for(int k=0;k<3;k++) score+=fabs(dot(patch.rn,cross(selection.points[(k+1)%3]-selection.points[k],reference-selection.points[k])));
    }
    double tie=1e-12*(selection.phase?patch.scale*patch.scale:patch.scale);
    if(!selection.found || score>selection.score+tie ||
       (fabs(score-selection.score)<=tie && lex_larger(reference,selection.points[selection.phase],patch))) {
        selection.points[selection.phase]=reference;
        selection.depths[selection.phase]=d;
        selection.score=score;
        selection.found=true;
    }
}

HIPGL_DEVICE inline bool patch_segment_plane(V p,V direction,V plane,double offset,
    const FacePatch& patch,double& lo,double& hi,double construction_magnitude=0) {
    // Projection puts an edge exactly in its incident plane mathematically.
    // Double cancellation can leave a tiny nonzero denominator there; dividing
    // by it invents an arbitrary clipping fraction and changes with topology.
    // Use a bound on arithmetic roundoff, NOT the geometric contact tolerance,
    // to classify a segment as parallel. Real near-flat slopes remain visible.
    double denominator=dot(plane,direction);
    double magnitude=construction_magnitude+fabs(offset)+fabs(plane.x*p.x)+fabs(plane.y*p.y)+fabs(plane.z*p.z)+
        fabs(plane.x*direction.x)+fabs(plane.y*direction.y)+fabs(plane.z*direction.z)+
        patch.scale*sqrt(dot(plane,plane));
    double roundoff=32*2.2204460492503131e-16*magnitude;
    // Surface tolerance belongs only to patch_surface_plane. In particular,
    // a depth-parallel edge with a real negative depth must not be admitted
    // merely because its separation is smaller than the surface tolerance.
    if(fabs(denominator)<=roundoff) return offset-dot(plane,p)>=-roundoff;
    return clip_plane(p,direction,plane,offset,patch.tolerance,lo,hi);
}

HIPGL_DEVICE inline bool patch_surface_plane(V p,V direction,V plane,double offset,
    const FacePatch& patch,double& lo,double& hi,double construction_magnitude=0) {
    // Float32 asset vertices can make an intended flat quad microscopically
    // noncoplanar. Face membership and final surface admission already use
    // this same length-scaled tolerance. Apply it consistently to a whole
    // near-coplanar segment, rather than slicing it at an arbitrary triangle
    // diagonal. Convexity bounds the entire segment by its two endpoints.
    // This policy is ONLY for solid surface planes, never the depth plane;
    // a materially separated endpoint still uses the exact clipping bound.
    double a=dot(plane,p)-offset,b=a+dot(plane,direction);
    if(fabs(a)<=patch.tolerance && fabs(b)<=patch.tolerance) return true;
    return patch_segment_plane(p,direction,plane,offset,patch,lo,hi,construction_magnitude);
}

HIPGL_DEVICE inline bool patch_clip_plane(V p,V direction,V plane,double offset,
    const FacePatch& patch,double& lo,double& hi) {
    // Project onto the reference plane along rn, then impose each entire
    // reference-solid halfspace. This includes the WHOLE coplanar face, not
    // an arbitrary first triangle or a fixed prefix of hull planes.
    double along=dot(plane,patch.rn);
    // A reference self-plane projects to zero. Its computed coefficients can
    // instead be tiny residuals because rn is unit only to double precision.
    // Carry the PRE-cancellation arithmetic scale into the parallel bound;
    // measuring error only from residual coefficients would miss this case.
    double construction=fabs(offset)+fabs(patch.rd*along)+
        sqrt(dot(plane,plane))*(sqrt(dot(p,p))+sqrt(dot(direction,direction))+patch.scale);
    return patch_surface_plane(p,direction,plane-patch.rn*along,offset-patch.rd*along,
                               patch,lo,hi,construction);
}

HIPGL_DEVICE inline void patch_edge(V a,V b,const FacePatch& patch,PatchSelection& selection) {
    V direction=b-a;
    double lo=0,hi=1;
    if(!patch_segment_plane(a,direction,patch.rn,patch.rd+patch.margin,patch,lo,hi)) return;
    if(patch.mesh_reference) {
        for(int f=0;f<patch.hull->nf;f++) {
            V n=face_normal(*patch.hull,f);
            double offset=dot(n,point(patch.hull->vertices,patch.hull->faces[3*f]));
            if(!patch_clip_plane(a,direction,n,offset,patch,lo,hi)) return;
        }
    } else {
        for(int k=0;k<3;k++) for(int sign=-1;sign<=1;sign+=2) {
            V n=patch.axes[k]*sign;
            if(!patch_clip_plane(a,direction,n,dot(n,patch.center)+patch.sizes[k],patch,lo,hi)) return;
        }
    }
    if(lo>hi) lo=hi=(lo+hi)*0.5;
    patch_consider(a+direction*lo,patch,selection);
    patch_consider(a+direction*hi,patch,selection);
}

HIPGL_DEVICE inline V box_face_vertex(const FacePatch& patch,int corner) {
    int k=patch.box_axis,u=(k+1)%3,v=(k+2)%3;
    return patch.center+patch.axes[k]*(patch.box_sign*patch.sizes[k])+
        patch.axes[u]*((corner&1)?patch.sizes[u]:-patch.sizes[u])+
        patch.axes[v]*((corner&2)?patch.sizes[v]:-patch.sizes[v]);
}

HIPGL_DEVICE inline void patch_reference_edge(V a,V b,const FacePatch& patch,PatchSelection& selection) {
    // The depth halfspace may cut a reference edge even when the incident
    // polygon contains the whole reference face. Reference vertices alone
    // miss both resulting boundary endpoints, so clip each projected edge.
    double denominator=dot(patch.in,patch.rn);
    V incident=a+patch.rn*((patch.id-dot(patch.in,a))/denominator);
    V direction=b-a;
    direction=direction-patch.rn*(dot(patch.in,direction)/denominator);
    double lo=0,hi=1;
    if(!patch_segment_plane(incident,direction,patch.rn,patch.rd+patch.margin,patch,lo,hi)) return;
    if(patch.mesh_reference) {
        for(int k=0;k<3;k++) for(int sign=-1;sign<=1;sign+=2) {
            V n=patch.axes[k]*sign;
            if(!patch_surface_plane(incident,direction,n,dot(n,patch.center)+patch.sizes[k],patch,lo,hi)) return;
        }
    } else {
        for(int f=0;f<patch.hull->nf;f++) {
            V n=face_normal(*patch.hull,f);
            double offset=dot(n,point(patch.hull->vertices,patch.hull->faces[3*f]));
            if(!patch_surface_plane(incident,direction,n,offset,patch,lo,hi)) return;
        }
    }
    if(lo>hi) lo=hi=(lo+hi)*0.5;
    patch_consider(incident+direction*lo,patch,selection);
    patch_consider(incident+direction*hi,patch,selection);
}

HIPGL_DEVICE inline void patch_stream(const FacePatch& patch,PatchSelection& selection) {
    if(patch.mesh_reference) {
        // Four incident box-face edges. Their clipped endpoints give every
        // edge-crossing vertex, plus incident vertices inside the reference.
        for(int c=0;c<4;c++) for(int k=0;k<2;k++) {
            int other=c^(1<<k);
            if(c<other) patch_edge(box_face_vertex(patch,c),box_face_vertex(patch,other),patch,selection);
        }
        // Projected reference edges complete the intersection, including
        // vertices introduced by the penetration halfspace in containment.
        for(int f=0;f<patch.hull->nf;f++) for(int e=0;e<3;e++) {
            int ia=patch.hull->faces[3*f+e],ib=patch.hull->faces[3*f+(e+1)%3];
            if(ia>=ib) continue;
            V a=point(patch.hull->vertices,ia),b=point(patch.hull->vertices,ib);
            if(fabs(dot(patch.rn,a)-patch.rd)>patch.tolerance ||
               fabs(dot(patch.rn,b)-patch.rd)>patch.tolerance) continue;
            patch_reference_edge(a,b,patch,selection);
        }
    } else {
        // Every actual incident face edge, including harmless coplanar
        // diagonals. Interior candidates cannot change the geometric extrema.
        for(int f=0;f<patch.hull->nf;f++) for(int e=0;e<3;e++) {
            int ia=patch.hull->faces[3*f+e],ib=patch.hull->faces[3*f+(e+1)%3];
            if(ia>=ib) continue;
            V a=point(patch.hull->vertices,ia),b=point(patch.hull->vertices,ib);
            if(fabs(dot(patch.in,a)-patch.id)>patch.tolerance ||
               fabs(dot(patch.in,b)-patch.id)>patch.tolerance) continue;
            patch_edge(a,b,patch,selection);
        }
        for(int c=0;c<4;c++) for(int k=0;k<2;k++) {
            int other=c^(1<<k);
            if(c<other) patch_reference_edge(box_face_vertex(patch,c),box_face_vertex(patch,other),patch,selection);
        }
    }
}

HIPGL_DEVICE inline int face_manifold(const Hull& hull,V center,const V* axes,const double* sizes,
    V sat_normal,double sat_depth,double tolerance,double span,double margin,
    const float* mp,const float* mq,float out_pos[][3],float out_norm[][3],float* out_depth) {
    constexpr double parallel_cosine=0.9961946980917455; // cos(5 degrees)
    // A true face-axis winner is parallel to one actual box axis or mesh
    // face normal. An edge-cross axis only enters here if it is geometrically
    // the same face direction, so genuinely unique edge contacts stay intact.
    int box_axis=-1,mesh_face=-1;
    for(int k=0;k<3;k++) if(fabs(dot(sat_normal,axes[k]))>1-1e-12) box_axis=k;
    if(box_axis<0) for(int f=0;f<hull.nf;f++)
        if(dot(sat_normal,face_normal(hull,f))>1-1e-12) {mesh_face=f;break;}
    if(box_axis<0 && mesh_face<0) return 0;
    FacePatch patch{};
    patch.hull=&hull;patch.center=center;patch.axes=axes;patch.sizes=sizes;
    patch.mesh_reference=box_axis<0;
    patch.tolerance=tolerance;patch.margin=margin;patch.scale=span;
    if(patch.mesh_reference) {
        patch.rn=face_normal(hull,mesh_face);
        patch.rd=dot(patch.rn,point(hull.vertices,hull.faces[3*mesh_face]));
        double alignment=-1;
        for(int k=0;k<3;k++) {
            double a=fabs(dot(patch.rn,axes[k]));
            if(a>alignment) {alignment=a;box_axis=k;}
        }
        if(alignment<parallel_cosine) return 0;
        patch.box_axis=box_axis;
        patch.box_sign=dot(patch.rn,axes[box_axis])>0?-1:1;
        patch.in=axes[box_axis]*patch.box_sign;
        patch.id=dot(patch.in,center)+sizes[box_axis];
    } else {
        patch.box_axis=box_axis;
        patch.box_sign=dot(sat_normal,axes[box_axis])>0?-1:1;
        patch.rn=axes[box_axis]*patch.box_sign;
        patch.rd=dot(patch.rn,center)+sizes[box_axis];
        double minimum=dot(patch.rn,point(hull.vertices));
        for(int i=1;i<hull.nv;i++) minimum=fmin(minimum,dot(patch.rn,point(hull.vertices,i)));
        double alignment=-1;
        for(int f=0;f<hull.nf;f++) {
            V fn=face_normal(hull,f);
            double a=-dot(fn,patch.rn);
            if(a<parallel_cosine || a<=alignment) continue;
            bool deepest=false;
            for(int k=0;k<3;k++) if(dot(patch.rn,point(hull.vertices,hull.faces[3*f+k]))<=minimum+tolerance) deepest=true;
            if(deepest) {alignment=a;mesh_face=f;patch.in=fn;}
        }
        if(mesh_face<0) return 0;
        patch.id=dot(patch.in,point(hull.vertices,hull.faces[3*mesh_face]));
    }
    V tangent=axes[(box_axis+1)%3];
    patch.u=unit(tangent-patch.rn*dot(tangent,patch.rn));
    patch.v=cross(patch.rn,patch.u);
    PatchSelection selection{};
    int count=0;
    for(int phase=0;phase<4;phase++) {
        selection.phase=phase;selection.found=false;
        patch_stream(patch,selection);
        if(!selection.found) break;
        if(phase==1 && selection.score<=tolerance*tolerance) break;
        if(phase==2 && selection.score<=tolerance*tolerance) break;
        if(phase==3) {
            double area=fabs(dot(patch.rn,cross(selection.points[1]-selection.points[0],selection.points[2]-selection.points[0])));
            if(selection.score<=area+tolerance*tolerance) break;
        }
        count++;
    }
    if(count<2 || fabs(selection.depths[0]-sat_depth)>2*tolerance) return 0;
    V n=patch.mesh_reference?patch.rn:patch.rn*(-1);
    V world_normal=rotate(mq,n);
    for(int c=0;c<count;c++) {
        V p=rotate(mq,selection.points[c]-patch.rn*(selection.depths[c]*.5))+point(mp);
        out_pos[c][0]=float(p.x);out_pos[c][1]=float(p.y);out_pos[c][2]=float(p.z);
        out_norm[c][0]=float(world_normal.x);out_norm[c][1]=float(world_normal.y);out_norm[c][2]=float(world_normal.z);
        out_depth[c]=float(selection.depths[c]+margin);
    }
    return count;
}

// Returns 0 for separation, 1-4 penetrating/touching convex contacts, and
// -1 only for separated positive-margin proximity passing the complete SAT
// (the caller retains its prior proximity policy there). Complete SAT may
// reject legacy false-positive proximity pairs before that fallback. depth
// includes the caller's margin offset;
// witness construction uses the geometric overlap, never that offset.
HIPGL_DEVICE inline int contact(const float* mp,const float* mq,
    const float* bp,const float* bq,const float* bs,
    int va,int nv,const float* vertices,int fa,int nf,
    const float* normals,const int* faces,float out_pos[][3],
    float out_norm[][3],float* out_depth,double margin) {
    if(nv<4 || nf<4 || !vertices || !faces || !normals) return 0;
    Hull h{vertices+3*va,faces+3*fa,normals+3*fa,nv,nf};
    V center=rotate(mq,point(bp)-point(mp),true);
    V axes[3];
    const V basis[3]={{1,0,0},{0,1,0},{0,0,1}};
    double sizes[3]={bs[0],bs[1],bs[2]};
    for(int k=0;k<3;k++) axes[k]=rotate(mq,rotate(bq,basis[k]),true);
    double best=1e100;
    V normal{0,0,0};

    // Complete SAT. Iterate ACTUAL counts, never prefix-limited faces/edges.
    for(int k=0;k<3;k++)
        if(!test_axis(axes[k],h,center,axes,sizes,margin,best,normal)) return 0;
    for(int f=0;f<nf;f++) {
        if(!test_axis(face_normal(h,f),h,center,axes,sizes,margin,best,normal)) return 0;
        for(int e=0;e<3;e++) {
            int a=h.faces[3*f+e],b=h.faces[3*f+(e+1)%3];
            // Closed, consistently oriented triangles contain each undirected
            // edge twice, in opposite orders. One direction suffices. Interior
            // triangulation diagonals add harmless redundant separating axes.
            if(a>=b) continue;
            V edge=unit(point(h.vertices,b)-point(h.vertices,a));
            for(int k=0;k<3;k++)
                if(!test_axis(cross(edge,axes[k]),h,center,axes,sizes,
                              margin,best,normal)) return 0;
        }
    }
    if(best==1e100) return 0;
    if(best<0) return -1;

    double support=dot(normal,point(h.vertices));
    double span2=0;
    V first=point(h.vertices);
    for(int i=0;i<nv;i++) {
        V v=point(h.vertices,i);
        support=fmax(support,dot(normal,v));
        span2=fmax(span2,dot(v-first,v-first));
    }
    double span=fmax(sqrt(span2),2*fmax(sizes[0],fmax(sizes[1],sizes[2])));
    double tolerance=2e-7*span+1e-12;
    int patch_count=face_manifold(h,center,axes,sizes,normal,best,tolerance,span,margin,
                                 mp,mq,out_pos,out_norm,out_depth);
    if(patch_count) return patch_count;
    V tangent_center=center+normal*best;
    V corners[8];
    for(int c=0;c<8;c++) {
        corners[c]=tangent_center;
        for(int k=0;k<3;k++)
            corners[c]=corners[c]+axes[k]*((c&(1<<k))?sizes[k]:-sizes[k]);
    }

    // Find the intersection of the two EXTREMAL features at tangency. All
    // coplanar triangles contribute, not just the first face. Streaming
    // admitted vertices and clipped edge endpoints has no witness-count cap.
    // Two-dimensional patches use an area centroid of clipped supporting
    // triangles. One-dimensional patches use their geometric endpoint
    // midpoint, independent of repeated vertices or triangulation diagonals.
    Witnesses witnesses{};
    double area=0;
    V moment{0,0,0};
    for(int i=0;i<nv;i++) {
        V v=point(h.vertices,i);
        if(fabs(dot(normal,v)-support)<=tolerance)
            admit(v,h,tangent_center,axes,sizes,normal,support,tolerance,witnesses);
    }
    for(int c=0;c<8;c++)
        if(fabs(dot(normal,corners[c])-support)<=tolerance)
            admit(corners[c],h,tangent_center,axes,sizes,normal,support,tolerance,witnesses);
    for(int f=0;f<nf;f++) {
        V a=point(h.vertices,h.faces[3*f]);
        V b=point(h.vertices,h.faces[3*f+1]);
        V c=point(h.vertices,h.faces[3*f+2]);
        if(fabs(dot(normal,a)-support)<=tolerance &&
           fabs(dot(normal,b)-support)<=tolerance &&
           fabs(dot(normal,c)-support)<=tolerance &&
           !triangle_moment(a,b,c,tangent_center,axes,sizes,normal,tolerance,area,moment)) return 0;
        for(int e=0;e<3;e++) {
            int ia=h.faces[3*f+e],ib=h.faces[3*f+(e+1)%3];
            if(ia>=ib) continue;
            V a=point(h.vertices,ia),b=point(h.vertices,ib);
            if(fabs(dot(normal,a)-support)>tolerance ||
               fabs(dot(normal,b)-support)>tolerance) continue;
            double lo=0,hi=1;
            V direction=b-a;
            if(!clip_box(a,direction,tangent_center,axes,sizes,tolerance,lo,hi)) continue;
            if(lo>hi) lo=hi=(lo+hi)*0.5;
            admit(a+direction*lo,h,tangent_center,axes,sizes,normal,support,tolerance,witnesses);
            admit(a+direction*hi,h,tangent_center,axes,sizes,normal,support,tolerance,witnesses);
        }
    }
    for(int c=0;c<8;c++) for(int k=0;k<3;k++) {
        int other=c^(1<<k);
        if(c>=other || fabs(dot(normal,corners[c])-support)>tolerance ||
           fabs(dot(normal,corners[other])-support)>tolerance) continue;
        double lo=0,hi=1;
        V direction=corners[other]-corners[c];
        if(!clip_hull(corners[c],direction,h,tolerance,lo,hi)) continue;
        if(lo>hi) lo=hi=(lo+hi)*0.5;
        admit(corners[c]+direction*lo,h,tangent_center,axes,sizes,normal,support,tolerance,witnesses);
        admit(corners[c]+direction*hi,h,tangent_center,axes,sizes,normal,support,tolerance,witnesses);
    }
    V tangent;
    if(area>tolerance*tolerance) tangent=moment*(1.0/area);
    else {
        if(!witnesses.count) return 0;
        // Any coordinate with nonzero span orders points on a line. Choose
        // the widest one for conditioning; first-distinct noisy directions
        // and point multiplicities cannot affect the endpoint midpoint.
        int widest=0;
        for(int k=1;k<3;k++)
            if(component(witnesses.high[k]-witnesses.low[k],k)>
               component(witnesses.high[widest]-witnesses.low[widest],widest)) widest=k;
        tangent=(witnesses.low[widest]+witnesses.high[widest])*0.5;
    }
    V p=rotate(mq,tangent-normal*(0.5*best))+point(mp);
    V n=rotate(mq,normal);
    out_pos[0][0]=float(p.x);out_pos[0][1]=float(p.y);out_pos[0][2]=float(p.z);
    out_norm[0][0]=float(n.x);out_norm[0][1]=float(n.y);out_norm[0][2]=float(n.z);
    out_depth[0]=float(best+margin);
    return 1;
}

}}} // namespace hipgl::physics::mesh_box_detail
