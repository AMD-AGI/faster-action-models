// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// Included inside hipgl::kernels after the shared fragment shader. Keeping
// this separate from the tuned single-sample kernel leaves its occupancy and
// dispatch unchanged. The framebuffer stores four native-resolution samples;
// colors are shaded at the pixel center, as llvmpipe's fixed-function GL does.

static_assert(TILE_SIZE == 16, "Native MSAA retains 16x16 binning tiles");

HIPGL_DEVICE inline uint32_t msaa_pack(float r, float g, float b, float a) {
    return (uint32_t)(clamp01(r) * 255.f + .5f)
         | ((uint32_t)(clamp01(g) * 255.f + .5f) << 8)
         | ((uint32_t)(clamp01(b) * 255.f + .5f) << 16)
         | ((uint32_t)(clamp01(a) * 255.f + .5f) << 24);
}

HIPGL_GLOBAL void kernel_clear_msaa(DeviceFramebuffer fb, RenderState st,
                                    bool color, bool depth) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= fb.width * fb.height) return;
    if (color) {
        float r = st.clear_color[0], g = st.clear_color[1], b = st.clear_color[2];
        if (st.gradient_bg) {
            const float t = fb.height > 1 ? float(p / fb.width) / (fb.height - 1) : 0.f;
            r = st.gradient_bot[0] + t * (st.gradient_top[0] - st.gradient_bot[0]);
            g = st.gradient_bot[1] + t * (st.gradient_top[1] - st.gradient_bot[1]);
            b = st.gradient_bot[2] + t * (st.gradient_top[2] - st.gradient_bot[2]);
        }
        const uint32_t c = msaa_pack(r, g, b, st.clear_color[3]);
        reinterpret_cast<uint4*>(fb.sample_color)[p] = make_uint4(c,c,c,c);
    }
    if (depth) {
        const float z = st.clear_depth;
        reinterpret_cast<float4*>(fb.sample_depth)[p] = make_float4(z,z,z,z);
    }
}

HIPGL_GLOBAL void kernel_resolve_msaa(DeviceFramebuffer fb) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= fb.width * fb.height) return;
    const uint4 c = reinterpret_cast<const uint4*>(fb.sample_color)[p];
    float v[4];
#pragma unroll
    for (int k=0; k<4; ++k) {
        const int shift = k * 8;
        const unsigned sum = ((c.x >> shift) & 255u) + ((c.y >> shift) & 255u)
                           + ((c.z >> shift) & 255u) + ((c.w >> shift) & 255u);
        v[k] = sum * (1.f / (4.f * 255.f));
    }
    reinterpret_cast<float4*>(fb.color_buffer)[p] = make_float4(v[0],v[1],v[2],v[3]);
    fb.depth_buffer[p] = fb.sample_depth[p * 4];
}

void launch_clear_msaa(DeviceFramebuffer fb, const RenderState& state,
                       bool color, bool depth, hipStream_t stream) {
    const int n = fb.width * fb.height;
    if (n <= 0 || !fb.sample_color || !fb.sample_depth) return;
    hipLaunchKernelGGL(kernel_clear_msaa, (n+255)/256, 256, 0, stream,
                       fb, state, color, depth);
}

void launch_resolve_msaa(DeviceFramebuffer fb, hipStream_t stream) {
    const int n = fb.width * fb.height;
    if (n <= 0 || !fb.sample_color || !fb.sample_depth) return;
    hipLaunchKernelGGL(kernel_resolve_msaa, (n+255)/256, 256, 0, stream, fb);
}

template <bool GOURAUD>
HIPGL_DEVICE inline bool msaa_shade_center(const TriangleSetup* tris, int id,
    float e0, float e1, float e2, int p, const DeviceFramebuffer& fb,
    const RenderState& state, const TextureState& texture,
    float& r, float& g, float& b, float& a) {
    float depth = 0.f, ar = 0.f, ag = 0.f, ab = 0.f, aa = 0.f, reveal = 1.f;
    bool wrote = false;
    hipgl_fine_shade_sample<false, false, GOURAUD, true, true>(tris, id,
        e0,e1,e2,p,p*4,fb,state,texture,depth,r,g,b,a,ar,ag,ab,aa,reveal,wrote);
    return wrote;
}

template <bool GOURAUD, bool BLEND, int RASTER_SIZE>
HIPGL_GLOBAL void __launch_bounds__(RASTER_SIZE * RASTER_SIZE) kernel_fine_msaa(
    const TriangleSetup* __restrict__ tris, const int* __restrict__ base,
    const int* __restrict__ counts, const unsigned long long* __restrict__ keys,
    int cap, int tiles_x, DeviceFramebuffer fb, RenderState state, TextureState texture) {
    constexpr int CHUNK = RASTER_SIZE < 16 ? RASTER_SIZE * RASTER_SIZE : 128;
    constexpr int SUBDIV = TILE_SIZE / RASTER_SIZE;
    static_assert(RASTER_SIZE == 4 || RASTER_SIZE == 8 || RASTER_SIZE == TILE_SIZE, "Unsupported MSAA raster size");
    const bool compact = RASTER_SIZE < 16 && warpSize == 64;
    __shared__ unsigned long long skey[CHUNK];
    __shared__ float4 edges[CHUNK][3];
    const int tid = threadIdx.x;
    const int tile = blockIdx.x / (SUBDIV * SUBDIV);
    const int sub = blockIdx.x % (SUBDIV * SUBDIV);
    const int tx = tile % tiles_x, ty = tile / tiles_x;
    const int sub_x = (sub % SUBDIV) * RASTER_SIZE;
    const int sub_y = (sub / SUBDIV) * RASTER_SIZE;
    const int local_x = sub_x + tid % RASTER_SIZE;
    const int local_y = sub_y + tid / RASTER_SIZE;
    const int px = tx * TILE_SIZE + local_x;
    const int py = ty * TILE_SIZE + local_y;
    const bool valid = px < fb.width && py < fb.height;
    const int p = valid ? py * fb.width + px : 0;
    const int off = base[tile];
    int count = counts[tile];
    if (off < 0 || off >= cap) return;
    count = min(count, cap-off);
    if (count <= 0) return;
    const float4 zin = reinterpret_cast<const float4*>(fb.sample_depth)[p];
    float z[4] = {zin.x, zin.y, zin.z, zin.w};
    int winner[4] = {-1,-1,-1,-1};
    uint32_t colors[4];
    if constexpr (BLEND) {
        const uint4 c = reinterpret_cast<const uint4*>(fb.sample_color)[p];
        colors[0]=c.x; colors[1]=c.y; colors[2]=c.z; colors[3]=c.w;
    }
    bool done = !valid || (state.reflect_mask && !state.reflect_mask[p]);
    if (__syncthreads_and(done)) return;
    // Source-over needs far-to-near traversal. The opaque mirror's additive
    // base pass writes depth, so nearest-first rejects hidden rear surfaces.
    const bool back_to_front=BLEND && !(state.blend_src==BLEND_ONE && state.blend_dst==BLEND_ONE);
    const float x0 = tx * TILE_SIZE + .5f, y0 = ty * TILE_SIZE + .5f;
    for (int c0=0; c0<count; c0+=CHUNK) {
        const int cn = min(CHUNK,count-c0);
        bool keep = false;
        if (tid < cn) {
            // Source-over blending composes the far surface first. Reverse
            // depth groups, preserving submission order inside each group.
            // Reversing the whole key would also reverse coplanar triangles.
            const int entry = back_to_front ? count-1-(c0+tid) : c0+tid;
            unsigned long long key = keys[off+entry];
            if constexpr (BLEND) {
              if (back_to_front) {
                const uint32_t depth_key = (uint32_t)(key >> 32);
                const bool tied = (entry > 0 && (uint32_t)(keys[off+entry-1] >> 32) == depth_key)
                    || (entry+1 < count && (uint32_t)(keys[off+entry+1] >> 32) == depth_key);
                if (tied) {
                    int lo=0, hi=entry;
                    while (lo < hi) {
                        const int mid=lo+(hi-lo)/2;
                        if ((uint32_t)(keys[off+mid] >> 32) < depth_key) lo=mid+1;
                        else hi=mid;
                    }
                    const int first=lo;
                    lo=entry+1; hi=count;
                    while (lo < hi) {
                        const int mid=lo+(hi-lo)/2;
                        if ((uint32_t)(keys[off+mid] >> 32) <= depth_key) lo=mid+1;
                        else hi=mid;
                    }
                    key=keys[off+first+(lo-1-entry)];
                }
              }
            }
            const int physical_id = hipgl_physical_triangle_id((uint32_t)key, state);
            key = (key & 0xFFFFFFFF00000000ull) | (uint32_t)physical_id;
            const TriangleSetup& t = tris[physical_id];
            keep = !t.degenerate && !triangle_is_culled(t,state);
            if (keep) {
#pragma unroll
                for (int e=0; e<3; ++e) {
                    const float a=t.edge_eq[e][0], b=t.edge_eq[e][1];
                    const float ec=(float)((double)a*x0+(double)b*y0+t.edge_eq[e][2]);
                    edges[tid][e]=make_float4(ec,a,b,t.edge_bias[e]);
                    // Include the entire sample footprint, not just centers.
                    const float pad=.375f*(fabsf(a)+fabsf(b));
                    const float hi=ec+a*sub_x+b*sub_y+fmaxf(a,0.f)*(RASTER_SIZE-1)
                        +fmaxf(b,0.f)*(RASTER_SIZE-1)+pad+t.edge_bias[e];
                    const float mag=fabsf(ec)+(fabsf(a)+fabsf(b))*TILE_SIZE+1.f;
                    if (hi < -1.e-5f*mag) keep=false;
                }
            }
            skey[tid] = keep ? key : ~0ull;
        }
        // Small jobs occupy at most one 64-lane wave. Walk retained triangles,
        // in the original sorted order, instead of scanning rejected entries
        // again in every pixel lane. The 16x16 fallback spans several waves.
        unsigned long long remaining = 0;
        if (compact) remaining = __ballot(keep);
        __syncthreads();
        for (int entry=0; entry<cn && !done; ++entry) {
            int j = entry;
            if (compact) {
                if (!remaining) break;
                j = __ffsll(remaining) - 1;
                remaining &= remaining - 1;
            }
            const unsigned long long key=skey[j];
            if (key == ~0ull) continue;
            if constexpr (!BLEND) {
                const float farthest=fmaxf(fmaxf(z[0],z[1]),fmaxf(z[2],z[3]));
                if (state.depth_test_enabled && state.depth_func==DEPTH_LEQUAL &&
                    (uint32_t)(key>>32)>hipgl_f2ord(farthest)) { done=true; break; }
            }
            const int id=(uint32_t)key;
            float center[3], u[3], v[3];
            unsigned coverage=15u;
#pragma unroll
            for (int e=0; e<3; ++e) {
                const float4 q=edges[j][e];
                // Retain the parent tile's anchor and arithmetic order so
                // changing job size does not change coverage or interpolation.
                center[e]=q.x+q.y*local_x+q.z*local_y;
                // GL sample locations with Y mirrored into HP screen space:
                // (.375,.875), (.875,.625), (.125,.375), (.625,.125).
                u[e]=(q.y-3.f*q.z)*.125f;
                v[e]=(3.f*q.y+q.z)*.125f;
                const float c=center[e]+q.w;
                coverage &= (c-u[e]>=0.f ? 1u : 0u) | (c+v[e]>=0.f ? 2u : 0u)
                          | (c-v[e]>=0.f ? 4u : 0u) | (c+u[e]>=0.f ? 8u : 0u);
            }
            if (state.reflect_mask) coverage &= state.reflect_mask[p];
            if (!coverage) continue;
            const TriangleSetup& t=tris[id];
            unsigned passing=0u;
#pragma unroll
            for (int s=0; s<4; ++s) {
                if (!(coverage & (1u<<s))) continue;
                const float b0=center[1]+(s==0 ? -u[1] : s==1 ? v[1] : s==2 ? -v[1] : u[1]);
                const float b1=center[2]+(s==0 ? -u[2] : s==1 ? v[2] : s==2 ? -v[2] : u[2]);
                const float b2=center[0]+(s==0 ? -u[0] : s==1 ? v[0] : s==2 ? -v[0] : u[0]);
                if (state.clip_plane_enabled) {
                    const float w0=b0*t.v[0].inv_w, w1=b1*t.v[1].inv_w, w2=b2*t.v[2].inv_w;
                    const float inv=1.f/(w0+w1+w2);
                    float d=-state.clip_plane[3];
#pragma unroll
                    for (int k=0;k<3;++k)
                        d+=state.clip_plane[k]*(w0*t.v[0].world_pos[k]+w1*t.v[1].world_pos[k]+w2*t.v[2].world_pos[k])*inv;
                    if (d<0.f) continue;
                }
                if (state.tri_stats) state.tri_stats[id*TRI_STAT_COUNT+TRI_STAT_COVERED]=1;
                const float depth=(b0*t.v[0].screen_z+b1*t.v[1].screen_z+b2*t.v[2].screen_z)/(b0+b1+b2);
                if (state.depth_test_enabled && !depth_test_passes(state.depth_func,depth,z[s])) continue;
                if (state.tri_stats) state.tri_stats[id*TRI_STAT_COUNT+TRI_STAT_DEPTH_PASS]=1;
                passing |= 1u<<s;
                if (state.depth_write_enabled) z[s]=depth;
                if constexpr (!BLEND) winner[s]=id;
            }
            if constexpr (BLEND) {
                if (passing) {
                    float r=0,g=0,b=0,a=0;
                    if (msaa_shade_center<GOURAUD>(tris,id,center[0],center[1],center[2],p,fb,state,texture,r,g,b,a)) {
#pragma unroll
                        for (int s=0;s<4;++s) if (passing&(1u<<s)) {
                            const uint32_t c=colors[s];
                            if (state.blend_src==BLEND_ONE && state.blend_dst==BLEND_ONE) {
                                constexpr float inv=1.f/255.f;
                                colors[s]=msaa_pack(r+(c&255u)*inv,g+((c>>8)&255u)*inv,
                                    b+((c>>16)&255u)*inv,a+((c>>24)&255u)*inv);
                            } else {
                                const float inv=(1.f-a)/255.f;
                                colors[s]=msaa_pack(r*a+(c&255u)*inv,g*a+((c>>8)&255u)*inv,
                                    b*a+((c>>16)&255u)*inv,a+((c>>24)&255u)*inv);
                            }
                        }
                    }
                }
            }
        }
        if (__syncthreads_and(done)) break;
    }
    if (!valid) return;
    if (state.depth_write_enabled)
        reinterpret_cast<float4*>(fb.sample_depth)[p]=make_float4(z[0],z[1],z[2],z[3]);
    if constexpr (BLEND) {
        reinterpret_cast<uint4*>(fb.sample_color)[p]=make_uint4(colors[0],colors[1],colors[2],colors[3]);
    } else {
        // Only final winners are shaded. One triangle winning several sample
        // slots is shaded once, then its packed result is replicated.
#pragma unroll
        for (int s=0;s<4;++s) {
            const int id=winner[s];
            bool seen=id<0;
#pragma unroll
            for (int k=0;k<s;++k) seen |= winner[k]==id;
            if (seen) continue;
            const TriangleSetup& t=tris[id];
            float e[3];
#pragma unroll
            for (int k=0;k<3;++k)
                e[k]=(float)((double)t.edge_eq[k][0]*(px+.5f)+(double)t.edge_eq[k][1]*(py+.5f)+t.edge_eq[k][2]);
            float r=0,g=0,b=0,a=0;
            if (msaa_shade_center<GOURAUD>(tris,id,e[0],e[1],e[2],p,fb,state,texture,r,g,b,a)) {
                const uint32_t c=msaa_pack(r,g,b,a);
#pragma unroll
                for (int k=s;k<4;++k) if (winner[k]==id) fb.sample_color[p*4+k]=c;
            }
        }
    }
}

template <int RASTER_SIZE>
void launch_fine_rasterize_msaa_sized(const TriangleSetup* tris,
    const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture, hipStream_t stream) {
    if (!tris || !tile_base || !tile_counts || !pool_key || pool_cap<=0) return;
    constexpr int subdiv=TILE_SIZE/RASTER_SIZE;
    constexpr int threads=RASTER_SIZE*RASTER_SIZE;
    const int n=tiles_x*tiles_y*subdiv*subdiv;
    const bool gouraud=state.gouraud_shading && state.lighting_enabled;
    if (gouraud && state.blend_enabled)
        hipLaunchKernelGGL((kernel_fine_msaa<true,true,RASTER_SIZE>),n,threads,0,stream,tris,tile_base,tile_counts,pool_key,pool_cap,tiles_x,fb,state,texture);
    else if (gouraud)
        hipLaunchKernelGGL((kernel_fine_msaa<true,false,RASTER_SIZE>),n,threads,0,stream,tris,tile_base,tile_counts,pool_key,pool_cap,tiles_x,fb,state,texture);
    else if (state.blend_enabled)
        hipLaunchKernelGGL((kernel_fine_msaa<false,true,RASTER_SIZE>),n,threads,0,stream,tris,tile_base,tile_counts,pool_key,pool_cap,tiles_x,fb,state,texture);
    else
        hipLaunchKernelGGL((kernel_fine_msaa<false,false,RASTER_SIZE>),n,threads,0,stream,tris,tile_base,tile_counts,pool_key,pool_cap,tiles_x,fb,state,texture);
}

void launch_fine_rasterize_msaa(const TriangleSetup* tris,
    const int* tile_base, const int* tile_counts,
    const unsigned long long* pool_key, int pool_cap,
    int tiles_x, int tiles_y, DeviceFramebuffer fb,
    const RenderState& state, const TextureState& texture, hipStream_t stream) {
    static const int raster_override=[] {
        const char* value=std::getenv("HIPGL_MSAA_SUBTILE");
        return value ? std::atoi(value) : 0;
    }();
    // Small native framebuffers have few, dense bins. Smaller jobs shorten
    // the shared traversal without changing the 16x16 binning/sort pipeline.
    // Keep the established mapping for larger grids until separately tuned.
    const int raster_size = raster_override == 4 || raster_override == 8 || raster_override == 16
        ? raster_override : (tiles_x * tiles_y <= 256 ? 4 : 16);
    if (raster_size==4)
        launch_fine_rasterize_msaa_sized<4>(tris,tile_base,tile_counts,pool_key,pool_cap,
            tiles_x,tiles_y,fb,state,texture,stream);
    else if (raster_size==8)
        launch_fine_rasterize_msaa_sized<8>(tris,tile_base,tile_counts,pool_key,pool_cap,
            tiles_x,tiles_y,fb,state,texture,stream);
    else
        launch_fine_rasterize_msaa_sized<16>(tris,tile_base,tile_counts,pool_key,pool_cap,
            tiles_x,tiles_y,fb,state,texture,stream);
}
