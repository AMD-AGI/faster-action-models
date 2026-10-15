// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// hipGL - OSMesa/GL Symbol Shim Library
// LD_PRELOAD target: exports standard OpenGL and OSMesa symbols that forward to hipgl*.
// MuJoCo and other OSMesa apps link against this for drop-in replacement.

#include "hipgl/gl_api.h"
#include "hipgl/osmesa_compat.h"
#include <cstdio>
#include <cstring>

// ── OSMesa/GL types (match what MuJoCo expects) ───────────────────────────────
typedef unsigned char GLboolean;
typedef unsigned int GLenum;
typedef unsigned int GLuint;
typedef int GLint;
typedef float GLfloat;
typedef double GLdouble;
typedef unsigned char GLubyte;
typedef void GLvoid;
typedef ptrdiff_t GLsizeiptr;
typedef size_t GLsizei;
typedef unsigned int GLbitfield;

struct osmesa_context {};
typedef struct osmesa_context* OSMesaContext;

// ── OSMesa format constants (match OSMesa headers) ─────────────────────────────
#define OSMESA_RGBA        0x1908
#define OSMESA_BGRA        0x80E1
#define OSMESA_RGB         0x1907
#define OSMESA_BGR         0x80E0
#define OSMESA_RGB_565     0x8363
#define OSMESA_COLOR_INDEX  0x1900

// ── Stub helpers for unimplemented functions ────────────────────────────────
static void stub_warn_once(const char* name) {
    static bool warned[256];
    static int idx = 0;
    static const char* names[256];
    for (int i = 0; i < idx; i++) {
        if (names[i] && strcmp(names[i], name) == 0) return;
    }
    if (idx < 256) {
        names[idx] = name;
        fprintf(stderr, "[hipGL shim] %s: stub (no-op)\n", name);
        idx++;
    }
}

extern "C" {

// ═══════════════════════════════════════════════════════════════════════════════
// OSMesa symbols
// ═══════════════════════════════════════════════════════════════════════════════

OSMesaContext OSMesaCreateContextExt(unsigned int format, int depthBits, int stencilBits,
                                     int accumBits, OSMesaContext sharelist) {
    return (OSMesaContext)HIPGLOSMesaCreateContextExt(format, depthBits, stencilBits,
                                                       accumBits, (HIPGLOSMesaContext)sharelist);
}

OSMesaContext OSMesaCreateContext(unsigned int format, OSMesaContext sharelist) {
    return (OSMesaContext)HIPGLOSMesaCreateContext(format, (HIPGLOSMesaContext)sharelist);
}

GLboolean OSMesaMakeCurrent(OSMesaContext ctx, void* buffer, unsigned int type,
                            int width, int height) {
    return HIPGLOSMesaMakeCurrent((HIPGLOSMesaContext)ctx, buffer, type, width, height) ? 1 : 0;
}

void OSMesaDestroyContext(OSMesaContext ctx) {
    HIPGLOSMesaDestroyContext((HIPGLOSMesaContext)ctx);
}

OSMesaContext OSMesaGetCurrentContext(void) {
    return (OSMesaContext)HIPGLOSMesaGetCurrentContext();
}

GLboolean OSMesaGetDepthBuffer(OSMesaContext ctx, int* width, int* height,
                               int* bytesPerValue, void** buffer) {
    return HIPGLOSMesaGetDepthBuffer((HIPGLOSMesaContext)ctx, width, height,
                                     bytesPerValue, buffer) ? 1 : 0;
}

GLboolean OSMesaGetColorBuffer(OSMesaContext ctx, int* width, int* height,
                               int* format, void** buffer) {
    return HIPGLOSMesaGetColorBuffer((HIPGLOSMesaContext)ctx, width, height,
                                     format, buffer) ? 1 : 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// GL State
// ═══════════════════════════════════════════════════════════════════════════════

void glEnable(GLenum cap) { hipglEnable(cap); }
void glDisable(GLenum cap) { hipglDisable(cap); }
GLboolean glIsEnabled(GLenum cap) { return hipglIsEnabled(cap); }
void glDepthFunc(GLenum func) { hipglDepthFunc(func); }
void glDepthMask(GLboolean flag) { hipglDepthMask(flag); }
void glColorMask(GLboolean r, GLboolean g, GLboolean b, GLboolean a) {
    hipglColorMask(r, g, b, a);
}
void glBlendFunc(GLenum sfactor, GLenum dfactor) { hipglBlendFunc(sfactor, dfactor); }
void glShadeModel(GLenum mode) { hipglShadeModel(mode); }
void glCullFace(GLenum mode) { hipglCullFace(mode); }
void glFrontFace(GLenum mode) { hipglFrontFace(mode); }
void glPolygonMode(GLenum face, GLenum mode) { hipglPolygonMode(face, mode); }
void glPointSize(GLfloat size) { hipglPointSize(size); }
void glLineWidth(GLfloat width) { hipglLineWidth(width); }
void glPolygonOffset(GLfloat factor, GLfloat units) { (void)factor;(void)units; stub_warn_once("glPolygonOffset"); }
void glAlphaFunc(GLenum func, GLfloat ref) { (void)func;(void)ref; stub_warn_once("glAlphaFunc"); }
void glStencilFunc(GLenum func, GLint ref, GLuint mask) { (void)func;(void)ref;(void)mask; stub_warn_once("glStencilFunc"); }
void glStencilOp(GLenum fail, GLenum zfail, GLenum zpass) { (void)fail;(void)zfail;(void)zpass; stub_warn_once("glStencilOp"); }
void glClipPlane(GLenum plane, const GLdouble* equation) { (void)plane;(void)equation; stub_warn_once("glClipPlane"); }

// ═══════════════════════════════════════════════════════════════════════════════
// Clear
// ═══════════════════════════════════════════════════════════════════════════════

void glClearColor(GLfloat r, GLfloat g, GLfloat b, GLfloat a) {
    hipglClearColor(r, g, b, a);
}
void glClearDepth(GLdouble depth) { hipglClearDepth(depth); }
void glClearStencil(GLint s) { hipglClearStencil(s); }
void glClear(GLbitfield mask) { hipglClear(mask); }

// ═══════════════════════════════════════════════════════════════════════════════
// Viewport
// ═══════════════════════════════════════════════════════════════════════════════

void glViewport(GLint x, GLint y, GLsizei width, GLsizei height) {
    hipglViewport(x, y, width, height);
}
void glScissor(GLint x, GLint y, GLsizei width, GLsizei height) {
    hipglScissor(x, y, width, height);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Matrix
// ═══════════════════════════════════════════════════════════════════════════════

void glMatrixMode(GLenum mode) { hipglMatrixMode(mode); }
void glLoadIdentity(void) { hipglLoadIdentity(); }
void glLoadMatrixf(const GLfloat* m) { hipglLoadMatrixf(m); }
void glMultMatrixf(const GLfloat* m) { hipglMultMatrixf(m); }
void glPushMatrix(void) { hipglPushMatrix(); }
void glPopMatrix(void) { hipglPopMatrix(); }
void glTranslatef(GLfloat x, GLfloat y, GLfloat z) { hipglTranslatef(x, y, z); }
void glRotatef(GLfloat angle, GLfloat x, GLfloat y, GLfloat z) {
    hipglRotatef(angle, x, y, z);
}
void glScalef(GLfloat x, GLfloat y, GLfloat z) { hipglScalef(x, y, z); }
void glOrtho(GLdouble left, GLdouble right, GLdouble bottom, GLdouble top,
             GLdouble near_val, GLdouble far_val) {
    hipglOrtho(left, right, bottom, top, near_val, far_val);
}
void glFrustum(GLdouble left, GLdouble right, GLdouble bottom, GLdouble top,
               GLdouble near_val, GLdouble far_val) {
    hipglFrustum(left, right, bottom, top, near_val, far_val);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Immediate mode
// ═══════════════════════════════════════════════════════════════════════════════

void glBegin(GLenum mode) { hipglBegin(mode); }
void glEnd(void) { hipglEnd(); }
void glVertex2f(GLfloat x, GLfloat y) { hipglVertex3f(x, y, 0.0f); }
void glVertex3f(GLfloat x, GLfloat y, GLfloat z) { hipglVertex3f(x, y, z); }
void glVertex3fv(const GLfloat* v) { hipglVertex3fv(v); }
void glNormal3f(GLfloat x, GLfloat y, GLfloat z) { hipglNormal3f(x, y, z); }
void glNormal3fv(const GLfloat* v) { hipglNormal3fv(v); }
void glColor3f(GLfloat r, GLfloat g, GLfloat b) { hipglColor3f(r, g, b); }
void glColor4f(GLfloat r, GLfloat g, GLfloat b, GLfloat a) { hipglColor4f(r, g, b, a); }
void glColor3fv(const GLfloat* v) { hipglColor3fv(v); }
void glColor4fv(const GLfloat* v) { hipglColor4fv(v); }
void glColor4ubv(const GLubyte* v) {
    if (v) hipglColor4f(v[0]/255.0f, v[1]/255.0f, v[2]/255.0f, v[3]/255.0f);
}
void glTexCoord2f(GLfloat s, GLfloat t) { hipglTexCoord2f(s, t); }
void glTexCoord2fv(const GLfloat* v) { if (v) hipglTexCoord2f(v[0], v[1]); }

// ═══════════════════════════════════════════════════════════════════════════════
// Lighting
// ═══════════════════════════════════════════════════════════════════════════════

void glLightfv(GLenum light, GLenum pname, const GLfloat* params) {
    hipglLightfv(light, pname, params);
}
void glLightf(GLenum light, GLenum pname, GLfloat param) { hipglLightf(light, pname, param); }
void glMaterialfv(GLenum face, GLenum pname, const GLfloat* params) {
    hipglMaterialfv(face, pname, params);
}
void glMaterialf(GLenum face, GLenum pname, GLfloat param) {
    hipglMaterialf(face, pname, param);
}
void glColorMaterial(GLenum face, GLenum mode) { hipglColorMaterial(face, mode); }
void glLightModelfv(GLenum pname, const GLfloat* params) {
    hipglLightModelfv(pname, params);
}
void glLightModeli(GLenum pname, GLint param) {
    hipglLightModeli(pname, param);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Fog
// ═══════════════════════════════════════════════════════════════════════════════

void glFogi(GLenum pname, GLint param) { (void)pname;(void)param; stub_warn_once("glFogi"); }
void glFogf(GLenum pname, GLfloat param) { (void)pname;(void)param; stub_warn_once("glFogf"); }
void glFogfv(GLenum pname, const GLfloat* params) { (void)pname;(void)params; stub_warn_once("glFogfv"); }

// ═══════════════════════════════════════════════════════════════════════════════
// Buffers
// ═══════════════════════════════════════════════════════════════════════════════

void glGenBuffers(GLsizei n, GLuint* buffers) { hipglGenBuffers(n, buffers); }
void glDeleteBuffers(GLsizei n, const GLuint* buffers) { hipglDeleteBuffers(n, buffers); }
void glBindBuffer(GLenum target, GLuint buffer) { hipglBindBuffer(target, buffer); }
void glBufferData(GLenum target, GLsizeiptr size, const GLvoid* data, GLenum usage) {
    hipglBufferData(target, size, data, usage);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Client state
// ═══════════════════════════════════════════════════════════════════════════════

void glEnableClientState(GLenum cap) { hipglEnableClientState(cap); }
void glDisableClientState(GLenum cap) { hipglDisableClientState(cap); }
void glVertexPointer(GLint size, GLenum type, GLsizei stride, const GLvoid* ptr) {
    hipglVertexPointer(size, type, stride, ptr);
}
void glNormalPointer(GLenum type, GLsizei stride, const GLvoid* ptr) {
    hipglNormalPointer(type, stride, ptr);
}
void glColorPointer(GLint size, GLenum type, GLsizei stride, const GLvoid* ptr) {
    hipglColorPointer(size, type, stride, ptr);
}
void glTexCoordPointer(GLint size, GLenum type, GLsizei stride, const GLvoid* ptr) {
    hipglTexCoordPointer(size, type, stride, ptr);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Draw
// ═══════════════════════════════════════════════════════════════════════════════

void glDrawArrays(GLenum mode, GLint first, GLsizei count) {
    hipglDrawArrays(mode, first, count);
}
void glDrawElements(GLenum mode, GLsizei count, GLenum type, const GLvoid* indices) {
    hipglDrawElements(mode, count, type, indices);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Textures
// ═══════════════════════════════════════════════════════════════════════════════

void glGenTextures(GLsizei n, GLuint* textures) { hipglGenTextures(n, textures); }
void glDeleteTextures(GLsizei n, const GLuint* textures) { hipglDeleteTextures(n, textures); }
void glBindTexture(GLenum target, GLuint texture) { hipglBindTexture(target, texture); }
void glActiveTexture(GLenum texture) { (void)texture; stub_warn_once("glActiveTexture"); }
void glTexImage2D(GLenum target, GLint level, GLint internalformat, GLsizei width,
                  GLsizei height, GLint border, GLenum format, GLenum type,
                  const GLvoid* data) {
    hipglTexImage2D(target, level, internalformat, width, height, border, format, type, data);
}
void glTexParameteri(GLenum target, GLenum pname, GLint param) {
    hipglTexParameteri(target, pname, param);
}
void glTexEnvi(GLenum target, GLenum pname, GLint param) { (void)target;(void)pname;(void)param; stub_warn_once("glTexEnvi"); }
void glTexGeni(GLenum coord, GLenum pname, GLint param) { (void)coord;(void)pname;(void)param; stub_warn_once("glTexGeni"); }
void glTexGenfv(GLenum coord, GLenum pname, const GLfloat* params) {
    (void)coord;(void)pname;(void)params; stub_warn_once("glTexGenfv");
}
void glGenerateMipmap(GLenum target) { (void)target; stub_warn_once("glGenerateMipmap"); }

// ═══════════════════════════════════════════════════════════════════════════════
// FBO (stubs - hipGL uses OSMesa buffer directly)
// ═══════════════════════════════════════════════════════════════════════════════

void glGenFramebuffers(GLsizei n, GLuint* framebuffers) {
    (void)n;(void)framebuffers; stub_warn_once("glGenFramebuffers");
}
void glBindFramebuffer(GLenum target, GLuint framebuffer) {
    (void)target;(void)framebuffer; stub_warn_once("glBindFramebuffer");
}
void glDeleteFramebuffers(GLsizei n, const GLuint* framebuffers) {
    (void)n;(void)framebuffers; stub_warn_once("glDeleteFramebuffers");
}
void glGenRenderbuffers(GLsizei n, GLuint* renderbuffers) {
    (void)n;(void)renderbuffers; stub_warn_once("glGenRenderbuffers");
}
void glBindRenderbuffer(GLenum target, GLuint renderbuffer) {
    (void)target;(void)renderbuffer; stub_warn_once("glBindRenderbuffer");
}
void glDeleteRenderbuffers(GLsizei n, const GLuint* renderbuffers) {
    (void)n;(void)renderbuffers; stub_warn_once("glDeleteRenderbuffers");
}
void glRenderbufferStorage(GLenum target, GLenum internalformat,
                           GLsizei width, GLsizei height) {
    (void)target;(void)internalformat;(void)width;(void)height; stub_warn_once("glRenderbufferStorage");
}
void glRenderbufferStorageMultisample(GLenum target, GLsizei samples,
                                      GLenum internalformat, GLsizei width, GLsizei height) {
    (void)target;(void)samples;(void)internalformat;(void)width;(void)height; stub_warn_once("glRenderbufferStorageMultisample");
}
void glFramebufferTexture2D(GLenum target, GLenum attachment, GLenum textarget,
                            GLuint texture, GLint level) {
    (void)target;(void)attachment;(void)textarget;(void)texture;(void)level; stub_warn_once("glFramebufferTexture2D");
}
void glFramebufferRenderbuffer(GLenum target, GLenum attachment,
                               GLenum renderbuffertarget, GLuint renderbuffer) {
    (void)target;(void)attachment;(void)renderbuffertarget;(void)renderbuffer; stub_warn_once("glFramebufferRenderbuffer");
}
GLenum glCheckFramebufferStatus(GLenum target) {
    (void)target; stub_warn_once("glCheckFramebufferStatus"); return 0x8CD5; /* GL_FRAMEBUFFER_COMPLETE */
}
void glGetRenderbufferParameteriv(GLenum target, GLenum pname, GLint* params) {
    (void)target;(void)pname;(void)params; stub_warn_once("glGetRenderbufferParameteriv");
}

// ═══════════════════════════════════════════════════════════════════════════════
// Display lists (stubs)
// ═══════════════════════════════════════════════════════════════════════════════

GLuint glGenLists(GLsizei range) { (void)range; stub_warn_once("glGenLists"); return 1; }
void glDeleteLists(GLuint list, GLsizei range) { (void)list;(void)range; stub_warn_once("glDeleteLists"); }
void glNewList(GLuint list, GLenum mode) { (void)list;(void)mode; stub_warn_once("glNewList"); }
void glEndList(void) { stub_warn_once("glEndList"); }
void glCallList(GLuint list) { (void)list; stub_warn_once("glCallList"); }

// ═══════════════════════════════════════════════════════════════════════════════
// Readback
// ═══════════════════════════════════════════════════════════════════════════════

void glReadPixels(GLint x, GLint y, GLsizei width, GLsizei height,
                  GLenum format, GLenum type, GLvoid* data) {
    hipglReadPixels(x, y, width, height, format, type, data);
}
void glPixelStorei(GLenum pname, GLint param) { hipglPixelStorei(pname, param); }
void glDrawBuffer(GLenum mode) { (void)mode; stub_warn_once("glDrawBuffer"); }
void glReadBuffer(GLenum mode) { (void)mode; stub_warn_once("glReadBuffer"); }

// ═══════════════════════════════════════════════════════════════════════════════
// Query
// ═══════════════════════════════════════════════════════════════════════════════

GLenum glGetError(void) { return hipglGetError(); }
const GLubyte* glGetString(GLenum name) { return hipglGetString(name); }
void glGetIntegerv(GLenum pname, GLint* params) { hipglGetIntegerv(pname, params); }
void glGetFloatv(GLenum pname, GLfloat* params) { hipglGetFloatv(pname, params); }
void glGetBooleanv(GLenum pname, GLboolean* params) {
    if (!params) return;
    (void)pname; stub_warn_once("glGetBooleanv"); *params = 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Sync
// ═══════════════════════════════════════════════════════════════════════════════

void glFlush(void) { hipglFlush(); }
void glFinish(void) { hipglFinish(); }

// ═══════════════════════════════════════════════════════════════════════════════
// Bitmap
// ═══════════════════════════════════════════════════════════════════════════════

void glBitmap(GLsizei width, GLsizei height, GLfloat xorig, GLfloat yorig,
              GLfloat xmove, GLfloat ymove, const GLubyte* bitmap) {
    (void)width;(void)height;(void)xorig;(void)yorig;(void)xmove;(void)ymove;(void)bitmap;
    stub_warn_once("glBitmap");
}

// ═══════════════════════════════════════════════════════════════════════════════
// VAO
// ═══════════════════════════════════════════════════════════════════════════════

void glGenVertexArrays(GLsizei n, GLuint* arrays) { hipglGenVertexArrays(n, arrays); }
void glBindVertexArray(GLuint array) { hipglBindVertexArray(array); }

} // extern "C"
