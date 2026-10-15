#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// hipGL - OpenGL-compatible C API
// Drop-in replacement for OpenGL on AMD Instinct™ GPUs (CDNA).
// Define HIPGL_REPLACE_GL before including to remap standard GL names.

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── GL Constants ────────────────────────────────────────────────────────────

// Boolean
#define HIPGL_FALSE               0
#define HIPGL_TRUE                1

// Primitive types
#define HIPGL_POINTS              0x0000
#define HIPGL_LINES               0x0001
#define HIPGL_LINE_STRIP          0x0002
#define HIPGL_LINE_LOOP           0x0003
#define HIPGL_TRIANGLES           0x0004
#define HIPGL_TRIANGLE_STRIP      0x0005
#define HIPGL_TRIANGLE_FAN        0x0006
#define HIPGL_QUADS               0x0007

// Buffer bits
#define HIPGL_DEPTH_BUFFER_BIT    0x00000100
#define HIPGL_STENCIL_BUFFER_BIT  0x00000400
#define HIPGL_COLOR_BUFFER_BIT    0x00004000

// Enable caps
#define HIPGL_CULL_FACE           0x0B44
#define HIPGL_DEPTH_TEST          0x0B71
#define HIPGL_STENCIL_TEST       0x0B90
#define HIPGL_BLEND               0x0BE2
#define HIPGL_SCISSOR_TEST        0x0C11
#define HIPGL_TEXTURE_2D          0x0DE1
#define HIPGL_LIGHTING            0x0B50
#define HIPGL_LIGHT0              0x4000
#define HIPGL_LIGHT1              0x4001
#define HIPGL_LIGHT2              0x4002
#define HIPGL_LIGHT3              0x4003
#define HIPGL_LIGHT4              0x4004
#define HIPGL_LIGHT5              0x4005
#define HIPGL_LIGHT6              0x4006
#define HIPGL_LIGHT7              0x4007
#define HIPGL_COLOR_MATERIAL      0x0B57

// Depth functions
#define HIPGL_NEVER               0x0200
#define HIPGL_LESS                0x0201
#define HIPGL_EQUAL               0x0202
#define HIPGL_LEQUAL              0x0203
#define HIPGL_GREATER             0x0204
#define HIPGL_NOTEQUAL            0x0205
#define HIPGL_GEQUAL              0x0206
#define HIPGL_ALWAYS              0x0207

// Blend factors
#define HIPGL_ZERO                0
#define HIPGL_ONE                 1
#define HIPGL_SRC_ALPHA           0x0302
#define HIPGL_ONE_MINUS_SRC_ALPHA 0x0303
#define HIPGL_DST_ALPHA           0x0304
#define HIPGL_ONE_MINUS_DST_ALPHA 0x0305
#define HIPGL_SRC_COLOR           0x0300
#define HIPGL_ONE_MINUS_SRC_COLOR 0x0301
#define HIPGL_DST_COLOR           0x0306
#define HIPGL_ONE_MINUS_DST_COLOR 0x0307

// Face
#define HIPGL_FRONT               0x0404
#define HIPGL_BACK                0x0405
#define HIPGL_FRONT_AND_BACK      0x0408

// Winding
#define HIPGL_CW                  0x0900
#define HIPGL_CCW                 0x0901

// Polygon mode
#define HIPGL_POINT               0x1B00
#define HIPGL_LINE                0x1B01
#define HIPGL_FILL                0x1B02

// Shade model
#define HIPGL_FLAT                0x1D00
#define HIPGL_SMOOTH              0x1D01

// Matrix mode
#define HIPGL_MODELVIEW           0x1700
#define HIPGL_PROJECTION          0x1701
#define HIPGL_TEXTURE             0x1702

// Data types
#define HIPGL_BYTE                0x1400
#define HIPGL_UNSIGNED_BYTE       0x1401
#define HIPGL_SHORT               0x1402
#define HIPGL_UNSIGNED_SHORT      0x1403
#define HIPGL_INT                 0x1404
#define HIPGL_UNSIGNED_INT        0x1405
#define HIPGL_FLOAT               0x1406
#define HIPGL_DOUBLE              0x140A

// Pixel formats
#define HIPGL_RED                 0x1903
#define HIPGL_GREEN               0x1904
#define HIPGL_BLUE                0x1905
#define HIPGL_ALPHA               0x1906
#define HIPGL_RGB                 0x1907
#define HIPGL_RGBA                0x1908
#define HIPGL_LUMINANCE           0x1909
#define HIPGL_LUMINANCE_ALPHA     0x190A
#define HIPGL_DEPTH_COMPONENT     0x1902

// Buffer targets
#define HIPGL_ARRAY_BUFFER        0x8892
#define HIPGL_ELEMENT_ARRAY_BUFFER 0x8893

// Buffer usage
#define HIPGL_STREAM_DRAW         0x88E0
#define HIPGL_STATIC_DRAW         0x88E4
#define HIPGL_DYNAMIC_DRAW        0x88E8

// Texture parameters
#define HIPGL_TEXTURE_MAG_FILTER  0x2800
#define HIPGL_TEXTURE_MIN_FILTER  0x2801
#define HIPGL_TEXTURE_WRAP_S      0x2802
#define HIPGL_TEXTURE_WRAP_T      0x2803
#define HIPGL_NEAREST             0x2600
#define HIPGL_LINEAR              0x2601
#define HIPGL_REPEAT              0x2901
#define HIPGL_CLAMP_TO_EDGE       0x812F

// Client state
#define HIPGL_VERTEX_ARRAY        0x8074
#define HIPGL_NORMAL_ARRAY        0x8075
#define HIPGL_COLOR_ARRAY         0x8076
#define HIPGL_TEXTURE_COORD_ARRAY 0x8078

// Get parameters
#define HIPGL_VIEWPORT            0x0BA2
#define HIPGL_SCISSOR_BOX         0x0C10
#define HIPGL_MAX_TEXTURE_SIZE    0x0D33
#define HIPGL_MODELVIEW_MATRIX    0x0BA6
#define HIPGL_PROJECTION_MATRIX   0x0BA7

// Light parameters
#define HIPGL_AMBIENT             0x1200
#define HIPGL_DIFFUSE             0x1201
#define HIPGL_SPECULAR            0x1202
#define HIPGL_POSITION            0x1203
#define HIPGL_SPOT_DIRECTION      0x1204
#define HIPGL_SPOT_EXPONENT       0x1205
#define HIPGL_SPOT_CUTOFF         0x1206
#define HIPGL_CONSTANT_ATTENUATION  0x1207
#define HIPGL_LINEAR_ATTENUATION    0x1208
#define HIPGL_QUADRATIC_ATTENUATION 0x1209

// Material parameters
#define HIPGL_EMISSION            0x1600
#define HIPGL_SHININESS           0x1601
#define HIPGL_AMBIENT_AND_DIFFUSE 0x1602

// String queries
#define HIPGL_VENDOR              0x1F00
#define HIPGL_RENDERER            0x1F01
#define HIPGL_VERSION             0x1F02
#define HIPGL_EXTENSIONS          0x1F03

// Pixel store
#define HIPGL_PACK_ALIGNMENT      0x0CF1
#define HIPGL_UNPACK_ALIGNMENT    0x0CF5

// Error codes
#define HIPGL_NO_ERROR            0

// ── Context Management ──────────────────────────────────────────────────────

void* hipglCreateContext(int width, int height);
void  hipglDestroyContext(void* ctx);
void  hipglMakeCurrent(void* ctx);

// ── State ───────────────────────────────────────────────────────────────────

void hipglEnable(unsigned int cap);
void hipglDisable(unsigned int cap);
void hipglDepthFunc(unsigned int func);
void hipglBlendFunc(unsigned int sfactor, unsigned int dfactor);
void hipglShadeModel(unsigned int mode);
void hipglCullFace(unsigned int mode);
void hipglFrontFace(unsigned int mode);
void hipglPolygonMode(unsigned int face, unsigned int mode);
void hipglPointSize(float size);
void hipglLineWidth(float width);
void hipglPolygonOffset(float factor, float units);
void hipglDepthMask(unsigned char flag);
void hipglColorMask(unsigned char r, unsigned char g, unsigned char b, unsigned char a);

// ── Clear ───────────────────────────────────────────────────────────────────

void hipglClearColor(float r, float g, float b, float a);
void hipglClearDepth(double depth);
void hipglClearStencil(int s);
void hipglClear(unsigned int mask);

// ── Viewport ────────────────────────────────────────────────────────────────

void hipglViewport(int x, int y, int width, int height);
void hipglScissor(int x, int y, int width, int height);

// ── Matrix Stack ────────────────────────────────────────────────────────────

void hipglMatrixMode(unsigned int mode);
void hipglLoadIdentity(void);
void hipglLoadMatrixf(const float* m);
void hipglMultMatrixf(const float* m);
void hipglPushMatrix(void);
void hipglPopMatrix(void);
void hipglTranslatef(float x, float y, float z);
void hipglRotatef(float angle, float x, float y, float z);
void hipglScalef(float x, float y, float z);
void hipglOrtho(double left, double right, double bottom, double top, double near_val, double far_val);
void hipglFrustum(double left, double right, double bottom, double top, double near_val, double far_val);

// ── Immediate Mode ──────────────────────────────────────────────────────────

void hipglBegin(unsigned int mode);
void hipglEnd(void);
void hipglVertex2f(float x, float y);
void hipglVertex3f(float x, float y, float z);
void hipglVertex3fv(const float* v);
void hipglNormal3f(float x, float y, float z);
void hipglNormal3fv(const float* v);
void hipglColor3f(float r, float g, float b);
void hipglColor4f(float r, float g, float b, float a);
void hipglColor3fv(const float* v);
void hipglColor4fv(const float* v);
void hipglColor4ubv(const unsigned char* v);
void hipglTexCoord2f(float s, float t);
void hipglTexCoord2fv(const float* v);

// ── Lighting ────────────────────────────────────────────────────────────────

void hipglLightfv(unsigned int light, unsigned int pname, const float* params);
void hipglLightf(unsigned int light, unsigned int pname, float param);
void hipglMaterialfv(unsigned int face, unsigned int pname, const float* params);
void hipglMaterialf(unsigned int face, unsigned int pname, float param);
void hipglColorMaterial(unsigned int face, unsigned int mode);
void hipglLightModelfv(unsigned int pname, const float* params);
void hipglLightModeli(unsigned int pname, int param);

// ── Buffer Objects ──────────────────────────────────────────────────────────

void hipglGenBuffers(int n, unsigned int* buffers);
void hipglDeleteBuffers(int n, const unsigned int* buffers);
void hipglBindBuffer(unsigned int target, unsigned int buffer);
void hipglBufferData(unsigned int target, ptrdiff_t size, const void* data, unsigned int usage);

// ── Client-Side Vertex Arrays ───────────────────────────────────────────────

void hipglEnableClientState(unsigned int cap);
void hipglDisableClientState(unsigned int cap);
void hipglVertexPointer(int size, unsigned int type, int stride, const void* ptr);
void hipglNormalPointer(unsigned int type, int stride, const void* ptr);
void hipglColorPointer(int size, unsigned int type, int stride, const void* ptr);
void hipglTexCoordPointer(int size, unsigned int type, int stride, const void* ptr);

// ── Modern Vertex Attribs ───────────────────────────────────────────────────

void hipglGenVertexArrays(int n, unsigned int* arrays);
void hipglBindVertexArray(unsigned int array);
void hipglVertexAttribPointer(unsigned int index, int size, unsigned int type,
                              unsigned char normalized, int stride, const void* pointer);
void hipglEnableVertexAttribArray(unsigned int index);
void hipglDisableVertexAttribArray(unsigned int index);

// ── Draw ────────────────────────────────────────────────────────────────────

void hipglDrawArrays(unsigned int mode, int first, int count);
void hipglDrawElements(unsigned int mode, int count, unsigned int type, const void* indices);

// ── Textures ────────────────────────────────────────────────────────────────

void hipglGenTextures(int n, unsigned int* textures);
void hipglDeleteTextures(int n, const unsigned int* textures);
void hipglBindTexture(unsigned int target, unsigned int texture);
void hipglTexImage2D(unsigned int target, int level, int internalformat,
                     int width, int height, int border,
                     unsigned int format, unsigned int type, const void* data);
void hipglTexParameteri(unsigned int target, unsigned int pname, int param);
void hipglActiveTexture(unsigned int texture);
void hipglTexEnvi(unsigned int target, unsigned int pname, int param);
void hipglTexGeni(unsigned int coord, unsigned int pname, int param);
void hipglTexGenfv(unsigned int coord, unsigned int pname, const float* params);
void hipglGenerateMipmap(unsigned int target);

// ── Fog, Alpha, Stencil, Clip ────────────────────────────────────────────────

void hipglFogi(unsigned int pname, int param);
void hipglFogf(unsigned int pname, float param);
void hipglFogfv(unsigned int pname, const float* params);
void hipglAlphaFunc(unsigned int func, float ref);
void hipglStencilFunc(unsigned int func, int ref, unsigned int mask);
void hipglStencilOp(unsigned int sfail, unsigned int dpfail, unsigned int dppass);
void hipglClipPlane(unsigned int plane, const double* equation);

// ── Misc ────────────────────────────────────────────────────────────────────

void hipglBitmap(int width, int height, float xorig, float yorig,
                 float xmove, float ymove, const unsigned char* bitmap);
void hipglDrawBuffer(unsigned int buf);
void hipglReadBuffer(unsigned int buf);
unsigned char hipglIsEnabled(unsigned int cap);

// ── Readback ────────────────────────────────────────────────────────────────

void hipglReadPixels(int x, int y, int width, int height,
                     unsigned int format, unsigned int type, void* data);
void hipglPixelStorei(unsigned int pname, int param);

// ── Display Lists ────────────────────────────────────────────────────────────

unsigned int hipglGenLists(int range);
void hipglDeleteLists(unsigned int list, int range);
void hipglNewList(unsigned int list, unsigned int mode);
void hipglEndList(void);
void hipglCallList(unsigned int list);

// ── Query ───────────────────────────────────────────────────────────────────

unsigned int        hipglGetError(void);
const unsigned char* hipglGetString(unsigned int name);
void hipglGetIntegerv(unsigned int pname, int* params);
void hipglGetIntegerv_extended(unsigned int pname, int* params);
void hipglGetFloatv(unsigned int pname, float* params);
void hipglGetBooleanv(unsigned int pname, unsigned char* params);
void hipglGetBooleanv(unsigned int pname, unsigned char* params);

// ── Sync ────────────────────────────────────────────────────────────────────

void hipglFlush(void);
void hipglFinish(void);

#ifdef __cplusplus
}
#endif

// ── GL Name Remapping ───────────────────────────────────────────────────────
// Define HIPGL_REPLACE_GL before including this header to remap standard
// OpenGL function and constant names to hipGL equivalents.

#ifdef HIPGL_REPLACE_GL

// Constants
#define GL_FALSE                  HIPGL_FALSE
#define GL_TRUE                   HIPGL_TRUE
#define GL_POINTS                 HIPGL_POINTS
#define GL_LINES                  HIPGL_LINES
#define GL_LINE_STRIP             HIPGL_LINE_STRIP
#define GL_LINE_LOOP              HIPGL_LINE_LOOP
#define GL_TRIANGLES              HIPGL_TRIANGLES
#define GL_TRIANGLE_STRIP         HIPGL_TRIANGLE_STRIP
#define GL_TRIANGLE_FAN           HIPGL_TRIANGLE_FAN
#define GL_QUADS                  HIPGL_QUADS
#define GL_DEPTH_BUFFER_BIT       HIPGL_DEPTH_BUFFER_BIT
#define GL_STENCIL_BUFFER_BIT     HIPGL_STENCIL_BUFFER_BIT
#define GL_COLOR_BUFFER_BIT       HIPGL_COLOR_BUFFER_BIT
#define GL_CULL_FACE              HIPGL_CULL_FACE
#define GL_DEPTH_TEST             HIPGL_DEPTH_TEST
#define GL_BLEND                  HIPGL_BLEND
#define GL_SCISSOR_TEST           HIPGL_SCISSOR_TEST
#define GL_TEXTURE_2D             HIPGL_TEXTURE_2D
#define GL_LIGHTING               HIPGL_LIGHTING
#define GL_LIGHT0                 HIPGL_LIGHT0
#define GL_LIGHT1                 HIPGL_LIGHT1
#define GL_LIGHT2                 HIPGL_LIGHT2
#define GL_LIGHT3                 HIPGL_LIGHT3
#define GL_COLOR_MATERIAL         HIPGL_COLOR_MATERIAL
#define GL_NEVER                  HIPGL_NEVER
#define GL_LESS                   HIPGL_LESS
#define GL_EQUAL                  HIPGL_EQUAL
#define GL_LEQUAL                 HIPGL_LEQUAL
#define GL_GREATER                HIPGL_GREATER
#define GL_NOTEQUAL               HIPGL_NOTEQUAL
#define GL_GEQUAL                 HIPGL_GEQUAL
#define GL_ALWAYS                 HIPGL_ALWAYS
#define GL_ZERO                   HIPGL_ZERO
#define GL_ONE                    HIPGL_ONE
#define GL_SRC_ALPHA              HIPGL_SRC_ALPHA
#define GL_ONE_MINUS_SRC_ALPHA    HIPGL_ONE_MINUS_SRC_ALPHA
#define GL_DST_ALPHA              HIPGL_DST_ALPHA
#define GL_ONE_MINUS_DST_ALPHA    HIPGL_ONE_MINUS_DST_ALPHA
#define GL_SRC_COLOR              HIPGL_SRC_COLOR
#define GL_ONE_MINUS_SRC_COLOR    HIPGL_ONE_MINUS_SRC_COLOR
#define GL_DST_COLOR              HIPGL_DST_COLOR
#define GL_ONE_MINUS_DST_COLOR    HIPGL_ONE_MINUS_DST_COLOR
#define GL_FRONT                  HIPGL_FRONT
#define GL_BACK                   HIPGL_BACK
#define GL_FRONT_AND_BACK         HIPGL_FRONT_AND_BACK
#define GL_CW                     HIPGL_CW
#define GL_CCW                    HIPGL_CCW
#define GL_POINT                  HIPGL_POINT
#define GL_LINE                   HIPGL_LINE
#define GL_FILL                   HIPGL_FILL
#define GL_FLAT                   HIPGL_FLAT
#define GL_SMOOTH                 HIPGL_SMOOTH
#define GL_MODELVIEW              HIPGL_MODELVIEW
#define GL_PROJECTION             HIPGL_PROJECTION
#define GL_TEXTURE                HIPGL_TEXTURE
#define GL_BYTE                   HIPGL_BYTE
#define GL_UNSIGNED_BYTE          HIPGL_UNSIGNED_BYTE
#define GL_SHORT                  HIPGL_SHORT
#define GL_UNSIGNED_SHORT         HIPGL_UNSIGNED_SHORT
#define GL_INT                    HIPGL_INT
#define GL_UNSIGNED_INT           HIPGL_UNSIGNED_INT
#define GL_FLOAT                  HIPGL_FLOAT
#define GL_DOUBLE                 HIPGL_DOUBLE
#define GL_RGB                    HIPGL_RGB
#define GL_RGBA                   HIPGL_RGBA
#define GL_DEPTH_COMPONENT        HIPGL_DEPTH_COMPONENT
#define GL_ARRAY_BUFFER           HIPGL_ARRAY_BUFFER
#define GL_ELEMENT_ARRAY_BUFFER   HIPGL_ELEMENT_ARRAY_BUFFER
#define GL_STREAM_DRAW            HIPGL_STREAM_DRAW
#define GL_STATIC_DRAW            HIPGL_STATIC_DRAW
#define GL_DYNAMIC_DRAW           HIPGL_DYNAMIC_DRAW
#define GL_TEXTURE_MAG_FILTER     HIPGL_TEXTURE_MAG_FILTER
#define GL_TEXTURE_MIN_FILTER     HIPGL_TEXTURE_MIN_FILTER
#define GL_TEXTURE_WRAP_S         HIPGL_TEXTURE_WRAP_S
#define GL_TEXTURE_WRAP_T         HIPGL_TEXTURE_WRAP_T
#define GL_NEAREST                HIPGL_NEAREST
#define GL_LINEAR                 HIPGL_LINEAR
#define GL_REPEAT                 HIPGL_REPEAT
#define GL_CLAMP_TO_EDGE          HIPGL_CLAMP_TO_EDGE
#define GL_VERTEX_ARRAY           HIPGL_VERTEX_ARRAY
#define GL_NORMAL_ARRAY           HIPGL_NORMAL_ARRAY
#define GL_COLOR_ARRAY            HIPGL_COLOR_ARRAY
#define GL_TEXTURE_COORD_ARRAY    HIPGL_TEXTURE_COORD_ARRAY
#define GL_VIEWPORT               HIPGL_VIEWPORT
#define GL_SCISSOR_BOX            HIPGL_SCISSOR_BOX
#define GL_AMBIENT                HIPGL_AMBIENT
#define GL_DIFFUSE                HIPGL_DIFFUSE
#define GL_SPECULAR               HIPGL_SPECULAR
#define GL_POSITION               HIPGL_POSITION
#define GL_EMISSION               HIPGL_EMISSION
#define GL_SHININESS              HIPGL_SHININESS
#define GL_AMBIENT_AND_DIFFUSE    HIPGL_AMBIENT_AND_DIFFUSE
#define GL_VENDOR                 HIPGL_VENDOR
#define GL_RENDERER               HIPGL_RENDERER
#define GL_VERSION                HIPGL_VERSION
#define GL_EXTENSIONS             HIPGL_EXTENSIONS
#define GL_PACK_ALIGNMENT         HIPGL_PACK_ALIGNMENT
#define GL_UNPACK_ALIGNMENT       HIPGL_UNPACK_ALIGNMENT
#define GL_NO_ERROR               HIPGL_NO_ERROR
#define GL_FRAMEBUFFER            HIPGL_FRAMEBUFFER
#define GL_RENDERBUFFER           HIPGL_RENDERBUFFER
#define GL_COLOR_ATTACHMENT0      HIPGL_COLOR_ATTACHMENT0
#define GL_DEPTH_ATTACHMENT       HIPGL_DEPTH_ATTACHMENT
#define GL_STENCIL_ATTACHMENT     HIPGL_STENCIL_ATTACHMENT
#define GL_FRAMEBUFFER_COMPLETE   HIPGL_FRAMEBUFFER_COMPLETE
#define GL_DEPTH_STENCIL          HIPGL_DEPTH_STENCIL
#define GL_DEPTH24_STENCIL8       HIPGL_DEPTH24_STENCIL8
#define GL_FOG                    HIPGL_FOG
#define GL_FOG_MODE               HIPGL_FOG_MODE
#define GL_FOG_DENSITY            HIPGL_FOG_DENSITY
#define GL_FOG_START              HIPGL_FOG_START
#define GL_FOG_END                HIPGL_FOG_END
#define GL_FOG_COLOR              HIPGL_FOG_COLOR
#define GL_EXP                    HIPGL_EXP
#define GL_EXP2                   HIPGL_EXP2
#define GL_ALPHA_TEST             HIPGL_ALPHA_TEST
#define GL_POLYGON_OFFSET_FILL    HIPGL_POLYGON_OFFSET_FILL
#define GL_POLYGON_OFFSET_LINE    HIPGL_POLYGON_OFFSET_LINE
#define GL_MULTISAMPLE            HIPGL_MULTISAMPLE
#define GL_CLIP_PLANE0            HIPGL_CLIP_PLANE0
#define GL_COMPILE                HIPGL_COMPILE
#define GL_COMPILE_AND_EXECUTE    HIPGL_COMPILE_AND_EXECUTE
#define GL_TEXTURE_GEN_S          HIPGL_TEXTURE_GEN_S
#define GL_TEXTURE_GEN_T          HIPGL_TEXTURE_GEN_T
#define GL_TEXTURE_GEN_MODE       HIPGL_TEXTURE_GEN_MODE
#define GL_OBJECT_LINEAR          HIPGL_OBJECT_LINEAR
#define GL_EYE_LINEAR             HIPGL_EYE_LINEAR
#define GL_SPHERE_MAP             HIPGL_SPHERE_MAP
#define GL_TEXTURE_ENV            HIPGL_TEXTURE_ENV
#define GL_TEXTURE_ENV_MODE       HIPGL_TEXTURE_ENV_MODE
#define GL_MODULATE               HIPGL_MODULATE
#define GL_DECAL                  HIPGL_DECAL
#define GL_REPLACE                HIPGL_REPLACE
#define GL_LIGHT_MODEL_AMBIENT    HIPGL_LIGHT_MODEL_AMBIENT
#define GL_LIGHT_MODEL_TWO_SIDE   HIPGL_LIGHT_MODEL_TWO_SIDE
#define GL_LIGHT_MODEL_LOCAL_VIEWER HIPGL_LIGHT_MODEL_LOCAL_VIEWER
#define GL_FRONT_LEFT             HIPGL_FRONT_LEFT
#define GL_BACK_LEFT              HIPGL_BACK_LEFT
#define GL_NONE                   HIPGL_NONE_
#define GL_TEXTURE0               HIPGL_TEXTURE0
#define GL_KEEP                   HIPGL_KEEP
#define GL_INCR                   HIPGL_INCR
#define GL_DECR                   HIPGL_DECR
#define GL_MAX_LIGHTS             HIPGL_MAX_LIGHTS
#define GL_MAX_CLIP_PLANES        HIPGL_MAX_CLIP_PLANES
#define GL_MAX_MODELVIEW_STACK_DEPTH HIPGL_MAX_MODELVIEW_STACK_DEPTH
#define GL_MAX_PROJECTION_STACK_DEPTH HIPGL_MAX_PROJECTION_STACK_DEPTH
#define GL_DEPTH_COMPONENT24      HIPGL_DEPTH_COMPONENT24
#define GL_DEPTH_COMPONENT16      HIPGL_DEPTH_COMPONENT16
#define GL_TEXTURE_CUBE_MAP       HIPGL_TEXTURE_CUBE_MAP
#define GL_DEBUG_OUTPUT           HIPGL_DEBUG_OUTPUT

// Functions
#define glEnable                  hipglEnable
#define glDisable                 hipglDisable
#define glIsEnabled               hipglIsEnabled
#define glDepthFunc               hipglDepthFunc
#define glDepthMask               hipglDepthMask
#define glColorMask               hipglColorMask
#define glBlendFunc               hipglBlendFunc
#define glShadeModel              hipglShadeModel
#define glCullFace                hipglCullFace
#define glFrontFace               hipglFrontFace
#define glPolygonMode             hipglPolygonMode
#define glPointSize               hipglPointSize
#define glLineWidth               hipglLineWidth
#define glClearColor              hipglClearColor
#define glClearDepth              hipglClearDepth
#define glClearStencil            hipglClearStencil
#define glClear                   hipglClear
#define glViewport                hipglViewport
#define glScissor                 hipglScissor
#define glMatrixMode              hipglMatrixMode
#define glLoadIdentity            hipglLoadIdentity
#define glLoadMatrixf             hipglLoadMatrixf
#define glMultMatrixf             hipglMultMatrixf
#define glPushMatrix              hipglPushMatrix
#define glPopMatrix               hipglPopMatrix
#define glTranslatef              hipglTranslatef
#define glRotatef                 hipglRotatef
#define glScalef                  hipglScalef
#define glOrtho                   hipglOrtho
#define glFrustum                 hipglFrustum
#define glBegin                   hipglBegin
#define glEnd                     hipglEnd
#define glVertex3f                hipglVertex3f
#define glVertex3fv               hipglVertex3fv
#define glNormal3f                hipglNormal3f
#define glNormal3fv               hipglNormal3fv
#define glColor3f                 hipglColor3f
#define glColor4f                 hipglColor4f
#define glColor3fv                hipglColor3fv
#define glColor4fv                hipglColor4fv
#define glTexCoord2f              hipglTexCoord2f
#define glLightfv                 hipglLightfv
#define glLightf                  hipglLightf
#define glMaterialfv              hipglMaterialfv
#define glMaterialf               hipglMaterialf
#define glColorMaterial           hipglColorMaterial
#define glGenBuffers              hipglGenBuffers
#define glDeleteBuffers           hipglDeleteBuffers
#define glBindBuffer              hipglBindBuffer
#define glBufferData              hipglBufferData
#define glEnableClientState       hipglEnableClientState
#define glDisableClientState      hipglDisableClientState
#define glVertexPointer           hipglVertexPointer
#define glNormalPointer           hipglNormalPointer
#define glColorPointer            hipglColorPointer
#define glTexCoordPointer         hipglTexCoordPointer
#define glGenVertexArrays         hipglGenVertexArrays
#define glBindVertexArray         hipglBindVertexArray
#define glVertexAttribPointer     hipglVertexAttribPointer
#define glEnableVertexAttribArray hipglEnableVertexAttribArray
#define glDisableVertexAttribArray hipglDisableVertexAttribArray
#define glDrawArrays              hipglDrawArrays
#define glDrawElements            hipglDrawElements
#define glGenTextures             hipglGenTextures
#define glDeleteTextures          hipglDeleteTextures
#define glBindTexture             hipglBindTexture
#define glTexImage2D              hipglTexImage2D
#define glTexParameteri           hipglTexParameteri
#define glReadPixels              hipglReadPixels
#define glPixelStorei             hipglPixelStorei
#define glGetError                hipglGetError
#define glGetString               hipglGetString
#define glGetIntegerv             hipglGetIntegerv
#define glGetFloatv               hipglGetFloatv
#define glGetBooleanv             hipglGetBooleanv
#define glFlush                   hipglFlush
#define glFinish                  hipglFinish

#endif // HIPGL_REPLACE_GL
