// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// hipGL - Display list support for MuJoCo compatibility

#include "hipgl/display_list.h"
#include "hipgl/gl_api.h"
#include <cstring>

namespace hipgl {

// GL_COMPILE = 0x1300, GL_COMPILE_AND_EXECUTE = 0x1301
static const unsigned int GL_COMPILE = 0x1300;
static const unsigned int GL_COMPILE_AND_EXECUTE = 0x1301;

void DisplayList::record(const DisplayListCommand& cmd) {
    commands_.push_back(cmd);
}

void DisplayList::replay() const {
    for (const auto& cmd : commands_) {
        switch (cmd.type) {
            case DisplayListCommand::CMD_ENABLE:
                hipglEnable(cmd.uargs[0]);
                break;
            case DisplayListCommand::CMD_DISABLE:
                hipglDisable(cmd.uargs[0]);
                break;
            case DisplayListCommand::CMD_DEPTH_FUNC:
                hipglDepthFunc(cmd.uargs[0]);
                break;
            case DisplayListCommand::CMD_BLEND_FUNC:
                hipglBlendFunc(cmd.uargs[0], cmd.uargs[1]);
                break;
            case DisplayListCommand::CMD_SHADE_MODEL:
                hipglShadeModel(cmd.uargs[0]);
                break;
            case DisplayListCommand::CMD_CULL_FACE:
                hipglCullFace(cmd.uargs[0]);
                break;
            case DisplayListCommand::CMD_FRONT_FACE:
                hipglFrontFace(cmd.uargs[0]);
                break;
            case DisplayListCommand::CMD_POLYGON_MODE:
                hipglPolygonMode(cmd.uargs[0], cmd.uargs[1]);
                break;
            case DisplayListCommand::CMD_MATRIX_MODE:
                hipglMatrixMode(cmd.uargs[0]);
                break;
            case DisplayListCommand::CMD_LOAD_IDENTITY:
                hipglLoadIdentity();
                break;
            case DisplayListCommand::CMD_LOAD_MATRIXF:
                if (!cmd.float_data.empty())
                    hipglLoadMatrixf(cmd.float_data.data());
                break;
            case DisplayListCommand::CMD_MULT_MATRIXF:
                if (!cmd.float_data.empty())
                    hipglMultMatrixf(cmd.float_data.data());
                break;
            case DisplayListCommand::CMD_PUSH_MATRIX:
                hipglPushMatrix();
                break;
            case DisplayListCommand::CMD_POP_MATRIX:
                hipglPopMatrix();
                break;
            case DisplayListCommand::CMD_TRANSLATEF:
                hipglTranslatef(cmd.fargs[0], cmd.fargs[1], cmd.fargs[2]);
                break;
            case DisplayListCommand::CMD_ROTATEF:
                hipglRotatef(cmd.fargs[0], cmd.fargs[1], cmd.fargs[2], cmd.fargs[3]);
                break;
            case DisplayListCommand::CMD_SCALEF:
                hipglScalef(cmd.fargs[0], cmd.fargs[1], cmd.fargs[2]);
                break;
            case DisplayListCommand::CMD_ORTHO:
                hipglOrtho(cmd.fargs[0], cmd.fargs[1], cmd.fargs[2], cmd.fargs[3],
                           cmd.fargs[4], cmd.fargs[5]);
                break;
            case DisplayListCommand::CMD_FRUSTUM:
                hipglFrustum(cmd.fargs[0], cmd.fargs[1], cmd.fargs[2], cmd.fargs[3],
                             cmd.fargs[4], cmd.fargs[5]);
                break;
            case DisplayListCommand::CMD_BEGIN:
                hipglBegin(cmd.uargs[0]);
                break;
            case DisplayListCommand::CMD_END:
                hipglEnd();
                break;
            case DisplayListCommand::CMD_VERTEX3F:
                hipglVertex3f(cmd.fargs[0], cmd.fargs[1], cmd.fargs[2]);
                break;
            case DisplayListCommand::CMD_VERTEX3FV:
                if (!cmd.float_data.empty())
                    hipglVertex3fv(cmd.float_data.data());
                break;
            case DisplayListCommand::CMD_VERTEX2F:
                hipglVertex2f(cmd.fargs[0], cmd.fargs[1]);
                break;
            case DisplayListCommand::CMD_NORMAL3F:
                hipglNormal3f(cmd.fargs[0], cmd.fargs[1], cmd.fargs[2]);
                break;
            case DisplayListCommand::CMD_NORMAL3FV:
                if (!cmd.float_data.empty())
                    hipglNormal3fv(cmd.float_data.data());
                break;
            case DisplayListCommand::CMD_COLOR3F:
                hipglColor3f(cmd.fargs[0], cmd.fargs[1], cmd.fargs[2]);
                break;
            case DisplayListCommand::CMD_COLOR4F:
                hipglColor4f(cmd.fargs[0], cmd.fargs[1], cmd.fargs[2], cmd.fargs[3]);
                break;
            case DisplayListCommand::CMD_COLOR4FV:
                if (!cmd.float_data.empty())
                    hipglColor4fv(cmd.float_data.data());
                break;
            case DisplayListCommand::CMD_COLOR4UBV:
                if (cmd.byte_data.size() >= 4)
                    hipglColor4ubv(cmd.byte_data.data());
                break;
            case DisplayListCommand::CMD_TEXCOORD2F:
                hipglTexCoord2f(cmd.fargs[0], cmd.fargs[1]);
                break;
            case DisplayListCommand::CMD_TEXCOORD2FV:
                if (!cmd.float_data.empty())
                    hipglTexCoord2fv(cmd.float_data.data());
                break;
            case DisplayListCommand::CMD_LIGHTFV:
                if (!cmd.float_data.empty())
                    hipglLightfv(cmd.uargs[0], cmd.uargs[1], cmd.float_data.data());
                break;
            case DisplayListCommand::CMD_LIGHTF:
                hipglLightf(cmd.uargs[0], cmd.uargs[1], cmd.fargs[0]);
                break;
            case DisplayListCommand::CMD_MATERIALFV:
                if (!cmd.float_data.empty())
                    hipglMaterialfv(cmd.uargs[0], cmd.uargs[1], cmd.float_data.data());
                break;
            case DisplayListCommand::CMD_MATERIALF:
                hipglMaterialf(cmd.uargs[0], cmd.uargs[1], cmd.fargs[0]);
                break;
            case DisplayListCommand::CMD_COLOR_MATERIAL:
                hipglColorMaterial(cmd.uargs[0], cmd.uargs[1]);
                break;
            case DisplayListCommand::CMD_LIGHT_MODELFV:
                if (!cmd.float_data.empty())
                    hipglLightModelfv(cmd.uargs[0], cmd.float_data.data());
                break;
            case DisplayListCommand::CMD_LIGHT_MODELI:
                hipglLightModeli(cmd.uargs[0], cmd.iargs[0]);
                break;
            case DisplayListCommand::CMD_DRAW_ARRAYS:
                hipglDrawArrays(cmd.uargs[0], cmd.iargs[0], cmd.iargs[1]);
                break;
            case DisplayListCommand::CMD_DRAW_ELEMENTS:
                hipglDrawElements(cmd.uargs[0], cmd.iargs[0], cmd.uargs[1],
                                 cmd.byte_data.empty() ? nullptr : cmd.byte_data.data());
                break;
            case DisplayListCommand::CMD_BIND_TEXTURE:
                hipglBindTexture(cmd.uargs[0], cmd.uargs[1]);
                break;
            case DisplayListCommand::CMD_ACTIVE_TEXTURE:
                hipglActiveTexture(cmd.uargs[0]);
                break;
            case DisplayListCommand::CMD_ENABLE_CLIENT_STATE:
                hipglEnableClientState(cmd.uargs[0]);
                break;
            case DisplayListCommand::CMD_DISABLE_CLIENT_STATE:
                hipglDisableClientState(cmd.uargs[0]);
                break;
            case DisplayListCommand::CMD_VERTEX_POINTER:
                hipglVertexPointer(cmd.iargs[0], cmd.uargs[0], cmd.iargs[1], nullptr);
                break;
            case DisplayListCommand::CMD_NORMAL_POINTER:
                hipglNormalPointer(cmd.uargs[0], cmd.iargs[0], nullptr);
                break;
            case DisplayListCommand::CMD_COLOR_POINTER:
                hipglColorPointer(cmd.iargs[0], cmd.uargs[0], cmd.iargs[1], nullptr);
                break;
            case DisplayListCommand::CMD_TEXCOORD_POINTER:
                hipglTexCoordPointer(cmd.iargs[0], cmd.uargs[0], cmd.iargs[1], nullptr);
                break;
            case DisplayListCommand::CMD_LINE_WIDTH:
                hipglLineWidth(cmd.fargs[0]);
                break;
            case DisplayListCommand::CMD_POINT_SIZE:
                hipglPointSize(cmd.fargs[0]);
                break;
            case DisplayListCommand::CMD_POLYGON_OFFSET:
                hipglPolygonOffset(cmd.fargs[0], cmd.fargs[1]);
                break;
            case DisplayListCommand::CMD_DEPTH_MASK:
                hipglDepthMask(cmd.uargs[0]);
                break;
            case DisplayListCommand::CMD_COLOR_MASK:
                hipglColorMask(cmd.uargs[0], cmd.uargs[1], cmd.uargs[2], cmd.uargs[3]);
                break;
            case DisplayListCommand::CMD_CLEAR_COLOR:
                hipglClearColor(cmd.fargs[0], cmd.fargs[1], cmd.fargs[2], cmd.fargs[3]);
                break;
            case DisplayListCommand::CMD_CLEAR_DEPTH:
                hipglClearDepth(cmd.fargs[0]);
                break;
            case DisplayListCommand::CMD_CLEAR:
                hipglClear(cmd.uargs[0]);
                break;
            case DisplayListCommand::CMD_VIEWPORT:
                hipglViewport(cmd.iargs[0], cmd.iargs[1], cmd.iargs[2], cmd.iargs[3]);
                break;
            case DisplayListCommand::CMD_SCISSOR:
                hipglScissor(cmd.iargs[0], cmd.iargs[1], cmd.iargs[2], cmd.iargs[3]);
                break;
            case DisplayListCommand::CMD_TEX_ENVI:
                hipglTexEnvi(cmd.uargs[0], cmd.uargs[1], cmd.iargs[0]);
                break;
            case DisplayListCommand::CMD_TEX_GENI:
                hipglTexGeni(cmd.uargs[0], cmd.uargs[1], cmd.iargs[0]);
                break;
            case DisplayListCommand::CMD_TEX_GENFV:
                if (!cmd.float_data.empty())
                    hipglTexGenfv(cmd.uargs[0], cmd.uargs[1], cmd.float_data.data());
                break;
            case DisplayListCommand::CMD_TEX_PARAMETERI:
                hipglTexParameteri(cmd.uargs[0], cmd.uargs[1], cmd.iargs[0]);
                break;
            case DisplayListCommand::CMD_FOGI:
                hipglFogi(cmd.uargs[0], cmd.iargs[0]);
                break;
            case DisplayListCommand::CMD_FOGF:
                hipglFogf(cmd.uargs[0], cmd.fargs[0]);
                break;
            case DisplayListCommand::CMD_FOGFV:
                if (!cmd.float_data.empty())
                    hipglFogfv(cmd.uargs[0], cmd.float_data.data());
                break;
            case DisplayListCommand::CMD_ALPHA_FUNC:
                hipglAlphaFunc(cmd.uargs[0], cmd.fargs[0]);
                break;
            case DisplayListCommand::CMD_STENCIL_FUNC:
                hipglStencilFunc(cmd.uargs[0], cmd.iargs[0], cmd.uargs[1]);
                break;
            case DisplayListCommand::CMD_STENCIL_OP:
                hipglStencilOp(cmd.uargs[0], cmd.uargs[1], cmd.uargs[2]);
                break;
            case DisplayListCommand::CMD_CLIP_PLANE:
                if (!cmd.float_data.empty()) {
                    double eq[4] = {cmd.float_data[0], cmd.float_data[1],
                                   cmd.float_data[2], cmd.float_data[3]};
                    hipglClipPlane(cmd.uargs[0], eq);
                }
                break;
            default:
                break;
        }
    }
}

// ── DisplayListManager ───────────────────────────────────────────────────────

static DisplayListManager s_manager;

DisplayListManager& get_display_list_manager() {
    return s_manager;
}

DisplayListManager::DisplayListManager()
    : next_id_(1)
    , recording_(false)
    , recording_id_(0)
    , recording_mode_(GL_COMPILE)
{
}

unsigned int DisplayListManager::gen_lists(int range) {
    if (range <= 0) return 0;
    unsigned int base = next_id_;
    next_id_ += static_cast<unsigned int>(range);
    for (int i = 0; i < range; i++) {
        lists_[base + i] = DisplayList();
    }
    return base;
}

void DisplayListManager::delete_lists(unsigned int list, int range) {
    for (int i = 0; i < range; i++) {
        lists_.erase(list + i);
    }
}

void DisplayListManager::new_list(unsigned int list, unsigned int mode) {
    if (recording_) return;
    recording_ = true;
    recording_id_ = list;
    recording_mode_ = mode;
    recording_list_ = DisplayList();
}

void DisplayListManager::end_list() {
    if (!recording_) return;
    lists_[recording_id_] = std::move(recording_list_);
    recording_ = false;
}

void DisplayListManager::call_list(unsigned int list) {
    auto it = lists_.find(list);
    if (it != lists_.end() && !it->second.empty()) {
        it->second.replay();
    }
}

// ── Display list GL API (extern "C") ─────────────────────────────────────────

extern "C" {

unsigned int hipglGenLists(int range) {
    return get_display_list_manager().gen_lists(range);
}

void hipglDeleteLists(unsigned int list, int range) {
    get_display_list_manager().delete_lists(list, range);
}

void hipglNewList(unsigned int list, unsigned int mode) {
    get_display_list_manager().new_list(list, mode);
}

void hipglEndList(void) {
    get_display_list_manager().end_list();
}

void hipglCallList(unsigned int list) {
    get_display_list_manager().call_list(list);
}

} // extern "C"

// ── maybe_record_command ─────────────────────────────────────────────────────

bool maybe_record_command(const DisplayListCommand& cmd) {
    auto& mgr = get_display_list_manager();
    if (!mgr.is_recording()) return false;

    DisplayList* dl = mgr.current_recording();
    if (!dl) return false;

    dl->record(cmd);

    // GL_COMPILE: record only, don't execute
    // GL_COMPILE_AND_EXECUTE: record and execute
    return (mgr.recording_mode() == GL_COMPILE);
}

} // namespace hipgl
