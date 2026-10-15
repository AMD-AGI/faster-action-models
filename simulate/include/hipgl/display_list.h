#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <functional>

namespace hipgl {

// A recorded GL command
struct DisplayListCommand {
    enum Type {
        CMD_ENABLE, CMD_DISABLE,
        CMD_DEPTH_FUNC, CMD_BLEND_FUNC, CMD_SHADE_MODEL,
        CMD_CULL_FACE, CMD_FRONT_FACE, CMD_POLYGON_MODE,
        CMD_MATRIX_MODE, CMD_LOAD_IDENTITY, CMD_LOAD_MATRIXF, CMD_MULT_MATRIXF,
        CMD_PUSH_MATRIX, CMD_POP_MATRIX,
        CMD_TRANSLATEF, CMD_ROTATEF, CMD_SCALEF,
        CMD_ORTHO, CMD_FRUSTUM,
        CMD_BEGIN, CMD_END,
        CMD_VERTEX3F, CMD_VERTEX3FV, CMD_VERTEX2F,
        CMD_NORMAL3F, CMD_NORMAL3FV,
        CMD_COLOR3F, CMD_COLOR4F, CMD_COLOR4FV, CMD_COLOR4UBV,
        CMD_TEXCOORD2F, CMD_TEXCOORD2FV,
        CMD_LIGHTFV, CMD_LIGHTF, CMD_MATERIALFV, CMD_MATERIALF,
        CMD_COLOR_MATERIAL, CMD_LIGHT_MODELFV, CMD_LIGHT_MODELI,
        CMD_DRAW_ARRAYS, CMD_DRAW_ELEMENTS,
        CMD_BIND_TEXTURE, CMD_ACTIVE_TEXTURE,
        CMD_ENABLE_CLIENT_STATE, CMD_DISABLE_CLIENT_STATE,
        CMD_VERTEX_POINTER, CMD_NORMAL_POINTER, CMD_COLOR_POINTER, CMD_TEXCOORD_POINTER,
        CMD_LINE_WIDTH, CMD_POINT_SIZE, CMD_POLYGON_OFFSET,
        CMD_DEPTH_MASK, CMD_COLOR_MASK,
        CMD_CLEAR_COLOR, CMD_CLEAR_DEPTH, CMD_CLEAR,
        CMD_VIEWPORT, CMD_SCISSOR,
        CMD_TEX_ENVI, CMD_TEX_GENI, CMD_TEX_GENFV,
        CMD_TEX_PARAMETERI,
        CMD_FOGI, CMD_FOGF, CMD_FOGFV,
        CMD_ALPHA_FUNC, CMD_STENCIL_FUNC, CMD_STENCIL_OP,
        CMD_CLIP_PLANE,
    };

    Type type;
    // Store up to 16 floats + 4 ints for any command's args
    float fargs[16];
    int   iargs[4];
    unsigned int uargs[4];
    std::vector<float> float_data; // for variable-length data (matrices, etc.)
    std::vector<unsigned char> byte_data; // for glColor4ubv etc.
};

class DisplayList {
public:
    void record(const DisplayListCommand& cmd);
    void replay() const;
    const std::vector<DisplayListCommand>& commands() const { return commands_; }
    bool empty() const { return commands_.empty(); }
private:
    std::vector<DisplayListCommand> commands_;
};

class DisplayListManager {
public:
    DisplayListManager();
    ~DisplayListManager() = default;

    unsigned int gen_lists(int range);
    void delete_lists(unsigned int list, int range);
    void new_list(unsigned int list, unsigned int mode); // mode = GL_COMPILE or GL_COMPILE_AND_EXECUTE
    void end_list();
    void call_list(unsigned int list);

    bool is_recording() const { return recording_; }
    unsigned int recording_mode() const { return recording_mode_; }
    DisplayList* current_recording() { return recording_ ? &recording_list_ : nullptr; }

private:
    std::unordered_map<unsigned int, DisplayList> lists_;
    unsigned int next_id_;
    bool recording_;
    unsigned int recording_id_;
    unsigned int recording_mode_; // GL_COMPILE or GL_COMPILE_AND_EXECUTE
    DisplayList recording_list_;
};

// Call this from each GL function that can be recorded in a display list.
// Returns true if the command was consumed by recording (and should NOT execute if mode=GL_COMPILE)
bool maybe_record_command(const DisplayListCommand& cmd);

// Global access
DisplayListManager& get_display_list_manager();

} // namespace hipgl
