#pragma once
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifdef __cplusplus
extern "C" {
#endif

void hipglTexImage2D_impl(unsigned int target, int level, int internalformat,
                          int width, int height, int border,
                          unsigned int format, unsigned int type, const void* data);

void hipglTexParameteri_impl(unsigned int target, unsigned int pname, int param);

#ifdef __cplusplus
}
#endif
