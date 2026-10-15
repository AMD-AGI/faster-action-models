# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# FindHIPGL.cmake - Config file for downstream projects to find_package(hipgl)
#
# Usage:
#   find_package(hipgl REQUIRED)
#   target_link_libraries(myapp PRIVATE hipgl::hipgl)
#
# Or when hipgl is installed:
#   set(CMAKE_PREFIX_PATH /path/to/hipgl/install)
#   find_package(hipgl REQUIRED)

include(FindPackageHandleStandardArgs)

# Try to find hipgl in common locations
find_path(HIPGL_INCLUDE_DIR
    NAMES hipgl/hipgl.h
    HINTS
        ${HIPGL_ROOT}
        $ENV{HIPGL_ROOT}
        ${CMAKE_INSTALL_PREFIX}
    PATH_SUFFIXES include
)

find_library(HIPGL_LIBRARY
    NAMES hipgl
    HINTS
        ${HIPGL_ROOT}
        $ENV{HIPGL_ROOT}
        ${CMAKE_INSTALL_PREFIX}
    PATH_SUFFIXES lib lib64
)

find_package_handle_standard_args(hipgl
    REQUIRED_VARS HIPGL_LIBRARY HIPGL_INCLUDE_DIR
    FAIL_MESSAGE "hipgl not found. Set HIPGL_ROOT or CMAKE_PREFIX_PATH to the hipgl installation."
)

if(hipgl_FOUND AND NOT TARGET hipgl::hipgl)
    add_library(hipgl::hipgl SHARED IMPORTED)
    set_target_properties(hipgl::hipgl PROPERTIES
        IMPORTED_LOCATION "${HIPGL_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${HIPGL_INCLUDE_DIR}"
    )
    # hipgl links against HIP - propagate dependency for downstream link
    find_package(hip QUIET)
    if(hip_FOUND)
        set_target_properties(hipgl::hipgl PROPERTIES
            INTERFACE_LINK_LIBRARIES "hip::device"
        )
    endif()
endif()
