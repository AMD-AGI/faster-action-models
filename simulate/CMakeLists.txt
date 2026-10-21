# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# hipGL - HIP-based OpenGL-compatible software rasterizer for AMD Instinct™ GPUs
# CMake 3.21+ required for HIP language support

cmake_minimum_required(VERSION 3.21)
project(hipgl
    VERSION 0.1.0
    LANGUAGES CXX
    DESCRIPTION "HIP-based OpenGL-compatible software rasterizer for AMD Instinct™ (CDNA)"
)

# ── Options ─────────────────────────────────────────────────────────────────
# This snapshot ships without tests/ and examples/ (see PROVENANCE.md), so both default OFF.
option(BUILD_TESTS   "Build test executables"   OFF)
option(BUILD_EXAMPLES "Build example executables" OFF)

# ── ROCm/HIP setup ──────────────────────────────────────────────────────────
# Default to the version-agnostic /opt/rocm symlink rather than a pinned
# point release. A hardcoded "/opt/rocm-7.1.1" configures on exactly one
# machine: a fresh clone on a 7.1.2 host fails with "Could not find a package
# configuration file provided by hip", because CMAKE_PREFIX_PATH points at a
# directory that does not exist. Existing build trees did not notice -- their
# CMakeCache.txt already carried an override.
# Order: explicit -DROCM_PATH=... > $ROCM_PATH in the environment > /opt/rocm.
if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(_rocm_default "$ENV{ROCM_PATH}")
    else()
        set(_rocm_default "/opt/rocm")
    endif()
else()
    set(_rocm_default "${ROCM_PATH}")
endif()
set(ROCM_PATH "${_rocm_default}" CACHE PATH "Path to ROCm installation")
if(NOT EXISTS "${ROCM_PATH}")
    message(FATAL_ERROR
        "ROCM_PATH does not exist: ${ROCM_PATH}\n"
        "Pass -DROCM_PATH=/path/to/rocm or set the ROCM_PATH environment variable.")
endif()
message(STATUS "instinct_sim: using ROCm at ${ROCM_PATH}")
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")

# ── GPU architecture ───────────────────────────────────────────────────────
# The simulator supports gfx950 (MI350X/MI355X) only: several kernels need more than the 64 KiB
# of LDS that gfx942 (MI300X/MI325X) provides, so a gfx942 build fails to compile. On MI300X/MI325X
# use --sim mujoco. Unless -DCMAKE_HIP_ARCHITECTURES=... is given, build for the gfx950 GPUs in
# this machine; when none is visible (a GPU-less build host or container, or another GPU) fall
# back to gfx950. Left to itself, CMake would fall back to gfx906, which runs on none of our
# cards. Must run before enable_language(HIP), which fixes the architecture list.
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
    set(_hp_archs "")
    find_program(_hp_agent_enum rocm_agent_enumerator HINTS "${ROCM_PATH}/bin" NO_CACHE)
    if(_hp_agent_enum)
        execute_process(COMMAND "${_hp_agent_enum}" OUTPUT_VARIABLE _hp_agents
                        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        string(REPLACE "\n" ";" _hp_agents "${_hp_agents}")
        foreach(_a IN LISTS _hp_agents)
            string(STRIP "${_a}" _a)
            if(_a STREQUAL "gfx950")
                list(APPEND _hp_archs "${_a}")
            elseif(_a MATCHES "^gfx[0-9a-f]+$" AND NOT _a STREQUAL "gfx000")
                message(WARNING "instinct_sim: ${_a} is not supported (gfx950 only); use --sim mujoco on this GPU")
            endif()
        endforeach()
        list(REMOVE_DUPLICATES _hp_archs)
    endif()
    if(NOT _hp_archs)
        set(_hp_archs "gfx950")
        message(STATUS "instinct_sim: no gfx950 GPU detected; building for ${_hp_archs}")
    endif()
    set(CMAKE_HIP_ARCHITECTURES "${_hp_archs}" CACHE STRING "HIP target architectures")
endif()
message(STATUS "instinct_sim: HIP architectures ${CMAKE_HIP_ARCHITECTURES}")

enable_language(HIP)
find_package(hip REQUIRED)

# ── C++ standard ───────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD 17)

# ── Collect sources ─────────────────────────────────────────────────────────
# CONFIGURE_DEPENDS so a NEW kernel translation unit (e.g. src/kernels/micro_draw.hip) is picked up by an incremental build without a
# manual cmake re-run.
file(GLOB HIP_SOURCES   CONFIGURE_DEPENDS "src/kernels/*.hip")
file(GLOB CORE_SOURCES  "src/core/*.cpp")
file(GLOB GL_SOURCES    "src/gl/*.cpp")

# Exclude gl_shim from main library (it goes into hipgl_osmesa)
list(FILTER GL_SOURCES EXCLUDE REGEX ".*gl_shim\\.cpp$")
set(MAIN_GL_SOURCES ${GL_SOURCES})

# tinyxml2 (for MJCF parsing)
set(TINYXML2_SOURCES "third_party/tinyxml2.cpp")

# All sources compiled as HIP since core/gl files use HIP runtime APIs
set(ALL_SOURCES ${HIP_SOURCES} ${CORE_SOURCES} ${MAIN_GL_SOURCES} ${TINYXML2_SOURCES})
set_source_files_properties(${ALL_SOURCES} PROPERTIES LANGUAGE HIP)
# Ray tracer kernels: the SLP vectorizer pairs scalar FP ops into packed-FP32
# ops and keeps duplicated operand copies live (measured: shade kernel 215 ->
# 188 VGPRs, ORDERED geometry kernel 96 -> 88 without it). The traversal's
# packed slab test uses explicit float2 vectors and is unaffected.
set_source_files_properties(src/kernels/rt_trace.hip PROPERTIES COMPILE_OPTIONS "-fno-slp-vectorize")

# ── hipgl shared library ────────────────────────────────────────────────────
add_library(hipgl SHARED ${ALL_SOURCES})

target_include_directories(hipgl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

target_link_libraries(hipgl PUBLIC hip::device)

set_target_properties(hipgl PROPERTIES
    OUTPUT_NAME hipgl
    SOVERSION ${PROJECT_VERSION_MAJOR}
    VERSION ${PROJECT_VERSION}
    LINKER_LANGUAGE HIP
)

# ── hipgl_osmesa shim library (LD_PRELOAD target) ────────────────────────────
add_library(hipgl_osmesa SHARED src/gl/gl_shim.cpp)
set_source_files_properties(src/gl/gl_shim.cpp PROPERTIES LANGUAGE HIP)
target_include_directories(hipgl_osmesa PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(hipgl_osmesa PRIVATE hipgl)
set_target_properties(hipgl_osmesa PROPERTIES
    OUTPUT_NAME hipgl_osmesa
    LINKER_LANGUAGE HIP
)

# ── Install ─────────────────────────────────────────────────────────────────
include(GNUInstallDirs)

install(TARGETS hipgl hipgl_osmesa
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/hipgl
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)

install(FILES cmake/FindHIPGL.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/hipgl
)

# ── Python bindings (pybind11) ───────────────────────────────────────────────
option(BUILD_PYTHON "Build Python bindings (requires pybind11)" ON)
if(BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development QUIET)
    if(Python3_FOUND)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
            OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE PYBIND11_RESULT
        )
        if(PYBIND11_RESULT EQUAL 0)
            list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
            find_package(pybind11 QUIET)
            if(pybind11_FOUND)
                pybind11_add_module(instinct_sim src/python/instinct_sim_bind.cpp)
                set_source_files_properties(src/python/instinct_sim_bind.cpp PROPERTIES LANGUAGE HIP)
                target_link_libraries(instinct_sim PRIVATE hipgl)
                target_include_directories(instinct_sim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
                set_target_properties(instinct_sim PROPERTIES
                    LINKER_LANGUAGE HIP
                    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
                )
                message(STATUS "Python bindings: instinct_sim module will be built")
            else()
                message(STATUS "pybind11 not found, skipping Python bindings")
            endif()
        else()
            message(STATUS "pybind11 Python package not found, skipping Python bindings")
        endif()
    else()
        message(STATUS "Python3 not found, skipping Python bindings")
    endif()
endif()

# ── Subdirectories ───────────────────────────────────────────────────────────
if(BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    add_subdirectory(tests)
endif()
if(BUILD_EXAMPLES AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples)
    add_subdirectory(examples)
endif()
