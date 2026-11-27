cmake_minimum_required(VERSION 3.22)
project(libinfer VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for language servers
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -----------------------------------------------------------------------------
# Configuration Flags for llama.cpp Submodule
# -----------------------------------------------------------------------------
set(GGML_BACKEND_DL ON CACHE BOOL "Enable dynamic backend loading" FORCE)
set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries for backends" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Do not build llama.cpp examples" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Do not build llama.cpp tests" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Do not build llama.cpp server" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "Do not build llama.cpp tools" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "Do not build llama.cpp common" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "Disable libcurl in llama.cpp" FORCE)

# Enable CUDA backend in llama.cpp if requested
option(GGML_CUDA "Enable CUDA backend" OFF)

# Enable Vulkan backend in llama.cpp if requested (For Intel/AMD/NVIDIA/ARM Mali/Adreno GPUs)
option(GGML_VULKAN "Enable Vulkan backend" OFF)

# Enable OpenCL backend in llama.cpp if requested (For ARM Mali/Adreno GPUs)
option(GGML_OPENCL "Enable OpenCL backend" OFF)

# Build CLI example executables when present
option(INFER_BUILD_EXAMPLES "Build example CLI executables" ON)

# Ensure sibling and parent shared libraries (libggml-base, libllama) resolve cleanly
# when dynamic backends are packaged in subdirectories (cuda/, vulkan/, opencl/).
if(UNIX AND NOT APPLE)
    set(CMAKE_BUILD_RPATH "$ORIGIN:$ORIGIN/..")
    set(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/..")
elseif(APPLE)
    set(CMAKE_BUILD_RPATH "@loader_path;@loader_path/..")
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/..")
endif()

# Add llama.cpp as third-party dependency
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
    message(WARNING "third_party/llama.cpp not found. Run 'git submodule update --init --recursive' before building.")
else()
    add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)
endif()

# -----------------------------------------------------------------------------
# Target: infer_core (Shared ABI Boundary)
# -----------------------------------------------------------------------------
add_library(infer_core SHARED
    src/infer_core.cpp
)

target_include_directories(infer_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include
)

if(TARGET llama AND TARGET ggml)
    target_link_libraries(infer_core PRIVATE llama ggml)
endif()

# Enforce strict symbol visibility: hide internal symbols by default
set_target_properties(infer_core PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "infer"
)

# Resolve sibling shared libraries (libllama, libggml) from the same directory when packaged
if(UNIX AND NOT APPLE)
    set_target_properties(infer_core PROPERTIES
        BUILD_RPATH "$ORIGIN:$ORIGIN/.."
        INSTALL_RPATH "$ORIGIN:$ORIGIN/.."
    )
elseif(APPLE)
    set_target_properties(infer_core PROPERTIES
        BUILD_RPATH "@loader_path;@loader_path/.."
        INSTALL_RPATH "@loader_path;@loader_path/.."
    )
endif()

# Export macro definitions for DLL/SO symbol visibility
target_compile_definitions(infer_core
    PRIVATE
        INFER_BUILD_SHARED
)

# -----------------------------------------------------------------------------
# Example CLI targets
# -----------------------------------------------------------------------------
if(INFER_BUILD_EXAMPLES)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/main.c")
        add_executable(infer_cli examples/main.c)
        target_link_libraries(infer_cli PRIVATE infer_core)
        target_include_directories(infer_cli PRIVATE include)
        if(UNIX AND NOT APPLE)
            set_target_properties(infer_cli PROPERTIES
                BUILD_RPATH "$ORIGIN"
                INSTALL_RPATH "$ORIGIN"
            )
        elseif(APPLE)
            set_target_properties(infer_cli PROPERTIES
                BUILD_RPATH "@loader_path"
                INSTALL_RPATH "@loader_path"
            )
        endif()
    endif()

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/orange_pi_chat.c")
        add_executable(infer_orange_pi examples/orange_pi_chat.c)
        target_link_libraries(infer_orange_pi PRIVATE infer_core)
        target_include_directories(infer_orange_pi PRIVATE include)
        if(UNIX AND NOT APPLE)
            set_target_properties(infer_orange_pi PROPERTIES
                BUILD_RPATH "$ORIGIN"
                INSTALL_RPATH "$ORIGIN"
            )
        elseif(APPLE)
            set_target_properties(infer_orange_pi PROPERTIES
                BUILD_RPATH "@loader_path"
                INSTALL_RPATH "@loader_path"
            )
        endif()
    endif()
endif()
