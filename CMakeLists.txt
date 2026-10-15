cmake_minimum_required(VERSION 3.22)
project(libinfer VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for language servers
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# -----------------------------------------------------------------------------
# Dependency Resolution: vcpkg vs Submodule
# -----------------------------------------------------------------------------
find_package(llama CONFIG QUIET)
find_package(ggml CONFIG QUIET)

if(llama_FOUND AND ggml_FOUND)
    message(STATUS "Found llama.cpp and ggml via vcpkg")
    set(USE_VCPKG ON)
else()
    message(STATUS "Falling back to third_party/llama.cpp submodule")
    set(USE_VCPKG OFF)
    
    # Configuration Flags for llama.cpp Submodule
    set(GGML_BACKEND_DL ON CACHE BOOL "Enable dynamic backend loading" FORCE)
    set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries for backends" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Do not build llama.cpp examples" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Do not build llama.cpp tests" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Do not build llama.cpp server" FORCE)
    
    # Enable CUDA backend in llama.cpp if requested
    option(GGML_CUDA "Enable CUDA backend" OFF)
    
    # Add llama.cpp as third-party dependency
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
        message(WARNING "third_party/llama.cpp not found. Run 'git submodule update --init --recursive' before building.")
    else()
        add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)
    endif()
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
)

if(USE_VCPKG)
    # Automatically pulls the include directories for both libraries
    target_link_libraries(infer_core PRIVATE llama ggml)
else()
    target_include_directories(infer_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/ggml/include
    )
    if(TARGET llama AND TARGET ggml)
        target_link_libraries(infer_core PRIVATE llama ggml)
    endif()
endif()

# Enforce strict symbol visibility: hide internal symbols by default
set_target_properties(infer_core PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME "infer"
)

# Export macro definitions for DLL/SO symbol visibility
target_compile_definitions(infer_core
    PRIVATE
        INFER_BUILD_SHARED
)

# -----------------------------------------------------------------------------
# Example CLI target
# -----------------------------------------------------------------------------
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/main.c")
    add_executable(infer_cli examples/main.c)
    target_link_libraries(infer_cli PRIVATE infer_core)
    target_include_directories(infer_cli PRIVATE include)
endif()

# -----------------------------------------------------------------------------
# Install rules for vcpkg
# -----------------------------------------------------------------------------
install(TARGETS infer_core
    EXPORT libinferTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES include/infer.h DESTINATION include)

# This EXPORT file is strictly required by vcpkg_cmake_config_fixup()
install(EXPORT libinferTargets
    FILE libinferTargets.cmake
    NAMESPACE libinfer::
    DESTINATION share/libinfer
)
