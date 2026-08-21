# [C#→C++] In .NET, the .csproj + MSBuild handle compilation, references and
# dependencies in one step. C++ has no standard build system: CMake is the de
# facto one. CMake does NOT compile — it generates build files (Makefiles,
# Ninja) that then drive the actual compiler (g++, clang, msvc).
cmake_minimum_required(VERSION 3.20)

project(vecdb
    VERSION 0.1.0
    DESCRIPTION "Hand-rolled vector search engine with a RAG demo on top"
    LANGUAGES CXX
)

# [C#→C++] In C# the language version is tied to the project SDK. In C++ the
# standard is requested explicitly and a compiler may only partially support
# it. CXX_STANDARD_REQUIRED stops CMake from silently downgrading.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)   # strict -std=c++20, no GNU extensions

# Default to Release: in C++ the Debug/Release gap is brutal (10-50x on
# numeric code) because Debug neither inlines nor vectorizes anything.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# High warnings as part of the project contract (same spirit as
# TreatWarningsAsErrors + analyzers in .NET).
add_compile_options(-Wall -Wextra -Wpedantic)

# ---- llama.cpp (optional, ON by default) ------------------------------------
# [C#→C++] FetchContent is the closest thing to a NuGet PackageReference:
# it downloads the dependency's SOURCE at configure time (pinned tag =
# reproducible) and builds it as part of this project — C++ has no standard
# binary package registry, so source-level vendoring is the portable path.
option(VECDB_WITH_LLAMA "Build the llama.cpp model backends" ON)
if(VECDB_WITH_LLAMA)
    include(FetchContent)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(LLAMA_CURL OFF CACHE BOOL "" FORCE)  # avoid a libcurl dev dependency
    FetchContent_Declare(
        llama
        GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
        GIT_TAG v0.2.0
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(llama)
endif()

add_subdirectory(src)
add_subdirectory(bench)
add_subdirectory(apps)

enable_testing()
add_subdirectory(tests)
