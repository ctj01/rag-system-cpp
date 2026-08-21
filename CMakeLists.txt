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

add_subdirectory(src)
add_subdirectory(bench)
add_subdirectory(apps)

enable_testing()
add_subdirectory(tests)
