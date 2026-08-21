# Static library with the numeric core.
# [C#→C++] A static lib is NOT an assembly: no metadata, no runtime identity.
# It is an archive of compiled object files the linker copies into the final
# executable. The public contract is the headers, not the lib.
add_library(vecdb_core STATIC
    core/dot_scalar.cpp
    core/dot_avx2.cpp
    core/dot_dispatch.cpp
    core/vector_index.cpp
    core/hnsw_index.cpp
    models/stub_models.cpp
    rag/chunker.cpp
    rag/rag_pipeline.cpp
)

# PUBLIC: anything linking vecdb_core inherits this include path (tests can
# #include "core/dot.hpp" with no extra setup).
target_include_directories(vecdb_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# CRITICAL: -mavx2/-mfma ONLY on this translation unit. Applied globally, the
# compiler could emit AVX2 instructions in any function of the project and the
# binary would SIGILL on a CPU without AVX2 — even with our runtime dispatch.
# Isolated here, only dot_avx2() contains AVX2, and it is only called after
# the CPU reports support.
set_source_files_properties(core/dot_avx2.cpp PROPERTIES
    COMPILE_OPTIONS "-mavx2;-mfma"
)

# llama.cpp backends: compiled only when the dependency was fetched. PUBLIC
# link/definition so consumers (apps) can #ifdef VECDB_WITH_LLAMA and call
# the factories.
if(VECDB_WITH_LLAMA)
    target_sources(vecdb_core PRIVATE models/llama_models.cpp)
    target_link_libraries(vecdb_core PUBLIC llama)
    target_compile_definitions(vecdb_core PUBLIC VECDB_WITH_LLAMA)
endif()
