add_executable(vecdb_bench bench_dot.cpp)
target_link_libraries(vecdb_bench PRIVATE vecdb_core)

add_executable(vecdb_bench_hnsw bench_hnsw.cpp)
target_link_libraries(vecdb_bench_hnsw PRIVATE vecdb_core)
