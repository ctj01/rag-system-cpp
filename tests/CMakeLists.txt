add_executable(test_dot test_dot.cpp)
target_link_libraries(test_dot PRIVATE vecdb_core)
target_include_directories(test_dot PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME dot_scalar_vs_simd COMMAND test_dot)

add_executable(test_index test_index.cpp)
target_link_libraries(test_index PRIVATE vecdb_core)
target_include_directories(test_index PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME vector_index COMMAND test_index)

add_executable(test_hnsw test_hnsw.cpp)
target_link_libraries(test_hnsw PRIVATE vecdb_core)
target_include_directories(test_hnsw PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME hnsw_recall COMMAND test_hnsw)

add_executable(test_models test_models.cpp)
target_link_libraries(test_models PRIVATE vecdb_core)
target_include_directories(test_models PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME stub_models COMMAND test_models)

add_executable(test_rag test_rag.cpp)
target_link_libraries(test_rag PRIVATE vecdb_core)
target_include_directories(test_rag PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME rag_pipeline COMMAND test_rag)
