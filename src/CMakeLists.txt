add_library (common Bigraph.cpp AdjList.cpp Vocab.cpp)
set (LINK_LIBS ${LINK_LIBS} gflags common)

add_executable (warplda main.cpp)
add_executable (transform transform.cpp)

target_link_libraries (warplda ${LINK_LIBS})
target_link_libraries (transform ${LINK_LIBS})
