add_library (common Bigraph.cpp AdjList.cpp Vocab.cpp NumaArray.cpp clock.cpp)
set (LINK_LIBS ${LINK_LIBS} common gflags numa )

add_executable (warplda main.cpp lda.cpp warplda.cpp)
add_executable (transform transform.cpp)

target_link_libraries (warplda ${LINK_LIBS})
target_link_libraries (transform ${LINK_LIBS})
