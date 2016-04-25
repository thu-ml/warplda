set (LINK_LIBS ${LINK_LIBS} gflags)

add_executable (warplda main.cpp)

target_link_libraries (warplda ${LINK_LIBS})
