#add_library (comm "corpus" "diagnostics" )
set (LINK_LIBS ${LINK_LIBS} comm)

add_executable (warplda main.cpp)

#target_link_libraries (${target} ${LINK_LIBS})
