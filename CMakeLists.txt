cmake_minimum_required (VERSION 2.8)

project (gridlda)

add_subdirectory(gflags)

# set compiling flags
set (CMAKE_CXX_FLAGS "-march=native -std=c++1y -Wall -g ${CMAKE_CXX_FLAGS}")

find_package(OpenMP)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


# set include dircetories
include_directories (./src)

add_subdirectory (src)

enable_testing ()
add_subdirectory (test)
