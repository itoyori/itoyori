cmake_minimum_required(VERSION 3.13)

project(itoyori VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_VERBOSE_MAKEFILE 1)

find_package(MPI REQUIRED)
include_directories(${MPI_CXX_INCLUDE_PATH})

add_library(itoyori INTERFACE)
target_include_directories(itoyori INTERFACE include)
target_link_libraries(itoyori INTERFACE ${MPI_CXX_LIBRARIES} rt)
target_compile_options(itoyori INTERFACE -Wall -Wextra -fno-stack-protector)

if(MPI_CXX_COMPILE_FLAGS)
  message(STATUS "MPI_CXX_COMPILE_FLAGS = ${MPI_CXX_COMPILE_FLAGS}")
  separate_arguments(MPI_CXX_COMPILE_OPTIONS UNIX_COMMAND "${MPI_CXX_COMPILE_FLAGS}")
  target_compile_options(itoyori INTERFACE ${MPI_CXX_COMPILE_OPTIONS})
endif()
if(MPI_CXX_LINK_FLAGS)
  message(STATUS "MPI_CXX_LINK_FLAGS = ${MPI_CXX_LINK_FLAGS}")
  separate_arguments(MPI_CXX_LINK_OPTIONS UNIX_COMMAND "${MPI_CXX_LINK_FLAGS}")
  target_link_options(itoyori INTERFACE ${MPI_CXX_LINK_OPTIONS})
endif()

include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

include(CTest)
if(BUILD_TESTING)
  list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
  add_subdirectory(tests)
endif()
