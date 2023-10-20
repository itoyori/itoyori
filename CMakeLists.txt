cmake_minimum_required(VERSION 3.13)

project(itoyori VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# set(CMAKE_VERBOSE_MAKEFILE 1)

# set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE RelWithDebInfo)
# set(CMAKE_BUILD_TYPE Debug)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -DNDEBUG -g")

find_package(MPI REQUIRED)

add_library(itoyori INTERFACE)
target_include_directories(itoyori INTERFACE
  ${MPI_CXX_INCLUDE_PATH}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/unordered_dense/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/lxm_random/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/massivelogger/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(itoyori INTERFACE ${MPI_CXX_LIBRARIES} rt)
target_compile_options(itoyori INTERFACE -Wall -Wextra -fno-stack-protector)

if(MPI_CXX_COMPILE_FLAGS)
  message(STATUS "MPI_CXX_COMPILE_FLAGS = ${MPI_CXX_COMPILE_FLAGS}")
  separate_arguments(MPI_CXX_COMPILE_OPTIONS UNIX_COMMAND "SHELL:${MPI_CXX_COMPILE_FLAGS}")
  target_compile_options(itoyori INTERFACE ${MPI_CXX_COMPILE_OPTIONS})
endif()
if(MPI_CXX_LINK_FLAGS)
  message(STATUS "MPI_CXX_LINK_FLAGS = ${MPI_CXX_LINK_FLAGS}")
  separate_arguments(MPI_CXX_LINK_OPTIONS UNIX_COMMAND "SHELL:${MPI_CXX_LINK_FLAGS}")
  target_link_options(itoyori INTERFACE ${MPI_CXX_LINK_OPTIONS})
endif()

option(ITYR_USE_BOOST_MEMORY_RESOURCE "Use boost::container::pmr instead of std::pmr" OFF)

include(CheckIncludeFileCXX)
check_include_file_cxx(memory_resource HAS_STD_MEMORY_RESOURCE)
if(NOT HAS_STD_MEMORY_RESOURCE)
  set(ITYR_USE_BOOST_MEMORY_RESOURCE ON)
endif()

if(ITYR_USE_BOOST_MEMORY_RESOURCE)
  find_package(Boost 1.80.0 REQUIRED COMPONENTS container)
  target_link_libraries(itoyori INTERFACE ${Boost_LIBRARIES})
  target_compile_definitions(itoyori INTERFACE -DITYR_ALLOCATOR_USE_BOOST=1)
endif()

check_include_file_cxx(utofu.h HAS_UTOFU)
if(HAS_UTOFU)
  target_link_libraries(itoyori INTERFACE tofucom)
endif()

install(TARGETS itoyori EXPORT itoyori-targets)

include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY unordered_dense/include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY lxm_random/include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY massivelogger/include/ DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

set(ConfigPackageLocation lib/cmake/itoyori)
install(EXPORT itoyori-targets
  FILE itoyori-targets.cmake
  DESTINATION ${ConfigPackageLocation}
  EXPORT_LINK_INTERFACE_LIBRARIES
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/itoyori-config.cmake"
  INSTALL_DESTINATION ${ConfigPackageLocation}
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/itoyori-config-version.cmake"
  VERSION ${itoyori_VERSION}
  COMPATIBILITY AnyNewerVersion
)
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/itoyori-config.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/itoyori-config-version.cmake"
  DESTINATION ${ConfigPackageLocation}
)

set(MAX_VERBOSE_LEVEL 0 CACHE STRING "Max verbose level for logging output")
add_definitions(-DITYR_MAX_VERBOSE_LEVEL=${MAX_VERBOSE_LEVEL})

include(CTest)
if(BUILD_TESTING)
  list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")
  add_subdirectory(tests)
endif()

option(BUILD_EXAMPLES "Build and install examples" ON)
if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
