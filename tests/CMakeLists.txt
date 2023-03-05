cmake_minimum_required(VERSION 3.1)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

add_executable(doctest.out doctest.cpp)
target_include_directories(doctest.out PUBLIC ../doctest)
target_link_libraries(doctest.out itoyori)

add_test(NAME doctest     COMMAND ${MPIEXEC} -n 1 setarch --addr-no-randomize ./doctest.out)
add_test(NAME doctest_np2 COMMAND ${MPIEXEC} -n 2 setarch --addr-no-randomize ./doctest.out)
add_test(NAME doctest_np4 COMMAND ${MPIEXEC} -n 4 setarch --addr-no-randomize ./doctest.out)
