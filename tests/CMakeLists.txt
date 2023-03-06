cmake_minimum_required(VERSION 3.1)

include_directories(../doctest)
add_executable(doctest.out doctest.cpp)
target_link_libraries(doctest.out itoyori)

add_test(NAME doctest     COMMAND ${MPIEXEC} -n 1 setarch ${CMAKE_HOST_SYSTEM_PROCESSOR} --addr-no-randomize ./doctest.out)
add_test(NAME doctest_np2 COMMAND ${MPIEXEC} -n 2 setarch ${CMAKE_HOST_SYSTEM_PROCESSOR} --addr-no-randomize ./doctest.out)
add_test(NAME doctest_np4 COMMAND ${MPIEXEC} -n 4 setarch ${CMAKE_HOST_SYSTEM_PROCESSOR} --addr-no-randomize ./doctest.out)
