cmake_minimum_required(VERSION 3.1)

add_executable(fib.out fib.cpp)
target_link_libraries(fib.out itoyori)

add_executable(nqueens.out nqueens.cpp)
target_link_libraries(nqueens.out itoyori)

install(TARGETS fib.out nqueens.out DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/ityr/examples")
