cmake_minimum_required(VERSION 3.1)

# set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

add_executable(fib.out fib.cpp)
target_link_libraries(fib.out itoyori)
install(TARGETS fib.out DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/ityr/examples")
