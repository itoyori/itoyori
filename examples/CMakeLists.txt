cmake_minimum_required(VERSION 3.1)

set(examples fib nqueens)

foreach(example IN LISTS examples)
  add_executable(${example}.out ${example}.cpp)
  target_link_libraries(${example}.out itoyori)

  add_executable(${example}_stats.out ${example}.cpp)
  target_link_libraries(${example}_stats.out itoyori)
  target_compile_options(${example}_stats.out PRIVATE -DITYR_PROFILER_MODE=stats)

  add_executable(${example}_trace.out ${example}.cpp)
  target_link_libraries(${example}_trace.out itoyori)
  target_compile_options(${example}_trace.out PRIVATE -DITYR_PROFILER_MODE=trace)

  install(TARGETS ${example}.out
                  ${example}_stats.out
                  ${example}_trace.out
          DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/ityr/examples")
endforeach()

