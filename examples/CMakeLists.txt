cmake_minimum_required(VERSION 3.1)

set(examples fib nqueens cilksort)

foreach(example IN LISTS examples)
  add_executable(${example}.out ${example}.cpp)
  target_link_libraries(${example}.out itoyori)

  add_executable(${example}_serial.out ${example}.cpp)
  target_link_libraries(${example}_serial.out itoyori)
  target_compile_options(${example}_serial.out PRIVATE -DITYR_ITO_SCHEDULER=serial)

  add_executable(${example}_prof_stats.out ${example}.cpp)
  target_link_libraries(${example}_prof_stats.out itoyori)
  target_compile_options(${example}_prof_stats.out PRIVATE -DITYR_PROFILER_MODE=stats)

  add_executable(${example}_prof_trace.out ${example}.cpp)
  target_link_libraries(${example}_prof_trace.out itoyori)
  target_compile_options(${example}_prof_trace.out PRIVATE -DITYR_PROFILER_MODE=trace)

  install(TARGETS ${example}.out
                  ${example}_serial.out
                  ${example}_prof_stats.out
                  ${example}_prof_trace.out
          DESTINATION "${CMAKE_INSTALL_LIBEXECDIR}/ityr/examples")
endforeach()

