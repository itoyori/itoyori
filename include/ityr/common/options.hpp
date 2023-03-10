#pragma once

#include <cstdio>

#define ITYR_STR_EXPAND(x) #x
#define ITYR_STR(x) ITYR_STR_EXPAND(x)
#define ITYR_PRINT_MACRO(x) printf(#x "=" ITYR_STR_EXPAND(x) "\n")

namespace ityr::common {

inline void print_compile_options() {
#ifndef ITYR_MAX_VERBOSE_LEVEL
#define ITYR_MAX_VERBOSE_LEVEL 0
#endif
  ITYR_PRINT_MACRO(ITYR_MAX_VERBOSE_LEVEL);

#ifndef ITYR_PROFILER_MODE
#define ITYR_PROFILER_MODE disabled
#endif
  ITYR_PRINT_MACRO(ITYR_PROFILER_MODE);

#ifndef ITYR_ALLOCATOR_USE_DYNAMIC_WIN
#define ITYR_ALLOCATOR_USE_DYNAMIC_WIN false
#endif
  ITYR_PRINT_MACRO(ITYR_ALLOCATOR_USE_DYNAMIC_WIN);
}

}
