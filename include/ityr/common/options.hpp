#pragma once

#include "ityr/common/util.hpp"

namespace ityr::common {

inline void print_compile_options() {
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
