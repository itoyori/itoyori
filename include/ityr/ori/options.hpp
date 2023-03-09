#pragma once

#include "ityr/common/util.hpp"

namespace ityr::ori {

inline void print_compile_options() {
#ifndef ITYR_ORI_BLOCK_SIZE
#define ITYR_ORI_BLOCK_SIZE 65536
#endif
  ITYR_PRINT_MACRO(ITYR_ORI_BLOCK_SIZE);

#ifndef ITYR_ORI_DEFAULT_MEM_MAPPER
#define ITYR_ORI_DEFAULT_MEM_MAPPER cyclic
#endif
  ITYR_PRINT_MACRO(ITYR_ORI_DEFAULT_MEM_MAPPER);
}

}
