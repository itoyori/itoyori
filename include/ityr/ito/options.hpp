#pragma once

#include "ityr/common/util.hpp"

namespace ityr::ito {

inline void print_compile_options() {
#ifndef ITYR_ITO_SCHEDULER
#define ITYR_ITO_SCHEDULER ws_workfirst
#endif
  ITYR_PRINT_MACRO(ITYR_ITO_SCHEDULER);
}

}
