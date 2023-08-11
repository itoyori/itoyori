#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/options.hpp"

namespace ityr::common {

inline constexpr int max_verbose_level = ITYR_MAX_VERBOSE_LEVEL;

template <int Level = 1>
inline void verbose(const char* fmt, ...) {
  if constexpr (Level <= max_verbose_level) {
    constexpr int slen = 256;
    static char msg[slen];

    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, slen, fmt, args);
    va_end(args);

    fprintf(stderr, "%ld: %s\n", clock_gettime_ns(), msg);
  }
}

}
