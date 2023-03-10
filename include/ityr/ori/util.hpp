#pragma once

#include <fstream>

#include "ityr/common/util.hpp"

namespace ityr::ori {

using block_size_t = uint32_t;

inline std::size_t sys_mmap_entry_limit() {
  std::ifstream ifs("/proc/sys/vm/max_map_count");
  if (!ifs) {
    common::die("Cannot open /proc/sys/vm/max_map_count");
  }
  std::size_t sys_limit;
  ifs >> sys_limit;
  return sys_limit;
}

}
