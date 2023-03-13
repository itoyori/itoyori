#pragma once

#include <fstream>

#include "ityr/common/util.hpp"

namespace ityr::ori {

namespace mode {
struct read_t {};
inline constexpr read_t read;
struct write_t {};
inline constexpr write_t write;
struct read_write_t {};
inline constexpr read_write_t read_write;
}

inline std::string str(mode::read_t) {
  return "read";
}

inline std::string str(mode::write_t) {
  return "write";
}

inline std::string str(mode::read_write_t) {
  return "read_write";
}

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
