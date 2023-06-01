#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/ori/util.hpp"
#include "ityr/ori/options.hpp"

namespace ityr::ori {

class home_profiler_disabled {
public:
  home_profiler_disabled() {}
  void record(std::size_t, bool) {}
  void record(std::byte*, std::size_t, std::byte*, std::size_t, bool) {}
  void start() {}
  void stop() {}
  void print() const {}
};

class home_profiler_stats {
public:
  home_profiler_stats() {}

  void record(std::size_t bytes, bool hit) {
    if (enabled_) {
      requested_bytes_ += bytes;
      if (hit) {
        seg_hit_count_++;
      } else {
        seg_miss_count_++;
      }
    }
  }

  void record(std::byte* seg_addr, std::size_t seg_size,
              std::byte* req_addr, std::size_t req_size, bool hit) {
    std::byte* addr_b = std::max(seg_addr, reinterpret_cast<std::byte*>(req_addr));
    std::byte* addr_e = std::min(seg_addr + seg_size, reinterpret_cast<std::byte*>(req_addr) + req_size);
    record(addr_e - addr_b, hit);
  }

  void start() {
    requested_bytes_ = 0;
    seg_hit_count_   = 0;
    seg_miss_count_  = 0;

    enabled_ = true;
  }

  void stop() {
    enabled_ = false;
  }

  void print() const {
    auto requested_bytes_all = common::mpi_reduce_value(requested_bytes_, 0, common::topology::mpicomm());
    auto seg_hit_count_all   = common::mpi_reduce_value(seg_hit_count_  , 0, common::topology::mpicomm());
    auto seg_miss_count_all  = common::mpi_reduce_value(seg_miss_count_ , 0, common::topology::mpicomm());

    if (common::topology::my_rank() == 0) {
      printf("[Home segments]\n");
      printf("  User requested:   %18ld bytes\n"   , requested_bytes_all);
      printf("  mmap hit count:   %18ld segments\n", seg_hit_count_all);
      printf("  mmap miss count:  %18ld segments\n", seg_miss_count_all);
      printf("\n");
      fflush(stdout);
    }
  }

private:
  std::size_t requested_bytes_ = 0;
  std::size_t seg_hit_count_   = 0;
  std::size_t seg_miss_count_  = 0;
  bool        enabled_         = false;
};

using home_profiler = ITYR_CONCAT(home_profiler_, ITYR_ORI_CACHE_PROF);

}
