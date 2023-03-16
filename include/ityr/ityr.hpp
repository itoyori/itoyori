#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/wallclock.hpp"
#include "ityr/common/profiler.hpp"
#include "ityr/common/options.hpp"
#include "ityr/ito/ito.hpp"
#include "ityr/ito/options.hpp"
#include "ityr/ori/ori.hpp"
#include "ityr/pattern/iterator.hpp"
#include "ityr/pattern/root_exec.hpp"
#include "ityr/pattern/parallel_loop.hpp"
#include "ityr/pattern/parallel_invoke.hpp"
#include "ityr/container/global_span.hpp"
#include "ityr/container/global_vector.hpp"

namespace ityr {

class ityr {
public:
  ityr(std::size_t cache_size, std::size_t sub_block_size, MPI_Comm comm)
    : topo_(comm),
      ito_(comm),
      ori_(cache_size, sub_block_size, comm) {}

private:
  common::mpi_initializer                                    mi_;
  common::singleton_initializer<common::topology::instance>  topo_;
  common::singleton_initializer<common::wallclock::instance> clock_;
  common::singleton_initializer<common::profiler::instance>  prof_;
  common::singleton_initializer<ito::instance>               ito_;
  common::singleton_initializer<ori::instance>               ori_;
};

using instance = common::singleton<ityr>;

inline void init(std::size_t cache_size     = std::size_t(16) * 1024 * 1024,
                 std::size_t sub_block_size = std::size_t(4) * 1024,
                 MPI_Comm comm              = MPI_COMM_WORLD) {
  instance::init(cache_size, sub_block_size, comm);
}

inline void fini() {
  instance::fini();
}

inline common::topology::rank_t my_rank() {
  return common::topology::my_rank();
}

inline common::topology::rank_t n_ranks() {
  return common::topology::n_ranks();
}

inline bool is_master() {
  return my_rank() == 0;
}

inline void barrier() {
  common::mpi_barrier(common::topology::mpicomm());
}

inline common::wallclock::wallclock_t gettime_ns() {
  return common::wallclock::gettime_ns();
}

inline void profiler_begin() {
  common::profiler::begin();
}

inline void profiler_end() {
  common::profiler::end();
}

inline void profiler_flush() {
  common::profiler::flush();
}

inline void print_compile_options() {
  common::print_compile_options();
  ito::print_compile_options();
}

}
