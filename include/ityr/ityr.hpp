#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/options.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/wallclock.hpp"
#include "ityr/common/profiler.hpp"
#include "ityr/ito/ito.hpp"
#include "ityr/ori/ori.hpp"
#include "ityr/pattern/count_iterator.hpp"
#include "ityr/pattern/root_exec.hpp"
#include "ityr/pattern/parallel_loop.hpp"
#include "ityr/pattern/parallel_invoke.hpp"
#include "ityr/container/global_span.hpp"
#include "ityr/container/global_vector.hpp"
#include "ityr/container/checkout_span.hpp"

namespace ityr {

namespace internal {

class ityr {
public:
  ityr(MPI_Comm comm)
    : mi_(comm),
      topo_(comm),
      ito_(comm),
      ori_(comm) {}

private:
  common::mpi_initializer                                    mi_;
  common::runtime_options                                    opts_;
  common::singleton_initializer<common::topology::instance>  topo_;
  common::singleton_initializer<common::wallclock::instance> clock_;
  common::singleton_initializer<common::profiler::instance>  prof_;
  common::singleton_initializer<ito::instance>               ito_;
  common::singleton_initializer<ori::instance>               ori_;
};

using instance = common::singleton<ityr>;

}

/**
 * @brief Initialize Itoyori (collective).
 *
 * @param comm MPI communicator to be used in Itoyori (default: `MPI_COMM_WORLD`).
 *
 * This function initializes the Itoyori runtime system.
 * Any itoyori APIs (except for runtime option settings) cannot be called prior to this call.
 * `ityr::fini()` must be called to properly release allocated resources.
 *
 * If MPI is not initialized at this point, Itoyori calls `MPI_Init()` and finalizes MPI by calling
 * `MPI_Finalize()` when `ityr::fini()` is called. If MPI is already initialized at this point,
 * Itoyori does not have the responsibility to finalize MPI.
 *
 * @see `ityr::fini()`
 */
inline void init(MPI_Comm comm = MPI_COMM_WORLD) {
  internal::instance::init(comm);
}

/**
 * @brief Finalize Itoyori (collective).
 *
 * This function finalizes the Itoyori runtime system.
 * Any itoyori APIs cannot be called after this call unless `ityr::init()` is called again.
 *
 * `MPI_Finalize()` may be called in this function if MPI is not initialized by the user when
 * `ityr::init()` is called.
 *
 * @see `ityr::init()`
 */
inline void fini() {
  internal::instance::fini();
}

/**
 * @brief Process rank (ID) starting from 0 (corresponding to an MPI rank).
 * @see `ityr::my_rank()`
 * @see `ityr::n_ranks()`
 */
using rank_t = common::topology::rank_t;

/**
 * @brief Return the rank of the process running the current thread.
 * @see `ityr::n_ranks()`
 */
inline rank_t my_rank() {
  return common::topology::my_rank();
}

/**
 * @brief Return the total number of processes.
 * @see `ityr::n_ranks()`
 */
inline rank_t n_ranks() {
  return common::topology::n_ranks();
}

/**
 * @brief Return true if `ityr::my_rank() == 0`.
 * @see `ityr::my_rank()`
 */
inline bool is_master() {
  return my_rank() == 0;
}

/**
 * @brief Return true if the current execution context is within the SPMD region.
 */
inline bool is_spmd() {
  return ito::is_spmd();
}

/**
 * @brief Barrier for all processes (collective).
 */
inline void barrier() {
  ITYR_CHECK(is_spmd());
  ori::release();
  common::mpi_barrier(common::topology::mpicomm());
  ori::acquire();
}

/**
 * @brief Wallclock time in nanoseconds.
 * @see `ityr::gettime_ns()`.
 */
using wallclock_t = common::wallclock::wallclock_t;

/**
 * @brief Return the current wallclock time in nanoseconds.
 *
 * The wallclock time is calibrated across different processes (that may reside in different machines)
 * at the program startup in a simple way, but the clock may be skewed due to various reasons.
 * To get an accurate execution time, it is recommended to call this function in the same process and
 * calculate the difference.
 */
inline wallclock_t gettime_ns() {
  return common::wallclock::gettime_ns();
}

/**
 * @brief Start the profiler (collective).
 * @see `ityr::profiler_end()`.
 * @see `ityr::profiler_flush()`.
 */
inline void profiler_begin() {
  ITYR_CHECK(is_spmd());
  ori::cache_prof_begin();
  ito::dag_prof_begin();
  common::profiler::begin();
}

/**
 * @brief Stop the profiler (collective).
 * @see `ityr::profiler_begin()`.
 * @see `ityr::profiler_flush()`.
 */
inline void profiler_end() {
  ITYR_CHECK(is_spmd());
  common::profiler::end();
  ito::dag_prof_end();
  ori::cache_prof_end();
}

/**
 * @brief Print the profiled results to stdout (collective).
 * @see `ityr::profiler_begin()`.
 * @see `ityr::profiler_end()`.
 */
inline void profiler_flush() {
  ITYR_CHECK(is_spmd());
  common::profiler::flush();
  ito::dag_prof_print();
  ori::cache_prof_print();
}

/**
 * @brief Print the compile-time options to stdout.
 * @see `ityr::print_runtime_options()`.
 */
inline void print_compile_options() {
  common::print_compile_options();
  ito::print_compile_options();
  ori::print_compile_options();
}

/**
 * @brief Print the runtime options to stdout.
 * @see `ityr::print_compile_options()`.
 */
inline void print_runtime_options() {
  common::print_runtime_options();
}

}
