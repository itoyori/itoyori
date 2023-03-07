#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/mpi_rma.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/profiler.hpp"
#include "ityr/common/prof_events.hpp"

namespace ityr::common {

class global_lock {
public:
  global_lock()
    : lock_win_(topology::mpicomm(), 1) {}

  bool trylock(topology::rank_t target_rank) const {
    ITYR_PROFILER_RECORD(prof_event_global_lock_trylock, target_rank);
    lock_t result = mpi_atomic_cas_value<lock_t>(1, 0, target_rank, 0, lock_win_.win());
    return result == 0;
  }

  void lock(topology::rank_t target_rank) const {
    while (!trylock(target_rank));
  }

  void unlock(topology::rank_t target_rank) const {
    ITYR_PROFILER_RECORD(prof_event_global_lock_unlock, target_rank);
    lock_t ret = mpi_atomic_put_value<lock_t>(0, target_rank, 0, lock_win_.win());
    ITYR_CHECK_MESSAGE(ret == 1, "should be locked before unlock");
  }

  bool is_locked(topology::rank_t target_rank) const {
    lock_t result = mpi_atomic_get_value<lock_t>(target_rank, 0, lock_win_.win());
    return result == 1;
  }

private:
  using lock_t = int;

  mpi_win_manager<lock_t> lock_win_;
};

ITYR_TEST_CASE("[ityr::common::global_lock] lock and unlock") {
  singleton_initializer<topology::instance> topo;

  global_lock lock;

  using value_t = std::size_t;
  mpi_win_manager<value_t> value_win(topology::mpicomm(), 1);

  ITYR_CHECK(value_win.local_buf()[0] == 0);

  auto n_ranks = topology::n_ranks();

  std::size_t n_updates = 100;

  for (topology::rank_t target_rank = 0; target_rank < n_ranks; target_rank++) {
    for (std::size_t i = 0; i < n_updates; i++) {
      lock.lock(target_rank);

      auto v = common::mpi_get_value<value_t>(target_rank, 0, value_win.win());
      common::mpi_put_value<value_t>(v + 1, target_rank, 0, value_win.win());

      lock.unlock(target_rank);
    }

    mpi_barrier(topology::mpicomm());
  }

  ITYR_CHECK(value_win.local_buf()[0] == n_updates * n_ranks);
}

}
