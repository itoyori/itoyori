#pragma once

#include <new>

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/mpi_rma.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/profiler.hpp"
#include "ityr/common/prof_events.hpp"

namespace ityr::common {

class global_lock {
public:
  global_lock(int n_locks = 1)
    : n_locks_(n_locks),
      lock_win_(topology::mpicomm(), n_locks_, 0) {}

  bool trylock(topology::rank_t target_rank, int idx = 0) const {
    ITYR_PROFILER_RECORD(prof_event_global_lock_trylock, target_rank);

    ITYR_CHECK(idx < n_locks_);

    lock_t result = mpi_atomic_cas_value<lock_t>(1, 0, target_rank, get_disp(idx), lock_win_.win());

    ITYR_CHECK(0 <= result);
    ITYR_CHECK(result <= 1);
    return result == 0;
  }

  void lock(topology::rank_t target_rank, int idx = 0) const {
    ITYR_CHECK(idx < n_locks_);
    while (!trylock(target_rank, idx));
  }

  void unlock(topology::rank_t target_rank, int idx = 0) const {
    ITYR_PROFILER_RECORD(prof_event_global_lock_unlock, target_rank);

    ITYR_CHECK(idx < n_locks_);

    lock_t ret = mpi_atomic_put_value<lock_t>(0, target_rank, get_disp(idx), lock_win_.win());
    ITYR_CHECK_MESSAGE(ret == 1, "should be locked before unlock");
  }

  bool is_locked(topology::rank_t target_rank, int idx = 0) const {
    ITYR_CHECK(idx < n_locks_);

    lock_t result = mpi_atomic_get_value<lock_t>(target_rank, get_disp(idx), lock_win_.win());
    return result == 1;
  }

private:
  using lock_t = int;

  struct alignas(common::hardware_destructive_interference_size) lock_wrapper {
    template <typename... Args>
    lock_wrapper(Args&&... args) : value(std::forward<Args>(args)...) {}
    lock_t value;
  };

  std::size_t get_disp(int idx) const {
    return idx * sizeof(lock_wrapper) + offsetof(lock_wrapper, value);
  }

  int                           n_locks_;
  mpi_win_manager<lock_wrapper> lock_win_;
};

ITYR_TEST_CASE("[ityr::common::global_lock] lock and unlock") {
  runtime_options opts;
  singleton_initializer<topology::instance> topo;

  ITYR_SUBCASE("single element") {
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

  ITYR_SUBCASE("multiple elements") {
    int n_elems = 3;
    global_lock lock(n_elems);

    using value_t = std::size_t;
    mpi_win_manager<value_t> value_win(topology::mpicomm(), n_elems);

    for (int i = 0; i < n_elems; i++) {
      ITYR_CHECK(value_win.local_buf()[i] == 0);
    }

    auto n_ranks = topology::n_ranks();

    std::size_t n_updates = 1000;

    for (topology::rank_t target_rank = 0; target_rank < n_ranks; target_rank++) {
      for (std::size_t i = 0; i < n_updates; i++) {
        int idx = i % n_elems;
        lock.lock(target_rank, idx);

        auto v = common::mpi_get_value<value_t>(target_rank, idx * sizeof(value_t), value_win.win());
        common::mpi_put_value<value_t>(v + 1, target_rank, idx * sizeof(value_t), value_win.win());

        lock.unlock(target_rank, idx);
      }

      mpi_barrier(topology::mpicomm());
    }

    value_t sum = 0;
    for (int i = 0; i < n_elems; i++) {
      sum += value_win.local_buf()[i];
    }

    ITYR_CHECK(sum == n_updates * n_ranks);
  }
}

}
