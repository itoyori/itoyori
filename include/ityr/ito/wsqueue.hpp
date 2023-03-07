#pragma once

#include <atomic>
#include <optional>
#include <memory>
#include <type_traits>
#include <algorithm>

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/mpi_rma.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/global_lock.hpp"
#include "ityr/common/profiler.hpp"
#include "ityr/ito/prof_events.hpp"

namespace ityr::ito {

class wsqueue_full_exception : public std::exception {
public:
  const char* what() const noexcept override { return "Work stealing queue is full."; }
};

template <typename Entry>
class wsqueue {
public:
  wsqueue(int n_entries)
    : n_entries_(n_entries),
      queue_state_win_(common::topology::mpicomm(), 1),
      entries_win_(common::topology::mpicomm(), n_entries_) {}

  void push(const Entry& entry) {
    ITYR_PROFILER_RECORD(prof_event_wsqueue_push);

    local_empty_ = false;

    queue_state& qs = local_queue_state();
    auto entries = local_entries();

    int t = qs.top.load(std::memory_order_relaxed);

    if (t == n_entries_) {
      queue_lock_.lock(common::topology::my_rank());

      int b = qs.base.load(std::memory_order_relaxed);

      if (b == 0) {
        throw wsqueue_full_exception{};
      }

      std::move(&entries[b], &entries[t], entries.begin());

      qs.top.store(t - b, std::memory_order_relaxed);
      qs.base.store(0, std::memory_order_relaxed);

      t = t - b;

      queue_lock_.unlock(common::topology::my_rank());
    }

    entries[t] = entry;

    qs.top.store(t + 1, std::memory_order_release);
  }

  std::optional<Entry> pop() {
    ITYR_PROFILER_RECORD(prof_event_wsqueue_pop);

    if (local_empty_) {
      return std::nullopt;
    }

    std::optional<Entry> ret;

    queue_state& qs = local_queue_state();
    auto entries = local_entries();

    int t = qs.top.load(std::memory_order_relaxed) - 1;
    qs.top.store(t, std::memory_order_relaxed);

    std::atomic_thread_fence(std::memory_order_seq_cst);

    int b = qs.base.load(std::memory_order_relaxed);

    if (b <= t) {
      ret = entries[t];
    } else {
      qs.top.store(t + 1, std::memory_order_relaxed);

      queue_lock_.lock(common::topology::my_rank());

      qs.top.store(t, std::memory_order_relaxed);
      int b = qs.base.load(std::memory_order_relaxed);

      if (b < t) {
        ret = entries[t];
      } else if (b == t) {
        ret = entries[t];

        qs.top.store(0, std::memory_order_relaxed);
        qs.base.store(0, std::memory_order_relaxed);

        local_empty_ = true;
      } else {
        ret = std::nullopt;

        qs.top.store(0, std::memory_order_relaxed);
        qs.base.store(0, std::memory_order_relaxed);

        local_empty_ = true;
      }

      queue_lock_.unlock(common::topology::my_rank());
    }

    return ret;
  }

  std::optional<Entry> steal_nolock(common::topology::rank_t target_rank) {
    ITYR_PROFILER_RECORD(prof_event_wsqueue_steal);

    ITYR_CHECK(queue_lock_.is_locked(target_rank));

    std::optional<Entry> ret;

    int b = common::mpi_atomic_faa_value<int>(1, target_rank, offsetof(queue_state, base), queue_state_win_.win());
    int t = common::mpi_get_value<int>(target_rank, offsetof(queue_state, top), queue_state_win_.win());

    if (b < t) {
      ret = common::mpi_get_value<Entry>(target_rank, sizeof(Entry) * b, entries_win_.win());
    } else {
      common::mpi_atomic_faa_value<int>(-1, target_rank, offsetof(queue_state, base), queue_state_win_.win());
      ret = std::nullopt;
    }

    return ret;
  }

  std::optional<Entry> steal(common::topology::rank_t target_rank) {
    queue_lock_.lock(target_rank);
    auto ret = steal_nolock(target_rank);
    queue_lock_.unlock(target_rank);
    return ret;
  }

  std::optional<Entry> steal_aborting(common::topology::rank_t target_rank) {
    if (!queue_lock_.trylock(target_rank)) {
      return std::nullopt;
    }
    auto ret = steal_nolock(target_rank);
    queue_lock_.unlock(target_rank);
    return ret;
  }

  int size() const {
    return local_queue_state().size();
  }

  bool empty() const {
    return local_empty_ || local_queue_state().empty();
  }

  bool empty(common::topology::rank_t target_rank) const {
    ITYR_PROFILER_RECORD(prof_event_wsqueue_empty);

    auto remote_qs = common::mpi_get_value<queue_state>(target_rank, 0, queue_state_win_.win());
    return remote_qs.empty();
  }

  const common::global_lock& lock() const { return queue_lock_; }

private:
  struct queue_state {
    std::atomic<int> top  = 0;
    std::atomic<int> base = 0;
    // Check if they are safe to be accessed by MPI RMA
    static_assert(sizeof(std::atomic<int>) == sizeof(int));

    queue_state() = default;

    // Copy constructors for std::atomic are deleted
    queue_state(const queue_state& qs) {
      top.store(qs.top.load(std::memory_order_relaxed), std::memory_order_relaxed);
      base.store(qs.base.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }
    queue_state& operator=(const queue_state& qs) {
      top.store(qs.top.load(std::memory_order_relaxed), std::memory_order_relaxed);
      base.store(qs.base.load(std::memory_order_relaxed), std::memory_order_relaxed);
    }

    queue_state(queue_state&& wm) = default;
    queue_state& operator=(queue_state&& wm) = default;

    int size() const {
      return std::max(0, top.load(std::memory_order_relaxed) -
                         base.load(std::memory_order_relaxed));
    }

    bool empty() const {
      return top.load(std::memory_order_relaxed) <=
             base.load(std::memory_order_relaxed);
    }
  };

  static_assert(std::is_standard_layout_v<queue_state>);
  // FIXME: queue_state is no longer trivially copyable.
  //        Thus, strictly speaking, using MPI RMA for queue_state is illegal.
  // static_assert(std::is_trivially_copyable_v<queue_state>);

  queue_state& local_queue_state() const {
    return queue_state_win_.local_buf()[0];
  }

  auto local_entries() const {
    return entries_win_.local_buf();
  }

  int                                  n_entries_;
  common::mpi_win_manager<queue_state> queue_state_win_;
  common::mpi_win_manager<Entry>       entries_win_;
  common::global_lock                  queue_lock_;
  bool                                 local_empty_ = false;
};

ITYR_TEST_CASE("[ityr::ito::wsqueue] work stealing queue") {
  int n_entries = 1000;
  using entry_t = int;

  common::singleton_initializer<common::topology::instance> topo;
  wsqueue<entry_t> wsq(n_entries);

  auto my_rank = common::topology::my_rank();
  auto n_ranks = common::topology::n_ranks();

  ITYR_SUBCASE("local push and pop") {
    int n_trial = 3;
    for (int t = 0; t < n_trial; t++) {
      for (int i = 0; i < n_entries; i++) {
        wsq.push(i);
      }
      for (int i = 0; i < n_entries; i++) {
        auto result = wsq.pop();
        ITYR_CHECK(result.has_value());
        ITYR_CHECK(*result == n_entries - i - 1); // LIFO order
      }
    }
  }

  ITYR_SUBCASE("should throw exception when full") {
    for (int i = 0; i < n_entries; i++) {
      wsq.push(i);
    }
    ITYR_CHECK_THROWS_AS(wsq.push(n_entries), wsqueue_full_exception);
  }

  ITYR_SUBCASE("steal") {
    if (n_ranks == 1) return;

    for (common::topology::rank_t target_rank = 0; target_rank < n_ranks; target_rank++) {
      ITYR_CHECK(wsq.empty(target_rank));

      common::mpi_barrier(common::topology::mpicomm());

      entry_t sum_expected = 0;
      if (target_rank == my_rank) {
        for (int i = 0; i < n_entries; i++) {
          wsq.push(i);
          sum_expected += i;
        }
      }

      common::mpi_barrier(common::topology::mpicomm());

      entry_t local_sum = 0;

      ITYR_SUBCASE("remote steal by only one process") {
        if ((target_rank + 1) % n_ranks == my_rank) {
          for (int i = 0; i < n_entries; i++) {
            auto result = wsq.steal(target_rank);
            ITYR_CHECK(result.has_value());
            ITYR_CHECK(*result == i); // FIFO order
            local_sum += *result;
          }
        }
      }

      ITYR_SUBCASE("remote steal concurrently") {
        if (target_rank != my_rank) {
          while (!wsq.empty(target_rank)) {
            auto result = wsq.steal(target_rank);
            if (result.has_value()) {
              local_sum += *result;
            }
          }
        }
      }

      ITYR_SUBCASE("local pop and remote steal concurrently") {
        if (target_rank == my_rank) {
          while (!wsq.empty()) {
            auto result = wsq.pop();
            if (result.has_value()) {
              local_sum += *result;
            }
          }
        } else {
          while (!wsq.empty(target_rank)) {
            auto result = wsq.steal_aborting(target_rank);
            if (result.has_value()) {
              local_sum += *result;
            }
          }
        }
      }

      common::mpi_barrier(common::topology::mpicomm());
      entry_t sum_all = common::mpi_reduce_value(local_sum, target_rank, common::topology::mpicomm());

      ITYR_CHECK(wsq.empty(target_rank));

      if (target_rank == my_rank) {
        ITYR_CHECK(sum_all == sum_expected);
      }

      common::mpi_barrier(common::topology::mpicomm());
    }
  }

  ITYR_SUBCASE("all operations concurrently") {
    int n_repeats = 5;

    for (common::topology::rank_t target_rank = 0; target_rank < n_ranks; target_rank++) {
      ITYR_CHECK(wsq.empty(target_rank));

      common::mpi_barrier(common::topology::mpicomm());

      if (target_rank == my_rank) {
        entry_t sum_expected = 0;
        entry_t local_sum = 0;

        // repeat push and pop
        for (int r = 0; r < n_repeats; r++) {
          for (int i = 0; i < n_entries; i++) {
            wsq.push(i);
            sum_expected += i;
          }
          while (!wsq.empty()) {
            auto result = wsq.pop();
            if (result.has_value()) {
              local_sum += *result;
            }
          }
        }

        auto req = common::mpi_ibarrier(common::topology::mpicomm());
        common::mpi_wait(req);

        entry_t sum_all = common::mpi_reduce_value(local_sum, target_rank, common::topology::mpicomm());

        ITYR_CHECK(sum_all == sum_expected);

      } else {
        entry_t local_sum = 0;

        auto req = common::mpi_ibarrier(common::topology::mpicomm());
        while (!common::mpi_test(req)) {
          auto result = wsq.steal_aborting(target_rank);
          if (result.has_value()) {
            local_sum += *result;
          }
        }

        ITYR_CHECK(wsq.empty(target_rank));

        common::mpi_reduce_value(local_sum, target_rank, common::topology::mpicomm());
      }

      common::mpi_barrier(common::topology::mpicomm());
    }
  }

  ITYR_SUBCASE("resize queue") {
    if (n_ranks == 1) return;

    for (common::topology::rank_t target_rank = 0; target_rank < n_ranks; target_rank++) {
      ITYR_CHECK(wsq.empty(target_rank));

      common::mpi_barrier(common::topology::mpicomm());

      if (target_rank == my_rank) {
        for (int i = 0; i < n_entries; i++) {
          wsq.push(i);
        }
      }

      common::mpi_barrier(common::topology::mpicomm());

      // only one process steals
      if ((target_rank + 1) % n_ranks == my_rank) {
        // steal half of the queue
        for (int i = 0; i < n_entries / 2; i++) {
          auto result = wsq.steal(target_rank);
          ITYR_CHECK(result.has_value());
          ITYR_CHECK(*result == i);
        }
      }

      common::mpi_barrier(common::topology::mpicomm());

      if (target_rank == my_rank) {
        // push half of the queue
        for (int i = 0; i < n_entries / 2; i++) {
          wsq.push(i);
        }
        // pop all
        for (int i = 0; i < n_entries; i++) {
          auto result = wsq.pop();
          ITYR_CHECK(result.has_value());
        }
      }

      common::mpi_barrier(common::topology::mpicomm());

      ITYR_CHECK(wsq.empty(target_rank));
    }
  }
}

}
