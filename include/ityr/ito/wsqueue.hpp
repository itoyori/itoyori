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

template <typename Entry, bool EnablePass = true>
class wsqueue {
public:
  wsqueue(int n_entries, int n_queues = 1)
    : n_entries_(n_entries),
      n_queues_(n_queues),
      initial_pos_(EnablePass ? n_entries / 2 : 0),
      queue_state_win_(common::topology::mpicomm(), n_queues_, initial_pos_),
      entries_win_(common::topology::mpicomm(), n_entries_ * n_queues_),
      queue_lock_(n_queues_) {}

  void push(const Entry& entry, int idx = 0) {
    ITYR_PROFILER_RECORD(prof_event_wsqueue_push);

    ITYR_CHECK(idx < n_queues_);

    queue_state& qs = local_queue_state(idx);
    auto entries = local_entries(idx);

    int t = qs.top.load(std::memory_order_relaxed);

    if (t == n_entries_) {
      queue_lock_.lock(common::topology::my_rank(), idx);

      int b = qs.base.load(std::memory_order_relaxed);
      int offset = -(b + 1) / 2;
      move_entries(offset, idx);
      t += offset;

      queue_lock_.unlock(common::topology::my_rank(), idx);
    }

    entries[t] = entry;

    qs.top.store(t + 1, std::memory_order_release);
  }

  std::optional<Entry> pop(int idx = 0) {
    ITYR_PROFILER_RECORD(prof_event_wsqueue_pop);

    ITYR_CHECK(idx < n_queues_);

    queue_state& qs = local_queue_state(idx);
    if constexpr (EnablePass) {
      // Move entries so that the base does not become too close to zero;
      // otherwise, remote pass operations may fail.
      // Check before the queue empty check.
      int b = qs.base.load(std::memory_order_relaxed);
      if (b < n_entries_ / 10) {
        int t = qs.top.load(std::memory_order_relaxed);
        if (n_entries_ - t > n_entries_ / 10) {
          queue_lock_.lock(common::topology::my_rank(), idx);

          int t = qs.top.load(std::memory_order_relaxed);
          int offset = (n_entries_ - t + 1) / 2;
          move_entries(offset, idx);

          queue_lock_.unlock(common::topology::my_rank(), idx);
        }
      }
    }

    if (qs.empty()) {
      // Wait until the thief releases the lock because the stack copy might be ongoing
      // TODO: any better way to handle this ordering?
      while (queue_lock_.is_locked(common::topology::my_rank(), idx));
      return std::nullopt;
    }

    std::optional<Entry> ret;
    auto entries = local_entries(idx);

    int t = qs.top.load(std::memory_order_relaxed) - 1;
    qs.top.store(t, std::memory_order_relaxed);

    std::atomic_thread_fence(std::memory_order_seq_cst);

    int b = qs.base.load(std::memory_order_relaxed);

    if (b <= t) {
      ret = entries[t];
    } else {
      qs.top.store(t + 1, std::memory_order_relaxed);

      queue_lock_.lock(common::topology::my_rank(), idx);

      qs.top.store(t, std::memory_order_relaxed);
      int b = qs.base.load(std::memory_order_relaxed);

      if (b < t) {
        ret = entries[t];
      } else if (b == t) {
        ret = entries[t];

        qs.top.store(initial_pos_, std::memory_order_relaxed);
        qs.base.store(initial_pos_, std::memory_order_relaxed);
      } else {
        ret = std::nullopt;

        qs.top.store(initial_pos_, std::memory_order_relaxed);
        qs.base.store(initial_pos_, std::memory_order_relaxed);
      }

      queue_lock_.unlock(common::topology::my_rank(), idx);
    }

    return ret;
  }

  std::optional<Entry> steal_nolock(common::topology::rank_t target_rank, int idx = 0) {
    ITYR_PROFILER_RECORD(prof_event_wsqueue_steal, target_rank);

    ITYR_CHECK(idx < n_queues_);

    ITYR_CHECK(queue_lock_.is_locked(target_rank, idx));

    std::optional<Entry> ret;

    int b = common::mpi_atomic_faa_value<int>(1, target_rank, queue_state_base_disp(idx), queue_state_win_.win());
    int t = common::mpi_get_value<int>(target_rank, queue_state_top_disp(idx), queue_state_win_.win());

    if (b < t) {
      ret = common::mpi_get_value<Entry>(target_rank, entries_disp(b, idx), entries_win_.win());
    } else {
      common::mpi_atomic_faa_value<int>(-1, target_rank, queue_state_base_disp(idx), queue_state_win_.win());
      ret = std::nullopt;
    }

    return ret;
  }

  std::optional<Entry> steal(common::topology::rank_t target_rank, int idx = 0) {
    ITYR_CHECK(idx < n_queues_);

    queue_lock_.lock(target_rank, idx);
    auto ret = steal_nolock(target_rank, idx);
    queue_lock_.unlock(target_rank, idx);
    return ret;
  }

  std::optional<Entry> steal_aborting(common::topology::rank_t target_rank, int idx = 0) {
    ITYR_CHECK(idx < n_queues_);

    if (!queue_lock_.trylock(target_rank, idx)) {
      return std::nullopt;
    }
    auto ret = steal_nolock(target_rank, idx);
    queue_lock_.unlock(target_rank, idx);
    return ret;
  }

  bool trypass(const Entry& entry, common::topology::rank_t target_rank, int idx = 0) {
    if constexpr (!EnablePass) {
      common::die("Pass operation is not allowed");
    }

    ITYR_CHECK(idx < n_queues_);

    queue_lock_.lock(target_rank, idx);

    int b = common::mpi_get_value<int>(target_rank, queue_state_base_disp(idx), queue_state_win_.win());

    if (b == 0) {
      queue_lock_.unlock(target_rank, idx);
      return false;
    }

    common::mpi_put_value<Entry>(entry, target_rank, entries_disp(b - 1, idx), entries_win_.win());

    common::mpi_put_value<int>(b - 1, target_rank, queue_state_base_disp(idx), queue_state_win_.win());

    queue_lock_.unlock(target_rank, idx);

    return true;
  }

  void pass(const Entry& entry, common::topology::rank_t target_rank, int idx = 0) {
    ITYR_PROFILER_RECORD(prof_event_wsqueue_pass, target_rank);

    ITYR_CHECK(idx < n_queues_);
    while (!trypass(entry, target_rank, idx));
  }

  template <typename Fn>
  void for_each_entry(Fn fn, int idx = 0) {
    ITYR_CHECK(idx < n_queues_);

    queue_state& qs = local_queue_state(idx);
    /* if (qs.empty()) { */
    /*   return; */
    /* } */

    auto entries = local_entries(idx);

    queue_lock_.lock(common::topology::my_rank(), idx);

    int t = qs.top.load(std::memory_order_relaxed);
    int b = qs.base.load(std::memory_order_relaxed);
    for (int i = b; i < t; i++) {
      fn(entries[i]);
    }

    queue_lock_.unlock(common::topology::my_rank(), idx);
  }

  int size(int idx = 0) const {
    ITYR_CHECK(idx < n_queues_);
    return local_queue_state(idx).size();
  }

  bool empty(common::topology::rank_t target_rank, int idx = 0) const {
    ITYR_PROFILER_RECORD(prof_event_wsqueue_empty, target_rank);

    ITYR_CHECK(idx < n_queues_);

    auto remote_qs = common::mpi_get_value<queue_state>(target_rank, queue_state_disp(idx), queue_state_win_.win());
    return remote_qs.empty();
  }

  const common::global_lock& lock() const { return queue_lock_; }
  int n_queues() const { return n_queues_; }

private:
  struct queue_state {
    std::atomic<int> top;
    std::atomic<int> base;
    // Check if they are safe to be accessed by MPI RMA
    static_assert(sizeof(std::atomic<int>) == sizeof(int));

    queue_state(int initial_pos = 0) : top(initial_pos), base(initial_pos) {}

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

  struct alignas(common::hardware_destructive_interference_size) queue_state_wrapper {
    template <typename... Args>
    queue_state_wrapper(Args&&... args) : value(std::forward<Args>(args)...) {}
    queue_state value;
  };

  std::size_t queue_state_disp(int idx) const {
    return idx * sizeof(queue_state_wrapper) + offsetof(queue_state_wrapper, value);
  }

  std::size_t queue_state_top_disp(int idx) const {
    return idx * sizeof(queue_state_wrapper) + offsetof(queue_state_wrapper, value) + offsetof(queue_state, top);
  }

  std::size_t queue_state_base_disp(int idx) const {
    return idx * sizeof(queue_state_wrapper) + offsetof(queue_state_wrapper, value) + offsetof(queue_state, base);
  }

  std::size_t entries_disp(int entry_num, int idx) const {
    return (entry_num + idx * n_entries_) * sizeof(Entry);
  }

  queue_state& local_queue_state(int idx) const {
    return queue_state_win_.local_buf()[idx].value;
  }

  auto local_entries(int idx) const {
    return entries_win_.local_buf().subspan(idx * n_entries_, n_entries_);
  }

  void move_entries(int offset, int idx) {
    ITYR_CHECK(queue_lock_.is_locked(common::topology::my_rank(), idx));

    queue_state& qs = local_queue_state(idx);
    auto entries = local_entries(idx);

    int t = qs.top.load(std::memory_order_relaxed);
    int b = qs.base.load(std::memory_order_relaxed);

    ITYR_CHECK(b <= t);

    int new_b = b + offset;
    int new_t = t + offset;

    if (offset == 0 || new_b < 0 || n_entries_ < new_t) {
      throw wsqueue_full_exception{};
    }

    std::move(&entries[b], &entries[t], &entries[new_b]);

    qs.top.store(new_t, std::memory_order_relaxed);
    qs.base.store(new_b, std::memory_order_relaxed);
  }

  int                                          n_entries_;
  int                                          n_queues_;
  int                                          initial_pos_;
  common::mpi_win_manager<queue_state_wrapper> queue_state_win_;
  common::mpi_win_manager<Entry>               entries_win_;
  common::global_lock                          queue_lock_;
  bool                                         local_empty_ = false;
};

ITYR_TEST_CASE("[ityr::ito::wsqueue] single queue") {
  int n_entries = 1000;
  using entry_t = int;

  common::runtime_options common_opts;
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
          while (!wsq.empty(my_rank)) {
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
          while (!wsq.empty(my_rank)) {
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

  ITYR_SUBCASE("pass") {
    for (common::topology::rank_t target_rank = 0; target_rank < n_ranks; target_rank++) {
      ITYR_CHECK(wsq.empty(target_rank));

      common::mpi_barrier(common::topology::mpicomm());

      if (target_rank == my_rank) {
        entry_t sum = 0;

        auto req = common::mpi_ibarrier(common::topology::mpicomm());
        while (!common::mpi_test(req) || !wsq.empty(my_rank)) {
          auto result = wsq.steal_aborting(target_rank);
          if (result.has_value()) {
            sum += *result;
          }
        }

        ITYR_CHECK(sum == (n_entries / n_ranks) * (n_entries / n_ranks - 1) / 2 * (n_ranks - 1));

      } else {
        for (int i = 0; i < n_entries / n_ranks; i++) {
          wsq.pass(i, target_rank);
        }

        auto req = common::mpi_ibarrier(common::topology::mpicomm());
        common::mpi_wait(req);
      }

      common::mpi_barrier(common::topology::mpicomm());
    }
  }
}

ITYR_TEST_CASE("[ityr::ito::wsqueue] multiple queues") {
  int n_entries = 1000;
  int n_queues = 3;
  using entry_t = int;

  common::runtime_options common_opts;
  common::singleton_initializer<common::topology::instance> topo;
  wsqueue<entry_t> wsq(n_entries, n_queues);

  auto my_rank = common::topology::my_rank();
  auto n_ranks = common::topology::n_ranks();

  int n_repeats = 5;

  for (common::topology::rank_t target_rank = 0; target_rank < n_ranks; target_rank++) {
    for (int q = 0; q < n_queues; q++) {
      ITYR_CHECK(wsq.empty(target_rank, q));
    }

    common::mpi_barrier(common::topology::mpicomm());

    if (target_rank == my_rank) {
      entry_t sum_expected = 0;
      entry_t local_sum = 0;

      // repeat push and pop
      for (int r = 0; r < n_repeats; r++) {
        for (int i = 0; i < n_entries; i++) {
          for (int q = 0; q < n_queues; q++) {
            wsq.push(i, q);
            sum_expected += i;
          }
        }
        for (int q = 0; q < n_queues; q++) {
          while (!wsq.empty(my_rank, q)) {
            auto result = wsq.pop(q);
            if (result.has_value()) {
              local_sum += *result;
            }
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
        for (int q = 0; q < n_queues; q++) {
          auto result = wsq.steal_aborting(target_rank, q);
          if (result.has_value()) {
            local_sum += *result;
          }
        }
      }

      for (int q = 0; q < n_queues; q++) {
        ITYR_CHECK(wsq.empty(target_rank, q));
      }

      common::mpi_reduce_value(local_sum, target_rank, common::topology::mpicomm());
    }

    common::mpi_barrier(common::topology::mpicomm());
  }
}

}
