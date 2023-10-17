#pragma once

#include <random>
#include <atomic>

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/mpi_rma.hpp"
#include "ityr/common/wallclock.hpp"
#include "ityr/ito/util.hpp"
#include "ityr/ito/options.hpp"
#include "ityr/ito/prof_events.hpp"

namespace ityr::ito {

/*
 * DAG profiler
 */

class dag_profiler_disabled {
public:
  static constexpr bool enabled = false;
  void start() {}
  void stop() {}
  bool is_stopped() const { return true; }
  void clear() {}
  void merge_serial(const dag_profiler_disabled&) {}
  void merge_parallel(const dag_profiler_disabled&) {}
  void increment_thread_count() {}
  void increment_strand_count() {}
  void print() const {}
};

class dag_profiler_workspan {
public:
  static constexpr bool enabled = true;

  void start() {
    ITYR_CHECK(is_stopped());
    t_start_ = common::wallclock::gettime_ns();
  }

  void stop() {
    ITYR_CHECK(!is_stopped());
    auto t_stop = common::wallclock::gettime_ns();
    work_ += t_stop - t_start_;
    span_ += t_stop - t_start_;
    t_start_ = 0;
  }

  bool is_stopped() const { return t_start_ == 0; }

  void clear() {
    t_start_   = 0;
    work_      = 0;
    span_      = 0;
    n_threads_ = 0;
    n_strands_ = 0;
  }

  void merge_serial(const dag_profiler_workspan& dp) {
    ITYR_CHECK(is_stopped());
    ITYR_CHECK(dp.is_stopped());

    work_ += dp.work_;
    span_ += dp.span_;
    n_threads_ += dp.n_threads_;
    n_strands_ += dp.n_strands_;
  }

  void merge_parallel(const dag_profiler_workspan& dp) {
    ITYR_CHECK(is_stopped());
    ITYR_CHECK(dp.is_stopped());

    work_ += dp.work_;
    span_ = std::max(span_, dp.span_);
    n_threads_ += dp.n_threads_;
    n_strands_ += dp.n_strands_;
  }

  void increment_thread_count() { n_threads_++; }
  void increment_strand_count() { n_strands_++; }

  void print() const {
    printf("work: %ld ns span: %ld ns parallelism: %f\n"
           "n_threads: %ld (ave: %ld ns) n_strands: %ld (ave: %ld ns)\n\n",
           work_, span_, static_cast<double>(work_) / span_,
           n_threads_, work_ / n_threads_, n_strands_, work_ / n_strands_);
    fflush(stdout);
  }

private:
  common::wallclock::wallclock_t t_start_   = 0;
  common::wallclock::wallclock_t work_      = 0;
  common::wallclock::wallclock_t span_      = 0;
  uint64_t                       n_threads_ = 0;
  uint64_t                       n_strands_ = 0;
};

using dag_profiler = ITYR_CONCAT(dag_profiler_, ITYR_ITO_DAG_PROF);

/*
 * Misc
 */

class task_general {
public:
  virtual ~task_general() = default;
  virtual void execute() = 0;
};

template <typename Fn, typename... Args>
class callable_task : public task_general {
public:
  template <typename Fn_, typename... Args_>
  callable_task(Fn_&& fn, Args_&&... args)
    : fn_(std::forward<Fn_>(fn)), arg_(std::forward<Args_>(args)...) {}
  void execute() { std::apply(std::move(fn_), std::move(arg_)); }
private:
  Fn                  fn_;
  std::tuple<Args...> arg_;
};

struct no_retval_t {};

inline common::topology::rank_t get_random_rank(common::topology::rank_t a,
                                                common::topology::rank_t b) {
  static std::mt19937 engine(std::random_device{}());

  ITYR_CHECK(0 <= a);
  ITYR_CHECK(a <= b);
  ITYR_CHECK(b < common::topology::n_ranks());
  std::uniform_int_distribution<common::topology::rank_t> dist(a, b);

  common::topology::rank_t rank;
  do {
    rank = dist(engine);
  } while (rank == common::topology::my_rank());

  ITYR_CHECK(a <= rank);
  ITYR_CHECK(rank != common::topology::my_rank());
  ITYR_CHECK(rank <= b);
  return rank;
}

template <typename T, typename Fn, typename ArgsTuple>
inline decltype(auto) invoke_fn(Fn&& fn, ArgsTuple&& args_tuple) {
  if constexpr (!std::is_same_v<T, no_retval_t>) {
    return std::apply(std::forward<Fn>(fn), std::forward<ArgsTuple>(args_tuple));
  } else {
    std::apply(std::forward<Fn>(fn), std::forward<ArgsTuple>(args_tuple));
    return no_retval_t{};
  }
}

/*
 * Call with profiler events
 */

template <typename Fn, typename... Args>
struct callback_retval {
  using type = std::invoke_result_t<Fn, Args...>;
};

template <typename... Args>
struct callback_retval<std::nullptr_t, Args...> {
  using type = void;
};

template <typename... Args>
struct callback_retval<std::nullptr_t&, Args...> {
  using type = void;
};

template <typename Fn, typename... Args>
using callback_retval_t = typename callback_retval<Fn, Args...>::type;

template <typename PhaseFrom, typename PhaseFn, typename PhaseTo,
          typename Fn, typename... Args>
inline auto call_with_prof_events(Fn&& fn, Args&&... args) {
  using retval_t = callback_retval_t<Fn, Args...>;

  if constexpr (!std::is_null_pointer_v<std::remove_reference_t<Fn>>) {
    common::profiler::switch_phase<PhaseFrom, PhaseFn>();

    if constexpr (!std::is_void_v<retval_t>) {
      auto ret = std::forward<Fn>(fn)(std::forward<Args>(args)...);
      common::profiler::switch_phase<PhaseFn, PhaseTo>();
      return ret;

    } else {
      std::forward<Fn>(fn)(std::forward<Args>(args)...);
      common::profiler::switch_phase<PhaseFn, PhaseTo>();
    }

  } else if constexpr (!std::is_same_v<PhaseFrom, PhaseTo>) {
    common::profiler::switch_phase<PhaseFrom, PhaseTo>();
  }

  if constexpr (!std::is_void_v<retval_t>) {
    return retval_t{};
  } else {
    return no_retval_t{};
  }
}

template <typename PhaseFrom, typename PhaseFn, typename PhaseTo,
          typename Fn, typename... Args>
inline auto call_with_prof_events(Fn&& fn, no_retval_t, Args&&... args) {
  // skip no_retval_t args
  return call_with_prof_events<PhaseFrom, PhaseFn, PhaseTo>(
      std::forward<Fn>(fn), std::forward<Args>(args)...);
}

/*
 * Mailbox
 */

template <typename Entry>
class oneslot_mailbox {
  static_assert(std::is_trivially_copyable_v<Entry>);

public:
  oneslot_mailbox()
    : win_(common::topology::mpicomm(), 1) {}

  void put(const Entry& entry, common::topology::rank_t target_rank) {
    ITYR_PROFILER_RECORD(prof_event_sched_mailbox_put, target_rank);

    ITYR_CHECK(!common::mpi_get_value<int>(target_rank, offsetof(mailbox, arrived), win_.win()));
    common::mpi_put_value(entry, target_rank, offsetof(mailbox, entry), win_.win());
    common::mpi_atomic_put_value(1, target_rank, offsetof(mailbox, arrived), win_.win());
  }

  std::optional<Entry> pop() {
    mailbox& mb = win_.local_buf()[0];
    if (mb.arrived.load(std::memory_order_acquire)) {
      mb.arrived.store(0, std::memory_order_relaxed);
      return mb.entry;
    } else {
      return std::nullopt;
    }
  }

  bool arrived() const {
    return win_.local_buf()[0].arrived.load(std::memory_order_relaxed);
  }

private:
  struct mailbox {
    Entry            entry;
    std::atomic<int> arrived = 0; // TODO: better to use std::atomic_ref in C++20
  };

  common::mpi_win_manager<mailbox> win_;
};

template <>
class oneslot_mailbox<void> {
public:
  oneslot_mailbox()
    : win_(common::topology::mpicomm(), 1) {}

  void put(common::topology::rank_t target_rank) {
    ITYR_PROFILER_RECORD(prof_event_sched_mailbox_put, target_rank);

    ITYR_CHECK(!common::mpi_get_value<int>(target_rank, offsetof(mailbox, arrived), win_.win()));
    common::mpi_atomic_put_value(1, target_rank, offsetof(mailbox, arrived), win_.win());
  }

  bool pop() {
    mailbox& mb = win_.local_buf()[0];
    if (mb.arrived.load(std::memory_order_acquire)) {
      mb.arrived.store(0, std::memory_order_relaxed);
      return true;
    } else {
      return false;
    }
  }

  bool arrived() const {
    return win_.local_buf()[0].arrived.load(std::memory_order_relaxed);
  }

private:
  struct mailbox {
    std::atomic<int> arrived = 0; // TODO: better to use std::atomic_ref in C++20
  };

  common::mpi_win_manager<mailbox> win_;
};

}
