#pragma once

#include <random>

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/wallclock.hpp"
#include "ityr/ito/util.hpp"
#include "ityr/ito/options.hpp"

namespace ityr::ito {

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

class task_general {
public:
  virtual ~task_general() = default;
  virtual void execute() = 0;
};

template <typename Fn, typename... Args>
class callable_task : task_general {
public:
  callable_task(Fn fn, Args... args) : fn_(fn), arg_(args...) {}
  void execute() { std::apply(fn_, arg_); }
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

template <typename T, typename Fn, typename... Args>
static T invoke_fn(Fn&& fn, Args&&... args) {
  T retval;
  if constexpr (!std::is_same_v<T, no_retval_t>) {
    retval = std::forward<Fn>(fn)(std::forward<Args>(args)...);
  } else {
    std::forward<Fn>(fn)(std::forward<Args>(args)...);
  }
  return retval;
}

}
