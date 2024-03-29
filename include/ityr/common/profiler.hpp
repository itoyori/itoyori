#pragma once

#include <limits>
#include <tuple>
#include <memory>

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/options.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/wallclock.hpp"
#include "ityr/common/options.hpp"

#if __has_include(<mlog/mlog.h>)
#include <mlog/mlog.h>
#else
#define MLOG_BEGIN(...) ((void*)0)
#define MLOG_END(...)
#define MLOG_READ_ARG(buf, type) ((type){})
namespace ityr::common::profiler {
using mlog_data_t = void*;
inline void mlog_die() {
  die("Trace profiler cannot be used because MassiveLogger is not loaded.");
}
inline void mlog_init(void**, int, size_t) { mlog_die(); }
inline void mlog_flush_all(void**, FILE*) { mlog_die(); }
inline void mlog_clear_all(void**) { mlog_die(); }
}
#endif

namespace ityr::common::profiler {

struct mode_disabled {
  using interval_begin_data = void*;
};

struct mode_stats {
  using interval_begin_data = wallclock::wallclock_t;
};

struct mode_trace {
  using interval_begin_data = void*;
};

struct profiler_state {
  bool                   enabled;
  wallclock::wallclock_t t_begin;
  wallclock::wallclock_t t_end;
  bool                   output_per_rank;
  mlog_data_t            trace_md;
};

class event {
public:
  event(profiler_state& state)
    : state_(state) {}

  virtual ~event() = default;

  mode_stats::interval_begin_data interval_begin(mode_disabled, wallclock::wallclock_t) { return {}; }
  void interval_end(mode_disabled, wallclock::wallclock_t, mode_disabled::interval_begin_data) {}

  mode_stats::interval_begin_data interval_begin(mode_stats, wallclock::wallclock_t t) {
    return t;
  }

  void interval_end(mode_stats, wallclock::wallclock_t t, mode_stats::interval_begin_data ibd) {
    do_acc(t - ibd);
  }

  mode_trace::interval_begin_data interval_begin(mode_trace, wallclock::wallclock_t t [[maybe_unused]]) {
    auto ibd = MLOG_BEGIN(&state_.trace_md, 0, t);
    return ibd;
  }

  void interval_end(mode_trace, wallclock::wallclock_t t [[maybe_unused]], mode_trace::interval_begin_data ibd [[maybe_unused]]) {
    MLOG_END(&state_.trace_md, 0, ibd, trace_decoder_base, this, t);
  }

  static void* trace_decoder_base(FILE* stream, int, int, void* buf0, void* buf1) {
    event* e = MLOG_READ_ARG(&buf1, event*);
    return e->trace_decoder(stream, buf0, buf1);
  }

  virtual void* trace_decoder(FILE* stream, void* buf0 [[maybe_unused]], void* buf1) {
    auto t0 = MLOG_READ_ARG(&buf0, wallclock::wallclock_t);
    auto t1 = MLOG_READ_ARG(&buf1, wallclock::wallclock_t);

    do_acc(t1 - t0);

    auto rank = topology::my_rank();
    fprintf(stream, "%d,%lu,%d,%lu,%s\n", rank, t0, rank, t1, str().c_str());
    return buf1;
  }

  virtual std::string str() const { return "UNKNOWN"; }

  virtual void clear() {
    sum_time_ = 0;
    max_time_ = 0;
    count_ = 0;
  }

  virtual void print_stats() {
    auto t_total = state_.t_end - state_.t_begin;
    if (state_.output_per_rank) {
      using msg_t = std::tuple<wallclock::wallclock_t, wallclock::wallclock_t, wallclock::wallclock_t, counter_t>;

      if (topology::my_rank() == 0) {
        print_stats_per_rank(0, sum_time_, max_time_, t_total, count_);
        for (topology::rank_t i = 1; i < topology::n_ranks(); i++) {
          mpi_barrier(topology::mpicomm());

          auto [s, m, t, c] = mpi_recv_value<msg_t>(i, 0, topology::mpicomm());
          print_stats_per_rank(i, s, m, t, c);
        }
      } else {
        for (topology::rank_t i = 1; i < topology::n_ranks(); i++) {
          // Insert a barrier to each iteration so that too many send requests do not go to rank 0 at the same time
          mpi_barrier(topology::mpicomm());

          if (i == topology::my_rank()) {
            mpi_send_value(std::make_tuple(sum_time_, max_time_, t_total, count_), 0, 0, topology::mpicomm());
          }
        }
      }
    } else {
      auto sum_time_all = mpi_reduce_value(sum_time_, 0, topology::mpicomm());
      auto max_time_all = mpi_reduce_value(max_time_, 0, topology::mpicomm(), MPI_MAX);
      auto t_total_all  = mpi_reduce_value(t_total, 0, topology::mpicomm());
      auto count_all    = mpi_reduce_value(count_, 0, topology::mpicomm());
      if (topology::my_rank() == 0) {
        print_stats_sum(sum_time_all, max_time_all, t_total_all, count_all);
      }
    }
  }

protected:
  void do_acc(wallclock::wallclock_t t) {
    sum_time_ += t;
    max_time_ = std::max(max_time_, t);
    count_++;
  }

  using counter_t = uint64_t;

  virtual void print_stats_per_rank(topology::rank_t       rank,
                                    wallclock::wallclock_t sum_time,
                                    wallclock::wallclock_t max_time,
                                    wallclock::wallclock_t t_total,
                                    counter_t              count) const {
    printf("  %-22s (rank %3d) : %10.6f %% ( %15ld ns / %15ld ns ) count: %10ld ave: %8ld ns max: %8ld ns\n",
           str().c_str(), rank,
           (double)sum_time / t_total * 100, sum_time, t_total,
           count, count == 0 ? 0 : (sum_time / count), max_time);
  }

  virtual void print_stats_sum(wallclock::wallclock_t sum_time,
                               wallclock::wallclock_t max_time,
                               wallclock::wallclock_t t_total,
                               counter_t              count) const {
    printf("  %-22s : %10.6f %% ( %15ld ns / %15ld ns ) count: %10ld ave: %8ld ns max: %8ld ns\n",
           str().c_str(),
           (double)sum_time / t_total * 100, sum_time, t_total,
           count, count == 0 ? 0 : (sum_time / count), max_time);
  }

  profiler_state&        state_;
  wallclock::wallclock_t sum_time_ = 0;
  wallclock::wallclock_t max_time_ = 0;
  counter_t              count_    = 0;
};

template <typename Mode>
class profiler {
public:
  profiler() {
    state_.output_per_rank = prof_output_per_rank_option::value();
    if constexpr (std::is_same_v<Mode, mode_trace>) {
      mlog_init(&state_.trace_md, 1, 1 << 20);
      trace_out_file_ = {std::fopen(trace_out_filename().c_str(), "w"), &std::fclose};
    }
  }

  void add(event* e) {
    events_.push_back(e);
  }

  void begin() {
    state_.enabled = true;
    state_.t_begin = wallclock::gettime_ns();
  }

  void end() {
    state_.enabled = false;
    state_.t_end = wallclock::gettime_ns();
    if (last_phase_ != nullptr) {
      last_phase_->interval_end(Mode{}, state_.t_end, phase_ibd_);
      last_phase_ = nullptr;
    }
  }

  void flush() {
    if constexpr (std::is_same_v<Mode, mode_trace>) {
      mlog_flush_all(&state_.trace_md, trace_out_file_.get());
    }
    for (auto&& e : events_) {
      e->print_stats();
    }
    if (topology::my_rank() == 0) {
      printf("\n");
      fflush(stdout);
    }
    for (auto&& e : events_) {
      e->clear();
    }
    if constexpr (std::is_same_v<Mode, mode_trace>) {
      mlog_clear_all(&state_.trace_md);
    }
  }

  profiler_state& get_state() { return state_; }

  template <typename PhaseFrom, typename PhaseTo>
  void switch_phase() {
    if (state_.enabled) {
      auto t = wallclock::gettime_ns();
      auto& phase_from = singleton<PhaseFrom>::get();
      auto& phase_to   = singleton<PhaseTo>::get();

      if (last_phase_ == nullptr) {
        phase_ibd_ = phase_from.interval_begin(Mode{}, state_.t_begin);
      } else {
        ITYR_CHECK(last_phase_ == &phase_from);
      }

      phase_from.interval_end(Mode{}, t, phase_ibd_);
      phase_ibd_ = phase_to.interval_begin(Mode{}, t);
      last_phase_ = &phase_to;
    }
  }

private:
  static std::string trace_out_filename() {
    std::stringstream ss;
    ss << "ityr_log_" << topology::my_rank() << ".ignore";
    return ss.str();
  }

  std::vector<event*>                   events_;
  profiler_state                        state_;
  event*                                last_phase_ = nullptr;
  typename Mode::interval_begin_data    phase_ibd_;
  std::unique_ptr<FILE, int (*)(FILE*)> trace_out_file_ = {nullptr, nullptr};
};

using mode = ITYR_CONCAT(mode_, ITYR_PROFILER_MODE);

using instance = singleton<profiler<mode>>;

using interval_begin_data = mode::interval_begin_data;

template <typename Event>
class event_initializer {
public:
  template <typename... Args>
  event_initializer(Args&&... args)
    : init_(instance::get().get_state(), std::forward<Args>(args)...) {
    if (init_.should_finalize()) {
      instance::get().add(&singleton<Event>::get());
    }
  }
private:
  singleton_initializer<singleton<Event>> init_;
};

template <typename Event, typename... Args>
inline interval_begin_data interval_begin(Args&&... args) {
  if constexpr (!std::is_same_v<mode, mode_disabled>) {
    auto t = wallclock::gettime_ns();
    return singleton<Event>::get().interval_begin(mode{}, t, std::forward<Args>(args)...);
  } else {
    return {};
  }
}

template <typename Event, typename... Args>
inline void interval_end(interval_begin_data ibd, Args&&... args) {
  if constexpr (!std::is_same_v<mode, mode_disabled>) {
    auto& state = instance::get().get_state();
    if (state.enabled) {
      auto t = wallclock::gettime_ns();
      singleton<Event>::get().interval_end(mode{}, t, ibd, std::forward<Args>(args)...);
    }
  }
}

template <typename Event>
class interval_scope {
public:
  template <typename... Args>
  interval_scope(Args&&... args) {
    ibd_ = interval_begin<Event>(std::forward<Args>(args)...);
  }
  ~interval_scope() {
    interval_end<Event>(ibd_);
  }

  interval_scope(const interval_scope&) = delete;
  interval_scope& operator=(const interval_scope&) = delete;

  interval_scope(interval_scope&&) = delete;
  interval_scope& operator=(interval_scope&&) = delete;

private:
  interval_begin_data ibd_;
};

#define ITYR_PROFILER_RECORD(event, ...) \
  ityr::common::profiler::interval_scope<event> ITYR_ANON_VAR {__VA_ARGS__};

template <typename PhaseFrom, typename PhaseTo>
inline void switch_phase() {
  if constexpr (!std::is_same_v<mode, mode_disabled>) {
    instance::get().switch_phase<PhaseFrom, PhaseTo>();
  }
}

inline void begin() {
  mpi_barrier(topology::mpicomm());
  instance::get().begin();
}

inline void end() {
  instance::get().end();
}

inline void flush() {
  if constexpr (!std::is_same_v<mode, mode_disabled>) {
    instance::get().flush();
  }
}

}
