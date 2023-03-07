#pragma once

#include <limits>
#include <tuple>

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/wallclock.hpp"

namespace ityr::common::profiler {

using interval_begin_data = wallclock::wallclock_t;

struct profiler_state {
  bool                   enabled;
  wallclock::wallclock_t t_begin;
  wallclock::wallclock_t t_end;
  bool                   output_per_rank;
};

class event_stats {
public:
  event_stats(profiler_state& state)
    : state_(state) {}

  interval_begin_data interval_begin(wallclock::wallclock_t t) {
    return t;
  }

  void interval_end(wallclock::wallclock_t t, interval_begin_data ibd) {
    /* auto t0 = std::max(state_.t_begin, ibd); */
    auto t0 = ibd;
    acc_time_ += t - t0;
    count_++;
  }

  virtual std::string str() const { return "UNKNOWN"; }

  virtual void clear() {
    acc_time_ = 0;
    count_ = 0;
  }

  virtual void flush() {
    auto t_total = state_.t_end - state_.t_begin;
    if (state_.output_per_rank) {
      if (topology::my_rank() == 0) {
        print_stats_per_rank(0, acc_time_, t_total, count_);
        for (topology::rank_t i = 1; i < topology::n_ranks(); i++) {
          auto [a, t, c] = mpi_recv_value<
            std::tuple<wallclock::wallclock_t, wallclock::wallclock_t, counter_t>>(i, 0, topology::mpicomm());
          print_stats_per_rank(i, a, t, c);
        }
      } else {
        mpi_send_value(std::make_tuple(acc_time_, t_total, count_), 0, 0, topology::mpicomm());
      }
    } else {
      auto acc_time_sum = mpi_reduce_value(acc_time_, 0, topology::mpicomm());
      auto t_total_sum = mpi_reduce_value(t_total, 0, topology::mpicomm());
      auto count_sum = mpi_reduce_value(count_, 0, topology::mpicomm());
      if (topology::my_rank() == 0) {
        print_stats_sum(acc_time_sum, t_total_sum, count_sum);
      }
    }
  }

protected:
  using counter_t = uint64_t;

  virtual void print_stats_per_rank(topology::rank_t       rank,
                                    wallclock::wallclock_t acc_time,
                                    wallclock::wallclock_t t_total,
                                    counter_t              count) const {
    printf("  %-21s (rank %3d) : %10.6f %% ( %15ld ns / %15ld ns ) count: %10ld ave: %8ld ns\n",
           str().c_str(), rank,
           (double)acc_time / t_total * 100, acc_time, t_total,
           count, count == 0 ? 0 : (acc_time / count));
  }

  virtual void print_stats_sum(wallclock::wallclock_t acc_time,
                               wallclock::wallclock_t t_total,
                               counter_t              count) const {
    printf("  %-21s : %10.6f %% ( %15ld ns / %15ld ns ) count: %10ld ave: %8ld ns\n",
           str().c_str(),
           (double)acc_time / t_total * 100, acc_time, t_total,
           count, count == 0 ? 0 : (acc_time / count));
  }

  profiler_state&        state_;
  wallclock::wallclock_t acc_time_ = 0;
  counter_t              count_    = 0;
};

using event = event_stats;

class profiler {
public:
  profiler() {
    state_.output_per_rank = getenv_coll("ITYR_PROF_OUTPUT_PER_RANK", false, topology::mpicomm());
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
      last_phase_->interval_end(state_.t_end, phase_ibd_);
      last_phase_ = nullptr;
    }
  }

  void flush() {
    for (auto&& e : events_) {
      e->flush();
    }
    if (topology::my_rank() == 0) {
      printf("\n");
      fflush(stdout);
    }
    for (auto&& e : events_) {
      e->clear();
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
        phase_ibd_ = phase_from.interval_begin(state_.t_begin);
      } else {
        ITYR_CHECK(last_phase_ == &phase_from);
      }

      phase_from.interval_end(t, phase_ibd_);
      phase_ibd_ = phase_to.interval_begin(t);
      last_phase_ = &phase_to;
    }
  }

private:
  std::vector<event*> events_;
  profiler_state      state_;
  event*              last_phase_ = nullptr;
  interval_begin_data phase_ibd_;
};

using instance = singleton<profiler>;

template <typename Event>
class event_initializer : public singleton_initializer<singleton<Event>> {
public:
  template <typename... Args>
  event_initializer(Args&&... args)
    : singleton_initializer<singleton<Event>>(instance::get().get_state(),
                                              std::forward<Args>(args)...) {
    instance::get().add(&singleton<Event>::get());
  }
};

template <typename Event, typename... Args>
inline interval_begin_data interval_begin(Args&&... args) {
  auto t = wallclock::gettime_ns();
  return singleton<Event>::get().interval_begin(t, std::forward<Args>(args)...);
}

template <typename Event, typename... Args>
inline void interval_end(interval_begin_data ibd, Args&&... args) {
  auto& state = instance::get().get_state();
  if (state.enabled) {
    auto t = wallclock::gettime_ns();
    singleton<Event>::get().interval_end(t, ibd, std::forward<Args>(args)...);
  }
}

template <typename PhaseFrom, typename PhaseTo>
inline void switch_phase() {
  instance::get().switch_phase<PhaseFrom, PhaseTo>();
}

inline void begin() {
  mpi_barrier(topology::mpicomm());
  instance::get().begin();
}

inline void end() {
  instance::get().end();
}

inline void flush() {
  instance::get().flush();
}

}
