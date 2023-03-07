#pragma once

#include <limits>
#include <tuple>

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/wallclock.hpp"

namespace ityr::common::profiler {

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

  using interval_begin_data = wallclock::wallclock_t;

  interval_begin_data interval_begin() {
    return wallclock::gettime_ns();
  }

  void interval_end(interval_begin_data ibd) {
    if (state_.enabled) {
      auto t = wallclock::gettime_ns();
      /* auto t0 = std::max(state_.t_begin, ibd); */
      auto t0 = ibd;
      acc_time_ += t - t0;
      count_++;
    }
  }

  virtual std::string str() const { return "UNKNOWN"; }

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
    acc_time_ = 0;
    count_ = 0;
  };

protected:
  using counter_t = uint64_t;

  void print_stats_per_rank(topology::rank_t       rank,
                            wallclock::wallclock_t acc_time,
                            wallclock::wallclock_t t_total,
                            counter_t              count) const {
    printf("  %-23s (rank %3d) : %10.6f %% ( %15ld ns / %15ld ns ) count: %8ld ave: %8ld ns\n",
           str().c_str(), rank,
           (double)acc_time / t_total * 100, acc_time, t_total,
           count, count == 0 ? 0 : (acc_time / count));
  }

  void print_stats_sum(wallclock::wallclock_t acc_time,
                       wallclock::wallclock_t t_total,
                       counter_t              count) const {
    printf("  %-23s : %10.6f %% ( %15ld ns / %15ld ns ) count: %8ld ave: %8ld ns\n",
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
  }

  void flush() {
    for (auto&& e : events_) {
      e->flush();
    }
    if (topology::my_rank() == 0) {
      printf("\n");
      fflush(stdout);
    }
  }

  profiler_state& get_state() { return state_; }

private:
  std::vector<event*> events_;
  profiler_state      state_;
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

template <typename Event>
inline auto interval_begin() {
  return singleton<Event>::get().interval_begin();
}

template <typename Event, typename IBD>
inline void interval_end(IBD ibd) {
  singleton<Event>::get().interval_end(ibd);
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
