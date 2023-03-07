#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/wallclock.hpp"
#include "ityr/common/profiler.hpp"

namespace ityr::ito {

struct prof_phase_sched : public common::profiler::event {
  using common::profiler::event::event;
  std::string str() const override { return "phase_sched"; }
};

struct prof_phase_thread : public common::profiler::event {
  using common::profiler::event::event;
  std::string str() const override { return "phase_thread"; }
};

struct prof_phase_spmd : public common::profiler::event {
  using common::profiler::event::event;
  std::string str() const override { return "phase_spmd"; }
};

struct prof_event_sched_steal : public common::profiler::event {
  using common::profiler::event::event;

  auto interval_begin(common::wallclock::wallclock_t t,
                      common::topology::rank_t       target_rank [[maybe_unused]]) {
    return common::profiler::event::interval_begin(t);
  }

  void interval_end(common::wallclock::wallclock_t        t,
                    common::profiler::interval_begin_data ibd,
                    bool                                  success) {
    if (state_.enabled) {
      auto t0 = ibd;
      if (success) {
        acc_time_success_ += t - t0;
        count_success_++;
      } else {
        acc_time_fail_ += t - t0;
        count_fail_++;
      }
    }
  }

  std::string str() const override {
    return success_mode ? "sched_steal (success)" : "sched_steal (fail)";
  }

  void flush() override {
    success_mode = true;
    acc_time_ = acc_time_success_;
    count_ = count_success_;
    common::profiler::event::flush();
    success_mode = false;
    acc_time_ = acc_time_fail_;
    count_ = count_fail_;
    common::profiler::event::flush();
  }

  void clear() override {
    acc_time_success_ = 0;
    acc_time_fail_    = 0;
    count_success_    = 0;
    count_fail_       = 0;
  }

private:
  common::wallclock::wallclock_t acc_time_success_ = 0;
  common::wallclock::wallclock_t acc_time_fail_    = 0;
  counter_t                      count_success_    = 0;
  counter_t                      count_fail_       = 0;
  bool                           success_mode;
};

struct prof_event_wsqueue_push : public common::profiler::event {
  using common::profiler::event::event;
  std::string str() const override { return "wsqueue_push"; }
};

struct prof_event_wsqueue_pop : public common::profiler::event {
  using common::profiler::event::event;
  std::string str() const override { return "wsqueue_pop"; }
};

struct prof_event_wsqueue_steal : public common::profiler::event {
  using common::profiler::event::event;
  std::string str() const override { return "wsqueue_steal"; }
};

struct prof_event_wsqueue_empty : public common::profiler::event {
  using common::profiler::event::event;
  std::string str() const override { return "wsqueue_empty"; }
};

class prof_events {
public:
  prof_events() {}

private:
  common::profiler::event_initializer<prof_phase_sched>         sched_;
  common::profiler::event_initializer<prof_phase_thread>        thread_;
  common::profiler::event_initializer<prof_phase_spmd>          spmd_;
  common::profiler::event_initializer<prof_event_sched_steal>   sched_steal_;
  common::profiler::event_initializer<prof_event_wsqueue_push>  wsqueue_push_;
  common::profiler::event_initializer<prof_event_wsqueue_pop>   wsqueue_pop_;
  common::profiler::event_initializer<prof_event_wsqueue_steal> wsqueue_steal_;
  common::profiler::event_initializer<prof_event_wsqueue_empty> wsqueue_empty_;
};

}
