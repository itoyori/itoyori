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

  auto interval_begin(common::profiler::mode_stats,
                      common::wallclock::wallclock_t t,
                      common::topology::rank_t       target_rank [[maybe_unused]]) {
    return t;
  }

  void interval_end(common::profiler::mode_stats,
                    common::wallclock::wallclock_t                    t,
                    common::profiler::mode_stats::interval_begin_data ibd,
                    bool                                              success) {
    do_acc(t - ibd, success);
  }

  auto interval_begin(common::profiler::mode_trace,
                      common::wallclock::wallclock_t t,
                      common::topology::rank_t       target_rank) {
    auto ibd = MLOG_BEGIN(&state_.trace_md, 0, t, target_rank);
    return ibd;
  }

  void interval_end(common::profiler::mode_trace,
                    common::wallclock::wallclock_t                    t,
                    common::profiler::mode_trace::interval_begin_data ibd,
                    bool                                              success) {
    MLOG_END(&state_.trace_md, 0, ibd, trace_decoder_base, this, t, success);
  }

  void* trace_decoder(FILE* stream, void* buf0, void* buf1) override {
    auto t0          = MLOG_READ_ARG(&buf0, common::wallclock::wallclock_t);
    auto target_rank = MLOG_READ_ARG(&buf0, common::topology::rank_t);
    auto t1          = MLOG_READ_ARG(&buf1, common::wallclock::wallclock_t);
    auto success     = MLOG_READ_ARG(&buf1, bool);

    do_acc(t1 - t0, success);

    success_mode_ = success;
    auto rank = common::topology::my_rank();
    fprintf(stream, "%d,%lu,%d,%lu,%s,%d\n", rank, t0, rank, t1, str().c_str(), target_rank);
    return buf1;
  }

  std::string str() const override {
    return success_mode_ ? "sched_steal (success)" : "sched_steal (fail)";
  }

  void print_stats() override {
    success_mode_ = true;
    acc_time_ = acc_time_success_;
    count_ = count_success_;
    common::profiler::event::print_stats();
    success_mode_ = false;
    acc_time_ = acc_time_fail_;
    count_ = count_fail_;
    common::profiler::event::print_stats();
  }

  void clear() override {
    acc_time_success_ = 0;
    acc_time_fail_    = 0;
    count_success_    = 0;
    count_fail_       = 0;
  }

private:
  void do_acc(common::wallclock::wallclock_t t, bool success) {
    if (success) {
      acc_time_success_ += t;
      count_success_++;
    } else {
      acc_time_fail_ += t;
      count_fail_++;
    }
  }

  common::wallclock::wallclock_t acc_time_success_ = 0;
  common::wallclock::wallclock_t acc_time_fail_    = 0;
  counter_t                      count_success_    = 0;
  counter_t                      count_fail_       = 0;
  bool                           success_mode_;
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
  common::profiler::event_initializer<prof_phase_sched>         phase_sched_;
  common::profiler::event_initializer<prof_phase_thread>        phase_thread_;
  common::profiler::event_initializer<prof_phase_spmd>          phase_spmd_;
  common::profiler::event_initializer<prof_event_sched_steal>   sched_steal_;
  common::profiler::event_initializer<prof_event_wsqueue_push>  wsqueue_push_;
  common::profiler::event_initializer<prof_event_wsqueue_pop>   wsqueue_pop_;
  common::profiler::event_initializer<prof_event_wsqueue_steal> wsqueue_steal_;
  common::profiler::event_initializer<prof_event_wsqueue_empty> wsqueue_empty_;
};

}
