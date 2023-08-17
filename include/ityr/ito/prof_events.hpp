#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/wallclock.hpp"
#include "ityr/common/profiler.hpp"
#include "ityr/common/prof_events.hpp"

namespace ityr::ito {

struct prof_event_sched_steal : public common::prof_event_target_base {
  using prof_event_target_base::prof_event_target_base;

  void interval_end(common::profiler::mode_stats,
                    common::wallclock::wallclock_t                    t,
                    common::profiler::mode_stats::interval_begin_data ibd,
                    bool                                              success) {
    do_acc(t - ibd, success);
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
    fprintf(stream, "%d,%lu,%d,%lu,%s,target=%d\n", rank, t0, rank, t1, str().c_str(), target_rank);
    return buf1;
  }

  std::string str() const override {
    return success_mode_ ? "sched_steal (success)" : "sched_steal (fail)";
  }

  void print_stats() override {
    success_mode_ = true;
    sum_time_ = sum_time_success_;
    max_time_ = max_time_success_;
    count_ = count_success_;
    common::profiler::event::print_stats();
    success_mode_ = false;
    sum_time_ = sum_time_fail_;
    max_time_ = max_time_fail_;
    count_ = count_fail_;
    common::profiler::event::print_stats();
  }

  void clear() override {
    sum_time_success_ = 0;
    sum_time_fail_    = 0;
    max_time_success_ = 0;
    max_time_fail_    = 0;
    count_success_    = 0;
    count_fail_       = 0;
  }

private:
  void do_acc(common::wallclock::wallclock_t t, bool success) {
    if (success) {
      sum_time_success_ += t;
      max_time_success_ = std::max(max_time_success_, t);
      count_success_++;
    } else {
      sum_time_fail_ += t;
      max_time_fail_ = std::max(max_time_fail_, t);
      count_fail_++;
    }
  }

  common::wallclock::wallclock_t sum_time_success_ = 0;
  common::wallclock::wallclock_t sum_time_fail_    = 0;
  common::wallclock::wallclock_t max_time_success_ = 0;
  common::wallclock::wallclock_t max_time_fail_    = 0;
  counter_t                      count_success_    = 0;
  counter_t                      count_fail_       = 0;
  bool                           success_mode_;
};

struct prof_event_sched_mailbox_put : public common::prof_event_target_base {
  using prof_event_target_base::prof_event_target_base;
  std::string str() const override { return "sched_mailbox_put"; }
};

struct prof_event_sched_adws_scan_tree : public common::profiler::event {
  using event::event;
  std::string str() const override { return "sched_adws_scan_tree"; }
};

struct prof_event_wsqueue_push : public common::profiler::event {
  using event::event;
  std::string str() const override { return "wsqueue_push"; }
};

struct prof_event_wsqueue_pop : public common::profiler::event {
  using event::event;
  std::string str() const override { return "wsqueue_pop"; }
};

struct prof_event_wsqueue_steal_nolock : public common::prof_event_target_base {
  using prof_event_target_base::prof_event_target_base;
  std::string str() const override { return "wsqueue_steal_nolock"; }
};

struct prof_event_wsqueue_steal_abort : public common::prof_event_target_base {
  using prof_event_target_base::prof_event_target_base;
  std::string str() const override { return "wsqueue_steal_abort"; }
};

struct prof_event_wsqueue_pass : public common::prof_event_target_base {
  using prof_event_target_base::prof_event_target_base;
  std::string str() const override { return "wsqueue_pass"; }
};

struct prof_event_wsqueue_empty : public common::prof_event_target_base {
  using prof_event_target_base::prof_event_target_base;
  std::string str() const override { return "wsqueue_empty"; }
};

struct prof_event_wsqueue_empty_batch : public common::prof_event_target_base {
  using prof_event_target_base::prof_event_target_base;
  std::string str() const override { return "wsqueue_empty_batch"; }
};

struct prof_phase_sched_loop : public common::profiler::event {
  using event::event;
  std::string str() const override { return "P_sched_loop"; }
};

struct prof_phase_sched_fork : public common::profiler::event {
  using event::event;
  std::string str() const override { return "P_sched_fork"; }
};

struct prof_phase_sched_join : public common::profiler::event {
  using event::event;
  std::string str() const override { return "P_sched_join"; }
};

struct prof_phase_sched_die : public common::profiler::event {
  using event::event;
  std::string str() const override { return "P_sched_die"; }
};

struct prof_phase_sched_migrate : public common::profiler::event {
  using event::event;
  std::string str() const override { return "P_sched_migrate"; }
};

struct prof_phase_sched_evacuate : public common::profiler::event {
  using event::event;
  std::string str() const override { return "P_sched_evacuate"; }
};

struct prof_phase_sched_resume_popped : public common::profiler::event {
  using event::event;
  std::string str() const override { return "P_sched_resume_popped"; }
};

struct prof_phase_sched_resume_join : public common::profiler::event {
  using event::event;
  std::string str() const override { return "P_sched_resume_join"; }
};

struct prof_phase_sched_resume_stolen : public common::profiler::event {
  using event::event;
  std::string str() const override { return "P_sched_resume_stolen"; }
};

struct prof_phase_sched_resume_migrate : public common::profiler::event {
  using event::event;
  std::string str() const override { return "P_sched_resume_migrate"; }
};

struct prof_phase_sched_start_new : public common::profiler::event {
  using event::event;
  std::string str() const override { return "P_sched_start_new"; }
};

struct prof_phase_cb_drift_fork : public common::profiler::event {
  using event::event;
  std::string str() const override { return "P_cb_drift_fork"; }
};

struct prof_phase_cb_drift_die : public common::profiler::event {
  using event::event;
  std::string str() const override { return "P_cb_drift_die"; }
};

struct prof_phase_cb_pre_suspend : public common::profiler::event {
  using event::event;
  std::string str() const override { return "P_cb_pre_suspend"; }
};

struct prof_phase_cb_post_suspend : public common::profiler::event {
  using event::event;
  std::string str() const override { return "P_cb_post_suspend"; }
};

struct prof_phase_thread : public common::profiler::event {
  using event::event;
  std::string str() const override { return "P_thread"; }
};

struct prof_phase_spmd : public common::profiler::event {
  using event::event;
  std::string str() const override { return "P_spmd"; }
};

class prof_events {
public:
  prof_events() {}

private:
  common::profiler::event_initializer<prof_event_sched_steal>          ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_event_sched_mailbox_put>    ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_event_sched_adws_scan_tree> ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_event_wsqueue_push>         ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_event_wsqueue_pop>          ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_event_wsqueue_steal_nolock> ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_event_wsqueue_steal_abort>  ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_event_wsqueue_pass>         ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_event_wsqueue_empty>        ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_event_wsqueue_empty_batch>  ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_phase_sched_loop>           ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_phase_sched_fork>           ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_phase_sched_join>           ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_phase_sched_die>            ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_phase_sched_migrate>        ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_phase_sched_evacuate>       ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_phase_sched_resume_popped>  ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_phase_sched_resume_join>    ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_phase_sched_resume_stolen>  ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_phase_sched_resume_migrate> ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_phase_sched_start_new>      ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_phase_cb_drift_fork>        ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_phase_cb_drift_die>         ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_phase_cb_pre_suspend>       ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_phase_cb_post_suspend>      ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_phase_thread>               ITYR_ANON_VAR;
  common::profiler::event_initializer<prof_phase_spmd>                 ITYR_ANON_VAR;
};

}
