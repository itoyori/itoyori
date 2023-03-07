#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/wallclock.hpp"
#include "ityr/common/profiler.hpp"

namespace ityr::common {

struct prof_event_target_base : public profiler::event {
  using event::event;

  auto interval_begin(profiler::mode_stats,
                      wallclock::wallclock_t t,
                      topology::rank_t       target_rank [[maybe_unused]]) {
    return t;
  }

  auto interval_begin(profiler::mode_trace,
                      wallclock::wallclock_t t,
                      topology::rank_t       target_rank) {
    auto ibd = MLOG_BEGIN(&state_.trace_md, 0, t, target_rank);
    return ibd;
  }

  void* trace_decoder(FILE* stream, void* buf0, void* buf1) override {
    auto t0          = MLOG_READ_ARG(&buf0, wallclock::wallclock_t);
    auto target_rank = MLOG_READ_ARG(&buf0, topology::rank_t);
    auto t1          = MLOG_READ_ARG(&buf1, wallclock::wallclock_t);

    do_acc(t1 - t0);

    auto rank = topology::my_rank();
    fprintf(stream, "%d,%lu,%d,%lu,%s,target=%d\n", rank, t0, rank, t1, str().c_str(), target_rank);
    return buf1;
  }
};

struct prof_event_rma_get : public prof_event_target_base {
  using prof_event_target_base::prof_event_target_base;
  std::string str() const override { return "rma_get"; }
};

struct prof_event_rma_put : public prof_event_target_base {
  using prof_event_target_base::prof_event_target_base;
  std::string str() const override { return "rma_put"; }
};

struct prof_event_rma_atomic_faa : public prof_event_target_base {
  using prof_event_target_base::prof_event_target_base;
  std::string str() const override { return "rma_atomic_faa"; }
};

struct prof_event_rma_atomic_cas : public prof_event_target_base {
  using prof_event_target_base::prof_event_target_base;
  std::string str() const override { return "rma_atomic_cas"; }
};

struct prof_event_rma_atomic_get : public prof_event_target_base {
  using prof_event_target_base::prof_event_target_base;
  std::string str() const override { return "rma_atomic_get"; }
};

struct prof_event_rma_atomic_put : public prof_event_target_base {
  using prof_event_target_base::prof_event_target_base;
  std::string str() const override { return "rma_atomic_put"; }
};

struct prof_event_rma_flush : public common::profiler::event {
  using event::event;
  std::string str() const override { return "rma_flush"; }
};

struct prof_event_global_lock_trylock : public common::prof_event_target_base {
  using prof_event_target_base::prof_event_target_base;
  std::string str() const override { return "global_lock_trylock"; }
};

struct prof_event_global_lock_unlock : public common::prof_event_target_base {
  using prof_event_target_base::prof_event_target_base;
  std::string str() const override { return "global_lock_unlock"; }
};

struct prof_event_allocator_alloc : public common::profiler::event {
  using event::event;
  std::string str() const override { return "allocator_alloc"; }
};

struct prof_event_allocator_free_local : public common::profiler::event {
  using event::event;
  std::string str() const override { return "allocator_free_local"; }
};

struct prof_event_allocator_free_remote : public common::prof_event_target_base {
  using prof_event_target_base::prof_event_target_base;
  std::string str() const override { return "allocator_free_remote"; }
};

struct prof_event_allocator_collect : public common::profiler::event {
  using event::event;
  std::string str() const override { return "allocator_collect"; }
};

class prof_events {
public:
  prof_events() {}

private:
  profiler::event_initializer<prof_event_rma_get>               rma_get_;
  profiler::event_initializer<prof_event_rma_put>               rma_put_;
  profiler::event_initializer<prof_event_rma_atomic_faa>        rma_atomic_faa_;
  profiler::event_initializer<prof_event_rma_atomic_cas>        rma_atomic_cas_;
  profiler::event_initializer<prof_event_rma_atomic_get>        rma_atomic_get_;
  profiler::event_initializer<prof_event_rma_atomic_put>        rma_atomic_put_;
  profiler::event_initializer<prof_event_rma_flush>             rma_flush_;
  profiler::event_initializer<prof_event_global_lock_trylock>   global_lock_trylock_;
  profiler::event_initializer<prof_event_global_lock_unlock>    global_lock_unlock_;
  profiler::event_initializer<prof_event_allocator_alloc>       allocator_alloc_;
  profiler::event_initializer<prof_event_allocator_free_local>  allocator_dealloc_local_;
  profiler::event_initializer<prof_event_allocator_free_remote> allocator_dealloc_remote_;
  profiler::event_initializer<prof_event_allocator_collect>     allocator_collect_;
};

}
