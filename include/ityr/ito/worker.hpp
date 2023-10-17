#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/ito/scheduler.hpp"

namespace ityr::ito::worker {

class worker {
public:
  worker()
    : sched_() {}

  template <typename SchedLoopCallback, typename Fn, typename... Args>
  auto root_exec(SchedLoopCallback cb, Fn&& fn, Args&&... args) {
    ITYR_CHECK(is_spmd_);
    is_spmd_ = false;

    using retval_t = std::invoke_result_t<Fn, Args...>;
    if constexpr (std::is_void_v<retval_t>) {
      if (common::topology::my_rank() == coll_master_) {
        sched_.root_exec<no_retval_t>(cb, std::forward<Fn>(fn), std::forward<Args>(args)...);
      } else {
        common::profiler::switch_phase<prof_phase_spmd, prof_phase_sched_loop>();
        sched_.sched_loop(cb);
        common::profiler::switch_phase<prof_phase_sched_loop, prof_phase_spmd>();
      }

      is_spmd_ = true;

      common::mpi_barrier(common::topology::mpicomm());

    } else {
      retval_t retval {};
      if (common::topology::my_rank() == coll_master_) {
        retval = sched_.root_exec<retval_t>(cb, std::forward<Fn>(fn), std::forward<Args>(args)...);
      } else {
        common::profiler::switch_phase<prof_phase_spmd, prof_phase_sched_loop>();
        sched_.sched_loop(cb);
        common::profiler::switch_phase<prof_phase_sched_loop, prof_phase_spmd>();
      }

      is_spmd_ = true;

      return common::mpi_bcast_value(retval, coll_master_, common::topology::mpicomm());
    }
  }

  template <typename Fn, typename... Args>
  auto coll_exec(const Fn& fn, const Args&... args) {
    ITYR_CHECK(!is_spmd_);

    using retval_t = std::invoke_result_t<Fn, Args...>;

    auto next_master = common::topology::my_rank();
    std::conditional_t<std::is_void_v<retval_t>, no_retval_t, retval_t> retv;

    auto coll_task_fn = [=, &retv]() {
      is_spmd_ = true;
      auto prev_coll_master = coll_master_;
      coll_master_ = next_master;
      if constexpr (std::is_void_v<retval_t>) {
        fn(args...);
        (void)retv;
      } else {
        auto&& ret = fn(args...);
        if (common::topology::my_rank() == next_master) {
          retv = std::forward<decltype(ret)>(ret);
        }
      }
      coll_master_ = prev_coll_master;
      is_spmd_ = false;
    };

    sched_.coll_exec(coll_task_fn);

    if constexpr (!std::is_void_v<retval_t>) {
      return retv;
    }
  }

  bool is_spmd() const { return is_spmd_; }

  scheduler& sched() { return sched_; }

private:
  scheduler                sched_;
  bool                     is_spmd_ = true;
  common::topology::rank_t coll_master_ = 0;
};

using instance = common::singleton<worker>;

}
