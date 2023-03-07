#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/ito/callstack.hpp"
#include "ityr/ito/scheduler.hpp"

namespace ityr::ito::worker {

class worker {
public:
  worker()
    : stack_(common::getenv_coll("ITYR_ITO_STACK_SIZE", std::size_t(2) * 1024 * 1024, common::topology::mpicomm())),
      sched_(stack_) {}

  template <typename Fn, typename... Args>
  auto root_exec(Fn&& fn, Args&&... args) {
    ITYR_CHECK(is_spmd_);
    is_spmd_ = false;

    common::profiler::switch_phase<prof_phase_spmd, prof_phase_sched>();

    using retval_t = std::invoke_result_t<Fn, Args...>;
    if constexpr (std::is_void_v<retval_t>) {
      if (common::topology::my_rank() == 0) {
        sched_.root_exec<scheduler::no_retval_t>(std::forward<Fn>(fn), std::forward<Args>(args)...);
      } else {
        sched_.sched_loop([]{ return true; });
      }
      common::mpi_barrier(common::topology::mpicomm());
    } else {
      retval_t retval {};
      if (common::topology::my_rank() == 0) {
        retval = sched_.root_exec<retval_t>(std::forward<Fn>(fn), std::forward<Args>(args)...);
      } else {
        sched_.sched_loop([]{ return true; });
      }
      return common::mpi_bcast_value(retval, 0, common::topology::mpicomm());
    }

    common::profiler::switch_phase<prof_phase_sched, prof_phase_spmd>();

    is_spmd_ = true;
  }

  bool is_spmd() const { return is_spmd_; }

  scheduler& sched() { return sched_; }

private:
  callstack stack_;
  scheduler sched_;
  bool      is_spmd_ = true;
};

using instance = common::singleton<worker>;

}
