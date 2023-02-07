#pragma once

#include <optional>

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/virtual_mem.hpp"
#include "ityr/common/allocator.hpp"
#include "ityr/ito/callstack.hpp"
#include "ityr/ito/scheduler.hpp"

namespace ityr::ito {

class worker {
public:
  worker(const common::topology& topo)
    : topo_(topo),
      stack_(topo, common::getenv_coll("ITYR_ITO_STACK_SIZE", std::size_t(2) * 1024 * 1024, topo_.mpicomm())),
      sched_(topo_, stack_) {}

  template <typename Fn, typename... Args>
  auto root_exec(Fn&& fn, Args&&... args) {
    using ret_t = std::invoke_result_t<Fn, Args...>;
    ret_t retval;
    if (topo_.my_rank() == 0) {
      retval = sched_.root_exec<ret_t>(std::forward<Fn>(fn), std::forward<Args>(args)...);
    } else {
      sched_.sched_loop([]{ return true; });
    }
    return common::mpi_bcast_value(retval, 0, topo_.mpicomm());
  }

  bool is_spmd() const { return is_spmd_; }

  scheduler& sched() { return sched_; }

private:
  const common::topology&            topo_;
  callstack                          stack_;
  bool                               is_spmd_ = true;
  scheduler                          sched_;
};

inline std::optional<worker>& worker_get_() {
  static std::optional<worker> instance;
  return instance;
}

inline worker& worker_get() {
  ITYR_CHECK(worker_get_().has_value());
  return *worker_get_();
}

template <typename... Args>
inline void worker_init(Args&&... args) {
  ITYR_CHECK(!worker_get_().has_value());
  worker_get_().emplace(std::forward<Args>(args)...);
}

inline void worker_fini() {
  ITYR_CHECK(worker_get_().has_value());
  worker_get_().reset();
}

}
