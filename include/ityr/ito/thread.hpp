#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/allocator.hpp"
#include "ityr/ito/worker.hpp"
#include "ityr/ito/scheduler.hpp"

namespace ityr::ito {

template <typename T>
class thread {
  // If the return value is void, set `no_retval_t` as the return type for the internal of the scheduler
  using sched_retval_t = std::conditional_t<std::is_void_v<T>, scheduler::no_retval_t, T>;

public:
  thread() {}
  template <typename Fn, typename... Args>
  thread(Fn&& fn, Args&&... args) {
    fork(std::forward<Fn>(fn), std::forward<Args>(args)...);
  }

  thread(const thread&) = delete;
  thread& operator=(const thread&) = delete;

  thread(thread&& th) : handler_(th.handler_) { th.handler_ = scheduler::thread_handler<sched_retval_t>{}; }
  thread& operator=(thread&& th) {
    this->~thread();
    handler_ = th.handler_;
    th.handler_ = scheduler::thread_handler<sched_retval_t>{};
    return *this;
  }

  template <typename Fn, typename... Args>
  void fork(Fn&& fn, Args&&... args) {
    worker& w = worker_get();
    handler_ = w.sched().fork<sched_retval_t>(std::forward<Fn>(fn), std::forward<Args>(args)...);
  }

  T join() {
    worker& w = worker_get();
    if constexpr (std::is_void_v<T>) {
      w.sched().join(handler_);
    } else {
      return w.sched().join(handler_);
    }
  }

private:
  scheduler::thread_handler<sched_retval_t> handler_;
};

}
