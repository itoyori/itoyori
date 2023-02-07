#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/allocator.hpp"
#include "ityr/ito/worker.hpp"
#include "ityr/ito/scheduler.hpp"

namespace ityr::ito {

template <typename T>
class thread {
public:
  thread() {}
  template <typename Fn, typename... Args>
  thread(Fn&& fn, Args&&... args) {
    fork(std::forward<Fn>(fn), std::forward<Args>(args)...);
  }

  thread(const thread&) = delete;
  thread& operator=(const thread&) = delete;

  thread(thread&& th) : handler_(th.handler_) { th.handler_ = scheduler::thread_handler<T>{}; }
  thread& operator=(thread&& th) {
    this->~thread();
    handler_ = th.handler_;
    th.handler_ = scheduler::thread_handler<T>{};
    return *this;
  }

  template <typename Fn, typename... Args>
  void fork(Fn&& fn, Args&&... args) {
    worker& w = worker_get();
    handler_ = w.sched().fork<T>(std::forward<Fn>(fn), std::forward<Args>(args)...);
  }

  T join() {
    worker& w = worker_get();
    return w.sched().join(handler_);
  }

private:
  scheduler::thread_handler<T> handler_;
};

}
