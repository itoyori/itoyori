#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/ito/worker.hpp"
#include "ityr/ito/scheduler.hpp"

namespace ityr::ito {

struct with_callback_t {};
inline constexpr with_callback_t with_callback;

template <typename W>
struct workhint {
  constexpr explicit workhint(W w1, W w2)
    : work_new(w1), work_rest(w2) {}

  W work_new;
  W work_rest;
};

template <typename T>
class thread {
  // If the return value is void, set `no_retval_t` as the return type for the internal of the scheduler
  using sched_retval_t = std::conditional_t<std::is_void_v<T>, no_retval_t, T>;

public:
  thread() {}

  template <typename Fn, typename... Args>
  explicit thread(Fn&& fn, Args&&... args) {
    fork(std::forward<Fn>(fn), std::forward<Args>(args)...);
  }

  template <typename OnDriftForkCallback, typename OnDriftDieCallback, typename Fn, typename... Args>
  thread(with_callback_t, OnDriftForkCallback on_drift_fork_cb, OnDriftDieCallback on_drift_die_cb,
         Fn&& fn, Args&&... args) {
    fork(with_callback, on_drift_fork_cb, on_drift_die_cb,
         std::forward<Fn>(fn), std::forward<Args>(args)...);
  }

  template <typename W, typename Fn, typename... Args>
  thread(workhint<W> wh, Fn&& fn, Args&&... args) {
    fork(wh, std::forward<Fn>(fn), std::forward<Args>(args)...);
  }

  template <typename OnDriftForkCallback, typename OnDriftDieCallback,
            typename W, typename Fn, typename... Args>
  thread(with_callback_t, OnDriftForkCallback on_drift_fork_cb, OnDriftDieCallback on_drift_die_cb,
         workhint<W> wh, Fn&& fn, Args&&... args) {
    fork(with_callback, on_drift_fork_cb, on_drift_die_cb,
         wh, std::forward<Fn>(fn), std::forward<Args>(args)...);
  }

  thread(const thread&) = delete;
  thread& operator=(const thread&) = delete;

  thread(thread&& th) = default;
  thread& operator=(thread&& th) = default;

  template <typename Fn, typename... Args>
  void fork(Fn&& fn, Args&&... args) {
    auto& w = worker::instance::get();
    ITYR_CHECK(!w.is_spmd());
    w.sched().fork(handler_, nullptr, nullptr,
                   1, 1, std::forward<Fn>(fn), std::forward<Args>(args)...);
  }

  template <typename OnDriftForkCallback, typename OnDriftDieCallback, typename Fn, typename... Args>
  void fork(with_callback_t, OnDriftForkCallback on_drift_fork_cb, OnDriftDieCallback on_drift_die_cb,
            Fn&& fn, Args&&... args) {
    auto& w = worker::instance::get();
    ITYR_CHECK(!w.is_spmd());
    w.sched().fork(handler_,
                   on_drift_fork_cb, on_drift_die_cb,
                   1, 1, std::forward<Fn>(fn), std::forward<Args>(args)...);
  }

  template <typename W, typename Fn, typename... Args>
  void fork(workhint<W> wh, Fn&& fn, Args&&... args) {
    auto& w = worker::instance::get();
    ITYR_CHECK(!w.is_spmd());
    w.sched().fork(handler_, nullptr, nullptr,
                   wh.work_new, wh.work_rest, std::forward<Fn>(fn), std::forward<Args>(args)...);
  }

  template <typename OnDriftForkCallback, typename OnDriftDieCallback,
            typename W, typename Fn, typename... Args>
  void fork(with_callback_t, OnDriftForkCallback on_drift_fork_cb, OnDriftDieCallback on_drift_die_cb,
            workhint<W> wh, Fn&& fn, Args&&... args) {
    auto& w = worker::instance::get();
    ITYR_CHECK(!w.is_spmd());
    w.sched().fork(handler_,
                   on_drift_fork_cb, on_drift_die_cb,
                   wh.work_new, wh.work_rest, std::forward<Fn>(fn), std::forward<Args>(args)...);
  }

  T join() {
    auto& w = worker::instance::get();
    ITYR_CHECK(!w.is_spmd());
    if constexpr (std::is_void_v<T>) {
      w.sched().join(handler_);
    } else {
      return w.sched().join(handler_);
    }
  }

  bool serialized() {
    return scheduler::is_serialized(handler_);
  }

private:
  scheduler::thread_handler<sched_retval_t> handler_;
};

}
