#pragma once

#include "ityr/common/util.hpp"
#include "ityr/ito/util.hpp"

namespace ityr::ito {

class scheduler_serial {
public:
  struct no_retval_t {};

  template <typename T>
  using thread_handler = T;

  scheduler_serial() {}

  template <typename T, typename Fn, typename... Args>
  T root_exec(Fn&& fn, Args&&... args) {
    return invoke_fn<T>(std::forward<Fn>(fn), std::forward<Args>(args)...);
  }

  template <typename T, typename OnDriftCallback, typename Fn, typename... Args>
  void fork(thread_handler<T>& th, OnDriftCallback&&, Fn&& fn, Args&&... args) {
    th = invoke_fn<T>(std::forward<Fn>(fn), std::forward<Args>(args)...);
  }

  template <typename T>
  T join(thread_handler<T>& th) {
    return th;
  }

  template <typename CondFn>
  void sched_loop(CondFn&&) {}

  template <typename T>
  static bool is_serialized(thread_handler<T>) {
    return true;
  }

  struct task_group_data {};
  task_group_data task_group_begin() { return {}; }
  void task_group_end(task_group_data&) {}

private:
  template <typename T, typename Fn, typename... Args>
  T invoke_fn(Fn&& fn, Args&&... args) {
    T retval;
    if constexpr (!std::is_same_v<T, no_retval_t>) {
      retval = std::forward<Fn>(fn)(std::forward<Args>(args)...);
    } else {
      fn(args...);
    }
    return retval;
  }
};

}
