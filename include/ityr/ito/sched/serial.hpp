#pragma once

#include "ityr/common/util.hpp"
#include "ityr/ito/util.hpp"
#include "ityr/ito/sched/util.hpp"

namespace ityr::ito {

class scheduler_serial {
public:
  template <typename T>
  using thread_handler = T;

  scheduler_serial() {}

  template <typename T, typename SchedLoopCallback, typename Fn, typename... Args>
  T root_exec(SchedLoopCallback, Fn&& fn, Args&&... args) {
    return invoke_fn<T>(std::forward<Fn>(fn), std::make_tuple(std::forward<Args>(args)...));
  }

  template <typename T, typename OnDriftForkCallback, typename OnDriftDieCallback,
            typename WorkHint, typename Fn, typename... Args>
  void fork(thread_handler<T>& th,
            OnDriftForkCallback, OnDriftDieCallback,
            WorkHint, WorkHint, Fn&& fn, Args&&... args) {
    th = invoke_fn<T>(std::forward<Fn>(fn), std::make_tuple(std::forward<Args>(args)...));
  }

  template <typename T>
  T join(thread_handler<T>& th) {
    return std::move(th);
  }

  template <typename SchedLoopCallback, typename CondFn>
  void sched_loop(SchedLoopCallback&&, CondFn) {}

  template <typename PreSuspendCallback, typename PostSuspendCallback>
  void poll(PreSuspendCallback&&, PostSuspendCallback&&) {}

  template <typename Fn, typename... Args>
  auto coll_exec(Fn&& fn, Args&&... args) {
    return std::forward<Fn>(fn)(std::forward<Args>(args)...);
  }

  bool is_executing_root() const {
    return true;
  }

  template <typename T>
  static bool is_serialized(const thread_handler<T>&) {
    return true;
  }

  struct task_group_data {};
  void task_group_begin(task_group_data*) {}
  template <typename PreSuspendCallback, typename PostSuspendCallback>
  void task_group_end(PreSuspendCallback&&, PostSuspendCallback&&) {}

  void dag_prof_begin() {}
  void dag_prof_end() {}
  void dag_prof_print() const {}
};

}
