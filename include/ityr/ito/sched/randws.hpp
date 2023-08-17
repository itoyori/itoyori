#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/mpi_rma.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/logger.hpp"
#include "ityr/common/allocator.hpp"
#include "ityr/common/profiler.hpp"
#include "ityr/ito/util.hpp"
#include "ityr/ito/options.hpp"
#include "ityr/ito/context.hpp"
#include "ityr/ito/callstack.hpp"
#include "ityr/ito/wsqueue.hpp"
#include "ityr/ito/prof_events.hpp"
#include "ityr/ito/sched/util.hpp"

namespace ityr::ito {

class scheduler_randws {
public:
  struct suspended_state {
    void*       evacuation_ptr;
    void*       frame_base;
    std::size_t frame_size;
  };

  template <typename T>
  struct thread_retval {
    T            value;
    dag_profiler dag_prof;
  };

  template <typename T>
  struct thread_state {
    thread_retval<T> retval;
    int              resume_flag = 0;
    suspended_state  suspended;
  };

  template <typename T>
  struct thread_handler {
    thread_state<T>* state      = nullptr;
    bool             serialized = false;
    thread_retval<T> retval_ser; // return the result by value if the thread is serialized
  };

  struct thread_local_storage {
    dag_profiler dag_prof;
  };

  struct task_group_data {
    dag_profiler dag_prof;
  };

  scheduler_randws()
    : stack_(stack_size_option::value()),
      wsq_(wsqueue_capacity_option::value()),
      thread_state_allocator_(thread_state_allocator_size_option::value()),
      suspended_thread_allocator_(suspended_thread_allocator_size_option::value()) {}

  template <typename T, typename SchedLoopCallback, typename Fn, typename... Args>
  T root_exec(SchedLoopCallback&& cb, Fn&& fn, Args&&... args) {
    common::profiler::switch_phase<prof_phase_spmd, prof_phase_sched_fork>();

    thread_state<T>* ts = new (thread_state_allocator_.allocate(sizeof(thread_state<T>))) thread_state<T>;

    suspend([&, ts](context_frame* cf) {
      sched_cf_ = cf;
      root_on_stack([&, ts, fn, args...]() {
        common::verbose("Starting root thread %p", ts);

        tls_ = new (alloca(sizeof(thread_local_storage))) thread_local_storage{};

        tls_->dag_prof.start();
        tls_->dag_prof.increment_thread_count();
        tls_->dag_prof.increment_strand_count();

        common::profiler::switch_phase<prof_phase_sched_fork, prof_phase_thread>();

        T ret = invoke_fn<T>(fn, args...);

        common::profiler::switch_phase<prof_phase_thread, prof_phase_sched_die>();
        common::verbose("Root thread %p is completed", ts);

        tls_->dag_prof.stop();

        on_root_die(ts, ret);
      });
    });

    sched_loop(std::forward<SchedLoopCallback>(cb),
               [=]() { return ts->resume_flag >= 1; });

    common::profiler::switch_phase<prof_phase_sched_loop, prof_phase_sched_join>();

    thread_retval<T> retval = ts->retval;
    std::destroy_at(ts);
    thread_state_allocator_.deallocate(ts, sizeof(thread_state<T>));

    if (dag_prof_enabled_) {
      dag_prof_result_ = retval.dag_prof;
    }

    common::profiler::switch_phase<prof_phase_sched_join, prof_phase_spmd>();

    return retval.value;
  }

  template <typename T, typename OnDriftForkCallback, typename OnDriftDieCallback,
            typename WorkHint, typename Fn, typename... Args>
  void fork(thread_handler<T>& th,
            OnDriftForkCallback&& on_drift_fork_cb, OnDriftDieCallback&& on_drift_die_cb,
            WorkHint, WorkHint, Fn&& fn, Args&&... args) {
    common::profiler::switch_phase<prof_phase_thread, prof_phase_sched_fork>();

    thread_state<T>* ts = new (thread_state_allocator_.allocate(sizeof(thread_state<T>))) thread_state<T>;
    th.state = ts;
    th.serialized = false;

    suspend([&, ts, fn, args...](context_frame* cf) mutable {
      common::verbose<2>("push context frame [%p, %p) into task queue", cf, cf->parent_frame);

      tls_ = new (alloca(sizeof(thread_local_storage))) thread_local_storage{};

      std::size_t cf_size = reinterpret_cast<uintptr_t>(cf->parent_frame) - reinterpret_cast<uintptr_t>(cf);
      wsq_.push(wsqueue_entry{cf, cf_size});

      tls_->dag_prof.start();
      tls_->dag_prof.increment_thread_count();
      tls_->dag_prof.increment_strand_count();

      common::verbose<2>("Starting new thread %p", ts);
      common::profiler::switch_phase<prof_phase_sched_fork, prof_phase_thread>();

      T ret = invoke_fn<T>(fn, args...);

      common::profiler::switch_phase<prof_phase_thread, prof_phase_sched_die>();
      common::verbose<2>("Thread %p is completed", ts);

      on_task_die();
      on_die(ts, ret, std::forward<OnDriftDieCallback>(on_drift_die_cb));

      common::verbose<2>("Thread %p is serialized (fast path)", ts);

      // The following is executed only when the thread is serialized
      std::destroy_at(ts);
      thread_state_allocator_.deallocate(ts, sizeof(thread_state<T>));
      th.state      = nullptr;
      th.serialized = true;
      th.retval_ser = {ret, tls_->dag_prof};

      common::verbose<2>("Resume parent context frame [%p, %p) (fast path)", cf, cf->parent_frame);

      common::profiler::switch_phase<prof_phase_sched_die, prof_phase_sched_resume_popped>();
    });

    if (th.serialized) {
      common::profiler::switch_phase<prof_phase_sched_resume_popped, prof_phase_thread>();
    } else {
      if constexpr (!std::is_null_pointer_v<std::remove_reference_t<OnDriftForkCallback>>) {
        common::profiler::switch_phase<prof_phase_sched_resume_stolen, prof_phase_cb_drift_fork>();
        on_drift_fork_cb();
        common::profiler::switch_phase<prof_phase_cb_drift_fork, prof_phase_thread>();
      } else {
        common::profiler::switch_phase<prof_phase_sched_resume_stolen, prof_phase_thread>();
      }
    }

    // restart to count only the last task in the task group
    tls_->dag_prof.clear();
    tls_->dag_prof.start();
    tls_->dag_prof.increment_strand_count();
  }

  template <typename T>
  T join(thread_handler<T>& th) {
    common::profiler::switch_phase<prof_phase_thread, prof_phase_sched_join>();

    on_task_die();

    thread_retval<T> retval;
    if (th.serialized) {
      common::verbose<2>("Skip join for serialized thread (fast path)");
      // We can skip deallocaton for its thread state because it has been already deallocated
      // when the thread is serialized (i.e., at a fork)
      retval = th.retval_ser;

    } else {
      ITYR_CHECK(th.state != nullptr);
      thread_state<T>* ts = th.state;

      if (remote_get_value(thread_state_allocator_, &ts->resume_flag) >= 1) {
        common::verbose("Thread %p is already joined", ts);
        if constexpr (!std::is_same_v<T, no_retval_t> || dag_profiler::enabled) {
          retval = remote_get_value(thread_state_allocator_, &ts->retval);
        }

      } else {
        bool migrated = true;
        suspend([&, ts](context_frame* cf) {
          suspended_state ss = evacuate(cf);

          remote_put_value(thread_state_allocator_, ss, &ts->suspended);

          // race
          if (remote_faa_value(thread_state_allocator_, 1, &ts->resume_flag) == 0) {
            common::verbose("Win the join race for thread %p (joining thread)", ts);
            common::profiler::switch_phase<prof_phase_sched_join, prof_phase_sched_loop>();
            resume_sched();
          } else {
            common::verbose("Lose the join race for thread %p (joining thread)", ts);
            suspended_thread_allocator_.deallocate(ss.evacuation_ptr, ss.frame_size);
            migrated = false;
          }
        });

        common::verbose("Resume continuation of join for thread %p", ts);

        if (migrated) {
          common::profiler::switch_phase<prof_phase_sched_resume_join, prof_phase_sched_join>();
        }

        if constexpr (!std::is_same_v<T, no_retval_t> || dag_profiler::enabled) {
          retval = remote_get_value(thread_state_allocator_, &ts->retval);
        }
      }

      std::destroy_at(ts);
      thread_state_allocator_.deallocate(ts, sizeof(thread_state<T>));
      th.state = nullptr;
    }

    tls_->dag_prof.merge_parallel(retval.dag_prof);

    common::profiler::switch_phase<prof_phase_sched_join, prof_phase_thread>();
    return retval.value;
  }

  template <typename SchedLoopCallback, typename CondFn>
  void sched_loop(SchedLoopCallback&& cb, CondFn&& cond_fn) {
    common::verbose("Enter scheduling loop");

    while (!should_exit_sched_loop(std::forward<CondFn>(cond_fn))) {
      steal();

      if constexpr (!std::is_null_pointer_v<std::remove_reference_t<SchedLoopCallback>>) {
        cb();
      }
    }

    common::verbose("Exit scheduling loop");
  }

  template <typename PreSuspendCallback, typename PostSuspendCallback>
  void poll(PreSuspendCallback&&, PostSuspendCallback&&) {}

  template <typename Fn, typename... Args>
  auto coll_exec(Fn&& fn, Args&&... args) {
    using retval_t = std::invoke_result_t<Fn, Args...>;

    auto begin_rank = common::topology::my_rank();
    std::conditional_t<std::is_void_v<retval_t>, no_retval_t, retval_t> retv;

    auto coll_task_fn = [&, fn, args...] {
      if constexpr (std::is_void_v<retval_t>) {
        fn(args...);
      } else {
        auto&& ret = fn(args...);
        if (common::topology::my_rank() == begin_rank) {
          retv = std::forward<decltype(ret)>(ret);
        }
      }
    };

    size_t task_size = sizeof(callable_task<decltype(coll_task_fn)>);
    void* task_ptr = suspended_thread_allocator_.allocate(task_size);

    auto t = new (task_ptr) callable_task(coll_task_fn);

    coll_task ct {.task_ptr = task_ptr, .task_size = task_size, .begin_rank = begin_rank};
    execute_coll_task(t, ct);

    suspended_thread_allocator_.deallocate(t, task_size);

    if constexpr (!std::is_void_v<retval_t>) {
      return retv;
    }
  }

  bool is_executing_root() const {
    return cf_top_ && cf_top_ == stack_top();
  }

  template <typename T>
  static bool is_serialized(thread_handler<T> th) {
    return th.serialized;
  }

  task_group_data task_group_begin() {
    tls_->dag_prof.stop();

    task_group_data tgdata {tls_->dag_prof};

    tls_->dag_prof.clear();
    tls_->dag_prof.start();
    tls_->dag_prof.increment_strand_count();

    return tgdata;
  }

  template <typename PreSuspendCallback, typename PostSuspendCallback>
  void task_group_end(task_group_data& tgdata, PreSuspendCallback&&, PostSuspendCallback&&) {
    on_task_die();

    tls_->dag_prof.merge_serial(tgdata.dag_prof);
    tls_->dag_prof.start();
    tls_->dag_prof.increment_strand_count();
  }

  void dag_prof_begin() { dag_prof_enabled_ = true; }
  void dag_prof_end() { dag_prof_enabled_ = false; }

  void dag_prof_print() const {
    if (common::topology::my_rank() == 0) {
      dag_prof_result_.print();
    }
  }

private:
  struct coll_task {
    void*                    task_ptr;
    std::size_t              task_size;
    common::topology::rank_t begin_rank;
  };

  void on_task_die() {
    if (!tls_->dag_prof.is_stopped()) {
      tls_->dag_prof.stop();
    }
  }

  template <typename T, typename OnDriftDieCallback>
  void on_die(thread_state<T>* ts, const T& ret, OnDriftDieCallback&& on_drift_die_cb) {
    auto qe = wsq_.pop();
    bool serialized = qe.has_value();

    if (!serialized) {
      if constexpr (!std::is_null_pointer_v<std::remove_reference_t<OnDriftDieCallback>>) {
        common::profiler::switch_phase<prof_phase_sched_die, prof_phase_cb_drift_die>();
        on_drift_die_cb();
        common::profiler::switch_phase<prof_phase_cb_drift_die, prof_phase_sched_die>();
      }

      if constexpr (!std::is_same_v<T, no_retval_t> || dag_profiler::enabled) {
        thread_retval<T> retval = {ret, tls_->dag_prof};
        remote_put_value(thread_state_allocator_, retval, &ts->retval);
      }

      // race
      if (remote_faa_value(thread_state_allocator_, 1, &ts->resume_flag) == 0) {
        common::verbose("Win the join race for thread %p (joined thread)", ts);
        common::profiler::switch_phase<prof_phase_sched_die, prof_phase_sched_loop>();
        resume_sched();
      } else {
        common::verbose("Lose the join race for thread %p (joined thread)", ts);
        common::profiler::switch_phase<prof_phase_sched_die, prof_phase_sched_resume_join>();
        suspended_state ss = remote_get_value(thread_state_allocator_, &ts->suspended);
        resume(ss);
      }
    }
  }

  template <typename T>
  void on_root_die(thread_state<T>* ts, const T& ret) {
    if constexpr (!std::is_same_v<T, no_retval_t> || dag_profiler::enabled) {
      thread_retval<T> retval = {ret, tls_->dag_prof};
      remote_put_value(thread_state_allocator_, retval, &ts->retval);
    }
    remote_put_value(thread_state_allocator_, 1, &ts->resume_flag);

    common::profiler::switch_phase<prof_phase_sched_die, prof_phase_sched_loop>();
    resume_sched();
  }

  void steal() {
    auto target_rank = get_random_rank(0, common::topology::n_ranks() - 1);

    auto ibd = common::profiler::interval_begin<prof_event_sched_steal>(target_rank);

    if (wsq_.empty(target_rank)) {
      common::profiler::interval_end<prof_event_sched_steal>(ibd, false);
      return;
    }

    if (!wsq_.lock().trylock(target_rank)) {
      common::profiler::interval_end<prof_event_sched_steal>(ibd, false);
      return;
    }

    auto we = wsq_.steal_nolock(target_rank);
    if (!we.has_value()) {
      wsq_.lock().unlock(target_rank);
      common::profiler::interval_end<prof_event_sched_steal>(ibd, false);
      return;
    }

    common::verbose("Steal context frame [%p, %p) from rank %d",
                    we->frame_base, reinterpret_cast<std::byte*>(we->frame_base) + we->frame_size, target_rank);

    stack_.direct_copy_from(we->frame_base, we->frame_size, target_rank);

    wsq_.lock().unlock(target_rank);

    common::profiler::interval_end<prof_event_sched_steal>(ibd, true);

    common::profiler::switch_phase<prof_phase_sched_loop, prof_phase_sched_resume_stolen>();

    context_frame* next_cf = reinterpret_cast<context_frame*>(we->frame_base);
    suspend([&](context_frame* cf) {
      sched_cf_ = cf;
      context::clear_parent_frame(next_cf);
      resume(next_cf);
    });
  }

  template <typename Fn>
  void suspend(Fn&& fn) {
    context_frame*        prev_cf_top = cf_top_;
    thread_local_storage* prev_tls    = tls_;

    context::save_context_with_call(prev_cf_top,
        [](context_frame* cf, void* cf_top_p, void* fn_p) {
      context_frame*& cf_top = *reinterpret_cast<context_frame**>(cf_top_p);
      Fn              fn     = *reinterpret_cast<Fn*>(fn_p); // copy closure to the new stack frame
      cf_top = cf;
      fn(cf);
    }, &cf_top_, &fn, prev_tls);

    cf_top_ = prev_cf_top;
    tls_    = prev_tls;
  }

  void resume(context_frame* cf) {
    common::verbose("Resume context frame [%p, %p) in the stack", cf, cf->parent_frame);
    context::resume(cf);
  }

  void resume(suspended_state ss) {
    common::verbose("Resume context frame [%p, %p) evacuated at %p",
                    ss.frame_base, ss.frame_size, ss.evacuation_ptr);

    // We pass the suspended thread states *by value* because the current local variables can be overwritten by the
    // new stack we will bring from remote nodes.
    context::jump_to_stack(ss.frame_base, [](void* allocator_, void* evacuation_ptr, void* frame_base, void* frame_size_) {
      common::remotable_resource& allocator  = *reinterpret_cast<common::remotable_resource*>(allocator_);
      std::size_t                 frame_size = reinterpret_cast<std::size_t>(frame_size_);
      common::remote_get(allocator,
                         reinterpret_cast<std::byte*>(frame_base),
                         reinterpret_cast<std::byte*>(evacuation_ptr),
                         frame_size);
      allocator.deallocate(evacuation_ptr, frame_size);

      context_frame* cf = reinterpret_cast<context_frame*>(frame_base);
      context::clear_parent_frame(cf);
      context::resume(cf);
    }, &suspended_thread_allocator_, ss.evacuation_ptr, ss.frame_base, reinterpret_cast<void*>(ss.frame_size));
  }

  void resume_sched() {
    cf_top_ = nullptr;
    tls_ = nullptr;
    common::verbose("Resume scheduler context");
    context::resume(sched_cf_);
  }

  suspended_state evacuate(context_frame* cf) {
    std::size_t cf_size = reinterpret_cast<uintptr_t>(cf->parent_frame) - reinterpret_cast<uintptr_t>(cf);
    void* evacuation_ptr = suspended_thread_allocator_.allocate(cf_size);
    std::memcpy(evacuation_ptr, cf, cf_size);

    common::verbose("Evacuate suspended thread context [%p, %p) to %p",
                    cf, cf->parent_frame, evacuation_ptr);

    return {evacuation_ptr, cf, cf_size};
  }

  context_frame* stack_top() const {
    // Add a margin of sizeof(context_frame) to the bottom of the stack, because
    // this region can be accessed by the clear_parent_frame() function later
    return reinterpret_cast<context_frame*>(stack_.bottom()) - 1;
  }

  template <typename Fn>
  void root_on_stack(Fn&& fn) {
    cf_top_ = stack_top();
    context::call_on_stack(stack_.top(), stack_.size() - sizeof(context_frame),
                           [](void* fn_, void*, void*, void*) {
      Fn fn = *reinterpret_cast<Fn*>(fn_); // copy closure to the new stack frame
      fn();
    }, &fn, nullptr, nullptr, nullptr);
  }

  void execute_coll_task(task_general* t, coll_task ct) {
    coll_task ct_ {.task_ptr = t, .task_size = ct.task_size, .begin_rank = ct.begin_rank};

    // pass coll task to other processes in a binary tree form
    auto n_ranks = common::topology::n_ranks();
    auto my_rank_shifted = (common::topology::my_rank() + n_ranks - ct.begin_rank) % n_ranks;
    for (common::topology::rank_t i = common::next_pow2(n_ranks); i > 1; i /= 2) {
      if (my_rank_shifted % i == 0) {
        auto target_rank_shifted = my_rank_shifted + i / 2;
        if (target_rank_shifted < n_ranks) {
          auto target_rank = (target_rank_shifted + ct.begin_rank) % n_ranks;
          coll_task_mailbox_.put(ct_, target_rank);
        }
      }
    }

    // Ensure all processes have finished coll task execution before deallocation
    common::mpi_barrier(common::topology::mpicomm());

    t->execute();

    // Ensure all processes have finished coll task execution before deallocation
    common::mpi_barrier(common::topology::mpicomm());
  }

  void execute_coll_task_if_arrived() {
    auto ct = coll_task_mailbox_.pop();
    if (ct.has_value()) {
      task_general* t = reinterpret_cast<task_general*>(
          suspended_thread_allocator_.allocate(ct->task_size));

      common::remote_get(suspended_thread_allocator_,
                         reinterpret_cast<std::byte*>(t),
                         reinterpret_cast<std::byte*>(ct->task_ptr),
                         ct->task_size);

      execute_coll_task(t, *ct);

      suspended_thread_allocator_.deallocate(t, ct->task_size);
    }
  }

  template <typename CondFn>
  bool should_exit_sched_loop(CondFn&& cond_fn) {
    if (sched_loop_make_mpi_progress_option::value()) {
      common::mpi_make_progress();
    }

    execute_coll_task_if_arrived();

    if (sched_loop_exit_req_ == MPI_REQUEST_NULL &&
        std::forward<CondFn>(cond_fn)()) {
      // If a given condition is met, enters a barrier
      sched_loop_exit_req_ = common::mpi_ibarrier(common::topology::mpicomm());
    }
    if (sched_loop_exit_req_ != MPI_REQUEST_NULL) {
      // If the barrier is resolved, the scheduler loop should terminate
      return common::mpi_test(sched_loop_exit_req_);
    }
    return false;
  }

  struct wsqueue_entry {
    void*       frame_base;
    std::size_t frame_size;
  };

  callstack                  stack_;
  oneslot_mailbox<coll_task> coll_task_mailbox_;
  wsqueue<wsqueue_entry>     wsq_;
  common::remotable_resource thread_state_allocator_;
  common::remotable_resource suspended_thread_allocator_;
  context_frame*             cf_top_              = nullptr;
  context_frame*             sched_cf_            = nullptr;
  MPI_Request                sched_loop_exit_req_ = MPI_REQUEST_NULL;
  thread_local_storage*      tls_                 = nullptr;
  bool                       dag_prof_enabled_    = false;
  dag_profiler               dag_prof_result_;
};

}
