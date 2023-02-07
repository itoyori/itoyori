#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/allocator.hpp"
#include "ityr/ito/context.hpp"
#include "ityr/ito/wsqueue.hpp"
#include "ityr/ito/callstack.hpp"

namespace ityr::ito {

class scheduler {
public:
  struct suspended_state {
    void* evacuation_ptr;
    void* stack_top;
    std::size_t size;
  };

  template <typename T>
  struct thread_state {
    T retval;
    int resume_flag = 0;
    suspended_state suspended;
  };

  template <typename T>
  struct thread_handler {
    thread_state<T>* state = nullptr;
    bool serialized = false;
  };

  scheduler(const common::topology& topo, const callstack& stack)
    : topo_(topo),
      stack_(stack),
      wsq_(topo_, common::getenv_coll("ITYR_ITO_WSQUEUE_CAPACITY", 1024, topo.mpicomm())),
      thread_state_allocator_(topo_),
      suspended_thread_allocator_(topo_) {}

  template <typename T, typename Fn, typename... Args>
  thread_handler<T> fork(Fn&& fn, Args&&... args) {
    thread_state<T>* ts = new (thread_state_allocator_.allocate(sizeof(thread_state<T>))) thread_state<T>;
    bool serialized = false;

    suspend([&, ts](context_frame* cf) {
      std::size_t cf_size = reinterpret_cast<uintptr_t>(cf->parent_frame) - reinterpret_cast<uintptr_t>(cf);
      wsq_.push(wsqueue_entry{cf, cf_size});

      T retval = std::forward<Fn>(fn)(std::forward<Args>(args)...);

      on_die(ts, retval);
      serialized = true;
    });

    return {ts, serialized};
  }

  template <typename T, typename Fn, typename... Args>
  thread_handler<T> fork_root(Fn&& fn, Args&&... args) {
    thread_state<T>* ts = new (thread_state_allocator_.allocate(sizeof(thread_state<T>))) thread_state<T>;

    suspend([&, ts](context_frame* cf) {
      sched_cf_ = cf;
      cf_top_ = reinterpret_cast<context_frame*>(stack_.bottom());
      root_on_stack([&, ts]() {
        T retval = std::forward<Fn>(fn)(std::forward<Args>(args)...);
        remote_put_value(thread_state_allocator_, retval, &ts->retval);
        remote_put_value(thread_state_allocator_, 1, &ts->resume_flag);
      });
    });

    sched_loop([=]() { return ts->resume_flag >= 1; });

    return {ts, true};
  }

  template <typename T>
  T join(thread_handler<T>& th) {
    ITYR_CHECK(th.state != nullptr);
    thread_state<T>* ts = th.state;

    T retval;
    if (th.serialized) {
      retval = ts->retval;

    } else if (remote_get_value(thread_state_allocator_, &ts->resume_flag) >= 1) {
      retval = remote_get_value(thread_state_allocator_, &ts->retval);

    } else {
      suspend([&, ts](context_frame* cf) {
        std::size_t cf_size = reinterpret_cast<uintptr_t>(cf->parent_frame) - reinterpret_cast<uintptr_t>(cf);
        void* evacuation_ptr = suspended_thread_allocator_.allocate(cf_size);
        std::memcpy(evacuation_ptr, cf, cf_size);

        suspended_state ss {evacuation_ptr, cf, cf_size};
        remote_put_value(thread_state_allocator_, ss, &ts->suspended);

        // race
        if (remote_faa_value(thread_state_allocator_, 1, &ts->resume_flag) == 0) {
          // win
          resume(sched_cf_);
        } else {
          // lose
          suspended_thread_allocator_.deallocate(ss.evacuation_ptr, ss.size);
          resume(cf);
        }
      });

      retval = remote_get_value(thread_state_allocator_, &ts->retval);
    }

    std::destroy_at(ts);
    thread_state_allocator_.deallocate(ts, sizeof(thread_state<T>));
    th.state = nullptr;
    return retval;
  }

  template <typename CondFn>
  void sched_loop(CondFn cond_fn) {
    while (!cond_fn()) {
    }
    auto req = common::mpi_ibarrier(topo_.mpicomm());
    while (!common::mpi_test(req)) {
    }
  }

private:
  template <typename T>
  void on_die(thread_state<T>* ts, const T& retval) {
    auto qe = wsq_.pop();
    if (qe.has_value()) {
      ITYR_CHECK(thread_state_allocator_.get_owner(ts) == topo_.my_rank());
      ts->retval = retval;

    } else {
      remote_put_value(thread_state_allocator_, retval, &ts->retval);
      // race
      if (remote_faa_value(thread_state_allocator_, 1, &ts->resume_flag) == 0) {
        // win
        resume(sched_cf_);
      } else {
        // lose
        suspended_state ss = remote_get_value(thread_state_allocator_, &ts->suspended);
        resume(ss);
      }
    }
  }

  template <typename Fn>
  void suspend(Fn&& fn) {
    context_frame* prev_cf_top = cf_top_;
    context::save_context_with_call(prev_cf_top,
        [](context_frame* cf, void* cf_top_p, void* fn_p) {
      context_frame*& cf_top = *reinterpret_cast<context_frame**>(cf_top_p);
      Fn&             fn     = *reinterpret_cast<Fn*>(fn_p);
      cf_top = cf;
      fn(cf);
    }, &cf_top_, &fn);
    cf_top_ = prev_cf_top;
  }

  void resume(context_frame* cf) {
    context::resume(cf);
  }

  void resume(const suspended_state& ss) {
    common::remote_get(suspended_thread_allocator_,
                       reinterpret_cast<std::byte*>(ss.stack_top),
                       reinterpret_cast<std::byte*>(ss.evacuation_ptr),
                       ss.size);
    suspended_thread_allocator_.deallocate(ss.evacuation_ptr, ss.size);
    context_frame* cf = reinterpret_cast<context_frame*>(ss.stack_top);
    resume(cf);
  }

  template <typename Fn>
  void root_on_stack(Fn&& fn) {
    context::call_on_stack(stack_.top(), stack_.size(), [](void* fn_p, void*, void*, void*) {
      Fn& fn = *reinterpret_cast<Fn*>(fn_p);
      fn();
    }, &fn, nullptr, nullptr, nullptr);
  }

  struct wsqueue_entry {
    void*       frame_base;
    std::size_t frame_size;
  };

  const common::topology&    topo_;
  const callstack&           stack_;
  wsqueue<wsqueue_entry>     wsq_;
  common::remotable_resource thread_state_allocator_;
  common::remotable_resource suspended_thread_allocator_;
  context_frame*             cf_top_   = nullptr;
  context_frame*             sched_cf_ = nullptr;
};

}
