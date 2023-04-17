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

namespace ityr::ito {

class task_general {
public:
  virtual ~task_general() = default;
  virtual void execute() = 0;
};

template <typename Fn, typename... Args>
class callable_task : task_general {
public:
  callable_task(Fn fn, Args... args) : fn_(fn), arg_(args...) {}
  void execute() { std::apply(fn_, arg_); }
private:
  Fn                  fn_;
  std::tuple<Args...> arg_;
};

class dist_range {
public:
  using value_type = double;

  dist_range() {}
  dist_range(common::topology::rank_t n_ranks)
    : begin_(0), end_(static_cast<value_type>(n_ranks)) {}
  dist_range(value_type begin, value_type end)
    : begin_(begin), end_(end) {}

  value_type begin() const { return begin_; }
  value_type end() const { return end_; }

  common::topology::rank_t begin_rank() const {
    return static_cast<common::topology::rank_t>(begin_);
  }

  common::topology::rank_t end_rank() const {
    auto end_rank = static_cast<common::topology::rank_t>(end_);
    if (static_cast<value_type>(end_rank) == end_) {
      // If the range is at the boundary of the end point, i.e., range = [x, y.0),
      // the end rank is determined to y-1.
      end_rank--;
    }
    return end_rank;
  }

  bool is_at_end_boundary() const {
    return static_cast<value_type>(static_cast<common::topology::rank_t>(end_)) == end_;
  }

  template <typename T>
  std::pair<dist_range, dist_range> divide(T r1, T r2) const {
    value_type at = begin_ + (end_ - begin_) * r1 / (r1 + r2);

    // Boundary condition for tasks at the very bottom of the task hierarchy.
    // A task with range [P, P) such that P = #workers would be assigned to worker P,
    // but worker P does not exist; thus we need to assign the task to worker P-1.
    if (at == end_) {
      constexpr value_type eps = 0.00001;
      at -= eps;
      if (at < begin_) at = begin_;
    }

    return std::make_pair(dist_range{begin_, at}, dist_range{at, end_});
  }

  common::topology::rank_t owner() const {
    return static_cast<common::topology::rank_t>(begin_);
  }

  bool is_cross_worker() const {
    return static_cast<common::topology::rank_t>(begin_) != static_cast<common::topology::rank_t>(end_);
  }

private:
  value_type begin_;
  value_type end_;
};

class dist_tree {
public:
  struct node_ref {
    common::topology::rank_t owner_rank  = -1;
    int                      local_depth = -1;
    int                      depth       = -1; // piggy-back to reduce communication
  };

  struct node {
    node_ref   parent;
    dist_range drange;
    bool       dominant;

    int depth() const { return parent.depth + 1; }
  };

  dist_tree()
    : max_local_depth_(adws_max_dist_tree_depth_option::value()),
      remote_buf_(max_local_depth_),
      win_(common::topology::mpicomm(), max_local_depth_) {}

  node_ref append(node_ref parent, dist_range drange) {
    auto my_rank = common::topology::my_rank();

    int local_depth;
    if (parent.owner_rank != my_rank) {
      // Append the top local node to the remote parent node
      local_depth = 0;
    } else {
      local_depth = parent.local_depth + 1;
      ITYR_REQUIRE_MESSAGE(parent.local_depth + 1 < max_local_depth_,
                           "Reached maximum local depth (%d) for dist_tree", max_local_depth_);
    }

    node& new_node = local_node(local_depth);
    new_node.parent   = parent;
    new_node.drange   = drange;
    new_node.dominant = false;

    return {my_rank, local_depth, parent.depth + 1};
  }

  void set_dominant(node_ref nr) {
    if (nr.owner_rank == common::topology::my_rank()) {
      get_local_node(nr).dominant = true;
    } else {
      std::size_t disp = nr.local_depth * sizeof(node) + offsetof(node, dominant);
      common::mpi_atomic_put_value(true, nr.owner_rank, disp, win_.win());
    }
  }

  std::optional<node> get_topmost_dominant(node_ref nr) {
    ITYR_PROFILER_RECORD(prof_event_sched_adws_scan_tree);

    // TODO: reduce comm
    std::optional<node> ret;

    node_ref parent;

    if (nr.owner_rank == common::topology::my_rank()) {
      for (int d = 0; d <= nr.local_depth; d++) {
        if (local_node(d).dominant) {
          ret = local_node(d);
          break;
        }
      }
      parent = local_node(0).parent;
    } else {
      parent = nr;
    }

    while (parent.owner_rank != -1) {
      common::mpi_get(remote_buf_.data(), parent.local_depth + 1,
                      parent.owner_rank, 0, win_.win());
      for (int d = 0; d <= parent.local_depth; d++) {
        if (remote_buf_[d].dominant) {
          ret = remote_buf_[d];
          break;
        }
      }
      parent = remote_buf_[0].parent;
    }

    return ret;
  }

  node& get_local_node(node_ref nr) {
    ITYR_CHECK(nr.owner_rank == common::topology::my_rank());
    return local_node(nr.local_depth);
  }

private:
  node& local_node(int local_depth) {
    ITYR_CHECK(0 <= local_depth);
    ITYR_CHECK(local_depth < max_local_depth_);
    return win_.local_buf()[local_depth];
  }

  int                           max_local_depth_;
  std::vector<node>             remote_buf_;
  common::mpi_win_manager<node> win_;
};

template <typename Entry>
class oneslot_mailbox {
  static_assert(std::is_trivially_copyable_v<Entry>);

public:
  oneslot_mailbox()
    : win_(common::topology::mpicomm(), 1) {}

  void put(const Entry& entry, common::topology::rank_t target_rank) {
    ITYR_CHECK(!common::mpi_get_value<bool>(target_rank, offsetof(mailbox, arrived), win_.win()));
    common::mpi_put_value(entry, target_rank, offsetof(mailbox, entry), win_.win());
    common::mpi_put_value(true, target_rank, offsetof(mailbox, arrived), win_.win());
  }

  std::optional<Entry> pop() {
    mailbox& mb = win_.local_buf()[0];
    if (mb.arrived) {
      mb.arrived = false;
      return mb.entry;
    } else {
      return std::nullopt;
    }
  }

private:
  struct mailbox {
    Entry entry;
    bool  arrived;
  };

  common::mpi_win_manager<mailbox> win_;
};

class scheduler_adws {
public:
  struct no_retval_t {};

  struct suspended_state {
    void*       evacuation_ptr;
    void*       frame_base;
    std::size_t frame_size;
  };

  template <typename T>
  struct thread_state {
    T               retval;
    int             resume_flag = 0;
    suspended_state suspended;
  };

  template <typename T>
  struct thread_handler {
    thread_state<T>* state      = nullptr;
    bool             serialized = false;
    T                retval_ser; // return the result by value if the thread is serialized
  };

  struct thread_local_storage {
    dist_range          drange;         // distribution range of this thread
    dist_tree::node_ref dtree_node_ref; // distribution tree node of the cross-worker task group that this thread belongs to
  };

  struct task_group_data {
    dist_range drange;
  };

  scheduler_adws()
    : stack_(stack_size_option::value()),
      primary_wsq_(adws_wsqueue_capacity_option::value(), adws_max_num_queue_option::value()),
      migration_wsq_(adws_wsqueue_capacity_option::value(), adws_max_num_queue_option::value()),
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

        dist_range root_drange {common::topology::n_ranks()};
        tls_ = new (alloca(sizeof(thread_local_storage)))
               thread_local_storage{.drange = root_drange, .dtree_node_ref = {}};

        common::profiler::switch_phase<prof_phase_sched_fork, prof_phase_thread>();

        T retval = invoke_fn<T>(fn, args...);

        common::profiler::switch_phase<prof_phase_thread, prof_phase_sched_die>();
        common::verbose("Root thread %p is completed", ts);

        on_root_die(ts, retval);
      });
    });

    sched_loop(std::forward<SchedLoopCallback>(cb),
               [=]() { return ts->resume_flag >= 1; });

    common::profiler::switch_phase<prof_phase_sched_loop, prof_phase_sched_join>();

    T retval = ts->retval;
    std::destroy_at(ts);
    thread_state_allocator_.deallocate(ts, sizeof(thread_state<T>));

    common::profiler::switch_phase<prof_phase_sched_join, prof_phase_spmd>();

    return retval;
  }

  task_group_data task_group_begin() {
    task_group_data tgdata {.drange = tls_->drange};

    if (tls_->drange.is_cross_worker()) {
      tls_->dtree_node_ref = dtree_.append(tls_->dtree_node_ref, tls_->drange);
      dtree_local_bottom_ref_ = tls_->dtree_node_ref;

      common::verbose("Begin a cross-worker task group of distribution range [%f, %f) at depth %d",
                      tls_->drange.begin(), tls_->drange.end(), tls_->dtree_node_ref.depth);
    }

    return tgdata;
  }

  void task_group_end(task_group_data& tgdata) {
    // restore the original distributed range of this thread at the beginning of the task group
    tls_->drange = tgdata.drange;

    if (tls_->drange.is_cross_worker()) {
      common::verbose("End a cross-worker task group of distribution range [%f, %f) at depth %d",
                      tls_->drange.begin(), tls_->drange.end(), tls_->dtree_node_ref.depth);

      // migrate the cross-worker-task to the owner
      auto target_rank = tls_->drange.owner();
      if (target_rank != common::topology::my_rank()) {
        common::profiler::switch_phase<prof_phase_thread, prof_phase_sched_migrate>();

        suspend([&](context_frame* cf) {
          suspended_state ss = evacuate(cf);

          common::verbose("Migrate continuation of cross-worker-task [%f, %f) to process %d",
                          tls_->drange.begin(), tls_->drange.end(), target_rank);

          cross_worker_mailbox_.put({ss.evacuation_ptr, ss.frame_base, ss.frame_size}, target_rank);

          common::profiler::switch_phase<prof_phase_sched_migrate, prof_phase_sched_loop>();

          resume_sched();
        });

        common::profiler::switch_phase<prof_phase_sched_resume_migrate, prof_phase_thread>();
      }

      // Set the parent dist_tree node to the current thread
      auto& dtree_node = dtree_.get_local_node(tls_->dtree_node_ref);
      tls_->dtree_node_ref = dtree_node.parent;
      dtree_local_bottom_ref_ = tls_->dtree_node_ref;
    }
  }

  template <typename T, typename OnDriftForkCallback, typename OnDriftDieCallback,
            typename WorkHint, typename Fn, typename... Args>
  void fork(thread_handler<T>& th,
            OnDriftForkCallback&& on_drift_fork_cb, OnDriftDieCallback&& on_drift_die_cb,
            WorkHint w1, WorkHint w2, Fn&& fn, Args&&... args) {
    common::profiler::switch_phase<prof_phase_thread, prof_phase_sched_fork>();

    auto my_rank = common::topology::my_rank();

    thread_state<T>* ts = new (thread_state_allocator_.allocate(sizeof(thread_state<T>))) thread_state<T>;
    th.state = ts;
    th.serialized = false;

    dist_range new_drange;
    common::topology::rank_t target_rank;
    if (tls_->drange.is_cross_worker()) {
      auto [dr1, dr2] = tls_->drange.divide(w1, w2);

      common::verbose("Distribution range [%f, %f) is divided into [%f, %f) and [%f, %f)",
                      tls_->drange.begin(), tls_->drange.end(),
                      dr1.begin(), dr1.end(), dr2.begin(), dr2.end());

      tls_->drange = dr1;
      new_drange = dr2;
      target_rank = dr2.owner();

    } else {
      // quick path for non-cross-worker tasks (without dividing the distribution range)
      new_drange = tls_->drange;
      // Since this task may have been stolen by workers outside of this task group,
      // the target rank should be itself.
      target_rank = my_rank;
    }

    if (target_rank == my_rank) {
      /* Put the continuation into the local queue and execute the new task (work-first) */

      suspend([&, ts, fn, args...](context_frame* cf) mutable {
        common::verbose<3>("push context frame [%p, %p) into task queue", cf, cf->parent_frame);

        tls_ = new (alloca(sizeof(thread_local_storage)))
               thread_local_storage{.drange = new_drange, .dtree_node_ref = tls_->dtree_node_ref};

        std::size_t cf_size = reinterpret_cast<uintptr_t>(cf->parent_frame) - reinterpret_cast<uintptr_t>(cf);

        if (use_primary_wsq_) {
          primary_wsq_.push({cf, cf_size}, tls_->dtree_node_ref.depth);
        } else {
          migration_wsq_.push({true, nullptr, cf, cf_size}, tls_->dtree_node_ref.depth);
        }

        common::verbose<3>("Starting new thread %p", ts);
        common::profiler::switch_phase<prof_phase_sched_fork, prof_phase_thread>();

        T retval = invoke_fn<T>(fn, args...);

        common::profiler::switch_phase<prof_phase_thread, prof_phase_sched_die>();
        common::verbose<3>("Thread %p is completed", ts);

        on_die_reached();
        on_die_workfirst(ts, retval, std::forward<OnDriftDieCallback>(on_drift_die_cb));

        common::verbose<3>("Thread %p is serialized (fast path)", ts);

        // The following is executed only when the thread is serialized
        std::destroy_at(ts);
        thread_state_allocator_.deallocate(ts, sizeof(thread_state<T>));
        th.state      = nullptr;
        th.serialized = true;
        th.retval_ser = retval;

        common::verbose<3>("Resume parent context frame [%p, %p) (fast path)", cf, cf->parent_frame);

        common::profiler::switch_phase<prof_phase_sched_die, prof_phase_sched_resume_parent>();
      });

      // reload my_rank because this thread might have been migrated
      if (target_rank == common::topology::my_rank()) {
        common::profiler::switch_phase<prof_phase_sched_resume_parent, prof_phase_thread>();
      } else {
        if constexpr (!std::is_null_pointer_v<std::remove_reference_t<OnDriftForkCallback>>) {
          common::profiler::switch_phase<prof_phase_sched_resume_stolen, prof_phase_sched_drift_fork_cb>();
          on_drift_fork_cb();
          common::profiler::switch_phase<prof_phase_sched_drift_fork_cb, prof_phase_thread>();
        } else {
          common::profiler::switch_phase<prof_phase_sched_resume_stolen, prof_phase_thread>();
        }
      }

    } else {
      /* Pass the new task to another worker and execute the continuation */

      auto new_task_fn = [&, my_rank, ts, new_drange, dtree_node_ref = tls_->dtree_node_ref,
                          on_drift_fork_cb, on_drift_die_cb, fn, args...]() mutable {
        common::verbose("Starting a migrated thread %p [%f, %f)",
                        ts, new_drange.begin(), new_drange.end());

        tls_ = new (alloca(sizeof(thread_local_storage)))
               thread_local_storage{.drange = new_drange, .dtree_node_ref = dtree_node_ref};

        if constexpr (!std::is_null_pointer_v<std::remove_reference_t<OnDriftForkCallback>>) {
          // If the new task is executed on another process
          if (my_rank != common::topology::my_rank()) {
            common::profiler::switch_phase<prof_phase_sched_start_new, prof_phase_sched_drift_fork_cb>();
            on_drift_fork_cb();
            common::profiler::switch_phase<prof_phase_sched_drift_fork_cb, prof_phase_thread>();
          } else {
            common::profiler::switch_phase<prof_phase_sched_start_new, prof_phase_thread>();
          }
        } else {
          common::profiler::switch_phase<prof_phase_sched_start_new, prof_phase_thread>();
        }

        T retval = invoke_fn<T>(fn, args...);

        common::profiler::switch_phase<prof_phase_thread, prof_phase_sched_die>();
        common::verbose("A migrated thread %p [%f, %f) is completed",
                        ts, new_drange.begin(), new_drange.end());

        on_die_reached();
        on_die_drifted(ts, retval, on_drift_die_cb);
      };

      size_t task_size = sizeof(callable_task<decltype(new_task_fn)>);
      void* task_ptr = suspended_thread_allocator_.allocate(task_size);

      auto t = new (task_ptr) callable_task(new_task_fn);

      if (new_drange.is_cross_worker()) {
        common::verbose("Migrate cross-worker-task %p [%f, %f) to process %d",
                        ts, new_drange.begin(), new_drange.end(), target_rank);

        cross_worker_mailbox_.put({nullptr, t, task_size}, target_rank);
      } else {
        common::verbose("Migrate non-cross-worker-task %p [%f, %f) to process %d",
                        ts, new_drange.begin(), new_drange.end(), target_rank);

        migration_wsq_.pass({false, nullptr, t, task_size}, target_rank,
                            tls_->dtree_node_ref.depth);
      }

      common::profiler::switch_phase<prof_phase_sched_fork, prof_phase_thread>();
    }
  }

  template <typename T>
  T join(thread_handler<T>& th) {
    common::profiler::switch_phase<prof_phase_thread, prof_phase_sched_join>();

    T retval;
    if (th.serialized) {
      common::verbose<3>("Skip join for serialized thread (fast path)");
      // We can skip deallocaton for its thread state because it has been already deallocated
      // when the thread is serialized (i.e., at a fork)
      retval = th.retval_ser;

    } else {
      ITYR_CHECK(th.state != nullptr);
      thread_state<T>* ts = th.state;

      if (remote_get_value(thread_state_allocator_, &ts->resume_flag) >= 1) {
        common::verbose("Thread %p is already joined", ts);
        if constexpr (!std::is_same_v<T, no_retval_t>) {
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

            // This procedure is not needed for purely work-first work stealing.
            // If it is combined with the help-first policy (migration of newly created tasks),
            // threads can block while their parents are in the local callstack, which does not
            // happen with the work-first policy alone.
            if (use_primary_wsq_) {
              auto qe = pop_from_primary_queues(tls_->dtree_node_ref.depth);
              if (qe.has_value()) {
                common::profiler::switch_phase<prof_phase_sched_join, prof_phase_sched_resume_parent>();

                context_frame* next_cf = reinterpret_cast<context_frame*>(qe->frame_base);
                resume(next_cf);
              }
            }

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

        if constexpr (!std::is_same_v<T, no_retval_t>) {
          retval = remote_get_value(thread_state_allocator_, &ts->retval);
        }
      }

      std::destroy_at(ts);
      thread_state_allocator_.deallocate(ts, sizeof(thread_state<T>));
      th.state = nullptr;
    }

    common::profiler::switch_phase<prof_phase_sched_join, prof_phase_thread>();
    return retval;
  }

  template <typename SchedLoopCallback, typename CondFn>
  void sched_loop(SchedLoopCallback&& cb, CondFn&& cond_fn) {
    common::verbose("Enter scheduling loop");

    while (!should_exit_sched_loop(std::forward<CondFn>(cond_fn))) {
      // TODO: immediately execute cross-worker tasks upon arrival
      auto cwt = cross_worker_mailbox_.pop();
      if (cwt.has_value()) {
        execute_cross_worker_task(*cwt);
        continue;
      }

      auto mwe = pop_from_migration_queues();
      if (mwe.has_value()) {
        use_primary_wsq_ = false;
        execute_migrated_task(*mwe);
        use_primary_wsq_ = true;
        continue;
      }

      if (adws_enable_steal_option::value()) {
        steal();
      }

      if constexpr (!std::is_null_pointer_v<std::remove_reference_t<SchedLoopCallback>>) {
        cb();
      }
      common::mpi_make_progress();
    }

    common::verbose("Exit scheduling loop");
  }

  template <typename T>
  static bool is_serialized(thread_handler<T> th) {
    return th.serialized;
  }

private:
  struct cross_worker_task {
    void*       evacuation_ptr;
    void*       frame_base;
    std::size_t frame_size;
  };

  struct primary_wsq_entry {
    void*       frame_base;
    std::size_t frame_size;
  };

  struct migration_wsq_entry {
    bool        is_continuation;
    void*       evacuation_ptr;
    void*       frame_base;
    std::size_t frame_size;
  };

  template <typename T, typename Fn, typename... Args>
  T invoke_fn(Fn&& fn, Args&&... args) {
    T retval;
    if constexpr (!std::is_same_v<T, no_retval_t>) {
      retval = std::forward<Fn>(fn)(std::forward<Args>(args)...);
    } else {
      std::forward<Fn>(fn)(std::forward<Args>(args)...);
    }
    return retval;
  }

  void on_die_reached() {
    // TODO: handle corner cases where cross-worker tasks finish without distributing
    // child cross-worker tasks to their owners
    if (tls_->drange.is_cross_worker()) {
      // Set the parent cross-worker task group as "dominant" task group, which allows for
      // work stealing within the range of workers within the task group.
      common::verbose("Distribution tree node (owner=%d, depth=%d) becomes dominant",
                      tls_->dtree_node_ref.owner_rank, tls_->dtree_node_ref.depth);

      dtree_.set_dominant(tls_->dtree_node_ref);
    }
  }

  template <typename T, typename OnDriftDieCallback>
  void on_die_workfirst(thread_state<T>* ts, const T& retval, OnDriftDieCallback&& on_drift_die_cb) {
    bool serialized;
    if (use_primary_wsq_) {
      auto qe = pop_from_primary_queues(tls_->dtree_node_ref.depth);
      serialized = qe.has_value();
    } else {
      auto qe = migration_wsq_.pop(tls_->dtree_node_ref.depth);
      serialized = qe.has_value();
      if (serialized) {
        ITYR_CHECK(qe->is_continuation);
      }
    }

    if (!serialized) {
      on_die_drifted(ts, retval, std::forward<OnDriftDieCallback>(on_drift_die_cb));
    }
  }

  template <typename T, typename OnDriftDieCallback>
  void on_die_drifted(thread_state<T>* ts, const T& retval, OnDriftDieCallback&& on_drift_die_cb) {
    if constexpr (!std::is_null_pointer_v<std::remove_reference_t<OnDriftDieCallback>>) {
      common::profiler::switch_phase<prof_phase_sched_die, prof_phase_sched_drift_die_cb>();
      on_drift_die_cb();
      common::profiler::switch_phase<prof_phase_sched_drift_die_cb, prof_phase_sched_die>();
    }

    if constexpr (!std::is_same_v<T, no_retval_t>) {
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

  template <typename T>
  void on_root_die(thread_state<T>* ts, const T& retval) {
    if constexpr (!std::is_same_v<T, no_retval_t>) {
      remote_put_value(thread_state_allocator_, retval, &ts->retval);
    }
    remote_put_value(thread_state_allocator_, 1, &ts->resume_flag);

    common::profiler::switch_phase<prof_phase_sched_die, prof_phase_sched_loop>();
    resume_sched();
  }

  common::topology::rank_t get_random_rank(common::topology::rank_t a, common::topology::rank_t b) {
    static std::mt19937 engine(std::random_device{}());

    ITYR_CHECK(0 <= a);
    ITYR_CHECK(a <= b);
    ITYR_CHECK(b < common::topology::n_ranks());
    std::uniform_int_distribution<common::topology::rank_t> dist(a, b);

    common::topology::rank_t rank;
    do {
      rank = dist(engine);
    } while (rank == common::topology::my_rank());

    ITYR_CHECK(a <= rank);
    ITYR_CHECK(rank != common::topology::my_rank());
    ITYR_CHECK(rank <= b);
    return rank;
  }

  void steal() {
    auto ne = dtree_.get_topmost_dominant(dtree_local_bottom_ref_);
    if (!ne.has_value()) {
      common::verbose<2>("Dominant dist_tree node not found");
      return;
    }
    dist_tree::node dominant_node = *ne;
    dist_range steal_range = dominant_node.drange;

    common::verbose<2>("Dominant dist_tree node found: drange=[%f, %f), depth=%d",
                       steal_range.begin(), steal_range.end(), dominant_node.depth());

    auto my_rank = common::topology::my_rank();

    auto begin_rank = steal_range.begin_rank();
    auto end_rank   = steal_range.end_rank();
    if (begin_rank == end_rank) {
      return;
    }

    ITYR_CHECK((begin_rank <= my_rank || my_rank <= end_rank));

    common::verbose<2>("Start work stealing for dominant task group [%f, %f)",
                       steal_range.begin(), steal_range.end());

    auto target_rank = get_random_rank(begin_rank, end_rank);

    common::verbose<2>("Target rank: %d", target_rank);

    if (target_rank != begin_rank) {
      bool success = steal_from_migration_queues(target_rank, dominant_node.depth());
      if (success) {
        return;
      }
    }

    if (target_rank != end_rank || (target_rank == end_rank && steal_range.is_at_end_boundary())) {
      bool success = steal_from_primary_queues(target_rank, dominant_node.depth());
      if (success) {
        return;
      }
    }
  }

  bool steal_from_primary_queues(common::topology::rank_t target_rank, int max_depth) {
    // TODO: quick check for the entire wsqueue array
    for (int d = max_depth; d < primary_wsq_.n_queues(); d++) {
      auto ibd = common::profiler::interval_begin<prof_event_sched_steal>(target_rank);

      if (primary_wsq_.empty(target_rank, d)) {
        common::profiler::interval_end<prof_event_sched_steal>(ibd, false);
        continue;
      }

      if (!primary_wsq_.lock().trylock(target_rank, d)) {
        common::profiler::interval_end<prof_event_sched_steal>(ibd, false);
        continue;
      }

      auto pwe = primary_wsq_.steal_nolock(target_rank, d);
      if (!pwe.has_value()) {
        primary_wsq_.lock().unlock(target_rank, d);
        common::profiler::interval_end<prof_event_sched_steal>(ibd, false);
        continue;
      }

      common::verbose("Steal context frame [%p, %p) from primary wsqueue (depth=%d) on rank %d",
                      pwe->frame_base, reinterpret_cast<std::byte*>(pwe->frame_base) + pwe->frame_size,
                      d, target_rank);

      stack_.direct_copy_from(pwe->frame_base, pwe->frame_size, target_rank);

      primary_wsq_.lock().unlock(target_rank, d);

      common::profiler::interval_end<prof_event_sched_steal>(ibd, true);

      common::profiler::switch_phase<prof_phase_sched_loop, prof_phase_sched_resume_stolen>();

      context_frame* next_cf = reinterpret_cast<context_frame*>(pwe->frame_base);
      suspend([&](context_frame* cf) {
        sched_cf_ = cf;
        context::clear_parent_frame(next_cf);
        resume(next_cf);
      });

      return true;
    }

    common::verbose<2>("Steal failed for primary queues on rank %d", target_rank);
    return false;
  }

  bool steal_from_migration_queues(common::topology::rank_t target_rank, int max_depth) {
    // TODO: quick check for the entire wsqueue array
    for (int d = migration_wsq_.n_queues() - 1; d >= max_depth; d--) {
      auto ibd = common::profiler::interval_begin<prof_event_sched_steal>(target_rank);

      if (migration_wsq_.empty(target_rank, d)) {
        common::profiler::interval_end<prof_event_sched_steal>(ibd, false);
        continue;
      }

      if (!migration_wsq_.lock().trylock(target_rank, d)) {
        common::profiler::interval_end<prof_event_sched_steal>(ibd, false);
        continue;
      }

      auto mwe = migration_wsq_.steal_nolock(target_rank, d);
      if (!mwe.has_value()) {
        migration_wsq_.lock().unlock(target_rank, d);
        common::profiler::interval_end<prof_event_sched_steal>(ibd, false);
        continue;
      }

      if (!mwe->is_continuation) {
        // This task is a new task
        common::verbose("Steal a new task from migration wsqueue (depth=%d) on rank %d",
                        d, target_rank);

        migration_wsq_.lock().unlock(target_rank, d);

        common::profiler::interval_end<prof_event_sched_steal>(ibd, true);

        common::profiler::switch_phase<prof_phase_sched_loop, prof_phase_sched_start_new>();

        suspend([&](context_frame* cf) {
          sched_cf_ = cf;
          start_new_task(mwe->frame_base, mwe->frame_size);
        });

      } else if (mwe->evacuation_ptr) {
        migration_wsq_.lock().unlock(target_rank, d);
        common::profiler::interval_end<prof_event_sched_steal>(ibd, true);
        common::die("unimplenented");

      } else {
        // This task is a continuation on the stack
        common::verbose("Steal context frame [%p, %p) from migration wsqueue (depth=%d) on rank %d",
                        mwe->frame_base, reinterpret_cast<std::byte*>(mwe->frame_base) + mwe->frame_size,
                        d, target_rank);

        stack_.direct_copy_from(mwe->frame_base, mwe->frame_size, target_rank);

        migration_wsq_.lock().unlock(target_rank, d);

        common::profiler::interval_end<prof_event_sched_steal>(ibd, true);

        common::profiler::switch_phase<prof_phase_sched_loop, prof_phase_sched_resume_stolen>();

        suspend([&](context_frame* cf) {
          sched_cf_ = cf;
          context_frame* next_cf = reinterpret_cast<context_frame*>(mwe->frame_base);
          resume(next_cf);
        });
      }

      return true;
    }

    common::verbose<2>("Steal failed for migration queues on rank %d", target_rank);
    return false;
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
    context::jump_to_stack(ss.frame_base, [](void* this_, void* evacuation_ptr, void* frame_base, void* frame_size_) {
      scheduler_adws& this_sched = *reinterpret_cast<scheduler_adws*>(this_);
      std::size_t     frame_size = reinterpret_cast<std::size_t>(frame_size_);
      common::remote_get(this_sched.suspended_thread_allocator_,
                         reinterpret_cast<std::byte*>(frame_base),
                         reinterpret_cast<std::byte*>(evacuation_ptr),
                         frame_size);
      this_sched.suspended_thread_allocator_.deallocate(evacuation_ptr, frame_size);

      context_frame* cf = reinterpret_cast<context_frame*>(frame_base);
      context::clear_parent_frame(cf);
      context::resume(cf);
    }, this, ss.evacuation_ptr, ss.frame_base, reinterpret_cast<void*>(ss.frame_size));
  }

  void resume_sched() {
    cf_top_ = nullptr;
    tls_ = nullptr;
    common::verbose("Resume scheduler context");
    context::resume(sched_cf_);
  }

  void start_new_task(void* task_ptr, std::size_t task_size) {
    root_on_stack([&]() {
      task_general* t = reinterpret_cast<task_general*>(alloca(task_size));

      common::remote_get(suspended_thread_allocator_,
                         reinterpret_cast<std::byte*>(t),
                         reinterpret_cast<std::byte*>(task_ptr),
                         task_size);
      suspended_thread_allocator_.deallocate(task_ptr, task_size);

      t->execute();
    });
  }

  void execute_cross_worker_task(const cross_worker_task& cwt) {
    if (cwt.evacuation_ptr == nullptr) {
      // This task is a new task
      common::verbose("Received a new cross-worker task");
      common::profiler::switch_phase<prof_phase_sched_loop, prof_phase_sched_start_new>();

      suspend([&](context_frame* cf) {
        sched_cf_ = cf;
        start_new_task(cwt.frame_base, cwt.frame_size);
      });

    } else {
      // This task is an evacuated continuation
      common::verbose("Received a continuation of a cross-worker task");
      common::profiler::switch_phase<prof_phase_sched_loop, prof_phase_sched_resume_migrate>();

      suspend([&](context_frame* cf) {
        sched_cf_ = cf;
        resume(suspended_state{cwt.evacuation_ptr, cwt.frame_base, cwt.frame_size});
      });
    }
  }

  void execute_migrated_task(const migration_wsq_entry& mwe) {
    if (!mwe.is_continuation) {
      // This task is a new task
      common::verbose("Popped a new task from local migration queues");
      common::profiler::switch_phase<prof_phase_sched_loop, prof_phase_sched_start_new>();

      suspend([&](context_frame* cf) {
        sched_cf_ = cf;
        start_new_task(mwe.frame_base, mwe.frame_size);
      });

    } else if (mwe.evacuation_ptr) {
      // This task is an evacuated continuation
      /* common::profiler::switch_phase<prof_phase_sched_loop, prof_phase_sched_resume_evac>(); */

      /* suspend([&](context_frame* cf) { */
      /*   sched_cf_ = cf; */
      /*   resume(suspended_state{mwe.evacuation_ptr, mwe.frame_base, mwe.frame_size}); */
      /* }); */
      common::die("unimplenented");

    } else {
      // This task is a continuation on the stack
      common::verbose("Popped a continuation from local migration queues");
      common::profiler::switch_phase<prof_phase_sched_loop, prof_phase_sched_resume_parent>();

      suspend([&](context_frame* cf) {
        sched_cf_ = cf;
        context_frame* next_cf = reinterpret_cast<context_frame*>(mwe.frame_base);
        resume(next_cf);
      });
    }
  }

  std::optional<primary_wsq_entry> pop_from_primary_queues(int depth_from) {
    // TODO: upper bound for depth can be tracked
    for (int d = depth_from; d >= 0; d--) {
      auto pwe = primary_wsq_.pop(d);
      if (pwe.has_value()) {
        return pwe;
      }
    }
    return std::nullopt;
  }

  std::optional<migration_wsq_entry> pop_from_migration_queues() {
    for (int d = 0; d < migration_wsq_.n_queues(); d++) {
      auto mwe = migration_wsq_.pop(d);
      if (mwe.has_value()) {
        return mwe;
      }
    }
    return std::nullopt;
  }

  suspended_state evacuate(context_frame* cf) {
    std::size_t cf_size = reinterpret_cast<uintptr_t>(cf->parent_frame) - reinterpret_cast<uintptr_t>(cf);
    void* evacuation_ptr = suspended_thread_allocator_.allocate(cf_size);
    std::memcpy(evacuation_ptr, cf, cf_size);

    common::verbose("Evacuate suspended thread context [%p, %p) to %p",
                    cf, cf->parent_frame, evacuation_ptr);

    return {evacuation_ptr, cf, cf_size};
  }

  template <typename Fn>
  void root_on_stack(Fn&& fn) {
    // Add a margin of sizeof(context_frame) to the bottom of the stack, because
    // this region can be accessed by the clear_parent_frame() function later
    cf_top_ = reinterpret_cast<context_frame*>(stack_.bottom()) - 1;
    context::call_on_stack(stack_.top(), stack_.size() - sizeof(context_frame),
                           [](void* fn_, void*, void*, void*) {
      Fn fn = *reinterpret_cast<Fn*>(fn_); // copy closure to the new stack frame
      fn();
    }, &fn, nullptr, nullptr, nullptr);
  }

  template <typename CondFn>
  bool should_exit_sched_loop(CondFn&& cond_fn) {
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

  callstack                          stack_;
  oneslot_mailbox<cross_worker_task> cross_worker_mailbox_;
  wsqueue<primary_wsq_entry, false>  primary_wsq_;
  wsqueue<migration_wsq_entry, true> migration_wsq_;
  common::remotable_resource         thread_state_allocator_;
  common::remotable_resource         suspended_thread_allocator_;
  context_frame*                     cf_top_              = nullptr;
  context_frame*                     sched_cf_            = nullptr;
  thread_local_storage*              tls_                 = nullptr;
  MPI_Request                        sched_loop_exit_req_ = MPI_REQUEST_NULL;
  bool                               use_primary_wsq_     = true;
  dist_tree                          dtree_;
  dist_tree::node_ref                dtree_local_bottom_ref_;
};

}
