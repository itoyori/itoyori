#pragma once

#include <mutex>

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/span.hpp"
#include "ityr/common/options.hpp"
#include "ityr/common/profiler.hpp"
#include "ityr/common/prof_events.hpp"

namespace ityr::common {

inline void mpi_win_flush(int target_rank, MPI_Win win) {
  ITYR_PROFILER_RECORD(prof_event_rma_flush);
  MPI_Win_flush(target_rank, win);
}

inline void mpi_win_flush_all(MPI_Win win) {
  ITYR_PROFILER_RECORD(prof_event_rma_flush);
  MPI_Win_flush_all(win);
}

template <typename T>
inline void mpi_get_nb(T*          origin,
                       std::size_t count,
                       int         target_rank,
                       std::size_t target_disp,
                       MPI_Win     win) {
  ITYR_PROFILER_RECORD(prof_event_rma_get, target_rank);
  ITYR_CHECK(win != MPI_WIN_NULL);
  MPI_Get(origin,
          sizeof(T) * count,
          MPI_BYTE,
          target_rank,
          target_disp,
          sizeof(T) * count,
          MPI_BYTE,
          win);
}

template <typename T>
inline void mpi_get(T*          origin,
                    std::size_t count,
                    int         target_rank,
                    std::size_t target_disp,
                    MPI_Win     win) {
  mpi_get_nb(origin, count, target_rank, target_disp, win);
  mpi_win_flush(target_rank, win);
}

template <typename T>
inline MPI_Request mpi_rget(T*          origin,
                            std::size_t count,
                            int         target_rank,
                            std::size_t target_disp,
                            MPI_Win     win) {
  ITYR_CHECK(win != MPI_WIN_NULL);
  MPI_Request req;
  MPI_Rget(origin,
           sizeof(T) * count,
           MPI_BYTE,
           target_rank,
           target_disp,
           sizeof(T) * count,
           MPI_BYTE,
           win,
           &req);
  return req;
}

template <typename T>
inline T mpi_get_value(int         target_rank,
                       std::size_t target_disp,
                       MPI_Win     win) {
  T value;
  mpi_get(&value, 1, target_rank, target_disp, win);
  return value;
}

template <typename T>
inline void mpi_put_nb(const T*    origin,
                       std::size_t count,
                       int         target_rank,
                       std::size_t target_disp,
                       MPI_Win     win) {
  ITYR_PROFILER_RECORD(prof_event_rma_put, target_rank);
  ITYR_CHECK(win != MPI_WIN_NULL);
  MPI_Put(origin,
          sizeof(T) * count,
          MPI_BYTE,
          target_rank,
          target_disp,
          sizeof(T) * count,
          MPI_BYTE,
          win);
}

template <typename T>
inline void mpi_put(const T*    origin,
                    std::size_t count,
                    int         target_rank,
                    std::size_t target_disp,
                    MPI_Win     win) {
  mpi_put_nb(origin, count, target_rank, target_disp, win);
  mpi_win_flush(target_rank, win);
}

template <typename T>
inline MPI_Request mpi_rput(const T*    origin,
                            std::size_t count,
                            int         target_rank,
                            std::size_t target_disp,
                            MPI_Win     win) {
  ITYR_CHECK(win != MPI_WIN_NULL);
  MPI_Request req;
  MPI_Rput(origin,
           sizeof(T) * count,
           MPI_BYTE,
           target_rank,
           target_disp,
           sizeof(T) * count,
           MPI_BYTE,
           win,
           &req);
  return req;
}

template <typename T>
inline void mpi_put_value(const T&    value,
                          int         target_rank,
                          std::size_t target_disp,
                          MPI_Win     win) {
  mpi_put(&value, 1, target_rank, target_disp, win);
}

template <typename T>
inline void mpi_atomic_faa_nb(const T*    origin,
                              T*          result,
                              int         target_rank,
                              std::size_t target_disp,
                              MPI_Win     win) {
  ITYR_PROFILER_RECORD(prof_event_rma_atomic_faa, target_rank);
  ITYR_CHECK(win != MPI_WIN_NULL);
  MPI_Fetch_and_op(origin,
                   result,
                   mpi_type<T>(),
                   target_rank,
                   target_disp,
                   MPI_SUM,
                   win);
}

template <typename T>
inline T mpi_atomic_faa_value(const T&    value,
                              int         target_rank,
                              std::size_t target_disp,
                              MPI_Win     win) {
  T result;
  mpi_atomic_faa_nb(&value, &result, target_rank, target_disp, win);
  mpi_win_flush(target_rank, win);
  return result;
}

template <typename T>
inline void mpi_atomic_cas_nb(const T*    origin,
                              const T*    compare,
                              T*          result,
                              int         target_rank,
                              std::size_t target_disp,
                              MPI_Win     win) {
  ITYR_PROFILER_RECORD(prof_event_rma_atomic_cas, target_rank);
  ITYR_CHECK(win != MPI_WIN_NULL);
  MPI_Compare_and_swap(origin,
                       compare,
                       result,
                       mpi_type<T>(),
                       target_rank,
                       target_disp,
                       win);
}

template <typename T>
inline T mpi_atomic_cas_value(const T&    value,
                              const T&    compare,
                              int         target_rank,
                              std::size_t target_disp,
                              MPI_Win     win) {
  T result;
  mpi_atomic_cas_nb(&value, &compare, &result, target_rank, target_disp, win);
  mpi_win_flush(target_rank, win);
  return result;
}

template <typename T>
inline void mpi_atomic_get_nb(T*          origin,
                              int         target_rank,
                              std::size_t target_disp,
                              MPI_Win     win) {
  ITYR_PROFILER_RECORD(prof_event_rma_atomic_get, target_rank);
  ITYR_CHECK(win != MPI_WIN_NULL);
  MPI_Fetch_and_op(nullptr,
                   origin,
                   mpi_type<T>(),
                   target_rank,
                   target_disp,
                   MPI_NO_OP,
                   win);
}

template <typename T>
inline T mpi_atomic_get_value(int         target_rank,
                              std::size_t target_disp,
                              MPI_Win     win) {
  T result;
  mpi_atomic_get_nb(&result, target_rank, target_disp, win);
  mpi_win_flush(target_rank, win);
  return result;
}

template <typename T>
inline void mpi_atomic_put_nb(const T*    origin,
                              T*          result,
                              int         target_rank,
                              std::size_t target_disp,
                              MPI_Win     win) {
  ITYR_PROFILER_RECORD(prof_event_rma_atomic_put, target_rank);
  ITYR_CHECK(win != MPI_WIN_NULL);
  MPI_Fetch_and_op(origin,
                   result,
                   mpi_type<T>(),
                   target_rank,
                   target_disp,
                   MPI_REPLACE,
                   win);
}

template <typename T>
inline T mpi_atomic_put_value(const T&    value,
                              int         target_rank,
                              std::size_t target_disp,
                              MPI_Win     win) {
  T result;
  mpi_atomic_put_nb(&value, &result, target_rank, target_disp, win);
  mpi_win_flush(target_rank, win);
  return result;
}

template <typename T>
class mpi_win_manager;

template <>
class mpi_win_manager<void> {
public:
  mpi_win_manager() {}
  mpi_win_manager(MPI_Comm comm) {
    MPI_Win_create_dynamic(MPI_INFO_NULL, comm, &win_);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win_);
    wireup(comm);
  }
  mpi_win_manager(MPI_Comm comm, std::size_t size, std::size_t alignment = alignof(max_align_t)) {
    if (rma_use_mpi_win_allocate::value()) {
      // TODO: handle alignment
      MPI_Win_allocate(size, 1, MPI_INFO_NULL, comm, &baseptr_, &win_);
    } else {
      // In Fujitsu MPI, a large communication latency was observed only when we used
      // MPI_Win_allocate, and here is a workaround for it.
      baseptr_ = std::aligned_alloc(alignment, size);
      MPI_Win_create(baseptr_, size, 1, MPI_INFO_NULL, comm, &win_);
    }
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win_);
    wireup(comm);
  }
  mpi_win_manager(MPI_Comm comm, void* baseptr, std::size_t size) : baseptr_(baseptr) {
    MPI_Win_create(baseptr,
                   size,
                   1,
                   MPI_INFO_NULL,
                   comm,
                   &win_);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, win_);
    wireup(comm);
  }

  ~mpi_win_manager() {
    if (win_ != MPI_WIN_NULL) {
      MPI_Win_unlock_all(win_);
      MPI_Win_free(&win_);
      // TODO: free baseptr_ when ITYR_RMA_USE_MPI_WIN_ALLOCATE=false and the user
      // did not provide a buffer (or remove the option)
    }
  }

  mpi_win_manager(const mpi_win_manager&) = delete;
  mpi_win_manager& operator=(const mpi_win_manager&) = delete;

  mpi_win_manager(mpi_win_manager&& wm) noexcept : win_(wm.win_) { wm.win_ = MPI_WIN_NULL; }
  mpi_win_manager& operator=(mpi_win_manager&& wm) noexcept {
    this->~mpi_win_manager();
    this->win_ = wm.win_;
    wm.win_ = MPI_WIN_NULL;
    return *this;
  }

  MPI_Win win() const { return win_; }
  void* baseptr() const { return baseptr_; }

private:
  void wireup(MPI_Comm comm) {
    static std::once_flag flag;
    std::call_once(flag, [&]() {
      // Invoke wireup routines in the internal of MPI, assuming that this is the first
      // one-sided communication since MPI_Init. MPI_MODE_NOCHECK will not involve communication.
      int my_rank = mpi_comm_rank(comm);
      int n_ranks = mpi_comm_size(comm);
      for (int i = 1; i <= n_ranks / 2; i++) {
        int target_rank = (my_rank + i) % n_ranks;
        mpi_get_value<char>(target_rank, 0, win_);
      }
    });
  }

  MPI_Win win_     = MPI_WIN_NULL;
  void*   baseptr_ = nullptr;
};

// This value should be larger than a cacheline size, because otherwise
// the buffers on different processes may be allocated to the same cacheline,
// which can cause unintended cache misses (false sharing).
// TODO: use hardware_destructive_interference_size?
inline constexpr std::size_t mpi_win_size_min = 1024;

template <typename T>
class mpi_win_manager {
public:
  mpi_win_manager() {}
  mpi_win_manager(MPI_Comm comm)
    : win_(comm),
      comm_(comm) {}
  mpi_win_manager(MPI_Comm comm, std::size_t count)
    : win_(comm, round_up_pow2(sizeof(T) * count, mpi_win_size_min), alignof(T)),
      comm_(comm),
      local_buf_(init_local_buf(count)) {}
  mpi_win_manager(MPI_Comm comm, T* baseptr, std::size_t count)
    : win_(comm, baseptr, sizeof(T) * count),
      comm_(comm) {}

  ~mpi_win_manager() {
    if (win_.win() != MPI_WIN_NULL) {
      destroy_local_buf();
    }
  }

  mpi_win_manager(const mpi_win_manager&) = delete;
  mpi_win_manager& operator=(const mpi_win_manager&) = delete;

  mpi_win_manager(mpi_win_manager&&) = default;
  mpi_win_manager& operator=(mpi_win_manager&&) = default;

  MPI_Win win() const { return win_.win(); }
  T* baseptr() const { return reinterpret_cast<T*>(win_.baseptr()); }

  span<T> local_buf() const { return local_buf_; }

private:
  span<T> init_local_buf(std::size_t count) const {
    T* local_base = baseptr();
    ITYR_REQUIRE(reinterpret_cast<uintptr_t>(local_base) % alignof(T) == 0);

    for (std::size_t i = 0; i < count; i++) {
      new (local_base + i) T();
    }
    mpi_barrier(comm_);
    return span<T>{local_base, count};
  }

  void destroy_local_buf() const {
    if (!local_buf_.empty()) {
      mpi_barrier(comm_);
      std::destroy(local_buf_.begin(), local_buf_.end());
    }
  }

  mpi_win_manager<void> win_;
  MPI_Comm              comm_;
  span<T>               local_buf_;
};

}
