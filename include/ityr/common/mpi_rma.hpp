#pragma once

#include <mpi.h>

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/span.hpp"

namespace ityr::common {

inline void mpi_win_flush(int target_rank, MPI_Win win) {
  MPI_Win_flush(target_rank, win);
}

inline void mpi_win_flush_all(MPI_Win win) {
  MPI_Win_flush_all(win);
}

template <typename T>
inline void mpi_get_nb(T*          origin,
                       std::size_t count,
                       int         target_rank,
                       std::size_t target_disp,
                       MPI_Win     win) {
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
  mpi_get_nb(&value, 1, target_rank, target_disp, win);
  mpi_win_flush(target_rank, win);
  return value;
}

template <typename T>
inline void mpi_put_nb(const T*    origin,
                       std::size_t count,
                       int         target_rank,
                       std::size_t target_disp,
                       MPI_Win     win) {
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
  mpi_put_nb(&value, 1, target_rank, target_disp, win);
  mpi_win_flush(target_rank, win);
}

template <typename T>
inline void mpi_atomic_faa_nb(const T*    origin,
                              T*          result,
                              int         target_rank,
                              std::size_t target_disp,
                              MPI_Win     win) {
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
    MPI_Win_lock_all(0, win_);
  }
  mpi_win_manager(MPI_Comm comm, std::size_t size) {
    MPI_Win_allocate(size, 1, MPI_INFO_NULL, comm, &baseptr_, &win_);
    MPI_Win_lock_all(0, win_);
  }
  mpi_win_manager(MPI_Comm comm, void* baseptr, std::size_t size) : baseptr_(baseptr) {
    MPI_Win_create(baseptr,
                   size,
                   1,
                   MPI_INFO_NULL,
                   comm,
                   &win_);
    MPI_Win_lock_all(0, win_);
  }

  ~mpi_win_manager() {
    if (win_ != MPI_WIN_NULL) {
      MPI_Win_unlock_all(win_);
      MPI_Win_free(&win_);
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
  MPI_Win win_     = MPI_WIN_NULL;
  void*   baseptr_ = nullptr;
};

template <typename T>
class mpi_win_manager {
public:
  mpi_win_manager() {}
  mpi_win_manager(MPI_Comm comm) : win_(comm), comm_(comm) {}
  mpi_win_manager(MPI_Comm comm, std::size_t count)
    : win_(comm, sizeof(T) * count), comm_(comm), local_buf_(init_local_buf(count)) {}
  mpi_win_manager(MPI_Comm comm, T* baseptr, std::size_t count)
    : win_(comm, baseptr, sizeof(T) * count), comm_(comm) {} // no initialization for local buf?

  ~mpi_win_manager() {
    if (win_.win() != MPI_WIN_NULL) {
      destroy_local_buf();
    }
  }

  mpi_win_manager(const mpi_win_manager&) = delete;
  mpi_win_manager& operator=(const mpi_win_manager&) = delete;

  mpi_win_manager(mpi_win_manager&& wm) = default;
  mpi_win_manager& operator=(mpi_win_manager&& wm) = default;

  MPI_Win win() const { return win_.win(); }
  T* baseptr() const { return reinterpret_cast<T*>(win_.baseptr()); }

  span<T> local_buf() const { return local_buf_; }

private:
  span<T> init_local_buf(std::size_t count) const {
    T* local_base = baseptr();
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

  const mpi_win_manager<void> win_;
  const MPI_Comm              comm_ = MPI_COMM_NULL;
  const span<T>               local_buf_;
};

}