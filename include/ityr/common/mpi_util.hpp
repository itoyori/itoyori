#pragma once

#include <mpi.h>

#include "ityr/common/util.hpp"
#include "ityr/common/span.hpp"

namespace ityr::common {

template <typename T> inline MPI_Datatype mpi_type();
template <>           inline MPI_Datatype mpi_type<int>()           { return MPI_INT;               }
template <>           inline MPI_Datatype mpi_type<unsigned int>()  { return MPI_UNSIGNED;          }
template <>           inline MPI_Datatype mpi_type<long>()          { return MPI_LONG;              }
template <>           inline MPI_Datatype mpi_type<unsigned long>() { return MPI_UNSIGNED_LONG;     }
template <>           inline MPI_Datatype mpi_type<bool>()          { return MPI_CXX_BOOL;          }
template <>           inline MPI_Datatype mpi_type<void*>()         { return mpi_type<uintptr_t>(); }

inline int mpi_comm_rank(MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  ITYR_CHECK(rank >= 0);
  return rank;
}

inline int mpi_comm_size(MPI_Comm comm) {
  int size;
  MPI_Comm_size(comm, &size);
  ITYR_CHECK(size >= 0);
  return size;
}

inline void mpi_barrier(MPI_Comm comm) {
  MPI_Barrier(comm);
}

inline MPI_Request mpi_ibarrier(MPI_Comm comm) {
  MPI_Request req;
  MPI_Ibarrier(comm, &req);
  return req;
}

template <typename T>
inline void mpi_bcast(T*          buf,
                      std::size_t count,
                      int         root_rank,
                      MPI_Comm    comm) {
  MPI_Bcast(buf,
            count,
            mpi_type<T>(),
            root_rank,
            comm);
}

template <typename T>
inline T mpi_bcast_value(const T& value,
                         int      root_rank,
                         MPI_Comm comm) {
  T result = value;
  mpi_bcast(&result, 1, root_rank, comm);
  return result;
}

template <typename T>
inline void mpi_reduce(const T*    sendbuf,
                       T*          recvbuf,
                       std::size_t count,
                       int         root_rank,
                       MPI_Comm    comm,
                       MPI_Op      op = MPI_SUM) {
  MPI_Reduce(sendbuf,
             recvbuf,
             count,
             mpi_type<T>(),
             op,
             root_rank,
             comm);
}

template <typename T>
inline T mpi_reduce_value(const T& value,
                          int      root_rank,
                          MPI_Comm comm,
                          MPI_Op   op = MPI_SUM) {
  T result;
  mpi_reduce(&value, &result, 1, root_rank, comm, op);
  return result;
}

template <typename T>
inline void mpi_allreduce(const T*    sendbuf,
                          T*          recvbuf,
                          std::size_t count,
                          MPI_Comm    comm,
                          MPI_Op      op = MPI_SUM) {
  MPI_Allreduce(sendbuf,
                recvbuf,
                count,
                mpi_type<T>(),
                op,
                comm);
}

template <typename T>
inline T mpi_allreduce_value(const T& value,
                             MPI_Comm comm,
                             MPI_Op   op = MPI_SUM) {
  T result;
  mpi_allreduce(&value, &result, 1, comm, op);
  return result;
}

inline void mpi_wait(MPI_Request& req) {
  MPI_Wait(&req, MPI_STATUS_IGNORE);
}

inline bool mpi_test(MPI_Request& req) {
  int flag;
  MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
  return flag;
}

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

template <typename T>
inline T getenv_coll(const char* env_var, T default_val, MPI_Comm comm) {
  static bool print_env = getenv_with_default("ITYR_PRINT_ENV", false);

  int rank = mpi_comm_rank(comm);
  T val = default_val;

  if (rank == 0) {
    val = getenv_with_default(env_var, default_val);
    if (print_env) {
      std::cout << env_var << " = " << val << std::endl;
    }
  }

  return mpi_bcast_value(val, 0, comm);
}

}
