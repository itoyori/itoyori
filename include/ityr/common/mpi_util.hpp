#pragma once

#include <mpi.h>

#include "ityr/common/util.hpp"

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
inline void mpi_send(const T*    buf,
                     std::size_t count,
                     int         target_rank,
                     int         tag,
                     MPI_Comm    comm) {
  MPI_Send(buf,
           sizeof(T) * count,
           MPI_BYTE,
           target_rank,
           tag,
           comm);
}

template <typename T>
inline MPI_Request mpi_isend(const T*    buf,
                             std::size_t count,
                             int         target_rank,
                             int         tag,
                             MPI_Comm    comm) {
  MPI_Request req;
  MPI_Isend(buf,
            sizeof(T) * count,
            MPI_BYTE,
            target_rank,
            tag,
            comm,
            &req);
  return req;
}

template <typename T>
inline void mpi_send_value(const T& value,
                           int      target_rank,
                           int      tag,
                           MPI_Comm comm) {
  mpi_send(&value, 1, target_rank, tag, comm);
}

template <typename T>
inline void mpi_recv(T*          buf,
                     std::size_t count,
                     int         target_rank,
                     int         tag,
                     MPI_Comm    comm) {
  MPI_Recv(buf,
           sizeof(T) * count,
           MPI_BYTE,
           target_rank,
           tag,
           comm,
           MPI_STATUS_IGNORE);
}

template <typename T>
inline MPI_Request mpi_irecv(T*          buf,
                             std::size_t count,
                             int         target_rank,
                             int         tag,
                             MPI_Comm    comm) {
  MPI_Request req;
  MPI_Irecv(buf,
            sizeof(T) * count,
            MPI_BYTE,
            target_rank,
            tag,
            comm,
            &req);
  return req;
}

template <typename T>
inline T mpi_recv_value(int      target_rank,
                        int      tag,
                        MPI_Comm comm) {
  T result {};
  mpi_recv(&result, 1, target_rank, tag, comm);
  return result;
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

template <typename T>
inline void mpi_scatter(const T*    sendbuf,
                        T*          recvbuf,
                        std::size_t count,
                        int         root_rank,
                        MPI_Comm    comm) {
  MPI_Scatter(sendbuf,
              count,
              mpi_type<T>(),
              recvbuf,
              count,
              mpi_type<T>(),
              root_rank,
              comm);
}

template <typename T>
inline T mpi_scatter_value(const T* sendbuf,
                           int      root_rank,
                           MPI_Comm comm) {
  T result {};
  mpi_scatter(sendbuf, &result, 1, root_rank, comm);
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

inline MPI_Comm& mpi_comm_root() {
  static MPI_Comm comm = MPI_COMM_WORLD;
  return comm;
}

class mpi_initializer {
public:
  mpi_initializer(MPI_Comm comm) {
    mpi_comm_root() = comm;
    MPI_Initialized(&initialized_outside_);
    if (!initialized_outside_) {
      MPI_Init(nullptr, nullptr);
    }
  }

  ~mpi_initializer() {
    if (!initialized_outside_) {
      MPI_Finalize();
    }
  }

  mpi_initializer(const mpi_initializer&) = delete;
  mpi_initializer& operator=(const mpi_initializer&) = delete;

  mpi_initializer(mpi_initializer&&) = delete;
  mpi_initializer& operator=(mpi_initializer&&) = delete;

private:
  int initialized_outside_ = 1;
};

template <typename T>
inline T getenv_coll(const std::string& env_var, T default_val) {
  MPI_Comm comm = mpi_comm_root();

  int rank = mpi_comm_rank(comm);
  T val = default_val;

  if (rank == 0) {
    val = getenv_with_default(env_var.c_str(), default_val);
  }

  return mpi_bcast_value(val, 0, comm);
}

}
