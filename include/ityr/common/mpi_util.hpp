#pragma once

#include <mpi.h>

#include "ityr/common/util.hpp"

#ifndef ITYR_DEBUG_UCX
#define ITYR_DEBUG_UCX 0
#endif

#if ITYR_DEBUG_UCX
#include <ucs/debug/log_def.h>
#include <sys/time.h>
#include <cstring>
#include <atomic>
#endif

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
            sizeof(T) * count,
            MPI_BYTE,
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
inline void mpi_allgather(const T*    sendbuf,
                          std::size_t sendcount,
                          T*          recvbuf,
                          std::size_t recvcount,
                          MPI_Comm    comm) {
  MPI_Allgather(sendbuf,
                sendcount,
                mpi_type<T>(),
                recvbuf,
                recvcount,
                mpi_type<T>(),
                comm);
}

template <typename T>
inline std::vector<T> mpi_allgather_value(const T& value,
                                          MPI_Comm comm) {
  std::vector<T> result(mpi_comm_size(comm));
  mpi_allgather(&value, 1, result.data(), 1, comm);
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

inline void mpi_make_progress() {
  int flag;
  MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, MPI_STATUS_IGNORE);
}

inline MPI_Comm& mpi_comm_root() {
  static MPI_Comm comm = MPI_COMM_WORLD;
  return comm;
}

#if ITYR_DEBUG_UCX

#define UCS_LOG_TIME_FMT        "[%lu.%06lu]"
#define UCS_LOG_METADATA_FMT    "%17s:%-4u %-4s %-5s %*s"
#define UCS_LOG_PROC_DATA_FMT   "[%s:%-5d:%s]"

#define UCS_LOG_FMT             UCS_LOG_TIME_FMT " " UCS_LOG_PROC_DATA_FMT " " \
                                UCS_LOG_METADATA_FMT "%s\n"

#define UCS_LOG_TIME_ARG(_tv)  (_tv).tv_sec, (_tv).tv_usec

#define UCS_LOG_METADATA_ARG(_short_file, _line, _level, _comp_conf) \
    (_short_file), (_line), (_comp_conf)->name, \
    ucs_log_level_names[_level], 0, ""

#define UCS_LOG_PROC_DATA_ARG() \
    ucs_get_host_name(), ucs_log_get_pid(), ucs_log_get_thread_name()

#define UCS_LOG_COMPACT_ARG(_tv)\
    UCS_LOG_TIME_ARG(_tv), UCS_LOG_PROC_DATA_ARG()

#define UCS_LOG_ARG(_short_file, _line, _level, _comp_conf, _tv, _message) \
    UCS_LOG_TIME_ARG(_tv), UCS_LOG_PROC_DATA_ARG(), \
    UCS_LOG_METADATA_ARG(_short_file, _line, _level, _comp_conf), (_message)

inline const char *ucs_log_level_names[] = {
  [UCS_LOG_LEVEL_FATAL]        = "FATAL",
  [UCS_LOG_LEVEL_ERROR]        = "ERROR",
  [UCS_LOG_LEVEL_WARN]         = "WARN",
  [UCS_LOG_LEVEL_DIAG]         = "DIAG",
  [UCS_LOG_LEVEL_INFO]         = "INFO",
  [UCS_LOG_LEVEL_DEBUG]        = "DEBUG",
  [UCS_LOG_LEVEL_TRACE]        = "TRACE",
  [UCS_LOG_LEVEL_TRACE_REQ]    = "REQ",
  [UCS_LOG_LEVEL_TRACE_DATA]   = "DATA",
  [UCS_LOG_LEVEL_TRACE_ASYNC]  = "ASYNC",
  [UCS_LOG_LEVEL_TRACE_FUNC]   = "FUNC",
  [UCS_LOG_LEVEL_TRACE_POLL]   = "POLL",
  [UCS_LOG_LEVEL_LAST]         = NULL,
  [UCS_LOG_LEVEL_PRINT]        = "PRINT"
};

inline const char* ucs_get_host_name() {
  static char hostname[256] = {0};
  if (*hostname == 0) {
    gethostname(hostname, sizeof(hostname));
    strtok(hostname, ".");
  }
  return hostname;
}

inline int ucs_log_get_pid() {
  static int ucs_log_pid = 0;
  if (ucs_log_pid == 0) {
    return getpid();
  }
  return ucs_log_pid;
}

inline const char* ucs_log_get_thread_name() {
  static thread_local char ucs_log_thread_name[32] = {0};
  static std::atomic<int> ucs_log_thread_count = 0;
  char *name = ucs_log_thread_name;
  uint32_t thread_num;

  if (name[0] == '\0') {
    int thread_num = std::atomic_fetch_add(&ucs_log_thread_count, 1);
    snprintf(ucs_log_thread_name, sizeof(ucs_log_thread_name), "%d", thread_num);
  }

  return name;
}

inline const char* ucs_basename(const char *path) {
  const char *name = strrchr(path, '/');
  return (name == NULL) ? path : name + 1;
}

inline FILE* ityr_ucx_log_fileptr() {
  static std::unique_ptr<char[]> outbuf;
  static std::unique_ptr<FILE, void(*)(FILE*)> outfile(NULL, [](FILE*){});
  std::size_t outbufsize = 1L * 1024 * 1024 * 1024;

  if (outfile == nullptr) {
    outbuf = std::make_unique<char[]>(outbufsize);

    char buf[256];
    snprintf(buf, sizeof(buf), "ityr_ucx.log.%d", mpi_comm_rank(MPI_COMM_WORLD));
    outfile = std::unique_ptr<FILE, void(*)(FILE*)>(fopen(buf, "w"),
        [](FILE* fp) { if (fp) ::fclose(fp); });
    if (outfile == nullptr) {
      perror("fopen");
      die("could not open file %s", buf);
    }

    int ret = setvbuf(outfile.get(), outbuf.get(), _IOFBF, outbufsize);
    if (ret != 0) {
      perror("setvbuf");
      die("setvbuf failed");
    }
  }

  return outfile.get();
}

inline bool ityr_ucx_log_enable(int mode = -1) {
  static bool enabled = false;
  if (mode == 0) {
    enabled = false;
  } else if (mode == 1) {
    enabled = true;
  }
  return enabled;
}

inline void ityr_ucx_log_flush() {
  fflush(ityr_ucx_log_fileptr());
}

inline ucs_log_func_rc_t
ityr_ucx_log_handler(const char *file, unsigned line, const char *function,
                     ucs_log_level_t level,
                     const ucs_log_component_config_t *comp_conf,
                     const char *format, va_list ap) {
  if (!ityr_ucx_log_enable()) {
    return UCS_LOG_FUNC_RC_CONTINUE;
  }

  if (!ucs_log_component_is_enabled(level, comp_conf) &&
      (level != UCS_LOG_LEVEL_PRINT)) {
    return UCS_LOG_FUNC_RC_CONTINUE;
  }

  size_t buffer_size = ucs_log_get_buffer_size();
  char* buf = reinterpret_cast<char*>(alloca(buffer_size + 1));
  buf[buffer_size] = 0;
  vsnprintf(buf, buffer_size, format, ap);

  const char* short_file = ucs_basename(file);
  struct timeval tv;
  gettimeofday(&tv, NULL);

  char* saveptr = "";
  char* log_line = strtok_r(buf, "\n", &saveptr);
  while (log_line != NULL) {
    fprintf(ityr_ucx_log_fileptr(), UCS_LOG_FMT,
        UCS_LOG_ARG(short_file, line, level,
          comp_conf, tv, log_line));
    log_line = strtok_r(NULL, "\n", &saveptr);
  }

  /* flush the log file if the log_level of this message is fatal or error */
  if (level <= UCS_LOG_LEVEL_ERROR) {
    ityr_ucx_log_flush();
  }

  return UCS_LOG_FUNC_RC_CONTINUE;
}
#endif

class mpi_initializer {
public:
  mpi_initializer(MPI_Comm comm) {
    mpi_comm_root() = comm;
    MPI_Initialized(&initialized_outside_);
    if (!initialized_outside_) {
      MPI_Init(nullptr, nullptr);
    }
#if ITYR_DEBUG_UCX
    while (ucs_log_num_handlers() > 0) {
      ucs_log_pop_handler();
    }
    ityr_ucx_log_fileptr();
    ucs_log_push_handler(ityr_ucx_log_handler);
#endif
  }

  ~mpi_initializer() {
#if ITYR_DEBUG_UCX
    ucs_log_pop_handler();
#endif
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
