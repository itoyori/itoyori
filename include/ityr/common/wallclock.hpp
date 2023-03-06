#pragma once

#include <limits>

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/topology.hpp"

namespace ityr::common::wallclock {

class global_clock {
public:
  global_clock()
    : n_sync_round_trips_(getenv_coll("ITYR_GLOBAL_CLOCK_N_SYNC_ROUND_TRIPS", 100, topology::mpicomm())) {
    sync();
  }

  void sync() {
    uint64_t t0 = clock_gettime_ns();
    mpi_barrier(topology::mpicomm());
    do_sync();
    mpi_barrier(topology::mpicomm());
    uint64_t t1 = clock_gettime_ns();
    verbose("Global clock synchronized (offset = %ld ns); took %ld ns", offset_, t1 - t0);
  }

  uint64_t gettime_ns() const {
    return clock_gettime_ns() - offset_;
  }

private:
  void do_sync() {
    // Only the leader of each node involves in clock synchronization
    if (topology::intra_my_rank() == 0) {
      int64_t* offsets = new int64_t[topology::inter_n_ranks()]();

      // uses the reference clock of the node of rank 0
      if (topology::inter_my_rank() == 0) {
        // takes O(n) time, where n = # of nodes
        for (int i = 1; i < topology::inter_n_ranks(); i++) {
          uint64_t min_gap = std::numeric_limits<uint64_t>::max();
          for (int j = 0; j < n_sync_round_trips_; j++) {
            uint64_t t0 = clock_gettime_ns();
            mpi_send_value(t0, i, j, topology::inter_mpicomm());
            uint64_t t1 = mpi_recv_value<uint64_t>(i, j, topology::inter_mpicomm());
            uint64_t t2 = clock_gettime_ns();

            // adopt the fastest communitation
            if (t2 - t0 < min_gap) {
              min_gap = t2 - t0;
              offsets[i] = t1 - static_cast<int64_t>((t0 + t2) / 2);
            }
          }
        }

        // Adjust the offset to begin with t=0
        int64_t begin_time = clock_gettime_ns();
        for (int i = 0; i < topology::inter_n_ranks(); i++) {
          offsets[i] += begin_time;
        }
      } else {
        for (int j = 0; j < n_sync_round_trips_; j++) {
          mpi_recv_value<uint64_t>(0, j, topology::inter_mpicomm());
          uint64_t t1 = clock_gettime_ns();
          mpi_send_value(t1, 0, j, topology::inter_mpicomm());
        }
      }

      offset_ = mpi_scatter_value(offsets, 0, topology::inter_mpicomm());

      delete[] offsets;
    }

    // Share the offset within the node
    offset_ = mpi_bcast_value(offset_, 0, topology::intra_mpicomm());
  }

  int     n_sync_round_trips_;
  int64_t offset_;
};

using instance = singleton<global_clock>;

inline uint64_t gettime_ns() { return instance::get().gettime_ns(); }

}
