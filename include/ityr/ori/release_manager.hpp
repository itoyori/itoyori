#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/mpi_rma.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/ori/util.hpp"
#include "ityr/ori/options.hpp"

namespace ityr::ori {

class release_manager {
public:
  release_manager()
    : win_(common::topology::mpicomm(), 1),
      remote_epochs_(common::topology::n_ranks(), 1),
      check_interval_(lazy_release_check_interval_option::value()),
      make_mpi_progress_(lazy_release_make_mpi_progress_option::value()) {}

  using epoch_t = uint64_t;

  struct release_handler {
    common::topology::rank_t target_rank    = 0;
    epoch_t                  required_epoch = 0;
  };

  MPI_Win win() const { return win_.win(); }

  epoch_t current_epoch() const { return data().current_epoch; }

  void increment_epoch() {
    data().current_epoch++;
  }

  release_handler get_release_handler() const {
    return {common::topology::my_rank(), current_epoch() + 1};
  }

  release_handler get_dummy_handler() const {
    return {0, 0};
  }

  void ensure_released(const release_handler& rh) {
    if (release_needed(rh)) {
      bool request_done = false;
      while (remote_epochs_[rh.target_rank] < rh.required_epoch) {
        if (!request_done) {
          request_release(rh.target_rank, rh.required_epoch);
          request_done = true;
        }
        usleep(check_interval_);
        if (make_mpi_progress_) {
          common::mpi_make_progress();
        }
        remote_epochs_[rh.target_rank] = get_remote_epoch(rh.target_rank);
      }
    }
  }

  bool release_requested() const {
    return data().requested_epoch > current_epoch();
  }

private:
  struct rma_data {
    epoch_t current_epoch   = 1;
    epoch_t requested_epoch = 1;
  };

  const rma_data& data() const { return *win_.baseptr(); }
  rma_data& data() { return *win_.baseptr(); }

  bool release_needed(const release_handler& rh) const {
    return rh.target_rank != common::topology::my_rank() && rh.required_epoch > 0;
  }

  epoch_t get_remote_epoch(common::topology::rank_t target_rank) const {
    return common::mpi_get_value<epoch_t>(target_rank, offsetof(rma_data, current_epoch), win());
  }

  void request_release(common::topology::rank_t target_rank,
                       epoch_t                  requested_epoch) const {
    // FIXME: As MPI_Fetch_and_op + MPI_MAX seems not offloaded to RDMA NICs, currently
    // MPI_Compare_and_swap is used instead.
    epoch_t remote_epoch = remote_epochs_[target_rank];
    while (remote_epoch < requested_epoch) {
      epoch_t result =
        common::mpi_atomic_cas_value(requested_epoch, remote_epoch, target_rank,
                                     offsetof(rma_data, requested_epoch), win());
      if (result == remote_epoch) {
        break; // success
      } else {
        remote_epoch = result;
      }
    }
  }

  common::mpi_win_manager<rma_data> win_;
  std::vector<epoch_t>              remote_epochs_;
  int                               check_interval_;
  bool                              make_mpi_progress_;
};

}
