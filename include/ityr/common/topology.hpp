#pragma once

#include <vector>
#include <mpi.h>

#include "ityr/common/util.hpp"

namespace ityr::common {

class topology {
public:
  using rank_t = int;

  topology(MPI_Comm comm, bool shared_memory_enabled = true) :
    cg_global_(comm, false),
    shared_memory_enabled_(get_env("ITYR_ENABLE_SHARED_MEMORY", shared_memory_enabled, my_rank())),
    cg_intra_(create_intra_comm(), shared_memory_enabled_),
    cg_inter_(create_inter_comm(), shared_memory_enabled_),
    process_map_(create_process_map()),
    intra2global_rank_(create_intra2global_rank()) {}

  MPI_Comm mpicomm() const { return cg_global_.mpicomm; }
  rank_t   my_rank() const { return cg_global_.my_rank; }
  rank_t   n_ranks() const { return cg_global_.n_ranks; }

  MPI_Comm intra_mpicomm() const { return cg_intra_.mpicomm; }
  rank_t   intra_my_rank() const { return cg_intra_.my_rank; }
  rank_t   intra_n_ranks() const { return cg_intra_.n_ranks; }

  MPI_Comm inter_mpicomm() const { return cg_inter_.mpicomm; }
  rank_t   inter_my_rank() const { return cg_inter_.my_rank; }
  rank_t   inter_n_ranks() const { return cg_inter_.n_ranks; }

  rank_t intra_rank(rank_t global_rank) const {
    ITYR_CHECK(0 <= global_rank);
    ITYR_CHECK(global_rank < n_ranks());
    return process_map_[global_rank].intra_rank;
  }

  rank_t inter_rank(rank_t global_rank) const {
    ITYR_CHECK(0 <= global_rank);
    ITYR_CHECK(global_rank < n_ranks());
    return process_map_[global_rank].inter_rank;
  }

  rank_t intra2global_rank(rank_t intra_rank) const {
    ITYR_CHECK(0 <= intra_rank);
    ITYR_CHECK(intra_rank < intra_n_ranks());
    return intra2global_rank_[intra_rank];
  }

  bool is_locally_accessible(topology::rank_t target_global_rank) const {
    return inter_rank(target_global_rank) == inter_my_rank();
  }

private:
  struct comm_group {
    rank_t   my_rank  = -1;
    rank_t   n_ranks  = -1;
    MPI_Comm mpicomm  = MPI_COMM_NULL;
    bool     own_comm = false;

    comm_group(MPI_Comm comm, bool own) : mpicomm(comm), own_comm(own) {
      MPI_Comm_rank(mpicomm, &my_rank);
      MPI_Comm_size(mpicomm, &n_ranks);
      ITYR_CHECK(my_rank != -1);
      ITYR_CHECK(n_ranks != -1);
    }

    ~comm_group() {
      if (own_comm) {
        MPI_Comm_free(&mpicomm);
      }
    }
  };

  struct process_map_entry {
    rank_t intra_rank;
    rank_t inter_rank;
  };

  MPI_Comm create_intra_comm() {
    if (shared_memory_enabled_) {
      MPI_Comm h;
      MPI_Comm_split_type(mpicomm(), MPI_COMM_TYPE_SHARED, my_rank(), MPI_INFO_NULL, &h);
      return h;
    } else {
      return MPI_COMM_SELF;
    }
  }

  MPI_Comm create_inter_comm() {
    if (shared_memory_enabled_) {
      MPI_Comm h;
      MPI_Comm_split(mpicomm(), intra_my_rank(), my_rank(), &h);
      return h;
    } else {
      return mpicomm();
    }
  }

  std::vector<process_map_entry> create_process_map() {
    process_map_entry my_entry {intra_my_rank(), inter_my_rank()};
    std::vector<process_map_entry> ret(n_ranks());
    MPI_Allgather(&my_entry,
                  sizeof(process_map_entry),
                  MPI_BYTE,
                  ret.data(),
                  sizeof(process_map_entry),
                  MPI_BYTE,
                  mpicomm());
    return ret;
  }

  std::vector<rank_t> create_intra2global_rank() {
    std::vector<rank_t> ret;
    for (rank_t i = 0; i < n_ranks(); i++) {
      if (process_map_[i].inter_rank == inter_my_rank()) {
        ret.push_back(i);
      }
    }
    ITYR_CHECK(ret.size() == std::size_t(intra_n_ranks()));
    return ret;
  }

  const comm_group                     cg_global_;
  const bool                           shared_memory_enabled_;
  const comm_group                     cg_intra_;
  const comm_group                     cg_inter_;
  const std::vector<process_map_entry> process_map_; // global_rank -> (intra, inter rank)
  const std::vector<rank_t>            intra2global_rank_;
};

}
