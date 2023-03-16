#pragma once

#include <vector>

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/options.hpp"

namespace ityr::common::topology {

using rank_t = int;

class topology {
public:
  topology() : topology(MPI_COMM_WORLD) {}
  topology(MPI_Comm comm)
    : enable_shared_memory_(enable_shared_memory_option::value()),
      cg_global_(comm, false),
      cg_intra_(create_intra_comm(), enable_shared_memory_),
      cg_inter_(create_inter_comm(), enable_shared_memory_),
      process_map_(create_process_map()),
      intra2global_rank_(create_intra2global_rank()) {}

  topology(const topology&) = delete;
  topology& operator=(const topology&) = delete;

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

  bool is_locally_accessible(rank_t target_global_rank) const {
    return inter_rank(target_global_rank) == inter_my_rank();
  }

private:
  struct comm_group {
    rank_t   my_rank;
    rank_t   n_ranks;
    MPI_Comm mpicomm  = MPI_COMM_NULL;
    bool     own_comm = false;

    comm_group(MPI_Comm comm, bool own)
      : my_rank(mpi_comm_rank(comm)), n_ranks(mpi_comm_size(comm)),
        mpicomm(comm), own_comm(own) {}

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
    if (enable_shared_memory_) {
      MPI_Comm h;
      MPI_Comm_split_type(mpicomm(), MPI_COMM_TYPE_SHARED, my_rank(), MPI_INFO_NULL, &h);
      return h;
    } else {
      return MPI_COMM_SELF;
    }
  }

  MPI_Comm create_inter_comm() {
    if (enable_shared_memory_) {
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

  bool                           enable_shared_memory_;
  comm_group                     cg_global_;
  comm_group                     cg_intra_;
  comm_group                     cg_inter_;
  std::vector<process_map_entry> process_map_; // global_rank -> (intra, inter rank)
  std::vector<rank_t>            intra2global_rank_;
};

using instance = singleton<topology>;

inline MPI_Comm mpicomm() { return instance::get().mpicomm(); }
inline rank_t   my_rank() { return instance::get().my_rank(); }
inline rank_t   n_ranks() { return instance::get().n_ranks(); }

inline MPI_Comm intra_mpicomm() { return instance::get().intra_mpicomm(); }
inline rank_t   intra_my_rank() { return instance::get().intra_my_rank(); }
inline rank_t   intra_n_ranks() { return instance::get().intra_n_ranks(); }

inline MPI_Comm inter_mpicomm() { return instance::get().inter_mpicomm(); }
inline rank_t   inter_my_rank() { return instance::get().inter_my_rank(); }
inline rank_t   inter_n_ranks() { return instance::get().inter_n_ranks(); }

inline rank_t intra_rank(rank_t global_rank) { return instance::get().intra_rank(global_rank); };
inline rank_t inter_rank(rank_t global_rank) { return instance::get().inter_rank(global_rank); };

inline rank_t intra2global_rank(rank_t intra_rank) { return instance::get().intra2global_rank(intra_rank); }

inline bool is_locally_accessible(rank_t target_global_rank) { return instance::get().is_locally_accessible(target_global_rank); };

}
