#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/ito/ito.hpp"
#include "ityr/pattern/iterator.hpp"
#include "ityr/pattern/parallel_loop.hpp"

namespace ityr {

class ityr {
public:
  ityr(MPI_Comm comm)
    : topo_(comm),
      ito_(comm) {}

private:
  common::mpi_initializer                                    mi_;
  common::singleton_initializer<common::topology::instance>  topo_;
  common::singleton_initializer<ito::instance>               ito_;
};

using instance = common::singleton<ityr>;

inline void init(MPI_Comm comm = MPI_COMM_WORLD) { instance::init(comm); }
inline void fini()                               { instance::fini();     }

inline common::topology::rank_t my_rank() {
  return common::topology::my_rank();
}

inline common::topology::rank_t n_ranks() {
  return common::topology::n_ranks();
}

inline bool is_master() {
  return my_rank() == 0;
}

inline void barrier() {
  common::mpi_barrier(common::topology::mpicomm());
}

}
