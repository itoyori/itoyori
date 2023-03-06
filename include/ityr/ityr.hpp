#pragma once

#include <mpi.h>

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/ito/ito.hpp"
#include "ityr/pattern/iterator.hpp"
#include "ityr/pattern/parallel_loop.hpp"

namespace ityr {

class ityr {
public:
  ityr(MPI_Comm comm)
    : mpi_initialized_outside_(init_mpi_if_needed()),
      topo_(comm, false) {}

  ~ityr() {
    if (!mpi_initialized_outside_) {
      MPI_Finalize();
    }
  }

  const common::topology& topology() const { return topo_; }

private:
  bool init_mpi_if_needed() {
    int initialized_outside;
    MPI_Initialized(&initialized_outside);
    if (!initialized_outside) {
      MPI_Init(nullptr, nullptr);
    }
    return initialized_outside;
  }

  bool             mpi_initialized_outside_;
  common::topology topo_;
};

inline std::optional<ityr>& get_instance_() {
  static std::optional<ityr> instance;
  return instance;
}

inline ityr& get_instance() {
  ITYR_CHECK(get_instance_().has_value());
  return *get_instance_();
}

inline void init(MPI_Comm comm = MPI_COMM_WORLD) {
  ITYR_CHECK(!get_instance_().has_value());
  get_instance_().emplace(comm);
  ito::init(get_instance().topology());
}

inline void fini() {
  ITYR_CHECK(get_instance_().has_value());
  ito::fini();
  get_instance_().reset();
}

inline common::topology::rank_t my_rank() {
  return get_instance().topology().my_rank();
}

inline common::topology::rank_t n_ranks() {
  return get_instance().topology().n_ranks();
}

inline bool is_master() {
  return my_rank() == 0;
}

inline void barrier() {
  common::mpi_barrier(get_instance().topology().mpicomm());
}

}
