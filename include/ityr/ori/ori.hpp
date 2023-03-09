#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/wallclock.hpp"
#include "ityr/common/profiler.hpp"
#include "ityr/common/prof_events.hpp"
#include "ityr/ori/core.hpp"
#include "ityr/ori/global_ptr.hpp"

namespace ityr::ori {

class ori {
public:
  ori(MPI_Comm comm)
    : topo_(comm) {}

private:
  common::mpi_initializer                                    mi_;
  common::singleton_initializer<common::topology::instance>  topo_;
  common::singleton_initializer<common::wallclock::instance> clock_;
  common::singleton_initializer<common::profiler::instance>  prof_;
  common::prof_events                                        common_prof_events_;
};

using instance = common::singleton<ori>;

inline void init(MPI_Comm comm = MPI_COMM_WORLD) { instance::init(comm); }
inline void fini()                               { instance::fini();     }

}
