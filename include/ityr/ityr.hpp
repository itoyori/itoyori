#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/ito/ito.hpp"

namespace ityr {

class ityr {
public:
  ityr(MPI_Comm comm)
    : topo_(comm, false) {}

  const common::topology& topology() const { return topo_; }

private:
  common::topology topo_;
};

inline std::optional<ityr>& ityr_get_() {
  static std::optional<ityr> instance;
  return instance;
}

inline ityr& ityr_get() {
  ITYR_CHECK(ityr_get_().has_value());
  return *ityr_get_();
}

inline void ityr_init(MPI_Comm comm = MPI_COMM_WORLD) {
  ITYR_CHECK(!ityr_get_().has_value());
  ityr_get_().emplace(comm);

  ito::ito_init(ityr_get().topology());
}

inline void ityr_fini() {
  ITYR_CHECK(ityr_get_().has_value());
  ityr_get_().reset();

  ito::ito_fini();
}

}
