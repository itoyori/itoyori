#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/virtual_mem.hpp"
#include "ityr/common/physical_mem.hpp"
#include "ityr/common/allocator.hpp"
#include "ityr/ito/thread.hpp"
#include "ityr/ito/worker.hpp"

namespace ityr::ito {

inline void ito_init(const common::topology& topo) {
  worker_init(topo);
}

inline void ito_fini() {
  worker_fini();
}

}
