#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/topology.hpp"

namespace ityr::ito {

// Check if address space layout randomization (ASLR) is disabled
class aslr_checker {
public:
  aslr_checker() {
    void* addr = get_static_local_var_addr();
    void* addr0 = common::mpi_bcast_value(addr, 0, common::topology::mpicomm());
    int ok = addr == addr0;
    int ok_sum = common::mpi_reduce_value(ok, 0, common::topology::mpicomm());
    if (common::topology::my_rank() == 0 &&
        ok_sum != common::topology::n_ranks()) {
      common::die("Error: address space layout randomization (ASLR) seems enabled.\n"
                  "To disable ASLR, please run your program with:\n"
                  "  $ mpiexec setarch $(uname -m) --addr-no-randomize [COMMANDS]...\n");
    }
  }

  aslr_checker(const aslr_checker&) = delete;
  aslr_checker& operator=(const aslr_checker&) = delete;

  aslr_checker(aslr_checker&&) = delete;
  aslr_checker& operator=(aslr_checker&&) = delete;

private:
  static void* get_static_local_var_addr() {
    static int var = 0;
    return &var;
  }
};

}
