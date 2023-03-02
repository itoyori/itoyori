#pragma once

#include <optional>

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/virtual_mem.hpp"
#include "ityr/common/physical_mem.hpp"

namespace ityr::ito {

class callstack {
public:
  callstack(const common::topology& topo, std::size_t size)
    : topo_(topo),
      vm_(common::reserve_same_vm_coll(topo, size, common::get_page_size())),
      pm_(init_stack_pm()),
      win_(topo_.mpicomm(), reinterpret_cast<std::byte*>(vm_.addr()), vm_.size()) {}

  void* top() const { return vm_.addr(); }
  void* bottom() const { return reinterpret_cast<std::byte*>(vm_.addr()) + vm_.size(); }
  std::size_t size() const { return vm_.size(); }

  void direct_copy_from(void*                    addr,
                        std::size_t              size,
                        common::topology::rank_t target_rank) const {
    ITYR_CHECK(target_rank != topo_.my_rank());
    ITYR_CHECK(target_rank < topo_.n_ranks());
    ITYR_CHECK(vm_.addr() <= addr);
    ITYR_CHECK(reinterpret_cast<std::byte*>(addr) + size <= reinterpret_cast<std::byte*>(vm_.addr()) + vm_.size());

    auto target_disp = reinterpret_cast<uintptr_t>(addr) - reinterpret_cast<uintptr_t>(vm_.addr());
    common::mpi_get(reinterpret_cast<std::byte*>(addr), size, target_rank, target_disp, win_.win());
  }

private:
  static std::string stack_shmem_name(int rank) {
    std::stringstream ss;
    ss << "/ityr_ito_stack_" << rank;
    return ss.str();
  }

  common::physical_mem init_stack_pm() {
    common::physical_mem pm(stack_shmem_name(topo_.my_rank()), vm_.size(), true);
    pm.map_to_vm(vm_.addr(), vm_.size(), 0);
    return pm;
  }

  const common::topology&            topo_;
  common::virtual_mem                vm_;
  common::physical_mem               pm_;
  common::mpi_win_manager<std::byte> win_;
};

}
