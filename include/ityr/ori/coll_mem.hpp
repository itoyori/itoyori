#pragma once

#include <vector>
#include <memory>

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/mpi_rma.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/virtual_mem.hpp"
#include "ityr/common/physical_mem.hpp"
#include "ityr/ori/mem_mapper.hpp"

namespace ityr::ori {

using coll_mem_id_t = uint64_t;

class coll_mem {
public:
  coll_mem(std::size_t                       size,
           coll_mem_id_t                     id,
           std::unique_ptr<mem_mapper::base> mmapper)
    : size_(size),
      id_(id),
      mmapper_(std::move(mmapper)),
      vm_(common::reserve_same_vm_coll(mmapper_->effective_size(), mmapper_->block_size())),
      intra_home_pms_(init_intra_home_pms()),
      intra_home_vms_(init_intra_home_vms()),
      win_(common::topology::mpicomm(), reinterpret_cast<std::byte*>(local_home_vm().addr()), local_home_vm().size()) {}

  coll_mem(coll_mem&&) = default;
  coll_mem& operator=(coll_mem&&) = default;

  coll_mem_id_t id() const { return id_; }
  std::size_t size() const { return size_; }
  std::size_t local_size() const { return local_home_vm().size(); }
  std::size_t effective_size() const { return vm_.size(); }

  const mem_mapper::base& mem_mapper() const { return *mmapper_; }

  const common::virtual_mem& vm() const { return vm_; }

  const common::physical_mem& local_home_pm() const {
    return intra_home_pm(common::topology::intra_my_rank());
  }

  const common::physical_mem& intra_home_pm(common::topology::rank_t intra_rank) const {
    ITYR_CHECK(intra_rank < common::topology::intra_n_ranks());
    return intra_home_pms_[intra_rank];
  }

  const common::virtual_mem& local_home_vm() const {
    return intra_home_vm(common::topology::intra_my_rank());
  }

  const common::virtual_mem& intra_home_vm(common::topology::rank_t intra_rank) const {
    ITYR_CHECK(intra_rank < common::topology::intra_n_ranks());
    return intra_home_vms_[intra_rank];
  }

  MPI_Win win() const { return win_.win(); }

private:
  static std::string home_shmem_name(coll_mem_id_t id, int global_rank) {
    std::stringstream ss;
    ss << "/ityr_ori_coll_mem_" << id << "_" << global_rank;
    return ss.str();
  }

  std::vector<common::physical_mem> init_intra_home_pms() const {
    common::physical_mem pm_local(home_shmem_name(id_, common::topology::my_rank()),
                                  mmapper_->local_size(common::topology::my_rank()),
                                  true);

    common::mpi_barrier(common::topology::intra_mpicomm());

    // Open home physical memory of other intra-node processes
    std::vector<common::physical_mem> home_pms(common::topology::intra_n_ranks());
    for (int i = 0; i < common::topology::intra_n_ranks(); i++) {
      if (i == common::topology::intra_my_rank()) {
        home_pms[i] = std::move(pm_local);
      } else {
        int target_rank = common::topology::intra2global_rank(i);
        int target_local_size = mmapper_->local_size(target_rank);
        common::physical_mem pm(home_shmem_name(id_, target_rank), target_local_size, false);
        home_pms[i] = std::move(pm);
      }
    }

    return home_pms;
  }

  std::vector<common::virtual_mem> init_intra_home_vms() const {
    std::vector<common::virtual_mem> home_vms;

    for (const auto& pm : intra_home_pms_) {
      common::virtual_mem& vm = home_vms.emplace_back(pm.size(), mmapper_->block_size());
      pm.map_to_vm(vm.addr(), vm.size(), 0);
    }

    return home_vms;
  }

  std::size_t                        size_;
  coll_mem_id_t                      id_;
  std::unique_ptr<mem_mapper::base>  mmapper_;
  common::virtual_mem                vm_;
  std::vector<common::physical_mem>  intra_home_pms_; // intra-rank -> pm
  std::vector<common::virtual_mem>   intra_home_vms_; // intra-rank -> vm
  common::mpi_win_manager<std::byte> win_;
};

template <typename Fn>
void for_each_mem_segment(const coll_mem& cm, const void* addr, std::size_t size, Fn fn) {
  ITYR_CHECK(addr >= cm.vm().addr());

  std::size_t offset_b = reinterpret_cast<uintptr_t>(addr) -
                         reinterpret_cast<uintptr_t>(cm.vm().addr());
  std::size_t offset_e = offset_b + size;

  ITYR_CHECK(offset_e <= cm.size());

  std::size_t offset = offset_b;
  while (offset < offset_e) {
    auto seg = cm.mem_mapper().get_segment(offset);
    fn(seg);
    offset = seg.offset_e;
  }
}

}
