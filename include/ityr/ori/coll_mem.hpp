#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/numa.hpp"
#include "ityr/common/rma.hpp"
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
      home_pm_(init_intra_home_pm()),
      home_vm_(init_intra_home_vm()),
      win_(common::rma::create_win(reinterpret_cast<std::byte*>(home_vm().addr()), home_vm().size())),
      home_all_mapped_(map_ahead_of_time()) {}

  coll_mem(coll_mem&&) = default;
  coll_mem& operator=(coll_mem&&) = default;

  coll_mem_id_t id() const { return id_; }
  std::size_t size() const { return size_; }
  std::size_t local_size() const { return home_vm().size(); }
  std::size_t effective_size() const { return vm_.size(); }
  bool home_all_mapped() const { return home_all_mapped_; }

  const mem_mapper::base& mem_mapper() const { return *mmapper_; }

  const common::virtual_mem& vm() const { return vm_; }

  const common::physical_mem& home_pm() const {
    return home_pm_;
  }

  const common::virtual_mem& home_vm() const {
    return home_vm_;
  }

  const common::rma::win& win() const { return *win_; }

private:
  static std::string home_shmem_name(coll_mem_id_t id, int inter_rank) {
    std::stringstream ss;
    ss << "/ityr_ori_coll_mem_" << id << "_" << inter_rank;
    return ss.str();
  }

  common::physical_mem init_intra_home_pm() const {
    if (common::topology::intra_my_rank() == 0) {
      common::physical_mem pm(home_shmem_name(id_, common::topology::inter_my_rank()),
                              mmapper_->local_size(common::topology::inter_my_rank()),
                              true);
      common::mpi_barrier(common::topology::intra_mpicomm());
      return pm;

    } else {
      common::mpi_barrier(common::topology::intra_mpicomm());
      common::physical_mem pm(home_shmem_name(id_, common::topology::inter_my_rank()),
                              mmapper_->local_size(common::topology::inter_my_rank()),
                              false);
      return pm;
    }
  }

  common::virtual_mem init_intra_home_vm() const {
    common::virtual_mem vm(home_pm_.size(), mmapper_->block_size());
    home_pm_.map_to_vm(vm.addr(), vm.size(), 0);

    common::mpi_barrier(common::topology::intra_mpicomm());

    if (common::topology::numa_enabled()) {
      std::size_t pm_offset = 0;
      while (pm_offset < vm.size()) {
        auto numa_seg = mmapper_->get_numa_segment(common::topology::inter_my_rank(), pm_offset);
        if (numa_seg.owner == -1 && common::topology::intra_my_rank() == 0) {
          // interleave all
          std::byte*  numa_seg_addr = reinterpret_cast<std::byte*>(vm.addr()) + numa_seg.pm_offset_b;
          std::size_t numa_seg_size = numa_seg.pm_offset_e - numa_seg.pm_offset_b;
          common::numa::interleave(numa_seg_addr, numa_seg_size, common::topology::numa_nodemask_all());
        }
        if (numa_seg.owner == common::topology::intra_my_rank()) {
          std::byte*  numa_seg_addr = reinterpret_cast<std::byte*>(vm.addr()) + numa_seg.pm_offset_b;
          std::size_t numa_seg_size = numa_seg.pm_offset_e - numa_seg.pm_offset_b;
          common::numa::bind_to(numa_seg_addr, numa_seg_size, common::topology::numa_node(numa_seg.owner));
        }
        pm_offset = numa_seg.pm_offset_e;
      }
      common::mpi_barrier(common::topology::intra_mpicomm());
    }

    return vm;
  }

  bool map_ahead_of_time() const {
    if (common::topology::inter_n_ranks() == 1) {
      home_pm_.map_to_vm(vm_.addr(), vm_.size(), 0);
      return true;

    } else if (mmapper_->should_map_all_home()) {
      // mmap all home blocks ahead of time if the mem mapper does not consume
      // too many mmap entries (e.g., for block distribution)
      std::size_t offset = 0;
      while (offset < size_) {
        auto seg = mmapper_->get_segment(offset);
        if (seg.owner == common::topology::inter_my_rank()) {
          std::byte*  seg_addr = reinterpret_cast<std::byte*>(vm_.addr()) + seg.offset_b;
          std::size_t seg_size = seg.offset_e - seg.offset_b;
          home_pm_.map_to_vm(seg_addr, seg_size, seg.pm_offset);
        }
        offset = seg.offset_e;
      }
      return true;
    }

    return false;
  }

  std::size_t                       size_;
  coll_mem_id_t                     id_;
  std::unique_ptr<mem_mapper::base> mmapper_;
  common::virtual_mem               vm_;
  common::physical_mem              home_pm_;
  common::virtual_mem               home_vm_;
  std::unique_ptr<common::rma::win> win_;
  bool                              home_all_mapped_;
};

template <typename Fn>
inline void for_each_mem_segment(const coll_mem& cm, const void* addr, std::size_t size, Fn fn) {
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
