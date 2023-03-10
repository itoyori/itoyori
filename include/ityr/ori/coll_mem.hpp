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
      local_home_vm_(mmapper_->local_size(common::topology::my_rank()), mmapper_->block_size()),
      intra_home_pms_(init_intra_home_pms()),
      win_(common::topology::mpicomm(), reinterpret_cast<std::byte*>(local_home_vm_.addr()), local_home_vm_.size()) {}

  coll_mem(coll_mem&&) = default;
  coll_mem& operator=(coll_mem&&) = default;

  coll_mem_id_t id() const { return id_; }
  std::size_t size() const { return size_; }
  std::size_t local_size() const { return local_home_vm_.size(); }
  std::size_t effective_size() const { return vm_.size(); }

  const mem_mapper::base& mem_mapper() const { return *mmapper_; }

  const common::virtual_mem& vm() const { return vm_; }

  const common::virtual_mem& local_home_vm() const { return local_home_vm_; }

  const common::physical_mem& intra_home_pm() const {
    return intra_home_pm(common::topology::intra_my_rank());
  }

  const common::physical_mem& intra_home_pm(common::topology::rank_t intra_rank) const {
    ITYR_CHECK(intra_rank < common::topology::intra_n_ranks());
    return intra_home_pms_[intra_rank];
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
                                  local_home_vm_.size(),
                                  true);
    pm_local.map_to_vm(local_home_vm_.addr(), local_home_vm_.size(), 0);

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

  std::size_t                        size_;
  coll_mem_id_t                      id_;
  std::unique_ptr<mem_mapper::base>  mmapper_;
  common::virtual_mem                vm_;
  common::virtual_mem                local_home_vm_;
  std::vector<common::physical_mem>  intra_home_pms_; // intra-rank -> pm
  common::mpi_win_manager<std::byte> win_;
};

template <typename Fn>
void for_each_mem_segment(const coll_mem& cm, void* addr, std::size_t size, Fn&& fn) {
  ITYR_CHECK(addr >= cm.vm().addr());

  std::size_t offset_b = reinterpret_cast<uintptr_t>(addr) -
                         reinterpret_cast<uintptr_t>(cm.vm().addr());
  std::size_t offset_e = offset_b + size;

  ITYR_CHECK(offset_e <= cm.size());

  std::size_t offset = offset_b;
  while (offset < offset_e) {
    auto seg = cm.mem_mapper().get_segment(offset);
    std::forward<Fn>(fn)(seg);
    offset = seg.offset_e;
  }
}

template <std::size_t BlockSize, typename HomeBlockFn, typename CacheBlockFn>
void for_each_mem_block(const coll_mem& cm, void* addr, std::size_t size,
                        HomeBlockFn&& home_block_fn, CacheBlockFn&& cache_block_fn) {
  for_each_mem_segment(cm, addr, size, [&](const auto& seg) {
    std::byte*  seg_addr = reinterpret_cast<std::byte*>(cm.vm().addr()) + seg.offset_b;
    std::size_t seg_size = seg.offset_e - seg.offset_b;

    if (common::topology::is_locally_accessible(seg.owner)) {
      // no need to iterate over memory blocks (of BlockSize) for home segments
      std::forward<HomeBlockFn>(home_block_fn)(seg_addr, seg_size, seg.owner, seg.pm_offset);

    } else {
      // iterate over memory blocks within the memory segment for cache blocks
      std::byte* blk_addr_b = std::max(seg_addr,
          reinterpret_cast<std::byte*>(common::round_down_pow2(reinterpret_cast<uintptr_t>(addr), BlockSize)));
      std::byte* blk_addr_e = std::min(seg_addr + seg_size, reinterpret_cast<std::byte*>(addr) + size);
      for (std::byte* blk_addr = blk_addr_b; blk_addr < blk_addr_e; blk_addr += BlockSize) {
        std::size_t pm_offset = seg.pm_offset + (blk_addr - seg_addr);
        ITYR_CHECK(pm_offset + BlockSize <= cm.mem_mapper().local_size(seg.owner));
        std::forward<CacheBlockFn>(cache_block_fn)(blk_addr, seg.owner, pm_offset);
      }
    }
  });
}

}
