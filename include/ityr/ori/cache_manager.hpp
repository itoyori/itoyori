#pragma once

#include <algorithm>

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/mpi_rma.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/virtual_mem.hpp"
#include "ityr/common/physical_mem.hpp"
#include "ityr/ori/util.hpp"
#include "ityr/ori/block_regions.hpp"
#include "ityr/ori/cache_system.hpp"
#include "ityr/ori/tlb.hpp"

namespace ityr::ori {

template <block_size_t BlockSize>
class cache_manager {
public:
  cache_manager(std::size_t cache_size, std::size_t sub_block_size)
    : cache_size_(cache_size),
      sub_block_size_(sub_block_size),
      vm_(cache_size_, BlockSize),
      pm_(init_cache_pm()),
      cs_(cache_size / BlockSize, cache_block(this)),
      win_(common::topology::mpicomm(), reinterpret_cast<std::byte*>(vm_.addr()), vm_.size()),
      max_dirty_cache_blocks_(common::getenv_coll("ITYR_ORI_MAX_DIRTY_CACHE_SIZE", cache_size / 2, common::topology::mpicomm()) / BlockSize) {
    ITYR_CHECK(cache_size_ > 0);
    ITYR_CHECK(common::is_pow2(cache_size_));
    ITYR_CHECK(cache_size_ % BlockSize == 0);
    ITYR_CHECK(common::is_pow2(sub_block_size_));
    ITYR_CHECK(sub_block_size_ <= BlockSize);
  }

  template <bool SkipFetch, bool IncrementRef>
  bool checkout_fast(std::byte* addr, std::size_t size) {
    ITYR_CHECK(addr);
    ITYR_CHECK(size > 0);

    std::byte* blk_addr = common::round_down_pow2(addr, BlockSize);
    if (blk_addr + BlockSize < addr + size) {
      // Fast path requires the requested region be within a single cache block
      return false;
    }

    auto cbo = cache_tlb_.get(blk_addr);
    if (!cbo.has_value()) {
      return false;
    }
    cache_block& cb = **cbo;

    block_region br = {addr - blk_addr, addr + size - blk_addr};

    if constexpr (SkipFetch) {
      cb.fresh_regions.add(br);
    } else {
      if (fetch_begin(cb, br)) {
        fetch_complete(cb.win);
      }
    }

    if constexpr (IncrementRef) {
      cb.ref_count++;
    }

    return true;
  }

  template <bool SkipFetch, bool IncrementRef>
  void checkout_blk(std::byte*               blk_addr,
                    std::byte*               req_addr_b,
                    std::byte*               req_addr_e,
                    MPI_Win                  win,
                    common::topology::rank_t owner,
                    std::size_t              pm_offset) {
    ITYR_CHECK(blk_addr <= req_addr_b);
    ITYR_CHECK(blk_addr <= req_addr_e);
    ITYR_CHECK(req_addr_b <= blk_addr + BlockSize);
    ITYR_CHECK(req_addr_e <= blk_addr + BlockSize);
    ITYR_CHECK(req_addr_b < req_addr_e);

    cache_block& cb = get_entry(blk_addr);

    if (blk_addr != cb.mapped_addr) {
      cb.addr      = blk_addr;
      cb.win       = win;
      cb.owner     = owner;
      cb.pm_offset = pm_offset;
      cache_blocks_to_map_.push_back(&cb);
    }

    block_region br = {req_addr_b - blk_addr, req_addr_e - blk_addr};

    if constexpr (SkipFetch) {
      cb.fresh_regions.add(br);
    } else {
      if (fetch_begin(cb, br)) {
        fetching_ = true;
      }
    }

    if constexpr (IncrementRef) {
      cb.ref_count++;
    }

    cache_tlb_.add(blk_addr, &cb);
  }

  void checkout_complete(MPI_Win win) {
    // Overlap communication and memory remapping
    if (!cache_blocks_to_map_.empty()) {
      for (cache_block* cb : cache_blocks_to_map_) {
        update_mapping(*cb);
      }
      cache_blocks_to_map_.clear();
    }

    if (fetching_) {
      fetch_complete(win);
      fetching_ = false;
    }
  }

  template <bool RegisterDirty, bool DecrementRef>
  bool checkin_fast(std::byte* addr, std::size_t size) {
    ITYR_CHECK(addr);
    ITYR_CHECK(size > 0);

    std::byte* blk_addr = common::round_down_pow2(addr, BlockSize);
    if (blk_addr + BlockSize < addr + size) {
      // Fast path requires the requested region be within a single cache block
      return false;
    }

    auto cbo = cache_tlb_.get(blk_addr);
    if (!cbo.has_value()) {
      return false;
    }
    cache_block& cb = **cbo;

    if constexpr (RegisterDirty) {
      block_region br = {addr - blk_addr, addr + size - blk_addr};
      add_dirty_region(cb, br);
    }

    if constexpr (DecrementRef) {
      cb.ref_count--;
    }

    return true;
  }

  template <bool RegisterDirty, bool DecrementRef>
  void checkin_blk(std::byte* blk_addr,
                   std::byte* req_addr_b,
                   std::byte* req_addr_e) {
    ITYR_CHECK(blk_addr <= req_addr_b);
    ITYR_CHECK(blk_addr <= req_addr_e);
    ITYR_CHECK(req_addr_b <= blk_addr + BlockSize);
    ITYR_CHECK(req_addr_e <= blk_addr + BlockSize);
    ITYR_CHECK(req_addr_b < req_addr_e);

    cache_block& cb = get_entry<false>(blk_addr);

    if constexpr (RegisterDirty) {
      block_region br = {req_addr_b - blk_addr, req_addr_e - blk_addr};
      add_dirty_region(cb, br);
    }

    if constexpr (DecrementRef) {
      cb.ref_count--;
    }
  }

  void release() {
    ensure_all_cache_clean();
  }

  void acquire() {
    invalidate_all();
  }

  void ensure_all_cache_clean() {
    writeback_begin();
    writeback_complete();

    if (cache_dirty_) {
      cache_dirty_ = false;
      /* rm_.increment_epoch(); */
    }
  }

  void ensure_evicted(void* addr) {
    cs_.ensure_evicted(cache_key(addr));
  }

  void discard_dirty(std::byte* blk_addr,
                     std::byte* req_addr_b,
                     std::byte* req_addr_e) {
    ITYR_CHECK(blk_addr <= req_addr_b);
    ITYR_CHECK(blk_addr <= req_addr_e);
    ITYR_CHECK(req_addr_b <= blk_addr + BlockSize);
    ITYR_CHECK(req_addr_e <= blk_addr + BlockSize);
    ITYR_CHECK(req_addr_b < req_addr_e);

    if (is_cached(blk_addr)) {
      cache_block& cb = get_entry(blk_addr);

      if (cb.is_writing_back()) {
        writeback_complete();
      }

      block_region br = {req_addr_b - blk_addr, req_addr_e - blk_addr};
      cb.dirty_regions.remove(br);
    }
  }

private:
  static constexpr bool enable_write_through = false;

  using epoch_t = uint64_t;

  struct cache_block {
    cache_entry_idx_t        entry_idx       = std::numeric_limits<cache_entry_idx_t>::max();
    std::byte*               addr            = nullptr;
    std::byte*               mapped_addr     = nullptr;
    MPI_Win                  win             = MPI_WIN_NULL;
    common::topology::rank_t owner           = -1;
    std::size_t              pm_offset       = 0;
    int                      ref_count       = 0;
    epoch_t                  writeback_epoch = 0;
    block_regions            fresh_regions;
    block_regions            dirty_regions;
    cache_manager*           outer;

    cache_block(cache_manager* outer_p) : outer(outer_p) {}

    bool is_writing_back() const {
      return writeback_epoch > outer->writeback_epoch_;
    }

    void invalidate() {
      ITYR_CHECK(!is_writing_back());
      ITYR_CHECK(dirty_regions.empty());
      fresh_regions.clear();
      ITYR_CHECK(is_evictable());

      common::verbose("Cache block %ld for [%p, %p) invalidated",
                      entry_idx, addr, addr + BlockSize);
    }

    /* Callback functions for cache_system class */

    bool is_evictable() const {
      return ref_count == 0 &&
             dirty_regions.empty() &&
             !is_writing_back();
    }

    void on_evict() {
      ITYR_CHECK(is_evictable());
      invalidate();
      entry_idx = std::numeric_limits<cache_entry_idx_t>::max();
      // for safety
      outer->cache_tlb_.clear();
    }

    void on_cache_map(cache_entry_idx_t idx) {
      entry_idx = idx;
    }
  };

  static std::string cache_shmem_name(int global_rank) {
    std::stringstream ss;
    ss << "/ityr_ori_cache_" << global_rank;
    return ss.str();
  }

  common::physical_mem init_cache_pm() {
    common::physical_mem pm(cache_shmem_name(common::topology::my_rank()), vm_.size(), true);
    pm.map_to_vm(vm_.addr(), vm_.size(), 0);
    return pm;
  }

  using cache_key_t = uintptr_t;

  cache_key_t cache_key(void* addr) const {
    ITYR_CHECK(addr);
    ITYR_CHECK(reinterpret_cast<uintptr_t>(addr) % BlockSize == 0);
    return reinterpret_cast<uintptr_t>(addr) / BlockSize;
  }

  block_region pad_fetch_region(block_region br) const {
    return {common::round_down_pow2(br.begin, sub_block_size_),
            common::round_up_pow2(br.end, sub_block_size_)};
  }

  template <bool UpdateLRU = true>
  cache_block& get_entry(void* addr) {
    try {
      return cs_.template ensure_cached<UpdateLRU>(cache_key(addr));
    } catch (cache_full_exception& e) {
      // write back all dirty cache and retry
      ensure_all_cache_clean();
      try {
        return cs_.template ensure_cached<UpdateLRU>(cache_key(addr));
      } catch (cache_full_exception& e) {
        common::die("cache is exhausted (too much checked-out memory)");
      }
    }
  }

  void update_mapping(cache_block& cb) {
    // save the number of mmap entries by unmapping previous virtual memory
    if (cb.mapped_addr) {
      common::verbose("Unmap cache block %d from [%p, %p) (size=%ld)",
                      cb.entry_idx, cb.mapped_addr, cb.mapped_addr + BlockSize, BlockSize);
      common::mmap_no_physical_mem(cb.mapped_addr, BlockSize, true);
    }
    ITYR_CHECK(cb.addr);
    common::verbose("Map cache block %d to [%p, %p) (size=%ld)",
                    cb.entry_idx, cb.addr, cb.addr + BlockSize, BlockSize);
    pm_.map_to_vm(cb.addr, BlockSize, cb.entry_idx * BlockSize);
    cb.mapped_addr = cb.addr;
  }

  bool fetch_begin(cache_block& cb, block_region br) {
    ITYR_CHECK(cb.owner < common::topology::n_ranks());

    if (cb.fresh_regions.include(br)) {
      // fast path (the requested region is already fetched)
      return false;
    }

    block_region br_pad = pad_fetch_region(br);

    std::byte* cache_begin = reinterpret_cast<std::byte*>(vm_.addr());

    // fetch only nondirty sections
    for (auto [blk_offset_b, blk_offset_e] : cb.fresh_regions.inverse(br_pad)) {
      ITYR_CHECK(cb.entry_idx < cs_.num_entries());

      std::byte*  addr      = cache_begin + cb.entry_idx * BlockSize + blk_offset_b;
      std::size_t size      = blk_offset_e - blk_offset_b;
      std::size_t pm_offset = cb.pm_offset + blk_offset_b;

      common::verbose("Fetching [%p, %p) (%ld bytes) to cache block %d from rank %d (win=%p, disp=%ld)",
                      cb.addr + blk_offset_b, cb.addr + blk_offset_e, size,
                      cb.entry_idx, cb.owner, cb.win, pm_offset);

      common::mpi_get_nb(addr, size, cb.owner, pm_offset, cb.win);
    }

    cb.fresh_regions.add(br_pad);

    return true;
  }

  void fetch_complete(MPI_Win win) {
    common::mpi_win_flush_all(win);
    common::verbose("Fetch complete (win=%p)", win);
  }

  void add_dirty_region(cache_block& cb, block_region br) {
    bool is_new_dirty_block = cb.dirty_regions.empty();

    cb.dirty_regions.add(br);

    if (is_new_dirty_block) {
      dirty_cache_blocks_.push_back(&cb);
      cache_dirty_ = true;

      if constexpr (enable_write_through) {
        ensure_all_cache_clean();

      } else if (dirty_cache_blocks_.size() >= max_dirty_cache_blocks_) {
        writeback_begin();
        dirty_cache_blocks_.clear();
      }
    }
  }

  void writeback_begin(cache_block& cb) {
    if (cb.writeback_epoch > writeback_epoch_) {
      // MPI_Put has been already started on this cache block.
      // As overlapping MPI_Put calls for the same location will cause undefined behaviour,
      // we need to insert MPI_Win_flush between overlapping MPI_Put calls here.
      writeback_complete();
      ITYR_CHECK(cb.writeback_epoch == writeback_epoch_);
    }

    std::byte* cache_begin = reinterpret_cast<std::byte*>(vm_.addr());

    for (auto [blk_offset_b, blk_offset_e] : cb.dirty_regions.get()) {
      ITYR_CHECK(cb.entry_idx < cs_.num_entries());

      std::byte*  addr      = cache_begin + cb.entry_idx * BlockSize + blk_offset_b;
      std::size_t size      = blk_offset_e - blk_offset_b;
      std::size_t pm_offset = cb.pm_offset + blk_offset_b;

      common::verbose("Writing back [%p, %p) (%ld bytes) to rank %d (win=%p, disp=%ld)",
                      cb.addr + blk_offset_b, cb.addr + blk_offset_e, size,
                      cb.owner, cb.win, pm_offset);

      common::mpi_put_nb(addr, size, cb.owner, pm_offset, cb.win);
    }

    cb.dirty_regions.clear();

    cb.writeback_epoch = writeback_epoch_ + 1;

    writing_back_wins_.push_back(cb.win);
  }

  void writeback_begin() {
    for (auto& cb : dirty_cache_blocks_) {
      if (!cb->dirty_regions.empty()) {
        writeback_begin(*cb);
      }
    }
  }

  void writeback_complete() {
    if (!writing_back_wins_.empty()) {
      // sort | uniq
      // FIXME: costly?
      std::sort(writing_back_wins_.begin(), writing_back_wins_.end());
      writing_back_wins_.erase(std::unique(writing_back_wins_.begin(), writing_back_wins_.end()), writing_back_wins_.end());

      for (auto win : writing_back_wins_) {
        MPI_Win_flush_all(win);
        common::verbose("Writing back complete (win=%p)", win);
      }
      writing_back_wins_.clear();

      writeback_epoch_++;
    }
  }

  bool is_cached(void* addr) const {
    return cs_.is_cached(cache_key(addr));
  }

  void invalidate_all() {
    ensure_all_cache_clean();

    cs_.for_each_entry([&](cache_block& cb) {
      // FIXME: this check is for prefetching
      cb.invalidate();
    });
  }

  std::size_t                            cache_size_;
  block_size_t                           sub_block_size_;

  common::virtual_mem                    vm_;
  common::physical_mem                   pm_;

  cache_system<cache_key_t, cache_block> cs_;

  // The cache region is not remotely accessible, but we make an MPI window here because
  // it will pre-register the memory region for RDMA and MPI_Get to the cache will speedup.
  common::mpi_win_manager<std::byte>     win_;

  tlb<std::byte*, cache_block*>          cache_tlb_;

  bool                                   fetching_ = false;
  std::vector<cache_block*>              cache_blocks_to_map_;

  std::vector<cache_block*>              dirty_cache_blocks_;
  std::size_t                            max_dirty_cache_blocks_;
  bool                                   cache_dirty_ = false;

  std::vector<MPI_Win>                   writing_back_wins_;
  epoch_t                                writeback_epoch_ = 0;
};

}
