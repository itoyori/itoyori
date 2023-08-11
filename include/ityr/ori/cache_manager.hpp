#pragma once

#include <cstring>
#include <algorithm>

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/logger.hpp"
#include "ityr/common/rma.hpp"
#include "ityr/common/virtual_mem.hpp"
#include "ityr/common/physical_mem.hpp"
#include "ityr/ori/util.hpp"
#include "ityr/ori/options.hpp"
#include "ityr/ori/prof_events.hpp"
#include "ityr/ori/block_regions.hpp"
#include "ityr/ori/cache_system.hpp"
#include "ityr/ori/tlb.hpp"
#include "ityr/ori/release_manager.hpp"
#include "ityr/ori/cache_profiler.hpp"

namespace ityr::ori {

template <block_size_t BlockSize>
class cache_manager {
  static constexpr bool enable_write_through = ITYR_ORI_ENABLE_WRITE_THROUGH;
  static constexpr bool enable_lazy_release = ITYR_ORI_ENABLE_LAZY_RELEASE;
  static constexpr bool enable_vm_map = ITYR_ORI_ENABLE_VM_MAP;

public:
  cache_manager(std::size_t cache_size, std::size_t sub_block_size)
    : cache_size_(cache_size),
      sub_block_size_(sub_block_size),
      vm_(cache_size_, BlockSize),
      pm_(init_cache_pm()),
      cs_(cache_size / BlockSize, cache_block(this)),
      cache_win_(common::rma::create_win(reinterpret_cast<std::byte*>(vm_.addr()), vm_.size())),
      max_dirty_cache_blocks_(max_dirty_cache_size_option::value() / BlockSize),
      cprof_(cs_.num_entries()) {
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
      cprof_.record_writeonly(cb.entry_idx, br, cb.valid_regions);
      cb.valid_regions.add(br);
    } else {
      if (fetch_begin(cb, br)) {
        add_fetching_win(*cb.win);
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
                    const common::rma::win&  win,
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
      cb.win       = &win;
      cb.owner     = owner;
      cb.pm_offset = pm_offset;
      if constexpr (enable_vm_map) {
        cache_blocks_to_map_.push_back(&cb);
      } else {
        cb.mapped_addr = blk_addr;
      }
    }

    block_region br = {req_addr_b - blk_addr, req_addr_e - blk_addr};

    if constexpr (SkipFetch) {
      cprof_.record_writeonly(cb.entry_idx, br, cb.valid_regions);
      cb.valid_regions.add(br);
    } else {
      if (fetch_begin(cb, br)) {
        add_fetching_win(win);
      }
    }

    if constexpr (IncrementRef) {
      cb.ref_count++;
    }

    cache_tlb_.add(blk_addr, &cb);
  }

  void checkout_complete() {
    // Overlap communication and memory remapping
    if constexpr (enable_vm_map) {
      if (!cache_blocks_to_map_.empty()) {
        for (cache_block* cb : cache_blocks_to_map_) {
          update_mapping(*cb);
        }
        cache_blocks_to_map_.clear();
      }
    }

    fetch_complete();
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
    ITYR_PROFILER_RECORD(prof_event_release);
    ensure_all_cache_clean();
  }

  using release_handler = std::conditional_t<enable_lazy_release, release_manager::release_handler, void*>;

  auto release_lazy() {
    if constexpr (enable_lazy_release) {
      if (has_dirty_cache_) {
        return rm_.get_release_handler();
      } else {
        return rm_.get_dummy_handler();
      }
    } else {
      release();
      return release_handler{};
    }
  }

  void acquire() {
    ITYR_PROFILER_RECORD(prof_event_acquire);

    // FIXME: no need to writeback dirty data here?
    ensure_all_cache_clean();
    invalidate_all();
  }

  template <typename ReleaseHandler>
  void acquire(ReleaseHandler rh) {
    ITYR_PROFILER_RECORD(prof_event_acquire);

    ensure_all_cache_clean();
    if constexpr (enable_lazy_release) {
      rm_.ensure_released(rh);
    }
    invalidate_all();
  }

  void poll() {
    if constexpr (enable_lazy_release) {
      if (rm_.release_requested()) {
        ITYR_PROFILER_RECORD(prof_event_release_lazy);

        ensure_all_cache_clean();
        ITYR_CHECK(!rm_.release_requested());
      }
    }
  }

  void ensure_all_cache_clean() {
    writeback_begin();
    writeback_complete();
    ITYR_CHECK(!has_dirty_cache_);
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

  void get_copy_blk(std::byte* blk_addr,
                    std::byte* req_addr_b,
                    std::byte* req_addr_e,
                    std::byte* to_addr) {
    ITYR_CHECK(blk_addr <= req_addr_b);
    ITYR_CHECK(blk_addr <= req_addr_e);
    ITYR_CHECK(req_addr_b <= blk_addr + BlockSize);
    ITYR_CHECK(req_addr_e <= blk_addr + BlockSize);
    ITYR_CHECK(req_addr_b < req_addr_e);

    ITYR_CHECK(is_cached(blk_addr));
    cache_block& cb = get_entry<false>(blk_addr);

    ITYR_CHECK(cb.entry_idx < cs_.num_entries());

    std::size_t blk_offset = req_addr_b - blk_addr;
    std::byte* from_addr = reinterpret_cast<std::byte*>(vm_.addr()) + cb.entry_idx * BlockSize + blk_offset;
    std::memcpy(to_addr, from_addr, req_addr_e - req_addr_b);
  }

  void put_copy_blk(std::byte*       blk_addr,
                    std::byte*       req_addr_b,
                    std::byte*       req_addr_e,
                    const std::byte* from_addr) {
    ITYR_CHECK(blk_addr <= req_addr_b);
    ITYR_CHECK(blk_addr <= req_addr_e);
    ITYR_CHECK(req_addr_b <= blk_addr + BlockSize);
    ITYR_CHECK(req_addr_e <= blk_addr + BlockSize);
    ITYR_CHECK(req_addr_b < req_addr_e);

    ITYR_CHECK(is_cached(blk_addr));
    cache_block& cb = get_entry<false>(blk_addr);

    ITYR_CHECK(cb.entry_idx < cs_.num_entries());

    std::size_t blk_offset = req_addr_b - blk_addr;
    std::byte* to_addr = reinterpret_cast<std::byte*>(vm_.addr()) + cb.entry_idx * BlockSize + blk_offset;
    std::memcpy(to_addr, from_addr, req_addr_e - req_addr_b);
  }

  void cache_prof_begin() { invalidate_all(); cprof_.start(); }
  void cache_prof_end() { cprof_.stop(); }
  void cache_prof_print() const { cprof_.print(); }

private:
  using writeback_epoch_t = uint64_t;

  struct cache_block {
    cache_entry_idx_t        entry_idx       = std::numeric_limits<cache_entry_idx_t>::max();
    std::byte*               addr            = nullptr;
    std::byte*               mapped_addr     = nullptr;
    const common::rma::win*  win             = nullptr;
    common::topology::rank_t owner           = -1;
    std::size_t              pm_offset       = 0;
    int                      ref_count       = 0;
    writeback_epoch_t        writeback_epoch = 0;
    block_regions            valid_regions;
    block_regions            dirty_regions;
    cache_manager*           outer;

    cache_block(cache_manager* outer_p) : outer(outer_p) {}

    bool is_writing_back() const {
      return writeback_epoch == outer->writeback_epoch_;
    }

    void invalidate() {
      outer->cprof_.invalidate(entry_idx, valid_regions);

      ITYR_CHECK(!is_writing_back());
      ITYR_CHECK(dirty_regions.empty());
      valid_regions.clear();
      ITYR_CHECK(is_evictable());

      common::verbose<3>("Cache block %ld for [%p, %p) invalidated",
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
      common::verbose<3>("Unmap cache block %d from [%p, %p) (size=%ld)",
                         cb.entry_idx, cb.mapped_addr, cb.mapped_addr + BlockSize, BlockSize);
      common::mmap_no_physical_mem(cb.mapped_addr, BlockSize, true);
    }
    ITYR_CHECK(cb.addr);
    common::verbose<3>("Map cache block %d to [%p, %p) (size=%ld)",
                       cb.entry_idx, cb.addr, cb.addr + BlockSize, BlockSize);
    pm_.map_to_vm(cb.addr, BlockSize, cb.entry_idx * BlockSize);
    cb.mapped_addr = cb.addr;
  }

  bool fetch_begin(cache_block& cb, block_region br) {
    ITYR_CHECK(cb.owner < common::topology::n_ranks());

    if (cb.valid_regions.include(br)) {
      // fast path (the requested region is already fetched)
      cprof_.record(cb.entry_idx, br, {});
      return false;
    }

    block_region br_pad = pad_fetch_region(br);

    std::byte* cache_begin = reinterpret_cast<std::byte*>(vm_.addr());

    block_regions fetch_regions = cb.valid_regions.inverse(br_pad);

    // fetch only nondirty sections
    for (auto [blk_offset_b, blk_offset_e] : fetch_regions) {
      ITYR_CHECK(cb.entry_idx < cs_.num_entries());

      std::byte*  addr      = cache_begin + cb.entry_idx * BlockSize + blk_offset_b;
      std::size_t size      = blk_offset_e - blk_offset_b;
      std::size_t pm_offset = cb.pm_offset + blk_offset_b;

      common::verbose<3>("Fetching [%p, %p) (%ld bytes) to cache block %d from rank %d (win=%p, disp=%ld)",
                         cb.addr + blk_offset_b, cb.addr + blk_offset_e, size,
                         cb.entry_idx, cb.owner, cb.win, pm_offset);

      common::rma::get_nb(*cache_win_, addr, size, *cb.win, cb.owner, pm_offset);
    }

    cb.valid_regions.add(br_pad);

    cprof_.record(cb.entry_idx, br, fetch_regions);

    return true;
  }

  void fetch_complete() {
    if (!fetching_wins_.empty()) {
      for (const common::rma::win* win : fetching_wins_) {
        // TODO: remove duplicates
        common::rma::flush(*win);
        common::verbose<3>("Fetch complete (win=%p)", win);
      }
      fetching_wins_.clear();
    }
  }

  void add_fetching_win(const common::rma::win& win) {
    if (fetching_wins_.empty() || fetching_wins_.back() != &win) {
      // best effort to avoid duplicates
      fetching_wins_.push_back(&win);
    }
  }

  void add_dirty_region(cache_block& cb, block_region br) {
    bool is_new_dirty_block = cb.dirty_regions.empty();

    cb.dirty_regions.add(br);

    if (is_new_dirty_block) {
      dirty_cache_blocks_.push_back(&cb);
      has_dirty_cache_ = true;

      if constexpr (enable_write_through) {
        ensure_all_cache_clean();

      } else if (dirty_cache_blocks_.size() >= max_dirty_cache_blocks_) {
        writeback_begin();
      }
    }
  }

  void writeback_begin() {
    for (auto& cb : dirty_cache_blocks_) {
      if (!cb->dirty_regions.empty()) {
        writeback_begin(*cb);
      }
    }
    dirty_cache_blocks_.clear();
  }

  void writeback_begin(cache_block& cb) {
    if (cb.writeback_epoch == writeback_epoch_) {
      // MPI_Put has been already started on this cache block.
      // As overlapping MPI_Put calls for the same location will cause undefined behaviour,
      // we need to insert MPI_Win_flush between overlapping MPI_Put calls here.
      writeback_complete();
      ITYR_CHECK(cb.writeback_epoch < writeback_epoch_);
    }

    std::byte* cache_begin = reinterpret_cast<std::byte*>(vm_.addr());

    for (auto [blk_offset_b, blk_offset_e] : cb.dirty_regions) {
      ITYR_CHECK(cb.entry_idx < cs_.num_entries());

      std::byte*  addr      = cache_begin + cb.entry_idx * BlockSize + blk_offset_b;
      std::size_t size      = blk_offset_e - blk_offset_b;
      std::size_t pm_offset = cb.pm_offset + blk_offset_b;

      common::verbose<3>("Writing back [%p, %p) (%ld bytes) to rank %d (win=%p, disp=%ld)",
                         cb.addr + blk_offset_b, cb.addr + blk_offset_e, size,
                         cb.owner, cb.win, pm_offset);

      common::rma::put_nb(*cache_win_, addr, size, *cb.win, cb.owner, pm_offset);
    }

    cb.dirty_regions.clear();

    cb.writeback_epoch = writeback_epoch_;

    writing_back_wins_.push_back(cb.win);
  }

  void writeback_complete() {
    if (!writing_back_wins_.empty()) {
      // sort | uniq
      // FIXME: costly?
      std::sort(writing_back_wins_.begin(), writing_back_wins_.end());
      writing_back_wins_.erase(std::unique(writing_back_wins_.begin(), writing_back_wins_.end()), writing_back_wins_.end());

      for (const common::rma::win* win : writing_back_wins_) {
        common::rma::flush(*win);
        common::verbose<3>("Writing back complete (win=%p)", win);
      }
      writing_back_wins_.clear();

      writeback_epoch_++;
    }

    if (dirty_cache_blocks_.empty() && has_dirty_cache_) {
      has_dirty_cache_ = false;
      rm_.increment_epoch();
    }
  }

  bool is_cached(void* addr) const {
    return cs_.is_cached(cache_key(addr));
  }

  void invalidate_all() {
    cs_.for_each_entry([&](cache_block& cb) {
      cb.invalidate();
    });
  }

  std::size_t                            cache_size_;
  block_size_t                           sub_block_size_;

  common::virtual_mem                    vm_;
  common::physical_mem                   pm_;

  cache_system<cache_key_t, cache_block> cs_;

  std::unique_ptr<common::rma::win>      cache_win_;

  tlb<std::byte*, cache_block*>          cache_tlb_;

  std::vector<const common::rma::win*>   fetching_wins_;
  std::vector<cache_block*>              cache_blocks_to_map_;

  std::vector<cache_block*>              dirty_cache_blocks_;
  std::size_t                            max_dirty_cache_blocks_;

  // A writeback epoch is an interval between writeback completion events.
  // Writeback epochs are conceptually different from epochs used in the lazy release manager.
  // Even if the writeback epoch is incremented, some cache blocks might be dirty.
  writeback_epoch_t                      writeback_epoch_ = 1;
  std::vector<const common::rma::win*>   writing_back_wins_;

  // A pending dirty cache block is marked dirty but not yet started to writeback.
  // Only if the writeback is completed and there is no pending dirty cache, we can say
  // all cache blocks are clean.
  bool                                   has_dirty_cache_ = false;

  // A release epoch is an interval between the events when all cache become clean.
  release_manager                        rm_;

  cache_profiler                         cprof_;
};

}
