#pragma once

#include <optional>
#include <algorithm>

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/options.hpp"
#include "ityr/common/logger.hpp"
#include "ityr/common/rma.hpp"
#include "ityr/common/allocator.hpp"
#include "ityr/ori/util.hpp"
#include "ityr/ori/options.hpp"
#include "ityr/ori/prof_events.hpp"
#include "ityr/ori/coll_mem.hpp"
#include "ityr/ori/coll_mem_manager.hpp"
#include "ityr/ori/noncoll_mem.hpp"
#include "ityr/ori/home_manager.hpp"
#include "ityr/ori/cache_manager.hpp"

namespace ityr::ori::core {

template <block_size_t BlockSize, typename Fn>
void for_each_block(void* addr, std::size_t size, Fn fn) {
  std::byte* blk_addr_b = common::round_down_pow2(reinterpret_cast<std::byte*>(addr), BlockSize);
  std::byte* blk_addr_e = reinterpret_cast<std::byte*>(addr) + size;

  for (std::byte* blk_addr = blk_addr_b; blk_addr < blk_addr_e; blk_addr += BlockSize) {
    std::byte* req_addr_b = std::max(reinterpret_cast<std::byte*>(addr), blk_addr);
    std::byte* req_addr_e = std::min(reinterpret_cast<std::byte*>(addr) + size, blk_addr + BlockSize);
    fn(blk_addr, req_addr_b, req_addr_e);
  }
}

template <block_size_t BlockSize, typename HomeSegFn, typename CacheBlkFn>
void for_each_seg_blk(const coll_mem& cm, void* addr, std::size_t size,
                      HomeSegFn home_seg_fn, CacheBlkFn cache_blk_fn) {
  for_each_mem_segment(cm, addr, size, [&](const auto& seg) {
    std::byte*  seg_addr = reinterpret_cast<std::byte*>(cm.vm().addr()) + seg.offset_b;
    std::size_t seg_size = seg.offset_e - seg.offset_b;

    if (common::topology::is_locally_accessible(seg.owner)) {
      // no need to iterate over memory blocks (of BlockSize) for home segments
      home_seg_fn(seg_addr, seg_size, seg.owner, seg.pm_offset);

    } else {
      // iterate over memory blocks within the memory segment for cache blocks
      std::byte* addr_b = std::max(seg_addr, reinterpret_cast<std::byte*>(addr));
      std::byte* addr_e = std::min(seg_addr + seg_size, reinterpret_cast<std::byte*>(addr) + size);
      for_each_block<BlockSize>(addr_b, addr_e - addr_b, [&](std::byte* blk_addr,
                                                             std::byte* req_addr_b,
                                                             std::byte* req_addr_e) {
        std::size_t pm_offset = seg.pm_offset + (blk_addr - seg_addr);
        ITYR_CHECK(pm_offset + BlockSize <= cm.mem_mapper().local_size(seg.owner));
        cache_blk_fn(blk_addr, req_addr_b, req_addr_e, seg.owner, pm_offset);
      });
    }
  });
}

template <block_size_t BlockSize>
class core_default {
  static constexpr bool enable_vm_map = ITYR_ORI_ENABLE_VM_MAP;

public:
  core_default(std::size_t cache_size, std::size_t sub_block_size)
    : noncoll_mem_(noncoll_allocator_size_option::value()),
      home_manager_(calc_home_mmap_limit(cache_size / BlockSize)),
      cache_manager_(cache_size, sub_block_size) {}

  static constexpr block_size_t block_size = BlockSize;

  void* malloc_coll(std::size_t size) { return malloc_coll<default_mem_mapper>(size); }

  template <template <block_size_t> typename MemMapper, typename... MemMapperArgs>
  void* malloc_coll(std::size_t size, MemMapperArgs&&... mmargs) {
    ITYR_REQUIRE_MESSAGE(size > 0, "Memory allocation size cannot be 0");
    ITYR_REQUIRE_MESSAGE(size == common::mpi_bcast_value(size, 0, common::topology::mpicomm()),
                         "The size passed to malloc_coll() is different among workers");

    auto mmapper = std::make_unique<MemMapper<BlockSize>>(size, common::topology::n_ranks(),
                                                          std::forward<MemMapperArgs>(mmargs)...);
    coll_mem& cm = cm_manager_.create(size, std::move(mmapper));
    void* addr = cm.vm().addr();

    common::verbose("Allocate collective memory [%p, %p) (%ld bytes) (win=%p)",
                    addr, reinterpret_cast<std::byte*>(addr) + size, size, &cm.win());

    return addr;
  }

  void* malloc(std::size_t size) {
    ITYR_CHECK_MESSAGE(size > 0, "Memory allocation size cannot be 0");

    void* addr = noncoll_mem_.allocate(size);

    common::verbose<2>("Allocate noncollective memory [%p, %p) (%ld bytes)",
                       addr, reinterpret_cast<std::byte*>(addr) + size, size);

    return addr;
  }

  void free_coll(void* addr) {
    ITYR_REQUIRE_MESSAGE(addr, "Null pointer was passed to free()");
    ITYR_REQUIRE_MESSAGE(addr == common::mpi_bcast_value(addr, 0, common::topology::mpicomm()),
                         "The address passed to free_coll() is different among workers");

    // ensure free safety
    cache_manager_.ensure_all_cache_clean();

    coll_mem& cm = cm_manager_.get(addr);
    ITYR_CHECK(addr == cm.vm().addr());

    // ensure all cache entries are evicted
    for (std::size_t o = 0; o < cm.effective_size(); o += BlockSize) {
      std::byte* addr = reinterpret_cast<std::byte*>(cm.vm().addr()) + o;
      home_manager_.ensure_evicted(addr);
      cache_manager_.ensure_evicted(addr);
    }

    home_manager_.clear_tlb();
    cache_manager_.clear_tlb();

    common::verbose("Deallocate collective memory [%p, %p) (%ld bytes) (win=%p)",
                    addr, reinterpret_cast<std::byte*>(addr) + cm.size(), cm.size(), &cm.win());

    cm_manager_.destroy(cm);
  }

  // TODO: remove size from parameters
  void free(void* addr, std::size_t size) {
    ITYR_CHECK_MESSAGE(addr, "Null pointer was passed to free()");
    ITYR_CHECK(noncoll_mem_.has(addr));

    common::topology::rank_t target_rank = noncoll_mem_.get_owner(addr);

    if (target_rank == common::topology::my_rank()) {
      noncoll_mem_.local_deallocate(addr, size);

      common::verbose<2>("Deallocate noncollective memory [%p, %p) (%ld bytes) locally",
                         addr, reinterpret_cast<std::byte*>(addr) + size, size);

    } else {
      // ensure dirty data of this memory object are discarded
      for_each_block<BlockSize>(addr, size, [&](std::byte* blk_addr,
                                                std::byte* req_addr_b,
                                                std::byte* req_addr_e) {
        cache_manager_.discard_dirty(blk_addr, req_addr_b, req_addr_e);
      });

      noncoll_mem_.remote_deallocate(addr, size, target_rank);

      common::verbose<2>("Deallocate noncollective memory [%p, %p) (%ld bytes) remotely (rank=%d)",
                         addr, reinterpret_cast<std::byte*>(addr) + size, size, target_rank);
    }
  }

  void get(const void* from_addr, void* to_addr, std::size_t size) {
    ITYR_PROFILER_RECORD(prof_event_get);

    std::byte* from_addr_ = reinterpret_cast<std::byte*>(const_cast<void*>(from_addr));

    // TODO: support get/put for data larger than the cache size
    if (common::round_down_pow2(from_addr_, BlockSize) ==
        common::round_down_pow2(from_addr_ + size, BlockSize)) {
      // if the size is sufficiently small, it is safe to skip incrementing reference count for cache blocks
      if (!checkout_impl_nb<mode::read_t, false>(from_addr_, size)) {
        checkout_complete_impl();
      }
      get_copy_impl(from_addr_, reinterpret_cast<std::byte*>(to_addr), size);
    } else {
      if (!checkout_impl_nb<mode::read_t, true>(from_addr_, size)) {
        checkout_complete_impl();
      }
      get_copy_impl(from_addr_, reinterpret_cast<std::byte*>(to_addr), size);
      checkin_impl<mode::read_t, true>(from_addr_, size);
    }
  }

  void put(const void* from_addr, void* to_addr, std::size_t size) {
    ITYR_PROFILER_RECORD(prof_event_put);

    std::byte* to_addr_ = reinterpret_cast<std::byte*>(to_addr);

    if (common::round_down_pow2(to_addr_, BlockSize) ==
        common::round_down_pow2(to_addr_ + size, BlockSize)) {
      // if the size is sufficiently small, it is safe to skip incrementing reference count for cache blocks
      if (!checkout_impl_nb<mode::write_t, false>(to_addr_, size)) {
        checkout_complete_impl();
      }
      put_copy_impl(reinterpret_cast<const std::byte*>(from_addr), to_addr_, size);
      checkin_impl<mode::write_t, false>(to_addr_, size);
    } else {
      if (!checkout_impl_nb<mode::write_t, true>(to_addr_, size)) {
        checkout_complete_impl();
      }
      put_copy_impl(reinterpret_cast<const std::byte*>(from_addr), to_addr_, size);
      checkin_impl<mode::write_t, true>(to_addr_, size);
    }
  }

  template <typename Mode>
  bool checkout_nb(void* addr, std::size_t size, Mode) {
    if constexpr (!enable_vm_map) {
      common::die("ITYR_ORI_ENABLE_VM_MAP must be true for core::checkout/checkin");
    }

    ITYR_PROFILER_RECORD(prof_event_checkout_nb);
    common::verbose<2>("Checkout request (mode: %s) for [%p, %p) (%ld bytes)",
                       str(Mode{}).c_str(), addr, reinterpret_cast<std::byte*>(addr) + size, size);

    ITYR_CHECK(addr);
    ITYR_CHECK(size > 0);

    return checkout_impl_nb<Mode, true>(reinterpret_cast<std::byte*>(addr), size);
  }

  template <typename Mode>
  void checkout(void* addr, std::size_t size, Mode mode) {
    if (!checkout_nb(addr, size, mode)) {
      checkout_complete();
    }
  }

  void checkout_complete() {
    ITYR_PROFILER_RECORD(prof_event_checkout_comp);
    checkout_complete_impl();
  }

  template <typename Mode>
  void checkin(void* addr, std::size_t size, Mode) {
    if constexpr (!enable_vm_map) {
      common::die("ITYR_ORI_ENABLE_VM_MAP must be true for core::checkout/checkin");
    }

    ITYR_PROFILER_RECORD(prof_event_checkin);
    common::verbose<2>("Checkin request (mode: %s) for [%p, %p) (%ld bytes)",
                       str(Mode{}).c_str(), addr, reinterpret_cast<std::byte*>(addr) + size, size);

    ITYR_CHECK(addr);
    ITYR_CHECK(size > 0);

    checkin_impl<Mode, true>(reinterpret_cast<std::byte*>(addr), size);
  }

  void release() {
    common::verbose("Release fence begin");

    cache_manager_.release();

    common::verbose("Release fence end");
  }

  using release_handler = typename cache_manager<BlockSize>::release_handler;

  release_handler release_lazy() {
    common::verbose<2>("Lazy release handler is created");

    return cache_manager_.release_lazy();
  }

  void acquire() {
    common::verbose("Acquire fence begin");

    cache_manager_.acquire();

    common::verbose("Acquire fence end");
  }

  void acquire(release_handler rh) {
    common::verbose("Acquire fence (lazy) begin");

    cache_manager_.acquire(rh);

    common::verbose("Acquire fence (lazy) end");
  }

  void set_readonly_coll(void* addr, std::size_t size) {
    release();
    common::mpi_barrier(common::topology::mpicomm());

    cache_manager_.set_readonly(addr, size);

    common::mpi_barrier(common::topology::mpicomm());
  }

  void unset_readonly_coll(void* addr, std::size_t size) {
    common::mpi_barrier(common::topology::mpicomm());

    cache_manager_.unset_readonly(addr, size);

    common::mpi_barrier(common::topology::mpicomm());
  }

  void poll() {
    cache_manager_.poll();
  }

  void collect_deallocated() {
    noncoll_mem_.collect_deallocated();
  }

  void cache_prof_begin() {
    home_manager_.home_prof_begin();
    cache_manager_.cache_prof_begin();
  }

  void cache_prof_end() {
    home_manager_.home_prof_end();
    cache_manager_.cache_prof_end();
  }

  void cache_prof_print() const {
    home_manager_.home_prof_print();
    cache_manager_.cache_prof_print();
  }

  /* APIs for debugging */

  void* get_local_mem(void* addr) {
    coll_mem& cm = cm_manager_.get(addr);
    return cm.local_home_vm().addr();
  }

private:
  std::size_t calc_home_mmap_limit(std::size_t n_cache_blocks) const {
    std::size_t sys_limit = sys_mmap_entry_limit();
    std::size_t margin = 1000;
    ITYR_CHECK(sys_limit > 2 * n_cache_blocks + margin);

    std::size_t candidate = (sys_limit - 2 * n_cache_blocks - margin) / 2;
    std::size_t max_val = 1024 * 1024; // some systems may have a too large vm.max_map_count value

    return std::min(max_val, candidate);
  }

  template <typename Mode, bool IncrementRef>
  bool checkout_impl_nb(std::byte* addr, std::size_t size) {
    constexpr bool skip_fetch = std::is_same_v<Mode, mode::write_t>;
    if (noncoll_mem_.has(addr)) {
      return checkout_noncoll_nb<skip_fetch, IncrementRef>(addr, size);
    } else {
      return checkout_coll_nb<skip_fetch, IncrementRef>(addr, size);
    }
  }

  template <bool SkipFetch, bool IncrementRef>
  bool checkout_coll_nb(std::byte* addr, std::size_t size) {
    if (home_manager_.template checkout_fast<IncrementRef>(addr, size)) {
      return true;
    }

    auto [entry_found, fetch_completed] =
      cache_manager_.template checkout_fast<SkipFetch, IncrementRef>(addr, size);
    if (entry_found) {
      return fetch_completed;
    }

    coll_mem& cm = cm_manager_.get(addr);

    bool checkout_completed = true;

    for_each_seg_blk<BlockSize>(cm, addr, size,
      // home segment
      [&](std::byte* seg_addr, std::size_t seg_size, common::topology::rank_t owner, std::size_t pm_offset) {
        checkout_completed &=
          home_manager_.template checkout_seg<IncrementRef>(
              seg_addr, seg_size, addr, size,
              cm.intra_home_pm(common::topology::intra_rank(owner)), pm_offset,
              cm.home_all_mapped());
      },
      // cache block
      [&](std::byte* blk_addr, std::byte* req_addr_b, std::byte* req_addr_e,
          common::topology::rank_t owner, std::size_t pm_offset) {
        checkout_completed &=
          cache_manager_.template checkout_blk<SkipFetch, IncrementRef>(
              blk_addr, req_addr_b, req_addr_e,
              cm.win(), owner, pm_offset);
      });

    return checkout_completed;
  }

  template <bool SkipFetch, bool IncrementRef>
  bool checkout_noncoll_nb(std::byte* addr, std::size_t size) {
    ITYR_CHECK(noncoll_mem_.has(addr));

    auto target_rank = noncoll_mem_.get_owner(addr);
    ITYR_CHECK(0 <= target_rank);
    ITYR_CHECK(target_rank < common::topology::n_ranks());

    if (common::topology::is_locally_accessible(target_rank)) {
      // There is no need to manage mmap entries for home blocks because
      // the remotable allocator employs block distribution policy.
      home_manager_.on_checkout_noncoll(size);
      return true;
    }

    auto [entry_found, fetch_completed] =
      cache_manager_.template checkout_fast<SkipFetch, IncrementRef>(addr, size);
    if (entry_found) {
      return fetch_completed;
    }

    bool checkout_completed = true;

    for_each_block<BlockSize>(addr, size, [&](std::byte* blk_addr,
                                              std::byte* req_addr_b,
                                              std::byte* req_addr_e) {
      checkout_completed &=
        cache_manager_.template checkout_blk<SkipFetch, IncrementRef>(
            blk_addr, req_addr_b, req_addr_e,
            noncoll_mem_.win(),
            target_rank,
            noncoll_mem_.get_disp(blk_addr));
    });

    return checkout_completed;
  }

  template <typename Mode, bool DecrementRef>
  void checkin_impl(std::byte* addr, std::size_t size) {
    constexpr bool register_dirty = !std::is_same_v<Mode, mode::read_t>;
    if (noncoll_mem_.has(addr)) {
      checkin_noncoll<register_dirty, DecrementRef>(addr, size);
    } else {
      checkin_coll<register_dirty, DecrementRef>(addr, size);
    }
  }

  void checkout_complete_impl() {
    home_manager_.checkout_complete();
    cache_manager_.checkout_complete();
  }

  template <bool RegisterDirty, bool DecrementRef>
  void checkin_coll(std::byte* addr, std::size_t size) {
    if (home_manager_.template checkin_fast<DecrementRef>(addr, size)) {
      return;
    }

    if (cache_manager_.template checkin_fast<RegisterDirty, DecrementRef>(addr, size)) {
      return;
    }

    coll_mem& cm = cm_manager_.get(addr);

    for_each_seg_blk<BlockSize>(cm, addr, size,
      // home segment
      [&](std::byte* seg_addr, std::size_t, common::topology::rank_t, std::size_t) {
        home_manager_.template checkin_seg<DecrementRef>(seg_addr, cm.home_all_mapped());
      },
      // cache block
      [&](std::byte* blk_addr, std::byte* req_addr_b, std::byte* req_addr_e,
          common::topology::rank_t, std::size_t) {
        cache_manager_.template checkin_blk<RegisterDirty, DecrementRef>(
            blk_addr, req_addr_b, req_addr_e);
      });
  }

  template <bool RegisterDirty, bool DecrementRef>
  void checkin_noncoll(std::byte* addr, std::size_t size) {
    ITYR_CHECK(noncoll_mem_.has(addr));

    auto target_rank = noncoll_mem_.get_owner(addr);
    ITYR_CHECK(0 <= target_rank);
    ITYR_CHECK(target_rank < common::topology::n_ranks());

    if (common::topology::is_locally_accessible(target_rank)) {
      // There is no need to manage mmap entries for home blocks because
      // the remotable allocator employs block distribution policy.
      return;
    }

    if (cache_manager_.template checkin_fast<RegisterDirty, DecrementRef>(addr, size)) {
      return;
    }

    for_each_block<BlockSize>(addr, size, [&](std::byte* blk_addr,
                                              std::byte* req_addr_b,
                                              std::byte* req_addr_e) {
      cache_manager_.template checkin_blk<RegisterDirty, DecrementRef>(
          blk_addr, req_addr_b, req_addr_e);
    });
  }

  /*
   * The following get/put_copy_* functions are mainly for performance evaluation for cases
   * in which virtual memory mapping is not used and instead the data are always copied between
   * the cache region and the user buffer via GET/PUT calls. Thus, checkout/checkin cannot be
   * used when enable_vm_map is false.
   */

  void get_copy_impl(std::byte* from_addr, std::byte* to_addr, std::size_t size) {
    if constexpr (enable_vm_map) {
      std::memcpy(to_addr, from_addr, size);
    } else if (noncoll_mem_.has(from_addr)) {
      get_copy_noncoll(from_addr, to_addr, size);
    } else {
      get_copy_coll(from_addr, to_addr, size);
    }
  }

  void get_copy_coll(std::byte* from_addr, std::byte* to_addr, std::size_t size) {
    ITYR_CHECK(!enable_vm_map);

    coll_mem& cm = cm_manager_.get(from_addr);

    for_each_seg_blk<BlockSize>(cm, from_addr, size,
      // home segment
      [&](std::byte* seg_addr, std::size_t seg_size, common::topology::rank_t owner, std::size_t pm_offset) {
        const common::virtual_mem& vm = cm.intra_home_vm(common::topology::intra_rank(owner));
        std::byte* seg_addr_b         = std::max(from_addr, seg_addr);
        std::byte* seg_addr_e         = std::min(seg_addr + seg_size, from_addr + size);
        std::size_t seg_offset        = seg_addr_b - seg_addr;
        std::byte* from_addr_         = reinterpret_cast<std::byte*>(vm.addr()) + pm_offset + seg_offset;
        std::byte* to_addr_           = to_addr + (seg_addr_b - from_addr);
        std::memcpy(to_addr_, from_addr_, seg_addr_e - seg_addr_b);
      },
      // cache block
      [&](std::byte* blk_addr, std::byte* req_addr_b, std::byte* req_addr_e,
          common::topology::rank_t, std::size_t) {
        cache_manager_.get_copy_blk(blk_addr, req_addr_b, req_addr_e, to_addr + (req_addr_b - from_addr));
      });
  }

  void get_copy_noncoll(std::byte* from_addr, std::byte* to_addr, std::size_t size) {
    ITYR_CHECK(!enable_vm_map);

    ITYR_CHECK(noncoll_mem_.has(from_addr));

    auto target_rank = noncoll_mem_.get_owner(from_addr);
    ITYR_CHECK(0 <= target_rank);
    ITYR_CHECK(target_rank < common::topology::n_ranks());

    if (common::topology::is_locally_accessible(target_rank)) {
      std::memcpy(to_addr, from_addr, size);
      return;
    }

    for_each_block<BlockSize>(from_addr, size, [&](std::byte* blk_addr,
                                                   std::byte* req_addr_b,
                                                   std::byte* req_addr_e) {
      cache_manager_.get_copy_blk(blk_addr, req_addr_b, req_addr_e, to_addr + (req_addr_b - from_addr));
    });
  }

  void put_copy_impl(const std::byte* from_addr, std::byte* to_addr, std::size_t size) {
    if constexpr (enable_vm_map) {
      std::memcpy(to_addr, from_addr, size);
    } else if (noncoll_mem_.has(to_addr)) {
      put_copy_noncoll(from_addr, to_addr, size);
    } else {
      put_copy_coll(from_addr, to_addr, size);
    }
  }

  void put_copy_coll(const std::byte* from_addr, std::byte* to_addr, std::size_t size) {
    ITYR_CHECK(!enable_vm_map);

    coll_mem& cm = cm_manager_.get(to_addr);

    for_each_seg_blk<BlockSize>(cm, to_addr, size,
      // home segment
      [&](std::byte* seg_addr, std::size_t seg_size, common::topology::rank_t owner, std::size_t pm_offset) {
        const common::virtual_mem& vm = cm.intra_home_vm(common::topology::intra_rank(owner));
        std::byte* seg_addr_b         = std::max(to_addr, seg_addr);
        std::byte* seg_addr_e         = std::min(seg_addr + seg_size, to_addr + size);
        std::size_t seg_offset        = seg_addr_b - seg_addr;
        const std::byte* from_addr_   = from_addr + (seg_addr_b - to_addr);
        std::byte* to_addr_           = reinterpret_cast<std::byte*>(vm.addr()) + pm_offset + seg_offset;
        std::memcpy(to_addr_, from_addr_, seg_addr_e - seg_addr_b);
      },
      // cache block
      [&](std::byte* blk_addr, std::byte* req_addr_b, std::byte* req_addr_e,
          common::topology::rank_t, std::size_t) {
        cache_manager_.put_copy_blk(blk_addr, req_addr_b, req_addr_e, from_addr + (req_addr_b - to_addr));
      });
  }

  void put_copy_noncoll(const std::byte* from_addr, std::byte* to_addr, std::size_t size) {
    ITYR_CHECK(!enable_vm_map);

    ITYR_CHECK(noncoll_mem_.has(to_addr));

    auto target_rank = noncoll_mem_.get_owner(to_addr);
    ITYR_CHECK(0 <= target_rank);
    ITYR_CHECK(target_rank < common::topology::n_ranks());

    if (common::topology::is_locally_accessible(target_rank)) {
      std::memcpy(to_addr, from_addr, size);
      return;
    }

    for_each_block<BlockSize>(to_addr, size, [&](std::byte* blk_addr,
                                                 std::byte* req_addr_b,
                                                 std::byte* req_addr_e) {
      cache_manager_.put_copy_blk(blk_addr, req_addr_b, req_addr_e, from_addr + (req_addr_b - to_addr));
    });
  }

  template <block_size_t BS>
  using default_mem_mapper = mem_mapper::ITYR_ORI_DEFAULT_MEM_MAPPER<BS>;

  coll_mem_manager         cm_manager_;
  noncoll_mem              noncoll_mem_;
  home_manager<BlockSize>  home_manager_;
  cache_manager<BlockSize> cache_manager_;
};

template <block_size_t BlockSize>
class core_nocache {
public:
  core_nocache(std::size_t, std::size_t)
    : noncoll_mem_(noncoll_allocator_size_option::value()) {}

  static constexpr block_size_t block_size = BlockSize;

  void* malloc_coll(std::size_t size) { return malloc_coll<default_mem_mapper>(size); }

  template <template <block_size_t> typename MemMapper, typename... MemMapperArgs>
  void* malloc_coll(std::size_t size, MemMapperArgs&&... mmargs) {
    if (size == 0) {
      common::die("Memory allocation size cannot be 0");
    }

    auto mmapper = std::make_unique<MemMapper<BlockSize>>(size, common::topology::n_ranks(),
                                                          std::forward<MemMapperArgs>(mmargs)...);
    coll_mem& cm = cm_manager_.create(size, std::move(mmapper));
    void* addr = cm.vm().addr();

    common::verbose("Allocate collective memory [%p, %p) (%ld bytes) (win=%p)",
                    addr, reinterpret_cast<std::byte*>(addr) + size, size, &cm.win());

    return addr;
  }

  void* malloc(std::size_t size) {
    ITYR_CHECK_MESSAGE(size > 0, "Memory allocation size cannot be 0");

    void* addr = noncoll_mem_.allocate(size);

    common::verbose<2>("Allocate noncollective memory [%p, %p) (%ld bytes)",
                       addr, reinterpret_cast<std::byte*>(addr) + size, size);

    return addr;
  }

  void free_coll(void* addr) {
    if (!addr) {
      common::die("Null pointer was passed to free_coll()");
    }

    coll_mem& cm = cm_manager_.get(addr);
    ITYR_CHECK(addr == cm.vm().addr());

    common::verbose("Deallocate collective memory [%p, %p) (%ld bytes) (win=%p)",
                    addr, reinterpret_cast<std::byte*>(addr) + cm.size(), cm.size(), &cm.win());

    cm_manager_.destroy(cm);
  }

  void free(void* addr, std::size_t size) {
    ITYR_CHECK_MESSAGE(addr, "Null pointer was passed to free()");
    ITYR_CHECK(noncoll_mem_.has(addr));

    common::topology::rank_t target_rank = noncoll_mem_.get_owner(addr);

    if (target_rank == common::topology::my_rank()) {
      noncoll_mem_.local_deallocate(addr, size);

      common::verbose<2>("Deallocate noncollective memory [%p, %p) (%ld bytes) locally",
                         addr, reinterpret_cast<std::byte*>(addr) + size, size);

    } else {
      noncoll_mem_.remote_deallocate(addr, size, target_rank);

      common::verbose<2>("Deallocate noncollective memory [%p, %p) (%ld bytes) remotely (rank=%d)",
                         addr, reinterpret_cast<std::byte*>(addr) + size, size, target_rank);
    }
  }

  void get(const void* from_addr, void* to_addr, std::size_t size) {
    ITYR_PROFILER_RECORD(prof_event_get);

    std::byte* from_addr_ = reinterpret_cast<std::byte*>(const_cast<void*>(from_addr));
    get_impl(from_addr_, reinterpret_cast<std::byte*>(to_addr), size);
  }

  void put(const void* from_addr, void* to_addr, std::size_t size) {
    ITYR_PROFILER_RECORD(prof_event_put);

    std::byte* to_addr_ = reinterpret_cast<std::byte*>(to_addr);
    put_impl(reinterpret_cast<const std::byte*>(from_addr), to_addr_, size);
  }

  template <typename Mode>
  bool checkout_nb(void*, std::size_t, Mode) {
    common::die("core::checkout/checkin is disabled");
  }

  template <typename Mode>
  void checkout(void*, std::size_t, Mode) {
    common::die("core::checkout/checkin is disabled");
  }

  void checkout_complete() {
    common::die("core::checkout/checkin is disabled");
  }

  template <typename Mode>
  void checkin(void*, std::size_t, Mode) {
    common::die("core::checkout/checkin is disabled");
  }

  void release() {}

  using release_handler = void*;

  release_handler release_lazy() { return {}; }

  void acquire() {}

  void acquire(release_handler) {}

  void set_readonly_coll(void*, std::size_t) {}
  void unset_readonly_coll(void*, std::size_t) {}

  void poll() {}

  void collect_deallocated() {
    noncoll_mem_.collect_deallocated();
  }

  void cache_prof_begin() {}
  void cache_prof_end() {}
  void cache_prof_print() const {}

  /* APIs for debugging */

  void* get_local_mem(void* addr) {
    coll_mem& cm = cm_manager_.get(addr);
    return cm.local_home_vm().addr();
  }

private:
  void get_impl(std::byte* from_addr, std::byte* to_addr, std::size_t size) {
    if (noncoll_mem_.has(from_addr)) {
      get_noncoll(from_addr, to_addr, size);
    } else {
      get_coll(from_addr, to_addr, size);
    }
  }

  void get_coll(std::byte* from_addr, std::byte* to_addr, std::size_t size) {
    coll_mem& cm = cm_manager_.get(from_addr);

    bool fetching = false;

    for_each_seg_blk<BlockSize>(cm, from_addr, size,
      // home segment
      [&](std::byte* seg_addr, std::size_t seg_size, common::topology::rank_t owner, std::size_t pm_offset) {
        const common::virtual_mem& vm = cm.intra_home_vm(common::topology::intra_rank(owner));
        std::byte* seg_addr_b         = std::max(from_addr, seg_addr);
        std::byte* seg_addr_e         = std::min(seg_addr + seg_size, from_addr + size);
        std::size_t seg_offset        = seg_addr_b - seg_addr;
        std::byte* from_addr_         = reinterpret_cast<std::byte*>(vm.addr()) + pm_offset + seg_offset;
        std::byte* to_addr_           = to_addr + (seg_addr_b - from_addr);
        std::memcpy(to_addr_, from_addr_, seg_addr_e - seg_addr_b);
      },
      // cache block
      [&](std::byte* blk_addr, std::byte* req_addr_b, std::byte* req_addr_e,
          common::topology::rank_t owner, std::size_t pm_offset) {
        common::rma::get_nb(to_addr + (req_addr_b - from_addr), req_addr_e - req_addr_b,
                            cm.win(), owner, pm_offset + (req_addr_b - blk_addr));
        fetching = true;
      });

    if (fetching) {
      common::rma::flush(cm.win());
    }
  }

  void get_noncoll(std::byte* from_addr, std::byte* to_addr, std::size_t size) {
    ITYR_CHECK(noncoll_mem_.has(from_addr));

    auto target_rank = noncoll_mem_.get_owner(from_addr);
    ITYR_CHECK(0 <= target_rank);
    ITYR_CHECK(target_rank < common::topology::n_ranks());

    if (common::topology::is_locally_accessible(target_rank)) {
      std::memcpy(to_addr, from_addr, size);
      return;
    }

    for_each_block<BlockSize>(from_addr, size, [&](std::byte* blk_addr,
                                                   std::byte* req_addr_b,
                                                   std::byte* req_addr_e) {
      common::rma::get_nb(to_addr + (req_addr_b - from_addr), req_addr_e - req_addr_b,
                          noncoll_mem_.win(), target_rank,
                          noncoll_mem_.get_disp(blk_addr) + (req_addr_b - blk_addr));
    });

    common::rma::flush(noncoll_mem_.win());
  }

  void put_impl(const std::byte* from_addr, std::byte* to_addr, std::size_t size) {
    if (noncoll_mem_.has(to_addr)) {
      put_noncoll(from_addr, to_addr, size);
    } else {
      put_coll(from_addr, to_addr, size);
    }
  }

  void put_coll(const std::byte* from_addr, std::byte* to_addr, std::size_t size) {
    coll_mem& cm = cm_manager_.get(to_addr);

    bool putting = false;

    for_each_seg_blk<BlockSize>(cm, to_addr, size,
      // home segment
      [&](std::byte* seg_addr, std::size_t seg_size, common::topology::rank_t owner, std::size_t pm_offset) {
        const common::virtual_mem& vm = cm.intra_home_vm(common::topology::intra_rank(owner));
        std::byte* seg_addr_b         = std::max(to_addr, seg_addr);
        std::byte* seg_addr_e         = std::min(seg_addr + seg_size, to_addr + size);
        std::size_t seg_offset        = seg_addr_b - seg_addr;
        const std::byte* from_addr_   = from_addr + (seg_addr_b - to_addr);
        std::byte* to_addr_           = reinterpret_cast<std::byte*>(vm.addr()) + pm_offset + seg_offset;
        std::memcpy(to_addr_, from_addr_, seg_addr_e - seg_addr_b);
      },
      // cache block
      [&](std::byte* blk_addr, std::byte* req_addr_b, std::byte* req_addr_e,
          common::topology::rank_t owner, std::size_t pm_offset) {
        common::rma::put_nb(from_addr + (req_addr_b - to_addr), req_addr_e - req_addr_b,
                            cm.win(), owner, pm_offset + (req_addr_b - blk_addr));
        putting = true;
      });

    if (putting) {
      common::rma::flush(cm.win());
    }
  }

  void put_noncoll(const std::byte* from_addr, std::byte* to_addr, std::size_t size) {
    ITYR_CHECK(noncoll_mem_.has(to_addr));

    auto target_rank = noncoll_mem_.get_owner(to_addr);
    ITYR_CHECK(0 <= target_rank);
    ITYR_CHECK(target_rank < common::topology::n_ranks());

    if (common::topology::is_locally_accessible(target_rank)) {
      std::memcpy(to_addr, from_addr, size);
      return;
    }

    for_each_block<BlockSize>(to_addr, size, [&](std::byte* blk_addr,
                                                 std::byte* req_addr_b,
                                                 std::byte* req_addr_e) {
      common::rma::put_nb(from_addr + (req_addr_b - to_addr), req_addr_e - req_addr_b,
                          noncoll_mem_.win(), target_rank,
                          noncoll_mem_.get_disp(blk_addr) + (req_addr_b - blk_addr));
    });

    common::rma::flush(noncoll_mem_.win());
  }

  template <block_size_t BS>
  using default_mem_mapper = mem_mapper::ITYR_ORI_DEFAULT_MEM_MAPPER<BS>;

  coll_mem_manager cm_manager_;
  noncoll_mem      noncoll_mem_;
};

template <block_size_t BlockSize>
class core_serial {
public:
  core_serial(std::size_t, std::size_t) {}

  static constexpr block_size_t block_size = BlockSize;

  void* malloc_coll(std::size_t size) { return std::malloc(size); }

  template <template <block_size_t> typename MemMapper, typename... MemMapperArgs>
  void* malloc_coll(std::size_t size, MemMapperArgs&&...) {
    return std::malloc(size);
  }

  void* malloc(std::size_t size) {
    return std::malloc(size);
  }

  void free_coll(void* addr) {
    std::free(addr);
  }

  void free(void* addr, std::size_t) {
    std::free(addr);
  }

  void get(const void* from_addr, void* to_addr, std::size_t size) {
    std::memcpy(to_addr, from_addr, size);
  }

  void put(const void* from_addr, void* to_addr, std::size_t size) {
    std::memcpy(to_addr, from_addr, size);
  }

  template <typename Mode>
  bool checkout_nb(void*, std::size_t, Mode) { return true; }

  template <typename Mode>
  void checkout(void*, std::size_t, Mode) {}

  void checkout_complete() {}

  template <typename Mode>
  void checkin(void*, std::size_t, Mode) {}

  void release() {}

  using release_handler = void*;

  release_handler release_lazy() { return {}; }

  void acquire() {}

  void acquire(release_handler) {}

  void set_readonly_coll(void*, std::size_t) {}
  void unset_readonly_coll(void*, std::size_t) {}

  void poll() {}

  void collect_deallocated() {}

  void cache_prof_begin() {}
  void cache_prof_end() {}
  void cache_prof_print() const {}

  /* APIs for debugging */

  void* get_local_mem(void* addr) { return addr; }
};

template <block_size_t BlockSize>
using core = ITYR_CONCAT(core_, ITYR_ORI_CORE)<BlockSize>;

using instance = common::singleton<core<ITYR_ORI_BLOCK_SIZE>>;

ITYR_TEST_CASE("[ityr::ori::core] malloc/free with block policy") {
  common::runtime_options common_opts;
  runtime_options opts;
  common::singleton_initializer<common::topology::instance> topo;
  common::singleton_initializer<common::rma::instance> rma;
  constexpr block_size_t bs = 65536;
  core<bs> c(16 * bs, bs / 4);

  ITYR_SUBCASE("free immediately") {
    int n = 10;
    for (int i = 1; i < n; i++) {
      auto p = c.malloc_coll<mem_mapper::block>(i * 1234);
      c.free_coll(p);
    }
  }

  ITYR_SUBCASE("free after accumulation") {
    int n = 10;
    void* ptrs[n];
    for (int i = 1; i < n; i++) {
      ptrs[i] = c.malloc_coll<mem_mapper::block>(i * 2743);
    }
    for (int i = 1; i < n; i++) {
      c.free_coll(ptrs[i]);
    }
  }
}

ITYR_TEST_CASE("[ityr::ori::core] malloc/free with cyclic policy") {
  common::runtime_options common_opts;
  runtime_options opts;
  common::singleton_initializer<common::topology::instance> topo;
  common::singleton_initializer<common::rma::instance> rma;
  constexpr block_size_t bs = 65536;
  core<bs> c(16 * bs, bs / 4);

  ITYR_SUBCASE("free immediately") {
    int n = 10;
    for (int i = 1; i < n; i++) {
      auto p = c.malloc_coll<mem_mapper::cyclic>(i * 123456);
      c.free_coll(p);
    }
  }

  ITYR_SUBCASE("free after accumulation") {
    int n = 10;
    void* ptrs[n];
    for (int i = 1; i < n; i++) {
      ptrs[i] = c.malloc_coll<mem_mapper::cyclic>(i * 27438, bs * i);
    }
    for (int i = 1; i < n; i++) {
      c.free_coll(ptrs[i]);
    }
  }
}

ITYR_TEST_CASE("[ityr::ori::core] malloc and free (noncollective)") {
  common::runtime_options common_opts;
  runtime_options opts;
  common::singleton_initializer<common::topology::instance> topo;
  common::singleton_initializer<common::rma::instance> rma;
  constexpr block_size_t bs = 65536;
  core<bs> c(16 * bs, bs / 4);

  constexpr int n = 10;
  ITYR_SUBCASE("free immediately") {
    for (int i = 0; i < n; i++) {
      void* p = c.malloc(std::size_t(1) << i);
      c.free(p, std::size_t(1) << i);
    }
  }

  ITYR_SUBCASE("free after accumulation") {
    void* ptrs[n];
    for (int i = 0; i < n; i++) {
      ptrs[i] = c.malloc(std::size_t(1) << i);
    }
    for (int i = 0; i < n; i++) {
      c.free(ptrs[i], std::size_t(1) << i);
    }
  }

  ITYR_SUBCASE("remote free") {
    void* ptrs_send[n];
    void* ptrs_recv[n];
    for (int i = 0; i < n; i++) {
      ptrs_send[i] = c.malloc(std::size_t(1) << i);
    }

    auto my_rank = common::topology::my_rank();
    auto n_ranks = common::topology::n_ranks();
    auto mpicomm = common::topology::mpicomm();

    auto req_send = common::mpi_isend(ptrs_send, n, (n_ranks + my_rank + 1) % n_ranks, 0, mpicomm);
    auto req_recv = common::mpi_irecv(ptrs_recv, n, (n_ranks + my_rank - 1) % n_ranks, 0, mpicomm);
    common::mpi_wait(req_send);
    common::mpi_wait(req_recv);

    for (int i = 0; i < n; i++) {
      c.free(ptrs_recv[i], std::size_t(1) << i);
    }
  }
}

ITYR_TEST_CASE("[ityr::ori::core] get/put") {
  common::runtime_options common_opts;
  runtime_options opts;
  common::singleton_initializer<common::topology::instance> topo;
  common::singleton_initializer<common::rma::instance> rma;
  constexpr block_size_t bs = 65536;
  int n_cb = 16;
  core<bs> c(n_cb * bs, bs / 4);

  auto my_rank = common::topology::my_rank();

  std::size_t n = n_cb * bs / sizeof(std::size_t);

  std::size_t* ps[2];
  ps[0] = reinterpret_cast<std::size_t*>(c.malloc_coll<mem_mapper::block >(n * sizeof(std::size_t)));
  ps[1] = reinterpret_cast<std::size_t*>(c.malloc_coll<mem_mapper::cyclic>(n * sizeof(std::size_t)));

  std::size_t* buf = new std::size_t[n + 2];

  auto barrier = [&]() {
    c.release();
    common::mpi_barrier(common::topology::mpicomm());
    c.acquire();
  };

  for (auto p : ps) {
    if (my_rank == 0) {
      for (std::size_t i = 0; i < n; i++) {
        buf[i] = i;
      }
      c.put(buf, p, n * sizeof(std::size_t));
    }

    barrier();

    ITYR_SUBCASE("get the entire array") {
      std::size_t special = 417;
      buf[0] = buf[n + 1] = special;

      c.get(p, buf + 1, n * sizeof(std::size_t));

      for (std::size_t i = 0; i < n; i++) {
        ITYR_CHECK(buf[i + 1] == i);
      }
      ITYR_CHECK(buf[0]     == special);
      ITYR_CHECK(buf[n + 1] == special);
    }

    ITYR_SUBCASE("get the partial array") {
      std::size_t ib = n / 5 * 2;
      std::size_t ie = n / 5 * 4;
      std::size_t s = ie - ib;

      std::size_t special = 417;
      buf[0] = buf[s + 1] = special;

      c.get(p + ib, buf + 1, s * sizeof(std::size_t));

      for (std::size_t i = 0; i < s; i++) {
        ITYR_CHECK(buf[i + 1] == i + ib);
      }
      ITYR_CHECK(buf[0]     == special);
      ITYR_CHECK(buf[s + 1] == special);
    }

    ITYR_SUBCASE("get each element") {
      for (std::size_t i = 0; i < n; i++) {
        std::size_t special = 417;
        buf[0] = buf[2] = special;
        c.get(p + i, &buf[1], sizeof(std::size_t));
        ITYR_CHECK(buf[0] == special);
        ITYR_CHECK(buf[1] == i);
        ITYR_CHECK(buf[2] == special);
      }
    }
  }

  delete[] buf;

  c.free_coll(ps[0]);
  c.free_coll(ps[1]);
}

ITYR_TEST_CASE("[ityr::ori::core] checkout/checkin (small, aligned)") {
  common::runtime_options common_opts;
  runtime_options opts;
  common::singleton_initializer<common::topology::instance> topo;
  common::singleton_initializer<common::rma::instance> rma;
  constexpr block_size_t bs = 65536;
  int n_cb = 16;
  core<bs> c(n_cb * bs, bs / 4);

  auto my_rank = common::topology::my_rank();
  auto n_ranks = common::topology::n_ranks();

  int n = bs * n_ranks;
  uint8_t* ps[2];
  ps[0] = reinterpret_cast<uint8_t*>(c.malloc_coll<mem_mapper::block >(n));
  ps[1] = reinterpret_cast<uint8_t*>(c.malloc_coll<mem_mapper::cyclic>(n));

  auto barrier = [&]() {
    c.release();
    common::mpi_barrier(common::topology::mpicomm());
    c.acquire();
  };

  for (auto p : ps) {
    uint8_t* home_ptr = reinterpret_cast<uint8_t*>(c.get_local_mem(p));
    for (std::size_t i = 0; i < bs; i++) {
      home_ptr[i] = my_rank;
    }

    barrier();

    ITYR_SUBCASE("read the entire array") {
      c.checkout(p, n, mode::read);
      for (int i = 0; i < n; i++) {
        ITYR_CHECK_MESSAGE(p[i] == i / bs, "rank: ", my_rank, ", i: ", i);
      }
      c.checkin(p, n, mode::read);
    }

    ITYR_SUBCASE("read and write the entire array") {
      for (int iter = 0; iter < n_ranks; iter++) {
        if (iter == my_rank) {
          c.checkout(p, n, mode::read_write);
          for (int i = 0; i < n; i++) {
            ITYR_CHECK_MESSAGE(p[i] == i / bs + iter, "iter: ", iter, ", rank: ", my_rank, ", i: ", i);
            p[i]++;
          }
          c.checkin(p, n, mode::read_write);
        }

        barrier();

        c.checkout(p, n, mode::read);
        for (int i = 0; i < n; i++) {
          ITYR_CHECK_MESSAGE(p[i] == i / bs + iter + 1, "iter: ", iter, ", rank: ", my_rank, ", i: ", i);
        }
        c.checkin(p, n, mode::read);

        barrier();
      }
    }

    ITYR_SUBCASE("read the partial array") {
      int ib = n / 5 * 2;
      int ie = n / 5 * 4;
      int s = ie - ib;

      c.checkout(p + ib, s, mode::read);
      for (int i = 0; i < s; i++) {
        ITYR_CHECK_MESSAGE(p[ib + i] == (i + ib) / bs, "rank: ", my_rank, ", i: ", i);
      }
      c.checkin(p + ib, s, mode::read);
    }
  }

  c.free_coll(ps[0]);
  c.free_coll(ps[1]);
}

ITYR_TEST_CASE("[ityr::ori::core] checkout/checkin (large, not aligned)") {
  common::runtime_options common_opts;
  runtime_options opts;
  common::singleton_initializer<common::topology::instance> topo;
  common::singleton_initializer<common::rma::instance> rma;
  constexpr block_size_t bs = 65536;
  int n_cb = 16;
  core<bs> c(n_cb * bs, bs / 4);

  auto my_rank = common::topology::my_rank();
  auto n_ranks = common::topology::n_ranks();

  std::size_t n = 10 * n_cb * bs / sizeof(std::size_t);

  std::size_t* ps[2];
  ps[0] = reinterpret_cast<std::size_t*>(c.malloc_coll<mem_mapper::block >(n * sizeof(std::size_t)));
  ps[1] = reinterpret_cast<std::size_t*>(c.malloc_coll<mem_mapper::cyclic>(n * sizeof(std::size_t)));

  std::size_t max_checkout_size = (16 - 2) * bs / sizeof(std::size_t);

  auto barrier = [&]() {
    c.release();
    common::mpi_barrier(common::topology::mpicomm());
    c.acquire();
  };

  for (auto p : ps) {
    if (my_rank == 0) {
      for (std::size_t i = 0; i < n; i += max_checkout_size) {
        std::size_t m = std::min(max_checkout_size, n - i);
        c.checkout(p + i, m * sizeof(std::size_t), mode::write);
        for (std::size_t j = i; j < i + m; j++) {
          p[j] = j;
        }
        c.checkin(p + i, m * sizeof(std::size_t), mode::write);
      }
    }

    barrier();

    ITYR_SUBCASE("read the entire array") {
      for (std::size_t i = 0; i < n; i += max_checkout_size) {
        std::size_t m = std::min(max_checkout_size, n - i);
        c.checkout(p + i, m * sizeof(std::size_t), mode::read);
        for (std::size_t j = i; j < i + m; j++) {
          ITYR_CHECK(p[j] == j);
        }
        c.checkin(p + i, m * sizeof(std::size_t), mode::read);
      }
    }

    ITYR_SUBCASE("read the partial array") {
      std::size_t ib = n / 5 * 2;
      std::size_t ie = n / 5 * 4;
      std::size_t s = ie - ib;

      for (std::size_t i = 0; i < s; i += max_checkout_size) {
        std::size_t m = std::min(max_checkout_size, s - i);
        c.checkout(p + ib + i, m * sizeof(std::size_t), mode::read);
        for (std::size_t j = ib + i; j < ib + i + m; j++) {
          ITYR_CHECK(p[j] == j);
        }
        c.checkin(p + ib + i, m * sizeof(std::size_t), mode::read);
      }
    }

    ITYR_SUBCASE("read and write the partial array") {
      std::size_t stride = 48;
      ITYR_REQUIRE(stride <= max_checkout_size);
      for (std::size_t i = my_rank * stride; i < n; i += n_ranks * stride) {
        std::size_t s = std::min(stride, n - i);
        c.checkout(p + i, s * sizeof(std::size_t), mode::read_write);
        for (std::size_t j = i; j < i + s; j++) {
          ITYR_CHECK(p[j] == j);
          p[j] *= 2;
        }
        c.checkin(p + i, s * sizeof(std::size_t), mode::read_write);
      }

      barrier();

      for (std::size_t i = 0; i < n; i += max_checkout_size) {
        std::size_t m = std::min(max_checkout_size, n - i);
        c.checkout(p + i, m * sizeof(std::size_t), mode::read);
        for (std::size_t j = i; j < i + m; j++) {
          ITYR_CHECK(p[j] == j * 2);
        }
        c.checkin(p + i, m * sizeof(std::size_t), mode::read);
      }
    }
  }

  c.free_coll(ps[0]);
  c.free_coll(ps[1]);
}

ITYR_TEST_CASE("[ityr::ori::core] checkout/checkin (noncontig)") {
  common::runtime_options common_opts;
  runtime_options opts;
  common::singleton_initializer<common::topology::instance> topo;
  common::singleton_initializer<common::rma::instance> rma;
  constexpr block_size_t bs = 65536;
  int n_cb = 8;
  core<bs> c(n_cb * bs, bs / 4);

  auto my_rank = common::topology::my_rank();
  auto n_ranks = common::topology::n_ranks();

  std::size_t n = 2 * n_cb * bs / sizeof(std::size_t);

  std::size_t* ps[2];
  ps[0] = reinterpret_cast<std::size_t*>(c.malloc_coll<mem_mapper::block >(n * sizeof(std::size_t)));
  ps[1] = reinterpret_cast<std::size_t*>(c.malloc_coll<mem_mapper::cyclic>(n * sizeof(std::size_t)));

  auto barrier = [&]() {
    c.release();
    common::mpi_barrier(common::topology::mpicomm());
    c.acquire();
  };

  for (auto p : ps) {
    for (std::size_t i = my_rank; i < n; i += n_ranks) {
      c.checkout(p + i, sizeof(std::size_t), mode::write);
      p[i] = i;
      c.checkin(p + i, sizeof(std::size_t), mode::write);
    }

    barrier();

    for (std::size_t i = (my_rank + 1) % n_ranks; i < n; i += n_ranks) {
      c.checkout(p + i, sizeof(std::size_t), mode::read_write);
      ITYR_CHECK(p[i] == i);
      p[i] *= 2;
      c.checkin(p + i, sizeof(std::size_t), mode::read_write);
    }

    barrier();

    for (std::size_t i = (my_rank + 2) % n_ranks; i < n; i += n_ranks) {
      if (i % 3 == 0) {
        c.checkout(p + i, sizeof(std::size_t), mode::write);
        p[i] = i * 10;
        c.checkin(p + i, sizeof(std::size_t), mode::write);
      } else {
        c.checkout(p + i, sizeof(std::size_t), mode::read);
        ITYR_CHECK(p[i] == i * 2);
        c.checkin(p + i, sizeof(std::size_t), mode::read);
      }
    }

    barrier();

    for (std::size_t i = (my_rank + 3) % n_ranks; i < n; i += n_ranks) {
      c.checkout(p + i, sizeof(std::size_t), mode::read);
      if (i % 3 == 0) {
        ITYR_CHECK(p[i] == i * 10);
      } else {
        ITYR_CHECK(p[i] == i * 2);
      }
      c.checkin(p + i, sizeof(std::size_t), mode::read);
    }
  }

  c.free_coll(ps[0]);
  c.free_coll(ps[1]);
}

ITYR_TEST_CASE("[ityr::ori::core] checkout/checkin (noncollective)") {
  common::runtime_options common_opts;
  runtime_options opts;
  common::singleton_initializer<common::topology::instance> topo;
  common::singleton_initializer<common::rma::instance> rma;
  constexpr block_size_t bs = 65536;
  int n_cb = 16;
  core<bs> c(n_cb * bs, bs / 4);

  auto my_rank = common::topology::my_rank();
  auto n_ranks = common::topology::n_ranks();
  auto mpicomm = common::topology::mpicomm();

  ITYR_SUBCASE("list creation") {
    int niter = 1000;
    int n_alloc_iter = 10;

    struct node_t {
      node_t* next = nullptr;
      int     value;
    };

    node_t* root_node = new (c.malloc(sizeof(node_t))) node_t;

    c.checkout(root_node, sizeof(node_t), mode::write);
    root_node->next = nullptr;
    root_node->value = my_rank;
    c.checkin(root_node, sizeof(node_t), mode::write);

    node_t* node = root_node;
    for (int i = 0; i < niter; i++) {
      for (int j = 0; j < n_alloc_iter; j++) {
        // append a new node
        node_t* new_node = new (c.malloc(sizeof(node_t))) node_t;

        c.checkout(&node->next, sizeof(node->next), mode::write);
        node->next = new_node;
        c.checkin(&node->next, sizeof(node->next), mode::write);

        c.checkout(&node->value, sizeof(node->value), mode::read);
        int val = node->value;
        c.checkin(&node->value, sizeof(node->value), mode::read);

        c.checkout(new_node, sizeof(node_t), mode::write);
        new_node->next = nullptr;
        new_node->value = val + 1;
        c.checkin(new_node, sizeof(node_t), mode::write);

        node = new_node;
      }

      c.release();

      // exchange nodes across nodes
      node_t* next_node;

      auto req_send = common::mpi_isend(&node     , 1, (n_ranks + my_rank + 1) % n_ranks, i, mpicomm);
      auto req_recv = common::mpi_irecv(&next_node, 1, (n_ranks + my_rank - 1) % n_ranks, i, mpicomm);
      common::mpi_wait(req_send);
      common::mpi_wait(req_recv);

      node = next_node;

      c.acquire();
    }

    c.release();
    common::mpi_barrier(mpicomm);
    c.acquire();

    int count = 0;
    node = root_node;
    while (node != nullptr) {
      c.checkout(node, sizeof(node_t), mode::read);

      ITYR_CHECK(node->value == my_rank + count);

      node_t* prev_node = node;
      node = node->next;

      c.checkin(prev_node, sizeof(node_t), mode::read);

      std::destroy_at(prev_node);
      c.free(prev_node, sizeof(node_t));

      count++;
    }
  }
}

ITYR_TEST_CASE("[ityr::ori::core] release/acquire fence") {
  common::runtime_options common_opts;
  runtime_options opts;
  common::singleton_initializer<common::topology::instance> topo;
  common::singleton_initializer<common::rma::instance> rma;
  constexpr block_size_t bs = 65536;
  int n_cb = 16;
  core<bs> c(n_cb * bs, bs / 4);

  auto my_rank = common::topology::my_rank();
  auto n_ranks = common::topology::n_ranks();
  auto mpicomm = common::topology::mpicomm();

  auto barrier = [&]() {
    c.release();
    common::mpi_barrier(mpicomm);
    c.acquire();
  };

  int* p = reinterpret_cast<int*>(c.malloc_coll(sizeof(int)));

  if (my_rank == 0) {
    c.checkout(p, sizeof(int), mode::write);
    p[0] = 3;
    c.checkin(p, sizeof(int), mode::write);
  }

  barrier();

  c.checkout(p, sizeof(int), mode::read);
  ITYR_CHECK(p[0] == 3);
  c.checkin(p, sizeof(int), mode::read);

  barrier();

  if (my_rank == (n_ranks + 1) % n_ranks) {
    c.checkout(p, sizeof(int), mode::read_write);
    p[0] += 5;
    c.checkin(p, sizeof(int), mode::read_write);
  }

  barrier();

  c.checkout(p, sizeof(int), mode::read);
  ITYR_CHECK(p[0] == 8);
  c.checkin(p, sizeof(int), mode::read);

  barrier();

  int n = 100;
  for (int i = 0; i < n; i++) {
    auto root_rank = (n_ranks + i) % n_ranks;
    if (my_rank == root_rank) {
      c.checkout(p, sizeof(int), mode::read_write);
      p[0] += 12;
      c.checkin(p, sizeof(int), mode::read_write);
    }

    core<bs>::release_handler rh;

    if (my_rank == root_rank) {
      rh = c.release_lazy();
    }

    rh = common::mpi_bcast_value(rh, root_rank, mpicomm);

    if (my_rank != root_rank) {
      c.acquire(rh);
    }

    c.checkout(p, sizeof(int), mode::read);
    ITYR_CHECK(p[0] == 20 + 12 * i);
    c.checkin(p, sizeof(int), mode::read);

    auto req = common::mpi_ibarrier(mpicomm);
    while (!common::mpi_test(req)) {
      c.poll();
    }
  }

  c.free_coll(p);
}

}
