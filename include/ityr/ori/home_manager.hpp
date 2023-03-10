#pragma once

#include <algorithm>

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/virtual_mem.hpp"
#include "ityr/common/physical_mem.hpp"
#include "ityr/ori/util.hpp"
#include "ityr/ori/block_regions.hpp"
#include "ityr/ori/cache_system.hpp"

namespace ityr::ori {

template <block_size_t BlockSize>
class home_manager {
public:
  home_manager(std::size_t mmap_entry_limit)
    : mmap_entry_limit_(mmap_entry_limit),
      cs_(mmap_entry_limit_, mmap_entry(this)) {}

  using epoch_t = uint64_t;

  struct mmap_entry {
    cache_entry_idx_t           entry_idx   = std::numeric_limits<cache_entry_idx_t>::max();
    std::byte*                  addr        = nullptr;
    std::byte*                  mapped_addr = nullptr;
    std::size_t                 size        = 0;
    std::size_t                 mapped_size = 0;
    const common::physical_mem* pm          = nullptr;
    std::size_t                 pm_offset   = 0;
    int                         ref_count   = 0;
    home_manager*               outer;

    mmap_entry(home_manager* outer_p) : outer(outer_p) {}

    /* Callback functions for cache_system class */

    bool is_evictable() const {
      return ref_count == 0;
    }

    void on_evict() {
      ITYR_CHECK(is_evictable());
      ITYR_CHECK(mapped_addr == addr);
      entry_idx = std::numeric_limits<cache_entry_idx_t>::max();
      // for safety
      /* outer->home_tlb_.clear(); */
    }

    void on_cache_map(cache_entry_idx_t idx) {
      entry_idx = idx;
    }
  };

  template <bool UpdateLRU = true>
  mmap_entry& get_entry(void* addr) {
    try {
      return cs_.template ensure_cached<UpdateLRU>(cache_key(addr));
    } catch (cache_full_exception& e) {
      common::die("home segments are exhausted (too much checked-out memory)");
    }
  }

  void update_mapping(mmap_entry& me) {
    if (me.mapped_addr) {
      common::verbose("Unmap home segment [%p, %p) (size=%ld)",
                      me.mapped_addr, me.mapped_addr + me.mapped_size, me.mapped_size);
      common::mmap_no_physical_mem(me.mapped_addr, me.mapped_size, true);
    }
    ITYR_CHECK(me.pm);
    ITYR_CHECK(me.addr);
    common::verbose("Map home segment [%p, %p) (size=%ld)",
                    me.addr, me.addr + me.size, me.size);
    me.pm->map_to_vm(me.addr, me.size, me.pm_offset);
    me.mapped_addr = me.addr;
    me.mapped_size = me.size;
  }

  void ensure_evicted(void* addr) {
    cs_.ensure_evicted(cache_key(addr));
  }

private:
  using cache_key_t = uintptr_t;

  cache_key_t cache_key(void* addr) const {
    ITYR_CHECK(addr);
    ITYR_CHECK(reinterpret_cast<uintptr_t>(addr) % BlockSize == 0);
    return reinterpret_cast<uintptr_t>(addr) / BlockSize;
  }

  std::size_t                           mmap_entry_limit_;
  cache_system<cache_key_t, mmap_entry> cs_;
};

}
