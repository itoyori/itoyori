#pragma once

#include <optional>
#include <algorithm>

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/ori/util.hpp"
#include "ityr/ori/cache_system.hpp"
#include "ityr/ori/coll_mem.hpp"
#include "ityr/ori/home_manager.hpp"
#include "ityr/ori/cache_manager.hpp"
#include "ityr/ori/options.hpp"

namespace ityr::ori {

enum class access_mode {
  read,
  write,
  read_write,
};

inline std::string str(access_mode mode) {
  switch (mode) {
    case access_mode::read:       return "read";
    case access_mode::write:      return "write";
    case access_mode::read_write: return "read_write";
  }
  return "UNKNOWN";
}

template <block_size_t BlockSize>
class core {
public:
  core(std::size_t cache_size, std::size_t sub_BlockSize)
    : home_manager_(calc_home_mmap_limit(cache_size / BlockSize)),
      cache_manager_(cache_size, sub_BlockSize) {}

  void* malloc_coll(std::size_t size) { return malloc_coll<default_mem_mapper>(size); }

  template <template <block_size_t> typename MemMapper, typename... MemMapperArgs>
  void* malloc_coll(std::size_t size, MemMapperArgs... mmargs) {
    if (size == 0) {
      common::die("Memory allocation size cannot be 0");
    }

    auto mmapper = std::make_unique<MemMapper<BlockSize>>(size, common::topology::n_ranks(), mmargs...);
    coll_mem& cm = coll_mem_create(size, std::move(mmapper));
    void* addr = cm.vm().addr();

    common::verbose("Allocate collective memory [%p, %p) (%ld bytes) (win=%p)",
                    addr, reinterpret_cast<std::byte*>(addr) + size, size, cm.win());

    return addr;
  }

  void* malloc(std::size_t size);

  void free_coll(void* addr) {
    if (!addr) {
      common::die("Null pointer was passed to free_coll()");
    }

    // ensure free safety
    cache_manager_.ensure_all_cache_clean();

    coll_mem& cm = coll_mem_get(addr);
    ITYR_CHECK(addr == cm.vm().addr());

    // ensure all cache entries are evicted
    for (std::size_t o = 0; o < cm.effective_size(); o += BlockSize) {
      std::byte* addr = reinterpret_cast<std::byte*>(cm.vm().addr()) + o;
      home_manager_.ensure_evicted(addr);
      cache_manager_.ensure_evicted(addr);
    }

    /* home_tlb_.clear(); */
    /* cache_tlb_.clear(); */

    common::verbose("Deallocate collective memory [%p, %p) (%ld bytes) (win=%p)",
                    addr, reinterpret_cast<std::byte*>(addr) + cm.size(), cm.size(), cm.win());

    coll_mem_destroy(cm);
  }

  void* free(void* addr);

  template <access_mode Mode>
  void checkout(void* addr, std::size_t size) {
    common::verbose("Checkout request (mode: %s) for [%p, %p) (%ld bytes) (win=%p)",
                    str(Mode).c_str(), addr, reinterpret_cast<std::byte*>(addr) + size, size);

    checkout_coll<Mode, true>(reinterpret_cast<std::byte*>(addr), size);
  }

  template <access_mode Mode>
  void checkin(void* addr, std::size_t size) {
    common::verbose("Checkin request (mode: %s) for [%p, %p) (%ld bytes) (win=%p)",
                    str(Mode).c_str(), addr, reinterpret_cast<std::byte*>(addr) + size, size);

    checkin_coll<Mode, true>(reinterpret_cast<std::byte*>(addr), size);
  }

  void release() {
    common::verbose("Release fence begin");

    cache_manager_.ensure_all_cache_clean();

    common::verbose("Release fence end");
  }

  void acquire() {
    common::verbose("Acquire fence begin");

    cache_manager_.invalidate_all();

    common::verbose("Acquire fence end");
  }

  /* APIs for debugging */

  void* get_local_mem(void* addr) {
    coll_mem& cm = coll_mem_get(addr);
    return cm.local_home_vm().addr();
  }

private:
  std::size_t calc_home_mmap_limit(std::size_t n_cache_blocks) const {
    std::size_t sys_limit = sys_mmap_entry_limit();
    std::size_t margin = 1000;
    ITYR_CHECK(sys_limit > n_cache_blocks + margin);
    return (sys_limit - n_cache_blocks - margin) / 2;
  }

  coll_mem& coll_mem_get(void* addr) {
    for (auto [addr_begin, addr_end, id] : coll_mem_ids_) {
      if (addr_begin <= addr && addr < addr_end) {
        return *coll_mems_[id];
      }
    }
    common::die("Address %p was passed but not allocated by Itoyori", addr);
  }

  coll_mem& coll_mem_create(std::size_t size, std::unique_ptr<mem_mapper::base> mmapper) {
    coll_mem_id_t id = coll_mems_.size();

    coll_mem& cm = *coll_mems_.emplace_back(std::in_place, size, id, std::move(mmapper));
    std::byte* raw_ptr = reinterpret_cast<std::byte*>(cm.vm().addr());

    coll_mem_ids_.emplace_back(std::make_tuple(raw_ptr, raw_ptr + size, id));

    return cm;
  }

  void coll_mem_destroy(coll_mem& cm) {
    std::byte* p = reinterpret_cast<std::byte*>(cm.vm().addr());
    auto it = std::find(coll_mem_ids_.begin(), coll_mem_ids_.end(),
                        std::make_tuple(p, p + cm.size(), cm.id()));
    ITYR_CHECK(it != coll_mem_ids_.end());
    coll_mem_ids_.erase(it);

    coll_mems_[cm.id()].reset();
  }

  template <access_mode Mode, bool IncrementRef>
  void checkout_coll(std::byte* addr, std::size_t size) {
    coll_mem& cm = coll_mem_get(addr);

    bool fetched = false;
    std::vector<mmap_entry*> home_segments_to_map;
    std::vector<cache_block*> cache_blocks_to_map;

    for_each_mem_block<BlockSize>(cm, addr, size,
      // home segment
      [&](std::byte* seg_addr, std::size_t seg_size, common::topology::rank_t owner, std::size_t pm_offset) {
        mmap_entry& me = home_manager_.get_entry(seg_addr);

        if (seg_addr != me.mapped_addr) {
          me.addr      = seg_addr;
          me.size      = seg_size;
          me.pm        = &cm.intra_home_pm(common::topology::intra_rank(owner));
          me.pm_offset = pm_offset;
          home_segments_to_map.push_back(&me);
        }

        if constexpr (IncrementRef) {
          me.ref_count++;
        }
      },
      // cache block
      [&](std::byte* blk_addr, common::topology::rank_t owner, std::size_t pm_offset) {
        cache_block& cb = cache_manager_.get_entry(blk_addr);

        if (blk_addr != cb.mapped_addr) {
          cb.addr      = blk_addr;
          cb.win       = cm.win();
          cb.owner     = owner;
          cb.pm_offset = pm_offset;
          cache_blocks_to_map.push_back(&cb);
        }

        block_region br = {std::max(addr, blk_addr) - blk_addr,
                           std::min(addr + size, blk_addr + BlockSize) - blk_addr};

        if constexpr (Mode == access_mode::write) {
          cb.fresh_regions.add(br);

        } else {
          if (cache_manager_.fetch_begin(cb, br)) {
            fetched = true;
          }
        }

        if constexpr (IncrementRef) {
          cb.ref_count++;
        }

        /* cache_tlb_.add(&cb); */
      });

    // Overlap communication and memory remapping
    for (mmap_entry* me : home_segments_to_map) {
      home_manager_.update_mapping(*me);
    }
    for (cache_block* cb : cache_blocks_to_map) {
      cache_manager_.update_mapping(*cb);
    }

    if (fetched) {
      cache_manager_.fetch_complete(cm.win());
    }
  }

  template <access_mode Mode, bool DecrementRef>
  void checkin_coll(std::byte* addr, std::size_t size) {
    coll_mem& cm = coll_mem_get(addr);

    for_each_mem_block<BlockSize>(cm, addr, size,
      // home segment
      [&](std::byte* seg_addr, std::size_t, common::topology::rank_t, std::size_t) {
        if constexpr (DecrementRef) {
          mmap_entry& me = home_manager_.template get_entry<false>(seg_addr);
          me.ref_count--;
        }
      },
      // cache block
      [&](std::byte* blk_addr, common::topology::rank_t, std::size_t) {
        cache_block& cb = cache_manager_.template get_entry<false>(blk_addr);

        if constexpr (Mode != access_mode::read) {
          block_region br = {std::max(addr, blk_addr) - blk_addr,
                             std::min(addr + size, blk_addr + BlockSize) - blk_addr};
          cache_manager_.add_dirty_region(cb, br);
        }

        if constexpr (DecrementRef) {
          cb.ref_count--;
        }
      });
  }

  template <block_size_t BS>
  using default_mem_mapper = mem_mapper::ITYR_ORI_DEFAULT_MEM_MAPPER<BS>;

  using mmap_entry = typename home_manager<BlockSize>::mmap_entry;
  using cache_block = typename cache_manager<BlockSize>::cache_block;

  std::vector<std::optional<coll_mem>>                 coll_mems_;
  std::vector<std::tuple<void*, void*, coll_mem_id_t>> coll_mem_ids_;

  home_manager<BlockSize>                              home_manager_;
  cache_manager<BlockSize>                             cache_manager_;
};

ITYR_TEST_CASE("[ityr::ori::core] malloc and free with block policy") {
  common::singleton_initializer<common::topology::instance> topo;
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

ITYR_TEST_CASE("[ityr::ori::core] malloc and free with cyclic policy") {
  common::singleton_initializer<common::topology::instance> topo;
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

ITYR_TEST_CASE("[ityr::ori::core] checkout and checkin (small, aligned)") {
  common::singleton_initializer<common::topology::instance> topo;
  constexpr block_size_t bs = 65536;
  core<bs> c(16 * bs, bs / 4);

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
      c.checkout<access_mode::read>(p, n);
      for (int i = 0; i < n; i++) {
        ITYR_CHECK_MESSAGE(p[i] == i / bs, "rank: ", my_rank, ", i: ", i);
      }
      c.checkin<access_mode::read>(p, n);
    }

    ITYR_SUBCASE("read and write the entire array") {
      for (int iter = 0; iter < n_ranks; iter++) {
        if (iter == my_rank) {
          c.checkout<access_mode::read_write>(p, n);
          for (int i = 0; i < n; i++) {
            ITYR_CHECK_MESSAGE(p[i] == i / bs + iter, "iter: ", iter, ", rank: ", my_rank, ", i: ", i);
            p[i]++;
          }
          c.checkin<access_mode::read_write>(p, n);
        }

        barrier();

        c.checkout<access_mode::read>(p, n);
        for (int i = 0; i < n; i++) {
          ITYR_CHECK_MESSAGE(p[i] == i / bs + iter + 1, "iter: ", iter, ", rank: ", my_rank, ", i: ", i);
        }
        c.checkin<access_mode::read>(p, n);

        barrier();
      }
    }

    ITYR_SUBCASE("read the partial array") {
      int ib = n / 5 * 2;
      int ie = n / 5 * 4;
      int s = ie - ib;

      c.checkout<access_mode::read>(p + ib, s);
      for (int i = 0; i < s; i++) {
        ITYR_CHECK_MESSAGE(p[ib + i] == (i + ib) / bs, "rank: ", my_rank, ", i: ", i);
      }
      c.checkin<access_mode::read>(p + ib, s);
    }
  }

  c.free_coll(ps[0]);
  c.free_coll(ps[1]);
}

ITYR_TEST_CASE("[ityr::ori::core] checkout and checkin (large, not aligned)") {
  common::singleton_initializer<common::topology::instance> topo;
  constexpr block_size_t bs = 65536;
  core<bs> c(16 * bs, bs / 4);

  auto my_rank = common::topology::my_rank();
  auto n_ranks = common::topology::n_ranks();

  std::size_t n = 10000000;

  int* ps[2];
  ps[0] = reinterpret_cast<int*>(c.malloc_coll<mem_mapper::block >(n * sizeof(int)));
  ps[1] = reinterpret_cast<int*>(c.malloc_coll<mem_mapper::cyclic>(n * sizeof(int)));

  std::size_t max_checkout_size = (16 - 2) * bs / sizeof(int);

  auto barrier = [&]() {
    c.release();
    common::mpi_barrier(common::topology::mpicomm());
    c.acquire();
  };

  for (auto p : ps) {
    if (my_rank == 0) {
      for (std::size_t i = 0; i < n; i += max_checkout_size) {
        int m = std::min(max_checkout_size, n - i);
        c.checkout<access_mode::write>(p + i, m * sizeof(int));
        for (std::size_t j = i; j < i + m; j++) {
          p[j] = j;
        }
        c.checkin<access_mode::write>(p + i, m * sizeof(int));
      }
    }

    barrier();

    ITYR_SUBCASE("read the entire array") {
      for (std::size_t i = 0; i < n; i += max_checkout_size) {
        std::size_t m = std::min(max_checkout_size, n - i);
        c.checkout<access_mode::read>(p + i, m * sizeof(int));
        for (std::size_t j = i; j < i + m; j++) {
          ITYR_CHECK(p[j] == j);
        }
        c.checkin<access_mode::read>(p + i, m * sizeof(int));
      }
    }

    ITYR_SUBCASE("read the partial array") {
      std::size_t ib = n / 5 * 2;
      std::size_t ie = n / 5 * 4;
      std::size_t s = ie - ib;

      for (std::size_t i = 0; i < s; i += max_checkout_size) {
        std::size_t m = std::min(max_checkout_size, s - i);
        c.checkout<access_mode::read>(p + ib + i, m * sizeof(int));
        for (std::size_t j = ib + i; j < ib + i + m; j++) {
          ITYR_CHECK(p[j] == j);
        }
        c.checkin<access_mode::read>(p + ib + i, m * sizeof(int));
      }
    }

    ITYR_SUBCASE("read and write the partial array") {
      std::size_t stride = 48;
      ITYR_REQUIRE(stride <= max_checkout_size);
      for (std::size_t i = my_rank * stride; i < n; i += n_ranks * stride) {
        std::size_t s = std::min(stride, n - i);
        c.checkout<access_mode::read_write>(p + i, s * sizeof(int));
        for (std::size_t j = i; j < i + s; j++) {
          ITYR_CHECK(p[j] == j);
          p[j] *= 2;
        }
        c.checkin<access_mode::read_write>(p + i, s * sizeof(int));
      }

      barrier();

      for (std::size_t i = 0; i < n; i += max_checkout_size) {
        std::size_t m = std::min(max_checkout_size, n - i);
        c.checkout<access_mode::read>(p + i, m * sizeof(int));
        for (std::size_t j = i; j < i + m; j++) {
          ITYR_CHECK(p[j] == j * 2);
        }
        c.checkin<access_mode::read>(p + i, m * sizeof(int));
      }
    }
  }

  c.free_coll(ps[0]);
  c.free_coll(ps[1]);
}

}
