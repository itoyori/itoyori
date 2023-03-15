#pragma once

#include <optional>
#include <algorithm>

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/allocator.hpp"
#include "ityr/ori/util.hpp"
#include "ityr/ori/cache_system.hpp"
#include "ityr/ori/coll_mem.hpp"
#include "ityr/ori/home_manager.hpp"
#include "ityr/ori/cache_manager.hpp"
#include "ityr/ori/options.hpp"

namespace ityr::ori::core {

template <block_size_t BlockSize>
class core {
public:
  core(std::size_t cache_size, std::size_t sub_block_size)
    : home_manager_(calc_home_mmap_limit(cache_size / BlockSize)),
      cache_manager_(cache_size, sub_block_size) {}

  static constexpr block_size_t block_size = BlockSize;

  void* malloc_coll(std::size_t size) { return malloc_coll<default_mem_mapper>(size); }

  template <template <block_size_t> typename MemMapper, typename... MemMapperArgs>
  void* malloc_coll(std::size_t size, MemMapperArgs&&... mmargs) {
    if (size == 0) {
      common::die("Memory allocation size cannot be 0");
    }

    auto mmapper = std::make_unique<MemMapper<BlockSize>>(size, common::topology::n_ranks(),
                                                          std::forward<MemMapperArgs>(mmargs)...);
    coll_mem& cm = coll_mem_create(size, std::move(mmapper));
    void* addr = cm.vm().addr();

    common::verbose("Allocate collective memory [%p, %p) (%ld bytes) (win=%p)",
                    addr, reinterpret_cast<std::byte*>(addr) + size, size, cm.win());

    return addr;
  }

  void* malloc(std::size_t size) {
    ITYR_CHECK_MESSAGE(size > 0, "Memory allocation size cannot be 0");

    void* addr = noncoll_allocator_.allocate(size);

    common::verbose("Allocate noncollective memory [%p, %p) (%ld bytes)",
                    addr, reinterpret_cast<std::byte*>(addr) + size, size);

    return addr;
  }

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

    common::verbose("Deallocate collective memory [%p, %p) (%ld bytes) (win=%p)",
                    addr, reinterpret_cast<std::byte*>(addr) + cm.size(), cm.size(), cm.win());

    coll_mem_destroy(cm);
  }

  // TODO: remove size from parameters
  void free(void* addr, std::size_t size) {
    ITYR_CHECK_MESSAGE(addr, "Null pointer was passed to free()");
    ITYR_CHECK(noncoll_allocator_.has(addr));

    common::topology::rank_t target_rank = noncoll_allocator_.get_owner(addr);

    if (target_rank == common::topology::my_rank()) {
      noncoll_allocator_.local_deallocate(addr, size);

      common::verbose("Deallocate noncollective memory [%p, %p) (%ld bytes) locally",
                      addr, reinterpret_cast<std::byte*>(addr) + size, size);

    } else {
      // ensure dirty data of this memory object are discarded
      for_each_block(addr, size, [&](std::byte* blk_addr,
                                     std::byte* req_addr_b,
                                     std::byte* req_addr_e) {
        cache_manager_.discard_dirty(blk_addr, req_addr_b, req_addr_e);
      });

      noncoll_allocator_.remote_deallocate(addr, size, target_rank);

      common::verbose("Deallocate noncollective memory [%p, %p) (%ld bytes) remotely (rank=%d)",
                      addr, reinterpret_cast<std::byte*>(addr) + size, size, target_rank);
    }
  }

  void get(const void* from_addr, void* to_addr, std::size_t size) {
    // TODO: support get/put for data larger than the cache size
    if (size <= BlockSize) {
      // if the size is sufficiently small, it is safe to skip incrementing reference count for cache blocks
      checkout_impl<mode::read_t, false>(reinterpret_cast<std::byte*>(const_cast<void*>(from_addr)), size);
      std::memcpy(to_addr, from_addr, size);
    } else {
      checkout_impl<mode::read_t, true>(reinterpret_cast<std::byte*>(const_cast<void*>(from_addr)), size);
      std::memcpy(to_addr, from_addr, size);
      checkin_impl<mode::read_t, true>(reinterpret_cast<std::byte*>(const_cast<void*>(from_addr)), size);
    }
  }

  void put(const void* from_addr, void* to_addr, std::size_t size) {
    if (size <= BlockSize) {
      // if the size is sufficiently small, it is safe to skip incrementing reference count for cache blocks
      checkout_impl<mode::write_t, false>(reinterpret_cast<std::byte*>(to_addr), size);
      std::memcpy(to_addr, from_addr, size);
      checkin_impl<mode::write_t, false>(reinterpret_cast<std::byte*>(to_addr), size);
    } else {
      checkout_impl<mode::write_t, true>(reinterpret_cast<std::byte*>(to_addr), size);
      std::memcpy(to_addr, from_addr, size);
      checkin_impl<mode::write_t, true>(reinterpret_cast<std::byte*>(to_addr), size);
    }
  }

  template <typename Mode>
  void checkout(void* addr, std::size_t size, Mode) {
    static_assert(!std::is_same_v<Mode, mode::no_access_t>);

    common::verbose("Checkout request (mode: %s) for [%p, %p) (%ld bytes)",
                    str(Mode{}).c_str(), addr, reinterpret_cast<std::byte*>(addr) + size, size);

    checkout_impl<Mode, true>(reinterpret_cast<std::byte*>(addr), size);
  }

  template <typename Mode>
  void checkin(void* addr, std::size_t size, Mode) {
    static_assert(!std::is_same_v<Mode, mode::no_access_t>);

    common::verbose("Checkin request (mode: %s) for [%p, %p) (%ld bytes)",
                    str(Mode{}).c_str(), addr, reinterpret_cast<std::byte*>(addr) + size, size);

    checkin_impl<Mode, true>(reinterpret_cast<std::byte*>(addr), size);
  }

  void release() {
    common::verbose("Release fence begin");

    cache_manager_.release();

    common::verbose("Release fence end");
  }

  void acquire() {
    common::verbose("Acquire fence begin");

    cache_manager_.acquire();

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

  template <typename Fn>
  void for_each_block(void* addr, std::size_t size, Fn&& fn) {
    std::byte* blk_addr_b = common::round_down_pow2(reinterpret_cast<std::byte*>(addr), BlockSize);
    std::byte* blk_addr_e = reinterpret_cast<std::byte*>(addr) + size;

    for (std::byte* blk_addr = blk_addr_b; blk_addr < blk_addr_e; blk_addr += BlockSize) {
      std::byte* req_addr_b = std::max(reinterpret_cast<std::byte*>(addr), blk_addr);
      std::byte* req_addr_e = std::min(reinterpret_cast<std::byte*>(addr) + size, blk_addr + BlockSize);
      std::forward<Fn>(fn)(blk_addr, req_addr_b, req_addr_e);
    }
  }

  template <typename HomeSegFn, typename CacheBlkFn>
  void for_each_seg(const coll_mem& cm, void* addr, std::size_t size,
                    HomeSegFn&& home_seg_fn, CacheBlkFn&& cache_blk_fn) {
    for_each_mem_segment(cm, addr, size, [&](const auto& seg) {
      std::byte*  seg_addr = reinterpret_cast<std::byte*>(cm.vm().addr()) + seg.offset_b;
      std::size_t seg_size = seg.offset_e - seg.offset_b;

      if (common::topology::is_locally_accessible(seg.owner)) {
        // no need to iterate over memory blocks (of BlockSize) for home segments
        std::forward<HomeSegFn>(home_seg_fn)(seg_addr, seg_size, seg.owner, seg.pm_offset);

      } else {
        // iterate over memory blocks within the memory segment for cache blocks
        std::byte* addr_b = std::max(seg_addr, reinterpret_cast<std::byte*>(addr));
        std::byte* addr_e = std::min(seg_addr + seg_size, reinterpret_cast<std::byte*>(addr) + size);
        for_each_block(addr_b, addr_e - addr_b, [&](std::byte* blk_addr,
                                                    std::byte* req_addr_b,
                                                    std::byte* req_addr_e) {
          std::size_t pm_offset = seg.pm_offset + (blk_addr - seg_addr);
          ITYR_CHECK(pm_offset + BlockSize <= cm.mem_mapper().local_size(seg.owner));
          std::forward<CacheBlkFn>(cache_blk_fn)(blk_addr, req_addr_b, req_addr_e, seg.owner, pm_offset);
        });
      }
    });
  }

  template <typename Mode, bool IncrementRef>
  void checkout_impl(std::byte* addr, std::size_t size) {
    constexpr bool skip_fetch = std::is_same_v<Mode, mode::write_t>;
    if (noncoll_allocator_.has(addr)) {
      checkout_noncoll<skip_fetch, IncrementRef>(addr, size);
    } else {
      checkout_coll<skip_fetch, IncrementRef>(addr, size);
    }
  }

  template <bool SkipFetch, bool IncrementRef>
  void checkout_coll(std::byte* addr, std::size_t size) {
    if (home_manager_.template checkout_fast<IncrementRef>(addr, size)) {
      return;
    }

    if (cache_manager_.template checkout_fast<SkipFetch, IncrementRef>(addr, size)) {
      return;
    }

    coll_mem& cm = coll_mem_get(addr);

    for_each_seg(cm, addr, size,
      // home segment
      [&](std::byte* seg_addr, std::size_t seg_size, common::topology::rank_t owner, std::size_t pm_offset) {
        home_manager_.template checkout_seg<IncrementRef>(
            seg_addr, seg_size,
            cm.intra_home_pm(common::topology::intra_rank(owner)),
            pm_offset);
      },
      // cache block
      [&](std::byte* blk_addr, std::byte* req_addr_b, std::byte* req_addr_e,
          common::topology::rank_t owner, std::size_t pm_offset) {
        cache_manager_.template checkout_blk<SkipFetch, IncrementRef>(
            blk_addr, req_addr_b, req_addr_e,
            cm.win(),
            owner,
            pm_offset);
      });

    home_manager_.checkout_complete();
    cache_manager_.checkout_complete(cm.win());
  }

  template <bool SkipFetch, bool IncrementRef>
  void checkout_noncoll(std::byte* addr, std::size_t size) {
    ITYR_CHECK(noncoll_allocator_.has(addr));

    auto target_rank = noncoll_allocator_.get_owner(addr);
    ITYR_CHECK(0 <= target_rank);
    ITYR_CHECK(target_rank < common::topology::n_ranks());

    if (common::topology::is_locally_accessible(target_rank)) {
      // There is no need to manage mmap entries for home blocks because
      // the remotable allocator employs block distribution policy.
      return;
    }

    if (cache_manager_.template checkout_fast<SkipFetch, IncrementRef>(addr, size)) {
      return;
    }

    for_each_block(addr, size, [&](std::byte* blk_addr,
                                   std::byte* req_addr_b,
                                   std::byte* req_addr_e) {
      cache_manager_.template checkout_blk<SkipFetch, IncrementRef>(
          blk_addr, req_addr_b, req_addr_e,
          noncoll_allocator_.win(),
          target_rank,
          noncoll_allocator_.get_disp(blk_addr));
    });

    cache_manager_.checkout_complete(noncoll_allocator_.win());
  }

  template <typename Mode, bool DecrementRef>
  void checkin_impl(std::byte* addr, std::size_t size) {
    constexpr bool register_dirty = !std::is_same_v<Mode, mode::read_t>;
    if (noncoll_allocator_.has(addr)) {
      checkin_noncoll<register_dirty, true>(addr, size);
    } else {
      checkin_coll<register_dirty, true>(addr, size);
    }
  }

  template <bool RegisterDirty, bool DecrementRef>
  void checkin_coll(std::byte* addr, std::size_t size) {
    if (home_manager_.template checkin_fast<DecrementRef>(addr, size)) {
      return;
    }

    if (cache_manager_.template checkin_fast<RegisterDirty, DecrementRef>(addr, size)) {
      return;
    }

    coll_mem& cm = coll_mem_get(addr);

    for_each_seg(cm, addr, size,
      // home segment
      [&](std::byte* seg_addr, std::size_t, common::topology::rank_t, std::size_t) {
        home_manager_.template checkin_seg<DecrementRef>(seg_addr);
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
    ITYR_CHECK(noncoll_allocator_.has(addr));

    auto target_rank = noncoll_allocator_.get_owner(addr);
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

    for_each_block(addr, size, [&](std::byte* blk_addr,
                                   std::byte* req_addr_b,
                                   std::byte* req_addr_e) {
      cache_manager_.template checkin_blk<RegisterDirty, DecrementRef>(
          blk_addr, req_addr_b, req_addr_e);
    });
  }

  template <block_size_t BS>
  using default_mem_mapper = mem_mapper::ITYR_ORI_DEFAULT_MEM_MAPPER<BS>;

  std::vector<std::optional<coll_mem>>                 coll_mems_;
  std::vector<std::tuple<void*, void*, coll_mem_id_t>> coll_mem_ids_;

  common::remotable_resource                           noncoll_allocator_;

  home_manager<BlockSize>                              home_manager_;
  cache_manager<BlockSize>                             cache_manager_;
};

using instance = common::singleton<core<ITYR_ORI_BLOCK_SIZE>>;

ITYR_TEST_CASE("[ityr::ori::core] malloc/free with block policy") {
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

ITYR_TEST_CASE("[ityr::ori::core] malloc/free with cyclic policy") {
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

ITYR_TEST_CASE("[ityr::ori::core] malloc and free (noncollective)") {
  common::singleton_initializer<common::topology::instance> topo;
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
  common::singleton_initializer<common::topology::instance> topo;
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
  common::singleton_initializer<common::topology::instance> topo;
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
  common::singleton_initializer<common::topology::instance> topo;
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
  common::singleton_initializer<common::topology::instance> topo;
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
  common::singleton_initializer<common::topology::instance> topo;
  constexpr block_size_t bs = 65536;
  int n_cb = 16;
  core<bs> c(n_cb * bs, bs / 4);

  auto my_rank = common::topology::my_rank();
  auto n_ranks = common::topology::n_ranks();
  auto mpicomm = common::topology::mpicomm();

  ITYR_SUBCASE("list creation") {
    int niter = 1000;
    int n_alloc_iter = 100;

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

}
