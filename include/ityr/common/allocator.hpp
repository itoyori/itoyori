#pragma once

#include <cstddef>
#include <cstdlib>
#include <sys/mman.h>

#define PCAS_HAS_MEMORY_RESOURCE __has_include(<memory_resource>)
#if PCAS_HAS_MEMORY_RESOURCE
#include <memory_resource>
namespace ityr::common { namespace pmr = std::pmr; }
#else
#include <boost/container/pmr/memory_resource.hpp>
#include <boost/container/pmr/unsynchronized_pool_resource.hpp>
#include <boost/container/pmr/pool_options.hpp>
namespace ityr::common { namespace pmr = boost::container::pmr; }
#endif

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/virtual_mem.hpp"
#include "ityr/common/physical_mem.hpp"
#include "ityr/common/freelist.hpp"

namespace ityr::common {

#ifndef ITYR_ALLOCATOR_USE_DYNAMIC_WIN
#define ITYR_ALLOCATOR_USE_DYNAMIC_WIN false
#endif
inline constexpr bool use_dynamic_win = ITYR_ALLOCATOR_USE_DYNAMIC_WIN;
#undef ITYR_ALLOCATOR_USE_DYNAMIC_WIN

class mpi_win_resource final : public pmr::memory_resource {
public:
  mpi_win_resource(void*       base_addr,
                   std::size_t max_size,
                   MPI_Win     win)
    : win_(win),
      freelist_(reinterpret_cast<uintptr_t>(base_addr), max_size) {}

  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    if (bytes % get_page_size() != 0 || alignment % get_page_size() != 0) {
      die("[ityr::common::allocator] Requests for mpi_win_resource must be page-aligned");
    }

    auto s = freelist_.get(bytes, alignment);
    if (!s.has_value()) {
      die("[ityr::common::allocator] Could not allocate memory for malloc_local()");
    }

    void* ret = reinterpret_cast<void*>(*s);

    if constexpr (use_dynamic_win) {
      MPI_Win_attach(win_, ret, bytes);
    }

    return ret;
  }

  void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
    if (bytes % get_page_size() != 0 || alignment % get_page_size() != 0) {
      die("[ityr::common::allocator] Requests for mpi_win_resource must be page-aligned");
    }

    if constexpr (use_dynamic_win) {
      MPI_Win_detach(win_, p);

      if (madvise(p, bytes, MADV_REMOVE) == -1) {
        perror("madvise");
        die("[ityr::common::allocator] madvise() failed");
      }
    }

    freelist_.add(reinterpret_cast<uintptr_t>(p), bytes);
  }

  bool do_is_equal(const pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

private:
  const MPI_Win win_;
  freelist      freelist_;
};

class block_resource final : public pmr::memory_resource {
public:
  block_resource(pmr::memory_resource* upstream_mr,
                 std::size_t           block_size)
    : upstream_mr_(upstream_mr),
      block_size_(block_size) {
    ITYR_CHECK(is_pow2(block_size));
  }

  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    if (bytes >= block_size_) {
      return upstream_mr_->allocate(bytes, alignment);
    }

    auto s = freelist_.get(bytes, alignment);
    if (!s.has_value()) {
      void* new_block = upstream_mr_->allocate(block_size_, block_size_);
      freelist_.add(reinterpret_cast<uintptr_t>(new_block), block_size_);
      s = freelist_.get(bytes, alignment);
      ITYR_CHECK(s.has_value());
    }

    return reinterpret_cast<void*>(*s);
  }

  void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
    if (bytes >= block_size_) {
      upstream_mr_->deallocate(p, bytes, alignment);
    }

    freelist_.add(reinterpret_cast<uintptr_t>(p), bytes);

    // TODO: return allocated blocks to upstream
  }

  bool do_is_equal(const pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

private:
  pmr::memory_resource* upstream_mr_;
  const std::size_t     block_size_;
  freelist              freelist_;
};

class remotable_resource final : public pmr::memory_resource {
public:
  remotable_resource(const topology& topo)
    : topo_(topo),
      local_max_size_(get_local_max_size()),
      global_max_size_(local_max_size_ * topo_.n_ranks()),
      vm_(reserve_same_vm_coll(topo, global_max_size_, local_max_size_)),
      pm_(init_pm()),
      local_base_addr_(reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(vm_.addr()) + local_max_size_ * topo_.my_rank())),
      win_(create_win()),
      win_mr_(local_base_addr_, local_max_size_, win()),
      block_mr_(&win_mr_, getenv_coll("ITYR_ALLOCATOR_BLOCK_SIZE", std::size_t(2), topo_.mpicomm()) * 1024 * 1024),
      std_pool_mr_(my_std_pool_options(), &block_mr_),
      max_unflushed_free_objs_(getenv_coll("ITYR_ALLOCATOR_MAX_UNFLUSHED_FREE_OBJS", 10, topo_.mpicomm())) {}

  MPI_Win win() const { return win_.win(); }

  bool belongs_to(const void* p) {
    return vm_.addr() <= p && p < reinterpret_cast<std::byte*>(vm_.addr()) + global_max_size_;
  }

  topology::rank_t get_owner(const void* p) const {
    return (reinterpret_cast<uintptr_t>(p) - reinterpret_cast<uintptr_t>(vm_.addr())) / local_max_size_;
  }

  std::size_t get_disp(const void* p) const {
    if constexpr (use_dynamic_win) {
      return reinterpret_cast<uintptr_t>(p);
    } else {
      return (reinterpret_cast<uintptr_t>(p) - reinterpret_cast<uintptr_t>(vm_.addr())) % local_max_size_;
    }
  }

  void* do_allocate(std::size_t bytes, std::size_t alignment = alignof(max_align_t)) override {
    std::size_t pad_bytes = round_up_pow2(sizeof(header), alignment);
    std::size_t real_bytes = bytes + pad_bytes;

    std::byte* p = reinterpret_cast<std::byte*>(std_pool_mr_.allocate(real_bytes, alignment));
    std::byte* ret = p + pad_bytes;

    printf("alloc: %p %ld\n", p, real_bytes);

    ITYR_CHECK(ret + bytes <= p + real_bytes);
    ITYR_CHECK(p + sizeof(header) <= ret);

    header* h = new(p) header {
      .prev = allocated_list_end_, .next = nullptr,
      .size = real_bytes, .alignment = alignment, .freed = 0};
    ITYR_CHECK(allocated_list_end_->next == nullptr);
    allocated_list_end_->next = h;
    allocated_list_end_ = h;

    return ret;
  }

  void do_deallocate(void* p, std::size_t bytes, std::size_t alignment = alignof(max_align_t)) override {
    std::size_t pad_bytes = round_up_pow2(sizeof(header), alignment);
    std::size_t real_bytes = bytes + pad_bytes;

    header* h = reinterpret_cast<header*>(reinterpret_cast<std::byte*>(p) - pad_bytes);
    remove_header_from_list(h);

    printf("dealloc: %p %ld\n", h, real_bytes);

    std_pool_mr_.deallocate(h, real_bytes, alignment);
  }

  bool do_is_equal(const pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

  void remote_deallocate(void* p, std::size_t bytes [[maybe_unused]], int target_rank, std::size_t alignment = alignof(max_align_t)) {
    ITYR_CHECK(topo_.my_rank() != target_rank);

    static constexpr int one = 1;
    static int ret; // dummy value; passing NULL to result_addr causes segfault on some MPI
    mpi_atomic_put_nb(&one, &ret, target_rank, get_header_disp(p, alignment), win());

    static int count = 0;
    count++;
    if (count >= max_unflushed_free_objs_) {
      mpi_win_flush_all(win());
      count = 0;
    }
  }

  void collect_deallocated() {
    header *h = allocated_list_.next;
    while (h) {
      if (h->freed) {
        header h_copy = *h;
        remove_header_from_list(h);
        std_pool_mr_.deallocate(reinterpret_cast<void*>(h), h_copy.size, h_copy.alignment);
        h = h_copy.next;
      } else {
        h = h->next;
      }
    }
  }

  // mainly for debugging
  bool empty() {
    return allocated_list_.next == nullptr;
  }

private:
  static std::string allocator_shmem_name(int inter_rank) {
    static int count = 0;
    std::stringstream ss;
    ss << "/ityr_allocator_" << count++ << "_" << inter_rank;
    return ss.str();
  }

  std::size_t get_local_max_size() const {
    std::size_t upper_limit = (std::size_t(1) << 40) / next_pow2(topo_.n_ranks());
    std::size_t default_local_size_mb;
    if constexpr (use_dynamic_win) {
      default_local_size_mb = upper_limit / 1024 / 1024;
    } else {
      default_local_size_mb = 128;
    }

    auto ret = std::size_t(getenv_coll("ITYR_ALLOCATOR_MAX_LOCAL_SIZE", default_local_size_mb, topo_.mpicomm())) * 1024 * 1024; // MB
    ITYR_CHECK(ret <= upper_limit);
    return ret;
  }

  physical_mem init_pm() const {
    physical_mem pm;

    if (topo_.intra_my_rank() == 0) {
      pm = physical_mem(allocator_shmem_name(topo_.inter_my_rank()), global_max_size_, true);
    }

    mpi_barrier(topo_.intra_mpicomm());

    if (topo_.intra_my_rank() != 0) {
      pm = physical_mem(allocator_shmem_name(topo_.inter_my_rank()), global_max_size_, false);
    }

    ITYR_CHECK(vm_.size() == global_max_size_);
    pm.map_to_vm(vm_.addr(), vm_.size(), 0);

    return pm;
  }

  mpi_win_manager<std::byte> create_win() const {
    if constexpr (use_dynamic_win) {
      return {topo_.mpicomm()};
    } else {
      auto local_base_addr = reinterpret_cast<std::byte*>(vm_.addr()) + local_max_size_ * topo_.my_rank();
      return {topo_.mpicomm(), local_base_addr, local_max_size_};
    }
  }

  // FIXME: workaround for boost
  // Ideally: pmr::pool_options{.max_blocks_per_chunk = (std::size_t)16 * 1024 * 1024 * 1024}
  pmr::pool_options my_std_pool_options() const {
    pmr::pool_options opts;
    opts.max_blocks_per_chunk = std::size_t(16) * 1024 * 1024 * 1024;
    return opts;
  }

  struct header {
    header*     prev      = nullptr;
    header*     next      = nullptr;
    std::size_t size      = 0;
    std::size_t alignment = 0;
    int         freed     = 0;
  };

  void remove_header_from_list(header* h) {
    ITYR_CHECK(h->prev);
    h->prev->next = h->next;

    if (h->next) {
      h->next->prev = h->prev;
    } else {
      ITYR_CHECK(h == allocated_list_end_);
      allocated_list_end_ = h->prev;
    }
  }

  std::size_t get_header_disp(const void* p, std::size_t alignment) const {
    std::size_t pad_bytes = round_up_pow2(sizeof(header), alignment);
    auto h = reinterpret_cast<const header*>(reinterpret_cast<const std::byte*>(p) - pad_bytes);
    const void* flag_addr = &h->freed;

    return get_disp(flag_addr);
  }

  const topology&                   topo_;
  const std::size_t                 local_max_size_;
  const std::size_t                 global_max_size_;
  virtual_mem                       vm_;
  physical_mem                      pm_;
  void*                             local_base_addr_;
  mpi_win_manager<std::byte>        win_;
  mpi_win_resource                  win_mr_;
  block_resource                    block_mr_;
  pmr::unsynchronized_pool_resource std_pool_mr_;
  int                               max_unflushed_free_objs_;
  header                            allocated_list_;
  header*                           allocated_list_end_ = &allocated_list_;
};

// Tests
// -----------------------------------------------------------------------------

ITYR_TEST_CASE("[ityr::common::allocator] basic test") {
  topology topo;
  remotable_resource allocator(topo);

  ITYR_SUBCASE("Local alloc/dealloc") {
    std::vector<std::size_t> sizes = {1, 2, 4, 8, 16, 32, 100, 200, 1000, 100000, 1000000};
    constexpr int N = 10;
    for (auto size : sizes) {
      void* ptrs[N];
      for (int i = 0; i < N; i++) {
        ptrs[i] = allocator.allocate(size);
        for (std::size_t j = 0; j < size; j += 128) {
          reinterpret_cast<char*>(ptrs[i])[j] = 0;
        }
      }
      for (int i = 0; i < N; i++) {
        allocator.deallocate(ptrs[i], size);
      }
    }
  }

  ITYR_SUBCASE("Remote access") {
    std::size_t size = 128;
    void* p = allocator.allocate(size);

    for (std::size_t i = 0; i < size; i++) {
      reinterpret_cast<uint8_t*>(p)[i] = topo.my_rank();
    }

    std::vector<void*> addrs(topo.n_ranks());
    addrs[topo.my_rank()] = p;

    // GET
    for (int target_rank = 0; target_rank < topo.n_ranks(); target_rank++) {
      addrs[target_rank] = mpi_bcast_value(addrs[target_rank], target_rank, topo.mpicomm());
      if (topo.my_rank() != target_rank) {
        std::vector<uint8_t> buf(size);
        mpi_get_nb(buf.data(), size, target_rank, allocator.get_disp(addrs[target_rank]), allocator.win());
        mpi_win_flush(target_rank, allocator.win());

        for (std::size_t i = 0; i < size; i++) {
          ITYR_CHECK(buf[i] == target_rank);
        }
      }
      mpi_barrier(topo.mpicomm());
    }

    // PUT
    std::vector<uint8_t> buf(size);
    for (std::size_t i = 0; i < size; i++) {
      buf[i] = topo.my_rank();
    }

    int target_rank = (topo.my_rank() + 1) % topo.n_ranks();
    mpi_put_nb(buf.data(), size, target_rank, allocator.get_disp(addrs[target_rank]), allocator.win());
    mpi_win_flush_all(allocator.win());

    mpi_barrier(topo.mpicomm());

    for (std::size_t i = 0; i < size; i++) {
      ITYR_CHECK(reinterpret_cast<uint8_t*>(p)[i] == (topo.n_ranks() + topo.my_rank() - 1) % topo.n_ranks());
    }

    ITYR_SUBCASE("Local free") {
      allocator.deallocate(p, size);
    }

    if (topo.n_ranks() > 1) {
      ITYR_SUBCASE("Remote free") {
        ITYR_CHECK(!allocator.empty());

        mpi_barrier(topo.mpicomm());

        int target_rank = (topo.my_rank() + 1) % topo.n_ranks();
        allocator.remote_deallocate(addrs[target_rank], size, target_rank);

        mpi_win_flush_all(allocator.win());
        mpi_barrier(topo.mpicomm());

        allocator.collect_deallocated();
      }
    }

    ITYR_CHECK(allocator.empty());
  }
}

}
