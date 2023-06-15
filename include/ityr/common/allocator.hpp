#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <atomic>
#include <sys/mman.h>

#if ITYR_ALLOCATOR_USE_BOOST
#include <boost/container/pmr/memory_resource.hpp>
#include <boost/container/pmr/unsynchronized_pool_resource.hpp>
#include <boost/container/pmr/pool_options.hpp>
namespace ityr::common { namespace pmr = boost::container::pmr; }
#else
#include <memory_resource>
namespace ityr::common { namespace pmr = std::pmr; }
#endif

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/mpi_rma.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/virtual_mem.hpp"
#include "ityr/common/physical_mem.hpp"
#include "ityr/common/freelist.hpp"
#include "ityr/common/profiler.hpp"
#include "ityr/common/prof_events.hpp"
#include "ityr/common/options.hpp"

namespace ityr::common {

inline constexpr bool use_dynamic_win = ITYR_ALLOCATOR_USE_DYNAMIC_WIN;

class mpi_win_resource final : public pmr::memory_resource {
public:
  mpi_win_resource(void*       base_addr,
                   std::size_t max_size,
                   MPI_Win     win)
    : win_(win),
      freelist_(reinterpret_cast<uintptr_t>(base_addr), max_size) {}

  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    if (alignment % get_page_size() != 0) {
      die("[ityr::common::allocator] Requests for mpi_win_resource must be page-aligned");
    }

    // Align with page size
    std::size_t real_bytes = round_up_pow2(bytes, get_page_size());

    auto s = freelist_.get(real_bytes, alignment);
    if (!s.has_value()) {
      die("[ityr::common::allocator] Could not allocate memory for malloc_local()");
    }

    void* ret = reinterpret_cast<void*>(*s);

    if constexpr (use_dynamic_win) {
      MPI_Win_attach(win_, ret, real_bytes);
    }

    return ret;
  }

  void do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override {
    if (alignment % get_page_size() != 0) {
      die("[ityr::common::allocator] Requests for mpi_win_resource must be page-aligned");
    }

    // Align with page size
    std::size_t real_bytes = round_up_pow2(bytes, get_page_size());

    if constexpr (use_dynamic_win) {
      MPI_Win_detach(win_, p);

      if (madvise(p, real_bytes, MADV_REMOVE) == -1) {
        perror("madvise");
        die("[ityr::common::allocator] madvise() failed");
      }
    }

    freelist_.add(reinterpret_cast<uintptr_t>(p), real_bytes);
  }

  bool do_is_equal(const pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

private:
  MPI_Win  win_;
  freelist freelist_;
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
      return upstream_mr_->allocate(bytes, std::max(alignment, block_size_));
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
      upstream_mr_->deallocate(p, bytes, std::max(alignment, block_size_));
      return;
    }

    freelist_.add(reinterpret_cast<uintptr_t>(p), bytes);

    // TODO: return allocated blocks to upstream
  }

  bool do_is_equal(const pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

private:
  pmr::memory_resource* upstream_mr_;
  std::size_t           block_size_;
  freelist              freelist_;
};

class remotable_resource final : public pmr::memory_resource {
public:
  remotable_resource(std::size_t local_max_size)
    : local_max_size_(calc_local_max_size(local_max_size)),
      global_max_size_(local_max_size_ * topology::n_ranks()),
      vm_(reserve_same_vm_coll(global_max_size_, local_max_size_)),
      pm_(init_pm()),
      local_base_addr_(reinterpret_cast<std::byte*>(vm_.addr()) + local_max_size_ * topology::my_rank()),
      win_(create_win()),
      win_mr_(local_base_addr_, local_max_size_, win()),
      block_mr_(&win_mr_, allocator_block_size_option::value()),
      std_pool_mr_(my_std_pool_options(), &block_mr_),
      max_unflushed_free_objs_(allocator_max_unflushed_free_objs_option::value()),
      allocated_size_(0),
      collect_threshold_(std::size_t(16) * 1024),
      collect_threshold_max_(local_max_size_ * 8 / 10) {}

  MPI_Win win() const { return win_.win(); }

  bool has(const void* p) const {
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
    ITYR_PROFILER_RECORD(prof_event_allocator_alloc);

    std::size_t pad_bytes = round_up_pow2(sizeof(header), alignment);
    std::size_t real_bytes = bytes + pad_bytes;

    if (allocated_size_ >= collect_threshold_) {
      collect_deallocated();
      collect_threshold_ = allocated_size_ * 2;
      if (collect_threshold_ > collect_threshold_max_) {
        collect_threshold_ = (collect_threshold_max_ + allocated_size_) / 2;
      }
    }

    std::byte* p = reinterpret_cast<std::byte*>(std_pool_mr_.allocate(real_bytes, alignment));
    std::byte* ret = p + pad_bytes;

    ITYR_CHECK(ret + bytes <= p + real_bytes);
    ITYR_CHECK(p + sizeof(header) <= ret);

    header* h = new (p) header {
      .prev = allocated_list_end_, .next = nullptr,
      .size = real_bytes, .alignment = alignment, .freed = 0};
    ITYR_CHECK(allocated_list_end_->next == nullptr);
    allocated_list_end_->next = h;
    allocated_list_end_ = h;

    allocated_size_ += real_bytes;

    return ret;
  }

  void do_deallocate(void* p, std::size_t bytes, std::size_t alignment = alignof(max_align_t)) override {
    auto target_rank = get_owner(p);
    if (target_rank == topology::my_rank()) {
      local_deallocate(p, bytes, alignment);
    } else {
      remote_deallocate(p, bytes, target_rank, alignment);
    }
  }

  bool do_is_equal(const pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

  void local_deallocate(void* p, std::size_t bytes, std::size_t alignment = alignof(max_align_t)) {
    ITYR_PROFILER_RECORD(prof_event_allocator_free_local);

    ITYR_CHECK(get_owner(p) == topology::my_rank());

    std::size_t pad_bytes = round_up_pow2(sizeof(header), alignment);
    std::size_t real_bytes = bytes + pad_bytes;

    header* h = reinterpret_cast<header*>(reinterpret_cast<std::byte*>(p) - pad_bytes);
    ITYR_CHECK(h->size == real_bytes);
    ITYR_CHECK(h->alignment == alignment);
    ITYR_CHECK(h->freed == 0);

    local_deallocate_impl(h, real_bytes, alignment);
  }

  void remote_deallocate(void* p, std::size_t bytes [[maybe_unused]], int target_rank, std::size_t alignment = alignof(max_align_t)) {
    ITYR_PROFILER_RECORD(prof_event_allocator_free_remote, target_rank);

    ITYR_CHECK(topology::my_rank() != target_rank);
    ITYR_CHECK(get_owner(p) == target_rank);

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
    ITYR_PROFILER_RECORD(prof_event_allocator_collect);

    header *h = allocated_list_.next;
    while (h) {
      if (h->freed.load(std::memory_order_acquire)) {
        header* h_next = h->next;
        local_deallocate_impl(h, h->size, h->alignment);
        h = h_next;
      } else {
        h = h->next;
      }
    }
  }

  bool is_locally_accessible(const void* p) const {
    return topology::is_locally_accessible(get_owner(p));
  }

  bool is_remotely_freed(void* p, std::size_t alignment = alignof(max_align_t)) {
    ITYR_CHECK(get_owner(p) == topology::my_rank());

    std::size_t pad_bytes = round_up_pow2(sizeof(header), alignment);
    header* h = reinterpret_cast<header*>(reinterpret_cast<std::byte*>(p) - pad_bytes);

    if (h->freed.load(std::memory_order_acquire)) {
      local_deallocate_impl(h, h->size, h->alignment);
      return true;
    }
    return false;
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

  std::size_t calc_local_max_size(std::size_t param) const {
    if (param == 0) {
      ITYR_CHECK(use_dynamic_win);
      return (std::size_t(1) << 40) / next_pow2(topology::n_ranks());
    } else {
      return param;
    }
  }

  physical_mem init_pm() const {
    physical_mem pm;

    if (topology::intra_my_rank() == 0) {
      pm = physical_mem(allocator_shmem_name(topology::inter_my_rank()), global_max_size_, true);
    }

    mpi_barrier(topology::intra_mpicomm());

    if (topology::intra_my_rank() != 0) {
      pm = physical_mem(allocator_shmem_name(topology::inter_my_rank()), global_max_size_, false);
    }

    ITYR_CHECK(vm_.size() == global_max_size_);

    for (topology::rank_t r = 0; r < topology::intra_n_ranks(); r++) {
      auto target_rank = topology::intra2global_rank(r);
      auto offset = local_max_size_ * target_rank;
      void* begin_addr = reinterpret_cast<std::byte*>(vm_.addr()) + offset;
      pm.map_to_vm(begin_addr, local_max_size_, offset);
    }

    return pm;
  }

  mpi_win_manager<std::byte> create_win() const {
    if constexpr (use_dynamic_win) {
      return {topology::mpicomm()};
    } else {
      auto local_base_addr = reinterpret_cast<std::byte*>(vm_.addr()) + local_max_size_ * topology::my_rank();
      return {topology::mpicomm(), local_base_addr, local_max_size_};
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
    header*          prev      = nullptr;
    header*          next      = nullptr;
    std::size_t      size      = 0;
    std::size_t      alignment = 0;
    std::atomic<int> freed     = 0;
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

  void local_deallocate_impl(header* h, std::size_t size, std::size_t alignment) {
    remove_header_from_list(h);
    std::destroy_at(h);
    std_pool_mr_.deallocate(h, size, alignment);
    allocated_size_ -= size;
  }

  std::size_t                       local_max_size_;
  std::size_t                       global_max_size_;
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
  std::size_t                       allocated_size_;
  std::size_t                       collect_threshold_;
  std::size_t                       collect_threshold_max_;
};

template <typename T>
void remote_get(const remotable_resource& rmr, T* origin_p, const T* target_p, std::size_t size) {
  if (rmr.is_locally_accessible(target_p)) {
    std::memcpy(origin_p, target_p, size * sizeof(T));
  } else {
    auto target_rank = rmr.get_owner(target_p);
    mpi_get(origin_p, size, target_rank, rmr.get_disp(target_p), rmr.win());
  }
}

template <typename T>
T remote_get_value(const remotable_resource& rmr, const T* target_p) {
  if (rmr.is_locally_accessible(target_p)) {
    return *target_p;
  } else {
    auto target_rank = rmr.get_owner(target_p);
    return mpi_get_value<T>(target_rank, rmr.get_disp(target_p), rmr.win());
  }
}

template <typename T>
void remote_put(const remotable_resource& rmr, const T* origin_p, T* target_p, std::size_t size) {
  if (rmr.is_locally_accessible(target_p)) {
    std::memcpy(target_p, origin_p, size * sizeof(T));
  } else {
    auto target_rank = rmr.get_owner(target_p);
    mpi_put(origin_p, size, target_rank, rmr.get_disp(target_p), rmr.win());
  }
}

template <typename T>
void remote_put_value(const remotable_resource& rmr, const T& val, T* target_p) {
  if (rmr.is_locally_accessible(target_p)) {
    *target_p = val;
  } else {
    auto target_rank = rmr.get_owner(target_p);
    mpi_put_value(val, target_rank, rmr.get_disp(target_p), rmr.win());
  }
}

template <typename T>
T remote_faa_value(const remotable_resource& rmr, const T& val, T* target_p) {
  auto target_rank = rmr.get_owner(target_p);
  return mpi_atomic_faa_value(val, target_rank, rmr.get_disp(target_p), rmr.win());
}

// Tests
// -----------------------------------------------------------------------------

ITYR_TEST_CASE("[ityr::common::allocator] basic test") {
  runtime_options opts;
  singleton_initializer<topology::instance> topo;

  remotable_resource allocator(std::size_t(16) * 1024 * 1024);

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
      reinterpret_cast<uint8_t*>(p)[i] = topology::my_rank();
    }

    std::vector<void*> addrs(topology::n_ranks());
    addrs[topology::my_rank()] = p;

    // GET
    for (int target_rank = 0; target_rank < topology::n_ranks(); target_rank++) {
      addrs[target_rank] = mpi_bcast_value(addrs[target_rank], target_rank, topology::mpicomm());
      if (topology::my_rank() != target_rank) {
        std::vector<uint8_t> buf(size);
        mpi_get_nb(buf.data(), size, target_rank, allocator.get_disp(addrs[target_rank]), allocator.win());
        mpi_win_flush(target_rank, allocator.win());

        for (std::size_t i = 0; i < size; i++) {
          ITYR_CHECK(buf[i] == target_rank);
        }
      }
      mpi_barrier(topology::mpicomm());
    }

    // PUT
    std::vector<uint8_t> buf(size);
    for (std::size_t i = 0; i < size; i++) {
      buf[i] = topology::my_rank();
    }

    int target_rank = (topology::my_rank() + 1) % topology::n_ranks();
    mpi_put_nb(buf.data(), size, target_rank, allocator.get_disp(addrs[target_rank]), allocator.win());
    mpi_win_flush_all(allocator.win());

    mpi_barrier(topology::mpicomm());

    for (std::size_t i = 0; i < size; i++) {
      ITYR_CHECK(reinterpret_cast<uint8_t*>(p)[i] == (topology::n_ranks() + topology::my_rank() - 1) % topology::n_ranks());
    }

    ITYR_SUBCASE("Local free") {
      allocator.deallocate(p, size);
    }

    if (topology::n_ranks() > 1) {
      ITYR_SUBCASE("Remote free") {
        ITYR_CHECK(!allocator.empty());

        mpi_barrier(topology::mpicomm());

        int target_rank = (topology::my_rank() + 1) % topology::n_ranks();
        allocator.remote_deallocate(addrs[target_rank], size, target_rank);

        mpi_win_flush_all(allocator.win());
        mpi_barrier(topology::mpicomm());

        allocator.collect_deallocated();
      }
    }

    ITYR_CHECK(allocator.empty());
  }
}

}
