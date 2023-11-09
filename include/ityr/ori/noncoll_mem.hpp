#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/rma.hpp"
#include "ityr/common/allocator.hpp"

namespace ityr::ori {

// TODO: unify these implementations with the common allocator

class root_resource final : public common::pmr::memory_resource {
public:
  root_resource(void* addr, std::size_t size)
    : addr_(addr),
      size_(size),
      freelist_(reinterpret_cast<uintptr_t>(addr_), size_) {}

  void* do_allocate(std::size_t bytes, std::size_t alignment) override {
    ITYR_CHECK(bytes <= size_);

    auto s = freelist_.get(bytes, alignment);
    if (!s.has_value()) {
      throw std::bad_alloc();
    }

    return reinterpret_cast<void*>(*s);
  }

  void do_deallocate(void* p, std::size_t bytes, std::size_t alignment [[maybe_unused]]) override {
    ITYR_CHECK(p);
    ITYR_CHECK(bytes <= size_);

    freelist_.add(reinterpret_cast<uintptr_t>(p), bytes);
  }

  bool do_is_equal(const common::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

private:
  void*            addr_;
  std::size_t      size_;
  common::freelist freelist_;
};

class noncoll_mem final : public common::pmr::memory_resource {
public:
  noncoll_mem(std::size_t local_max_size)
    : local_max_size_(local_max_size),
      global_max_size_(local_max_size_ * common::topology::n_ranks()),
      vm_(common::reserve_same_vm_coll(global_max_size_, local_max_size_)),
      pm_(init_pm()),
      local_base_addr_(reinterpret_cast<std::byte*>(vm_.addr()) + local_max_size_ * common::topology::my_rank()),
      win_(common::rma::create_win(local_base_addr_, local_max_size_)),
      root_mr_(local_base_addr_, local_max_size_ - sizeof(int)), // The last element is used for a flag value for deallocation
      std_pool_mr_(my_std_pool_options(), &root_mr_),
      max_unflushed_free_objs_(common::allocator_max_unflushed_free_objs_option::value()),
      allocated_size_(0),
      collect_threshold_(std::size_t(16) * 1024),
      collect_threshold_max_(local_max_size_ * 8 / 10) {
    // Set the flag value for deallocation
    *reinterpret_cast<int*>(
      reinterpret_cast<std::byte*>(local_base_addr_) + local_max_size_ - sizeof(int)) = remote_free_flag_value;
  }

  const common::rma::win& win() const { return *win_; }

  bool has(const void* p) const {
    return vm_.addr() <= p && p < reinterpret_cast<std::byte*>(vm_.addr()) + global_max_size_;
  }

  common::topology::rank_t get_owner(const void* p) const {
    return (reinterpret_cast<uintptr_t>(p) - reinterpret_cast<uintptr_t>(vm_.addr())) / local_max_size_;
  }

  std::size_t get_disp(const void* p) const {
    return (reinterpret_cast<uintptr_t>(p) - reinterpret_cast<uintptr_t>(vm_.addr())) % local_max_size_;
  }

  void* do_allocate(std::size_t bytes, std::size_t alignment = alignof(max_align_t)) override {
    ITYR_PROFILER_RECORD(common::prof_event_allocator_alloc);

    std::size_t pad_bytes = common::round_up_pow2(sizeof(header), alignment);
    std::size_t real_bytes = bytes + pad_bytes;

    if (allocated_size_ >= collect_threshold_) {
      collect_deallocated();
    }

    std::byte* p;
    try {
      p = reinterpret_cast<std::byte*>(std_pool_mr_.allocate(real_bytes, alignment));
    } catch (std::bad_alloc& e) {
      // collect remotely freed objects and try allocation again
      collect_deallocated();
      try {
        p = reinterpret_cast<std::byte*>(std_pool_mr_.allocate(real_bytes, alignment));
      } catch (std::bad_alloc& e) {
        // TODO: throw std::bad_alloc?
        common::die("[ityr::ori::noncoll_mem] Could not allocate memory for malloc_local()");
      }
    };

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
    if (target_rank == common::topology::my_rank()) {
      local_deallocate(p, bytes, alignment);
    } else {
      remote_deallocate(p, bytes, target_rank, alignment);
    }
  }

  bool do_is_equal(const common::pmr::memory_resource& other) const noexcept override {
    return this == &other;
  }

  void local_deallocate(void* p, std::size_t bytes, std::size_t alignment = alignof(max_align_t)) {
    ITYR_PROFILER_RECORD(common::prof_event_allocator_free_local);

    ITYR_CHECK(get_owner(p) == common::topology::my_rank());

    std::size_t pad_bytes = common::round_up_pow2(sizeof(header), alignment);
    std::size_t real_bytes = bytes + pad_bytes;

    header* h = reinterpret_cast<header*>(reinterpret_cast<std::byte*>(p) - pad_bytes);
    ITYR_CHECK(h->size == real_bytes);
    ITYR_CHECK(h->alignment == alignment);
    ITYR_CHECK(h->freed == 0);

    local_deallocate_impl(h, real_bytes, alignment);
  }

  void remote_deallocate(void* p, std::size_t bytes [[maybe_unused]], int target_rank, std::size_t alignment = alignof(max_align_t)) {
    ITYR_PROFILER_RECORD(common::prof_event_allocator_free_remote, target_rank);

    ITYR_CHECK(common::topology::my_rank() != target_rank);
    ITYR_CHECK(get_owner(p) == target_rank);

    int* flag_val_p = reinterpret_cast<int*>(
      reinterpret_cast<std::byte*>(local_base_addr_) + local_max_size_ - sizeof(int));
    common::rma::put_nb(win(), flag_val_p, 1, win(), target_rank, get_header_disp(p, alignment));

    static int count = 0;
    count++;
    if (count >= max_unflushed_free_objs_) {
      common::rma::flush(win());
      count = 0;
    }
  }

  void collect_deallocated() {
    ITYR_PROFILER_RECORD(common::prof_event_allocator_collect);

    header *h = allocated_list_.next;
    while (h) {
      int flag = h->freed.load(std::memory_order_acquire);
      if (flag) {
        ITYR_CHECK_MESSAGE(flag == remote_free_flag_value, "noncoll memory corruption");
        header* h_next = h->next;
        local_deallocate_impl(h, h->size, h->alignment);
        h = h_next;
      } else {
        h = h->next;
      }
    }

    collect_threshold_ = allocated_size_ * 2;
    if (collect_threshold_ > collect_threshold_max_) {
      collect_threshold_ = (collect_threshold_max_ + allocated_size_) / 2;
    }
  }

  bool is_locally_accessible(const void* p) const {
    return common::topology::is_locally_accessible(get_owner(p));
  }

  bool is_remotely_freed(void* p, std::size_t alignment = alignof(max_align_t)) {
    ITYR_CHECK(get_owner(p) == common::topology::my_rank());

    std::size_t pad_bytes = common::round_up_pow2(sizeof(header), alignment);
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
    ss << "/ityr_noncoll_" << count++ << "_" << inter_rank;
    return ss.str();
  }

  common::physical_mem init_pm() const {
    common::physical_mem pm;

    if (common::topology::intra_my_rank() == 0) {
      pm = common::physical_mem(allocator_shmem_name(common::topology::inter_my_rank()), global_max_size_, true);
    }

    common::mpi_barrier(common::topology::intra_mpicomm());

    if (common::topology::intra_my_rank() != 0) {
      pm = common::physical_mem(allocator_shmem_name(common::topology::inter_my_rank()), global_max_size_, false);
    }

    ITYR_CHECK(vm_.size() == global_max_size_);

    for (common::topology::rank_t r = 0; r < common::topology::intra_n_ranks(); r++) {
      auto target_rank = common::topology::intra2global_rank(r);
      auto offset = local_max_size_ * target_rank;
      void* begin_addr = reinterpret_cast<std::byte*>(vm_.addr()) + offset;
      pm.map_to_vm(begin_addr, local_max_size_, offset);
    }

    return pm;
  }

  common::pmr::pool_options my_std_pool_options() const {
    common::pmr::pool_options opts;
    opts.max_blocks_per_chunk = local_max_size_ / 10;
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
    std::size_t pad_bytes = common::round_up_pow2(sizeof(header), alignment);
    auto h = reinterpret_cast<const header*>(reinterpret_cast<const std::byte*>(p) - pad_bytes);
    const void* flag_addr = &h->freed;

    return get_disp(flag_addr);
  }

  void local_deallocate_impl(header* h, std::size_t size, std::size_t alignment) {
    remove_header_from_list(h);
    std::destroy_at(h);
    std_pool_mr_.deallocate(h, size, alignment);

    ITYR_CHECK(allocated_size_ >= size);
    allocated_size_ -= size;
  }

  static constexpr int remote_free_flag_value = 417;

  std::size_t                               local_max_size_;
  std::size_t                               global_max_size_;
  common::virtual_mem                       vm_;
  common::physical_mem                      pm_;
  void*                                     local_base_addr_;
  std::unique_ptr<common::rma::win>         win_;
  root_resource                             root_mr_;
  common::pmr::unsynchronized_pool_resource std_pool_mr_;
  int                                       max_unflushed_free_objs_;
  header                                    allocated_list_;
  header*                                   allocated_list_end_ = &allocated_list_;
  std::size_t                               allocated_size_;
  std::size_t                               collect_threshold_;
  std::size_t                               collect_threshold_max_;
};

}
