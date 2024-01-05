#pragma once

#include "ityr/common/util.hpp"
#include "ityr/ito/ito.hpp"
#include "ityr/ori/ori.hpp"
#include "ityr/pattern/parallel_loop.hpp"

namespace ityr {

/*
 * @brief Unique pointer for an mmaped file.
 *
 * This class manages a virtual address space mapped to a file on the disk.
 * The specified file is opened and mapped to the same virtual address at construction,
 * and closed at destruction.
 * The pointer to the mmapped file (which can be obtained via `get()` method) can be uniformly used
 * across different processes.
 *
 * The constructor and destructor must be called collectively by all processes
 * (i.e., in the SPMD region or in the root thread).
 * At construction, the same virtual address space is allocated among all processes and
 * the file content is directly mapped to the virtual address (via `mmap()`).
 *
 * Currently, only the read-only mode is supported.
 *
 * @see `ityr::make_unique_ptr()`
 */
template <typename T>
class unique_file_ptr {
public:
  using element_type = T;
  using pointer      = element_type*;
  using reference    = std::add_lvalue_reference_t<element_type>;

  constexpr unique_file_ptr() noexcept {}

  explicit unique_file_ptr(const std::string& fpath, bool mlock = false)
    : ptr_(reinterpret_cast<pointer>(alloc_coll(fpath, mlock))) {
    ITYR_CHECK(ori::file_mem_get(ptr_).size() % sizeof(T) == 0);
  }

  unique_file_ptr(const unique_file_ptr&) = delete;
  unique_file_ptr& operator=(const unique_file_ptr&) = delete;

  unique_file_ptr(std::nullptr_t) noexcept {}
  unique_file_ptr& operator=(std::nullptr_t) noexcept {
    destroy();
    ptr_ = nullptr;
    return *this;
  }

  unique_file_ptr(unique_file_ptr&& ufp) noexcept
    : ptr_(ufp.ptr_) { ufp.ptr_ = nullptr; }
  unique_file_ptr& operator=(unique_file_ptr&& ufp) noexcept {
    destroy();
    ptr_ = ufp.ptr_;
    ufp.ptr_ = nullptr;
    return *this;
  }

  template <typename U>
  unique_file_ptr(unique_file_ptr<U>&& ufp) noexcept
    : ptr_(ufp.ptr_) { ufp.ptr_ = nullptr; }
  template <typename U>
  unique_file_ptr& operator=(unique_file_ptr<U>&& ufp) noexcept {
    destroy();
    ptr_ = ufp.ptr_;
    ufp.ptr_ = nullptr;
    return *this;
  }

  ~unique_file_ptr() { destroy(); }

  explicit operator bool() const noexcept {
    return ptr_;
  }

  pointer get() const noexcept {
    return ptr_;
  }

  std::size_t size() const noexcept {
    auto&& fm = ori::file_mem_get(ptr_);
    return fm.size() / sizeof(T);
  }

  reference operator*() const {
    ITYR_CHECK(get() != nullptr);
    return *get();
  }

  reference operator[](std::size_t i) const {
    ITYR_CHECK(get() != nullptr);
    ITYR_CHECK(i <= size());
    return *(get() + i);
  }

  pointer operator->() const noexcept {
    return get();
  }

  pointer release() noexcept {
    pointer p = ptr_;
    ptr_ = nullptr;
    return p;
  }

  void swap(unique_file_ptr<T> ufp) {
    std::swap(ptr_, ufp.ptr_);
  }

private:
  void destroy() {
    if (ptr_) {
      free_coll(ptr_);
    }
  }

  static void* alloc_coll(const std::string& fpath, bool mlock) {
    if (ito::is_spmd()) {
      return ori::file_mem_alloc_coll(fpath, mlock);
    } else if (ito::is_root()) {
      // FIXME: ugly hack to pass heap-allocated string to other processes
      constexpr std::size_t max_chars = 256;
      if (fpath.size() >= max_chars) {
        common::die("File path length for unique_file_ptr must be less than %ld.", max_chars);
      }
      std::array<char, max_chars> buf;
      std::strncpy(buf.data(), fpath.c_str(), max_chars - 1);
      buf.back() = '\0';
      return ito::coll_exec([=] {
        return ori::file_mem_alloc_coll(buf.data(), mlock);
      });
    } else {
      common::die("Collective operations for ityr::global_vector must be executed on the root thread or SPMD region.");
    }
  }

  static void free_coll(void* addr) {
    if (ito::is_spmd()) {
      ori::file_mem_free_coll(addr);
    } else if (ito::is_root()) {
      ito::coll_exec([=] { ori::file_mem_free_coll(addr); });
    } else {
      common::die("Collective operations for ityr::global_vector must be executed on the root thread or SPMD region.");
    }
  }

  pointer ptr_ = nullptr;
};

template <typename T1, typename T2>
inline bool operator==(const unique_file_ptr<T1>& ufp1, const unique_file_ptr<T2>& ufp2) {
  return ufp1.get() == ufp2.get();
}

template <typename T>
inline bool operator==(const unique_file_ptr<T>& ufp, std::nullptr_t) {
  return ufp.get() == nullptr;
}

template <typename T>
inline bool operator==(std::nullptr_t, const unique_file_ptr<T>& ufp) {
  return nullptr == ufp.get();
}

template <typename T1, typename T2>
inline bool operator!=(const unique_file_ptr<T1>& ufp1, const unique_file_ptr<T2>& ufp2) {
  return ufp1.get() != ufp2.get();
}

template <typename T>
inline bool operator!=(const unique_file_ptr<T>& ufp, std::nullptr_t) {
  return ufp.get() != nullptr;
}

template <typename T>
inline bool operator!=(std::nullptr_t, const unique_file_ptr<T>& ufp) {
  return nullptr != ufp.get();
}

template <typename T1, typename T2>
inline bool operator>(const unique_file_ptr<T1>& ufp1, const unique_file_ptr<T2>& ufp2) {
  return ufp1.get() > ufp2.get();
}

template <typename T>
inline bool operator>(const unique_file_ptr<T>& ufp, std::nullptr_t) {
  return ufp.get() > nullptr;
}

template <typename T>
inline bool operator>(std::nullptr_t, const unique_file_ptr<T>& ufp) {
  return nullptr > ufp.get();
}

template <typename T1, typename T2>
inline bool operator>=(const unique_file_ptr<T1>& ufp1, const unique_file_ptr<T2>& ufp2) {
  return ufp1.get() >= ufp2.get();
}

template <typename T>
inline bool operator>=(const unique_file_ptr<T>& ufp, std::nullptr_t) {
  return ufp.get() >= nullptr;
}

template <typename T>
inline bool operator>=(std::nullptr_t, const unique_file_ptr<T>& ufp) {
  return nullptr >= ufp.get();
}

template <typename T1, typename T2>
inline bool operator<(const unique_file_ptr<T1>& ufp1, const unique_file_ptr<T2>& ufp2) {
  return ufp1.get() < ufp2.get();
}

template <typename T>
inline bool operator<(const unique_file_ptr<T>& ufp, std::nullptr_t) {
  return ufp.get() < nullptr;
}

template <typename T>
inline bool operator<(std::nullptr_t, const unique_file_ptr<T>& ufp) {
  return nullptr < ufp.get();
}

template <typename T1, typename T2>
inline bool operator<=(const unique_file_ptr<T1>& ufp1, const unique_file_ptr<T2>& ufp2) {
  return ufp1.get() <= ufp2.get();
}

template <typename T>
inline bool operator<=(const unique_file_ptr<T>& ufp, std::nullptr_t) {
  return ufp.get() <= nullptr;
}

template <typename T>
inline bool operator<=(std::nullptr_t, const unique_file_ptr<T>& ufp) {
  return nullptr <= ufp.get();
}

template <typename T>
inline void swap(unique_file_ptr<T> ufp1, unique_file_ptr<T> ufp2) {
  ufp1.swap(ufp2);
}

template <typename CharT, typename Traits, typename T>
inline std::basic_ostream<CharT, Traits>&
operator<<(std::basic_ostream<CharT, Traits>& ostream, const unique_file_ptr<T>& ufp) {
  ostream << ufp.get();
  return ostream;
}

/*
 * @brief Create a unique pointer for an mmaped file.
 *
 * @see `ityr::unique_file_ptr`
 */
template <typename T>
inline unique_file_ptr<T> make_unique_file(const std::string& fpath, bool mlock = false) {
  return unique_file_ptr<T>(fpath, mlock);
}

ITYR_TEST_CASE("[ityr::unique_file_ptr] unique_file_ptr") {
  ito::init();
  ori::init();

  auto my_rank = common::topology::my_rank();

  long n = 100000;
  std::string filename = "test.bin";

  if (my_rank == 0) {
    std::vector<long> buf(n);
    for (long i = 0; i < n; i++) {
      buf[i] = i;
    }
    std::ofstream ostream(filename, std::ios::binary);
    ostream.write(reinterpret_cast<char*>(buf.data()), buf.size() * sizeof(long));
  }

  common::mpi_barrier(common::topology::mpicomm());

  ITYR_SUBCASE("spmd") {
    unique_file_ptr<long> fp = make_unique_file<long>(filename);
    ITYR_CHECK(fp.size() == n);

    for (long i = 0; i < n; i++) {
      ITYR_CHECK(fp[i] == i);
    }

    common::mpi_barrier(common::topology::mpicomm());

    fp = {}; // destroy;
  }

  ITYR_SUBCASE("root exec") {
    ito::root_exec([&] {
      unique_file_ptr<long> fp = make_unique_file<long>(filename);
      ITYR_CHECK(fp.size() == n);

      for_each(
          execution::parallel_policy(100),
          count_iterator<long>(0),
          count_iterator<long>(n),
          fp.get(),
          [](long i, long v) { ITYR_CHECK(v == i); });
    });
  }

  common::mpi_barrier(common::topology::mpicomm());

  if (my_rank == 0) {
    remove(filename.c_str());
  }

  ori::fini();
  ito::fini();
}

}
