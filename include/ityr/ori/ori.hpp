#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/options.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/wallclock.hpp"
#include "ityr/common/profiler.hpp"
#include "ityr/common/prof_events.hpp"
#include "ityr/ori/util.hpp"
#include "ityr/ori/options.hpp"
#include "ityr/ori/core.hpp"
#include "ityr/ori/global_ptr.hpp"
#include "ityr/ori/prof_events.hpp"

namespace ityr::ori {

inline constexpr block_size_t block_size = ITYR_ORI_BLOCK_SIZE;

class ori {
public:
  ori(MPI_Comm comm)
    : mi_(comm),
      topo_(comm),
      core_(cache_size_option::value(), sub_block_size_option::value()) {}

private:
  common::mpi_initializer                                    mi_;
  common::runtime_options                                    common_opts_;
  common::singleton_initializer<common::topology::instance>  topo_;
  common::singleton_initializer<common::wallclock::instance> clock_;
  common::singleton_initializer<common::profiler::instance>  prof_;
  common::prof_events                                        common_prof_events_;

  runtime_options                                            ori_opts_;
  common::singleton_initializer<core::instance>              core_;
  prof_events                                                prof_events_;
};

using instance = common::singleton<ori>;

inline void init(MPI_Comm comm = MPI_COMM_WORLD) {
  instance::init(comm);
}

inline void fini() {
  instance::fini();
}

template <typename T>
inline global_ptr<T> malloc_coll(std::size_t count) {
  return global_ptr<T>(reinterpret_cast<T*>(core::instance::get().malloc_coll(count * sizeof(T))));
}

template <typename T, template <block_size_t> typename MemMapper, typename... MemMapperArgs>
inline global_ptr<T> malloc_coll(std::size_t count, MemMapperArgs&&... mmargs) {
  return global_ptr<T>(reinterpret_cast<T*>(core::instance::get().malloc_coll<MemMapper>(count * sizeof(T),
                                                                                         std::forward<MemMapperArgs>(mmargs)...)));
}

template <typename T>
inline global_ptr<T> malloc(std::size_t count) {
  return global_ptr<T>(reinterpret_cast<T*>(core::instance::get().malloc(count * sizeof(T))));
}

template <typename T>
inline void free_coll(global_ptr<T> ptr) {
  core::instance::get().free_coll(ptr.raw_ptr());
}

template <typename T>
inline void free(global_ptr<T> ptr, std::size_t count) {
  core::instance::get().free(ptr.raw_ptr(), count * sizeof(T));
}

template <typename ConstT, typename T>
inline void get(global_ptr<ConstT> from_ptr, T* to_ptr, std::size_t count) {
  static_assert(std::is_same_v<std::remove_const_t<ConstT>, T>,
                "from_ptr must be of the same type as to_ptr ignoring const");
  static_assert(std::is_trivially_copyable_v<T>, "GET requires trivially copyable types");
  core::instance::get().get(from_ptr.raw_ptr(), to_ptr, count * sizeof(T));
}

template <typename T>
inline void put(const T* from_ptr, global_ptr<T> to_ptr, std::size_t count) {
  static_assert(std::is_trivially_copyable_v<T>, "PUT requires trivially copyable types");
  core::instance::get().put(from_ptr, to_ptr.raw_ptr(), count * sizeof(T));
}

inline constexpr bool force_getput = ITYR_ORI_FORCE_GETPUT;

template <bool SkipFetch, typename T>
inline T* checkout_with_getput(global_ptr<T> ptr, std::size_t count) {
  std::size_t size = count * sizeof(T);
  auto ret = reinterpret_cast<std::remove_const_t<T>*>(std::malloc(size + sizeof(void*)));
  if (!SkipFetch) {
    core::instance::get().get(ptr.raw_ptr(), ret, size);
  }
  *reinterpret_cast<void**>(reinterpret_cast<std::byte*>(ret) + size) = ptr.raw_ptr();
  return ret;
}

template <typename T>
inline const T* checkout(global_ptr<T> ptr, std::size_t count, mode::read_t) {
  if constexpr (force_getput) {
    return checkout_with_getput<false>(ptr, count);
  }
  core::instance::get().checkout(ptr.raw_ptr(), count * sizeof(T), mode::read);
  return ptr.raw_ptr();
}

template <typename T>
inline T* checkout(global_ptr<T> ptr, std::size_t count, mode::write_t) {
  static_assert(!std::is_const_v<T>, "Const pointers cannot be checked out with write access mode");
  if constexpr (force_getput) {
    return checkout_with_getput<true>(ptr, count);
  }
  core::instance::get().checkout(ptr.raw_ptr(), count * sizeof(T), mode::write);
  return ptr.raw_ptr();
}

template <typename T>
inline T* checkout(global_ptr<T> ptr, std::size_t count, mode::read_write_t) {
  static_assert(!std::is_const_v<T>, "Const pointers cannot be checked out with read+write access mode");
  if constexpr (force_getput) {
    return checkout_with_getput<false>(ptr, count);
  }
  core::instance::get().checkout(ptr.raw_ptr(), count * sizeof(T), mode::read_write);
  return ptr.raw_ptr();
}

template <bool RegisterDirty, typename T>
inline void checkin_with_getput(T* raw_ptr, std::size_t count) {
  std::size_t size = count * sizeof(T);
  void* gptr = *reinterpret_cast<void**>(reinterpret_cast<std::byte*>(
        const_cast<std::remove_const_t<T>*>(raw_ptr)) + size);
  if constexpr (RegisterDirty) {
    core::instance::get().put(raw_ptr, gptr, size);
  }
  std::free(const_cast<std::remove_const_t<T>*>(raw_ptr));
}

template <typename T>
inline void checkin(const T* raw_ptr, std::size_t count, mode::read_t) {
  if constexpr (force_getput) {
    checkin_with_getput<false>(raw_ptr, count);
    return;
  }
  core::instance::get().checkin(const_cast<T*>(raw_ptr), count * sizeof(T), mode::read);
}

template <typename T>
inline void checkin(T* raw_ptr, std::size_t count, mode::write_t) {
  static_assert(!std::is_const_v<T>, "Const pointers cannot be checked in with write access mode");
  if constexpr (force_getput) {
    checkin_with_getput<true>(raw_ptr, count);
    return;
  }
  core::instance::get().checkin(raw_ptr, count * sizeof(T), mode::write);
}

template <typename T>
inline void checkin(T* raw_ptr, std::size_t count, mode::read_write_t) {
  static_assert(!std::is_const_v<T>, "Const pointers cannot be checked in with read+write access mode");
  if constexpr (force_getput) {
    checkin_with_getput<true>(raw_ptr, count);
    return;
  }
  core::instance::get().checkin(raw_ptr, count * sizeof(T), mode::read_write);
}

template <typename T, typename Mode, typename Fn>
inline auto with_checkout(global_ptr<T> ptr, std::size_t count, Mode, Fn&& fn) {
  // TODO: use smart-pointer-like class to automatically call checkin at destructor
  auto raw_ptr = checkout(ptr, count, Mode{});
  if constexpr (std::is_void_v<std::invoke_result_t<Fn, decltype(raw_ptr)>>) {
    std::forward<Fn>(fn)(raw_ptr);
    checkin(raw_ptr, count, Mode{});
  } else {
    auto ret = std::forward<Fn>(fn)(raw_ptr);
    checkin(raw_ptr, count, Mode{});
    return ret;
  }
}

template <typename T1, typename Mode1,
          typename T2, typename Mode2, typename Fn>
inline auto with_checkout(global_ptr<T1> ptr1, std::size_t count1, Mode1,
                          global_ptr<T2> ptr2, std::size_t count2, Mode2, Fn&& f) {
  return with_checkout(ptr1, count1, Mode1{}, [&](auto&& p1) {
    return with_checkout(ptr2, count2, Mode2{}, [&](auto&& p2) {
      return std::forward<Fn>(f)(std::forward<decltype(p1)>(p1),
                                 std::forward<decltype(p2)>(p2));
    });
  });
}

template <typename T1, typename Mode1,
          typename T2, typename Mode2,
          typename T3, typename Mode3, typename Fn>
inline auto with_checkout(global_ptr<T1> ptr1, std::size_t count1, Mode1,
                          global_ptr<T2> ptr2, std::size_t count2, Mode2,
                          global_ptr<T3> ptr3, std::size_t count3, Mode3, Fn&& f) {
  return with_checkout(ptr1, count1, Mode1{}, [&](auto&& p1) {
    return with_checkout(ptr2, count2, Mode2{}, [&](auto&& p2) {
      return with_checkout(ptr3, count3, Mode3{}, [&](auto&& p3) {
        return std::forward<Fn>(f)(std::forward<decltype(p1)>(p1),
                                   std::forward<decltype(p2)>(p2),
                                   std::forward<decltype(p3)>(p3));
      });
    });
  });
}

inline void release() {
  core::instance::get().release();
}

inline auto release_lazy() {
  return core::instance::get().release_lazy();
}

using release_handler = core::instance::instance_type::release_handler;

inline void acquire() {
  core::instance::get().acquire();
}

inline void acquire(release_handler rh) {
  core::instance::get().acquire(rh);
}

inline void poll() {
  core::instance::get().poll();
}

inline void collect_deallocated() {
  core::instance::get().collect_deallocated();
}

}
