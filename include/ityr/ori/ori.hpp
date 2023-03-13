#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/wallclock.hpp"
#include "ityr/common/profiler.hpp"
#include "ityr/common/prof_events.hpp"
#include "ityr/ori/util.hpp"
#include "ityr/ori/options.hpp"
#include "ityr/ori/core.hpp"
#include "ityr/ori/global_ptr.hpp"

namespace ityr::ori {

inline constexpr block_size_t block_size = ITYR_ORI_BLOCK_SIZE;

class ori {
public:
  ori(std::size_t cache_size, std::size_t sub_block_size, MPI_Comm comm)
    : topo_(comm),
      core_(cache_size, sub_block_size) {}

private:
  common::mpi_initializer                                    mi_;
  common::singleton_initializer<common::topology::instance>  topo_;
  common::singleton_initializer<common::wallclock::instance> clock_;
  common::singleton_initializer<common::profiler::instance>  prof_;
  common::prof_events                                        common_prof_events_;
  common::singleton_initializer<core::instance>              core_;
};

using instance = common::singleton<ori>;

inline void init(std::size_t cache_size     = std::size_t(16) * 1024 * 1024,
                 std::size_t sub_block_size = std::size_t(4) * 1024,
                 MPI_Comm comm              = MPI_COMM_WORLD) {
  instance::init(cache_size, sub_block_size, comm);
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
  return global_ptr<T>(core::instance::get().malloc_coll<MemMapper>(count * sizeof(T),
                                                                    std::forward<MemMapperArgs>(mmargs)...));
}

template <typename T>
inline void free_coll(global_ptr<T> ptr) {
  core::instance::get().free_coll(ptr.raw_ptr());
}

template <typename T>
inline void free(global_ptr<T> ptr, std::size_t count) {
  core::instance::get().free(ptr.raw_ptr(), count * sizeof(T));
}

template <typename T>
inline const T* checkout(global_ptr<T> ptr, std::size_t count, mode::read_t) {
  core::instance::get().checkout(ptr.raw_ptr(), count * sizeof(T), mode::read);
  return ptr.raw_ptr();
}

template <typename T>
inline T* checkout(global_ptr<T> ptr, std::size_t count, mode::write_t) {
  static_assert(!std::is_const_v<T>, "Const pointers cannot be checked out with write access mode");
  core::instance::get().checkout(ptr.raw_ptr(), count * sizeof(T), mode::write);
  return ptr.raw_ptr();
}

template <typename T>
inline T* checkout(global_ptr<T> ptr, std::size_t count, mode::read_write_t) {
  static_assert(!std::is_const_v<T>, "Const pointers cannot be checked out with read+write access mode");
  core::instance::get().checkout(ptr.raw_ptr(), count * sizeof(T), mode::read_write);
  return ptr.raw_ptr();
}

template <typename T>
inline void checkin(const T* raw_ptr, std::size_t count, mode::read_t) {
  core::instance::get().checkin(const_cast<T*>(raw_ptr), count * sizeof(T), mode::read);
}

template <typename T>
inline void checkin(T* raw_ptr, std::size_t count, mode::write_t) {
  static_assert(!std::is_const_v<T>, "Const pointers cannot be checked in with write access mode");
  core::instance::get().checkin(raw_ptr, count * sizeof(T), mode::write);
}

template <typename T>
inline void checkin(T* raw_ptr, std::size_t count, mode::read_write_t) {
  static_assert(!std::is_const_v<T>, "Const pointers cannot be checked in with read+write access mode");
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
                          global_ptr<T2> ptr3, std::size_t count3, Mode3, Fn&& f) {
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

inline void acquire() {
  core::instance::get().acquire();
}

}