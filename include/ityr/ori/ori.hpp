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
  return global_ptr<T>(core::instance::get().malloc_coll(count * sizeof(T)));
}

template <typename T, template <block_size_t> typename MemMapper, typename... MemMapperArgs>
inline global_ptr<T> malloc_coll(std::size_t count, MemMapperArgs&&... mmargs) {
  return global_ptr<T>(core::instance::get().malloc_coll<MemMapper>(count * sizeof(T),
                                                                    std::forward<MemMapperArgs>(mmargs)...));
}

template <typename T>
inline void free_coll(global_ptr<T> ptr) {
  global_ptr<T>(core::instance::get().free_coll(ptr.raw_ptr()));
}

template <typename T>
inline void free(global_ptr<T> ptr, std::size_t count) {
  global_ptr<T>(core::instance::get().free(ptr.raw_ptr(), count * sizeof(T)));
}

template <typename T>
inline const T* checkout(global_ptr<T> ptr, std::size_t count, mode::read_t) {
  return global_ptr<T>(core::instance::get().checkout(ptr.raw_ptr(), count * sizeof(T), mode::read));
}

template <typename T>
inline T* checkout(global_ptr<T> ptr, std::size_t count, mode::write_t) {
  static_assert(!std::is_const_v<T>, "Const pointers cannot be checked out with write access mode");
  return global_ptr<T>(core::instance::get().checkout(ptr.raw_ptr(), count * sizeof(T), mode::write));
}

template <typename T>
inline T* checkout(global_ptr<T> ptr, std::size_t count, mode::read_write_t) {
  static_assert(!std::is_const_v<T>, "Const pointers cannot be checked out with read+write access mode");
  return global_ptr<T>(core::instance::get().checkout(ptr.raw_ptr(), count * sizeof(T), mode::read_write));
}

template <typename T>
inline void checkin(const T* raw_ptr, std::size_t count, mode::read_t) {
  global_ptr<T>(core::instance::get().checkin(raw_ptr, count * sizeof(T), mode::read));
}

template <typename T>
inline void checkin(T* raw_ptr, std::size_t count, mode::write_t) {
  static_assert(!std::is_const_v<T>, "Const pointers cannot be checked in with write access mode");
  global_ptr<T>(core::instance::get().checkin(raw_ptr, count * sizeof(T), mode::write));
}

template <typename T>
inline void checkin(T* raw_ptr, std::size_t count, mode::read_write_t) {
  static_assert(!std::is_const_v<T>, "Const pointers cannot be checked in with read+write access mode");
  global_ptr<T>(core::instance::get().checkin(raw_ptr, count * sizeof(T), mode::read_write));
}

}
