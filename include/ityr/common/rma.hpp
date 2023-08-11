#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/options.hpp"
#include "ityr/common/rma/mpi.hpp"
#include "ityr/common/rma/utofu.hpp"

namespace ityr::common::rma {

using instance = singleton<ITYR_RMA_IMPL>;
using win = ITYR_RMA_IMPL::win;

template <typename T>
inline std::unique_ptr<win> create_win(T* baseptr, std::size_t count) {
  if constexpr (std::is_void_v<T>) {
    return std::make_unique<win>(instance::get().create_win(baseptr, count));
  } else {
    return std::make_unique<win>(instance::get().create_win(baseptr, sizeof(T) * count));
  }
}

template <typename T>
inline void get_nb(const win&  origin_win,
                   T*          origin_addr,
                   std::size_t count,
                   const win&  target_win,
                   int         target_rank,
                   std::size_t target_disp) {
  static_assert(!std::is_void_v<T>);
  instance::get().get_nb(origin_win, reinterpret_cast<std::byte*>(origin_addr), sizeof(T) * count,
                         target_win, target_rank, target_disp);
}

// Needed only for evaluation of get/put APIs (nocache), where `origin_win` is not provided
template <typename T>
inline void get_nb(T*          origin_addr,
                   std::size_t count,
                   const win&  target_win,
                   int         target_rank,
                   std::size_t target_disp) {
  static_assert(!std::is_void_v<T>);
  instance::get().get_nb(reinterpret_cast<std::byte*>(origin_addr), sizeof(T) * count,
                         target_win, target_rank, target_disp);
}

template <typename T>
inline void put_nb(const win&  origin_win,
                   const T*    origin_addr,
                   std::size_t count,
                   const win&  target_win,
                   int         target_rank,
                   std::size_t target_disp) {
  static_assert(!std::is_void_v<T>);
  instance::get().put_nb(origin_win, reinterpret_cast<const std::byte*>(origin_addr), sizeof(T) * count,
                         target_win, target_rank, target_disp);
}

// Needed only for evaluation of get/put APIs (nocache), where `origin_win` is not provided
template <typename T>
inline void put_nb(const T*    origin_addr,
                   std::size_t count,
                   const win&  target_win,
                   int         target_rank,
                   std::size_t target_disp) {
  static_assert(!std::is_void_v<T>);
  instance::get().put_nb(reinterpret_cast<const std::byte*>(origin_addr), sizeof(T) * count,
                         target_win, target_rank, target_disp);
}

inline void flush(const win& target_win) {
  instance::get().flush(target_win);
}

}
