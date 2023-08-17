#pragma once

#include "ityr/common/util.hpp"
#include "ityr/ito/ito.hpp"
#include "ityr/ori/ori.hpp"

namespace ityr {

/**
 * @brief Spawn the root thread (collective).
 *
 * @param fn      Function object to be called by the root thread.
 * @param args... Argments to be passed to `fn` (optional).
 *
 * @return The return value of the root thread function `fn(args...)`.
 *
 * This function switches from the SPMD region to the root thread executing `fn(args...)`.
 * This function must be called by all processes collectively (i.e., in the SPMD region).
 * This function returns when the root thread is completed and its return value is broadcasted to
 * all processes.
 *
 * The root thread is created and started by the master process (of rank 0), but it is not guaranteed
 * to be continuously executed by the master process, because it can be migrated to other processes at
 * fork/join calls.
 *
 * Example:
 * ```
 * int main() {
 *   ityr::init();
 *
 *   // SPMD region
 *
 *   ityr::root_exec([=] {
 *     // Only one root thread is spawned globally
 *   });
 *   // returns when the root thread is completed
 *
 *   // SPMD region
 *
 *   ityr::fini();
 * }
 * ```
 *
 * @see `ityr::coll_exec()`
 */
template <typename Fn, typename... Args>
inline auto root_exec(Fn&& fn, Args&&... args) {
  ITYR_CHECK(ito::is_spmd());

  ori::release();
  common::mpi_barrier(common::topology::mpicomm());
  ori::acquire();

  using retval_t = std::invoke_result_t<Fn, Args...>;
  if constexpr (std::is_void_v<retval_t>) {
    ito::root_exec(ito::with_callback,
                   []() { ori::poll(); },
                   std::forward<Fn>(fn), std::forward<Args>(args)...);
    // TODO: release() is needed only for the last worker which executed the root thread
    ori::release();
    common::mpi_barrier(common::topology::mpicomm());
    ori::acquire();
  } else {
    auto ret = ito::root_exec(ito::with_callback,
                              []() { ori::poll(); },
                              std::forward<Fn>(fn), std::forward<Args>(args)...);
    ori::release();
    common::mpi_barrier(common::topology::mpicomm());
    ori::acquire();
    return ret;
  }
}

/**
 * @brief Execute the same function collectively on all processes.
 *
 * @param fn      Function object to be called on all processes.
 * @param args... Argments to be passed to `fn` (optional).
 *
 * @return The return value of `fn(args...)` executed on the calling process.
 *
 * This function switches from the root thread to the SPMD region to execute `fn(args...)` on all
 * processes. This function must be called by the root thread (within `ityr::root_exec()`).
 * The return value of `fn(args...)` on the process calling this function is returned, and those
 * on other processes are discarded. This function returns when all processes complete the function.
 *
 * Example:
 * ```
 * ityr::root_exec([=] {
 *   auto my_rank = ityr::my_rank();
 *
 *   auto ret = ityr::coll_exec([] {
 *     // SPMD region (all processes execute this function)
 *     return ityr::my_rank();
 *   });
 *
 *   // The return value on the calling process is returned
 *   assert(ret == my_rank);
 * });
 * ```
 *
 * @see `ityr::root_exec()`
 */

template <typename Fn, typename... Args>
inline auto coll_exec(Fn&& fn, Args&&... args) {
  ITYR_CHECK(ito::is_root());

  ori::release();

  using retval_t = std::invoke_result_t<Fn, Args...>;
  if constexpr (std::is_void_v<retval_t>) {
    ito::coll_exec([=]() {
      ori::acquire();
      fn(args...);
      ori::release();
    });

    ori::acquire();
  } else {
    auto&& ret = ito::coll_exec([=]() {
      ori::acquire();
      auto&& ret = fn(args...);
      ori::release();
      return std::forward<decltype(ret)>(ret);
    });

    ori::acquire();

    return std::forward<decltype(ret)>(ret);
  }
}

}
