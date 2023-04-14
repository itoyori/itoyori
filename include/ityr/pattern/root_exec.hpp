#pragma once

#include "ityr/common/util.hpp"
#include "ityr/ito/ito.hpp"
#include "ityr/ori/ori.hpp"

namespace ityr {

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
    ori::acquire();
  } else {
    auto ret = ito::root_exec(ito::with_callback,
                              []() { ori::poll(); },
                              std::forward<Fn>(fn), std::forward<Args>(args)...);
    ori::acquire();
    return ret;
  }
}

}
