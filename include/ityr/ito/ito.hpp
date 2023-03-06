#pragma once

#include <functional>

#include "ityr/common/util.hpp"
#include "ityr/common/virtual_mem.hpp"
#include "ityr/common/physical_mem.hpp"
#include "ityr/common/allocator.hpp"
#include "ityr/ito/thread.hpp"
#include "ityr/ito/worker.hpp"

namespace ityr::ito {

class ito {
public:
  ito(MPI_Comm comm)
    : topo_(comm) {}

private:
  common::mpi_initializer                                    mi_;
  common::singleton_initializer<common::topology::instance>  topo_;
  common::singleton_initializer<worker::instance>            worker_;
};

using instance = common::singleton<ito>;

inline void init(MPI_Comm comm = MPI_COMM_WORLD) { instance::init(comm); }
inline void fini()                               { instance::fini();     }

template <typename Fn, typename... Args>
inline auto root_exec(Fn&& fn, Args&&... args) {
  auto& w = worker::instance::get();
  return w.root_exec(std::forward<Fn>(fn), std::forward<Args>(args)...);
}

ITYR_TEST_CASE("[ityr::ito] fib") {
  init();

  std::function<int(int)> fib = [&](int n) -> int {
    if (n <= 1) {
      return 1;
    } else {
      thread<int> th([=]{ return fib(n - 1); });
      int y = fib(n - 2);
      int x = th.join();
      return x + y;
    }
  };

  int r = root_exec(fib, 10);
  ITYR_CHECK(r == 89);

  fini();
}

ITYR_TEST_CASE("[ityr::ito] load balancing") {
  init();

  std::function<void(int)> lb = [&](int n) {
    if (n == 0) {
      return;
    } else if (n == 1) {
      common::mpi_barrier(common::topology::mpicomm());
    } else {
      thread<void> th([=]{ return lb(n / 2); });
      lb(n - n / 2);
      th.join();
    }
  };

  root_exec(lb, common::topology::n_ranks());

  fini();
}

}
