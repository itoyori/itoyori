#pragma once

#include <functional>

#include "ityr/common/util.hpp"
#include "ityr/common/virtual_mem.hpp"
#include "ityr/common/physical_mem.hpp"
#include "ityr/common/allocator.hpp"
#include "ityr/ito/thread.hpp"
#include "ityr/ito/worker.hpp"

namespace ityr::ito {

inline void ito_init(const common::topology& topo) {
  worker_init(topo);
}

inline void ito_fini() {
  worker_fini();
}

template <typename Fn, typename... Args>
inline auto root_exec(Fn&& fn, Args&&... args) {
  worker& w = worker_get();
  return w.root_exec(std::forward<Fn>(fn), std::forward<Args>(args)...);
}

ITYR_TEST_CASE("[ityr::ito] fib") {
  common::topology topo;
  ito_init(topo);

  std::function<int(int)> fib = [&](int n) -> int {
    if (n <= 1) {
      return 1;
    } else {
      ito::thread<int> th([=]{ return fib(n - 1); });
      int y = fib(n - 2);
      int x = th.join();
      return x + y;
    }
  };

  int r = root_exec(fib, 10);
  ITYR_CHECK(r == 89);

  ito_fini();
}

}
