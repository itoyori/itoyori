#pragma once

#include <functional>

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/wallclock.hpp"
#include "ityr/common/profiler.hpp"
#include "ityr/common/prof_events.hpp"
#include "ityr/ito/util.hpp"
#include "ityr/ito/thread.hpp"
#include "ityr/ito/worker.hpp"
#include "ityr/ito/prof_events.hpp"

namespace ityr::ito {

class ito {
public:
  ito(MPI_Comm comm)
    : topo_(comm) {}

private:
  common::mpi_initializer                                    mi_;
  common::singleton_initializer<common::topology::instance>  topo_;
  aslr_checker                                               aslr_checker_;
  common::singleton_initializer<common::wallclock::instance> clock_;
  common::singleton_initializer<common::profiler::instance>  prof_;
  common::singleton_initializer<worker::instance>            worker_;
  common::prof_events                                        common_prof_events_;
  prof_events                                                ito_prof_events_;
};

using instance = common::singleton<ito>;

inline void init(MPI_Comm comm = MPI_COMM_WORLD) {
  instance::init(comm);
}

inline void fini() {
  instance::fini();
}

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
