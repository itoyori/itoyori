#pragma once

#include <functional>

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/ito/ito.hpp"

namespace ityr {

class ityr {
public:
  ityr(MPI_Comm comm)
    : topo_(comm, false) {}

  const common::topology& topology() const { return topo_; }

private:
  common::topology topo_;
};

inline std::optional<ityr>& ityr_get_() {
  static std::optional<ityr> instance;
  return instance;
}

inline ityr& ityr_get() {
  ITYR_CHECK(ityr_get_().has_value());
  return *ityr_get_();
}

inline void ityr_init(MPI_Comm comm = MPI_COMM_WORLD) {
  ITYR_CHECK(!ityr_get_().has_value());
  ityr_get_().emplace(comm);

  ito::ito_init(ityr_get().topology());
}

inline void ityr_fini() {
  ITYR_CHECK(ityr_get_().has_value());
  ityr_get_().reset();

  ito::ito_fini();
}

template <typename Fn, typename... Args>
inline auto master_do(Fn&& fn, Args&&... args) {
  using ret_t = std::invoke_result_t<Fn, Args...>;
  ret_t retval;
  if (ityr_get().topology().my_rank() == 0) {
    ito::thread<ret_t> th;
    th.fork_root(std::forward<Fn>(fn), std::forward<Args>(args)...);
    retval = th.join();
  } else {
    ito::worker_get().sched_loop();
  }
  return common::mpi_bcast_value(retval, 0, ityr_get().topology().mpicomm());
}

ITYR_TEST_CASE("[ityr::ito::thread] fib") {
  ityr_init();

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

  int r = master_do(fib, 10);
  ITYR_CHECK(r == 89);

  ityr_fini();
}

}
