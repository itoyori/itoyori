#pragma once

#include "ityr/common/util.hpp"
#include "ityr/ito/ito.hpp"
#include "ityr/ori/ori.hpp"
#include "ityr/pattern/iterator.hpp"
#include "ityr/pattern/global_iterator.hpp"

namespace ityr {

struct no_retval_t {};
inline constexpr no_retval_t no_retval;

inline bool operator==(no_retval_t, no_retval_t) noexcept {
  return true;
}

inline bool operator!=(no_retval_t, no_retval_t) noexcept {
  return false;
}

template <typename ReleaseHandler>
struct parallel_invoke_state {
public:
  parallel_invoke_state(ReleaseHandler rh) : rh_(rh) {}

  bool all_serialized() const { return all_serialized_; }

  inline auto parallel_invoke_aux() {
    return std::make_tuple();
  }

  template <typename Fn>
  inline auto parallel_invoke_aux(Fn&& fn) {
    // insert an empty arg tuple
    return parallel_invoke_aux(std::forward<Fn>(fn),
                               std::make_tuple());
  }

  template <typename Fn1, typename Fn2, typename... Rest>
  inline auto parallel_invoke_aux(Fn1&& fn1, Fn2&& fn2, Rest&&... rest) {
    // insert an empty arg tuple
    return parallel_invoke_aux(std::forward<Fn1>(fn1),
                               std::make_tuple(),
                               std::forward<Fn2>(fn2),
                               std::forward<Rest>(rest)...);
  }

  template <typename Fn, typename... Args, typename... Rest>
  inline auto parallel_invoke_aux(Fn&& fn, std::tuple<Args...>&& args) {
    using retval_t = std::invoke_result_t<Fn, Args...>;

    if constexpr (std::is_void_v<retval_t>) {
      std::apply(std::forward<Fn>(fn),
                 std::forward<std::tuple<Args...>>(args));
      return std::make_tuple(no_retval);
    } else {
      auto ret = std::apply(std::forward<Fn>(fn),
                            std::forward<std::tuple<Args...>>(args));
      return std::make_tuple(ret);
    }
  }

  template <typename Fn, typename... Args, typename... Rest>
  inline auto parallel_invoke_aux(Fn&& fn, std::tuple<Args...>&& args, Rest&&... rest) {
    using retval_t = std::invoke_result_t<Fn, Args...>;

    ori::poll();

    ito::thread<retval_t> th(ito::with_callback,
                             []() { ori::release(); },
                             std::apply<const Fn&, const std::tuple<Args...>&>,
                             std::forward<Fn>(fn),
                             std::forward<std::tuple<Args...>>(args));
    if (!th.serialized()) {
      ori::acquire(rh_);
    }
    all_serialized_ &= th.serialized();

    auto ret_rest = parallel_invoke_aux(std::forward<Rest>(rest)...);

    if constexpr (std::is_void_v<retval_t>) {
      if (!th.serialized()) {
        ori::release();
      }

      th.join();
      return std::tuple_cat(std::make_tuple(no_retval), ret_rest);
    } else {
      if (!th.serialized()) {
        ori::release();
      }

      auto ret = th.join();
      return std::tuple_cat(std::make_tuple(ret), ret_rest);
    }
  }

private:
  ReleaseHandler rh_;
  bool           all_serialized_ = true;
};

template <typename... Args>
inline auto parallel_invoke(Args&&... args) {
  auto rh = ori::release_lazy();
  parallel_invoke_state s(rh);
  auto ret = s.parallel_invoke_aux(std::forward<Args>(args)...);
  if (!s.all_serialized()) {
    ori::acquire();
  }
  return ret;
}

ITYR_TEST_CASE("[ityr::pattern::parallel_invoke] parallel invoke") {
  ito::init();
  ori::init();

  ITYR_SUBCASE("with functions") {
    ito::root_exec([=] {
      auto [x, y] = parallel_invoke(
        []() { return 1; },
        []() { return 2; }
      );
      ITYR_CHECK(x == 1);
      ITYR_CHECK(y == 2);
    });

    ito::root_exec([=] {
      auto [x, y, z] = parallel_invoke(
        []() { return 1;   },
        []() { return 2;   },
        []() { return 4.8; }
      );
      ITYR_CHECK(x == 1);
      ITYR_CHECK(y == 2);
      ITYR_CHECK(z == 4.8);
    });
  }

  ITYR_SUBCASE("with args") {
    ito::root_exec([=] {
      auto [x, y] = parallel_invoke(
        [](int i) { return i    ; }, std::make_tuple(1),
        [](int i) { return i * 2; }, std::make_tuple(2)
      );
      ITYR_CHECK(x == 1);
      ITYR_CHECK(y == 4);
    });

    ito::root_exec([=] {
      auto [x, y] = parallel_invoke(
        [](int i, int j       ) { return i + j    ; }, std::make_tuple(1, 2),
        [](int i, int j, int k) { return i + j + k; }, std::make_tuple(3, 4, 5)
      );
      ITYR_CHECK(x == 3);
      ITYR_CHECK(y == 12);
    });
  }

  ITYR_SUBCASE("corner cases") {
    ito::root_exec([=] {
      ITYR_CHECK(parallel_invoke() == std::make_tuple());

      // The following is not allowed before C++20: Lambda expression in an unevaluated operand
      // ITYR_CHECK(std::make_tuple(no_retval) == parallel_invoke([]{}));

      {
        auto ret = parallel_invoke([]{});
        ITYR_CHECK(ret == std::make_tuple(no_retval));
      }

      {
        auto ret = parallel_invoke([]{}, []{});
        ITYR_CHECK(ret == std::make_tuple(no_retval, no_retval));
      }

      {
        auto ret = parallel_invoke([]{}, []{}, []{ return 1; });
        ITYR_CHECK(ret == std::make_tuple(no_retval, no_retval, 1));
      }

      {
        auto ret = parallel_invoke([]{ return 1; }, []{}, []{});
        ITYR_CHECK(ret == std::make_tuple(1, no_retval, no_retval));
      }

      {
        auto ret = parallel_invoke([](int){}, std::make_tuple(1), []{ return 1; }, []{});
        ITYR_CHECK(ret == std::make_tuple(no_retval, 1, no_retval));
      }
    });
  }

  ori::fini();
  ito::fini();
}

}
