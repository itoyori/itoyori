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

template <typename... Args>
struct count_num_tasks;

template <typename Fn>
struct count_num_tasks<Fn> {
  static constexpr int value = 1;
};

template <typename Fn1, typename Fn2, typename... Rest>
struct count_num_tasks<Fn1, Fn2, Rest...> {
  static constexpr int value = 1 + count_num_tasks<Fn2, Rest...>::value;
};

template <typename Fn, typename... Args>
struct count_num_tasks<Fn, std::tuple<Args...>> {
  static constexpr int value = 1;
};

template <typename Fn, typename... Args, typename... Rest>
struct count_num_tasks<Fn, std::tuple<Args...>, Rest...> {
  static constexpr int value = 1 + count_num_tasks<Rest...>::value;
};

static_assert(count_num_tasks<void (*)()>::value                                            == 1);
static_assert(count_num_tasks<void (*)(int), std::tuple<int>>::value                        == 1);
static_assert(count_num_tasks<void (*)(), int (*)()>::value                                 == 2);
static_assert(count_num_tasks<void (*)(int, int), std::tuple<int, int>, void (*)()>::value  == 2);
static_assert(count_num_tasks<void (*)(int), std::tuple<int>, int (*)(), void (*)()>::value == 3);

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

  template <typename Fn, typename... Args>
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

    constexpr int n_rest_tasks = count_num_tasks<Rest...>::value;
    static_assert(n_rest_tasks > 0);

    ori::poll();

    // for immediately executing cross-worker tasks in ADWS
    // TODO: remove one of these two acquire calls?
    ito::poll([&]() { all_serialized_ = false; return ori::release_lazy(); },
              [&](ori::release_handler rh) { ori::acquire(rh); ori::acquire(rh_); });

    ito::thread<retval_t> th(ito::with_callback,
                             [rh = rh_]() { ori::acquire(rh); },
                             []() { ori::release(); },
                             ito::with_workhint, 1, n_rest_tasks,
                             std::apply<const Fn&, const std::tuple<Args...>&>,
                             std::forward<Fn>(fn),
                             std::forward<std::tuple<Args...>>(args));
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

  auto tgdata = ito::task_group_begin();

  parallel_invoke_state s(rh);
  auto ret = s.parallel_invoke_aux(std::forward<Args>(args)...);

  // No lazy release here because the suspended thread (cross-worker tasks in ADWS) is
  // always resumed by another process.
  ito::task_group_end(tgdata,
                      []() { ori::release(); },
                      []() { ori::acquire(); });

  // TODO: avoid duplicated acquire calls
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
