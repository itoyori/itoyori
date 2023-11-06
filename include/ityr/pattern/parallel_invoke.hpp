#pragma once

#include <variant>

#include "ityr/common/util.hpp"
#include "ityr/ito/ito.hpp"
#include "ityr/ori/ori.hpp"
#include "ityr/pattern/count_iterator.hpp"
#include "ityr/pattern/global_iterator.hpp"

namespace ityr {

namespace internal {

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

  template <typename Fn, typename... Args, typename... Rest>
  inline auto parallel_invoke_aux(Fn&& fn, const std::tuple<Args...>& args, Rest&&... rest) {
    return parallel_invoke_with_args(std::forward<Fn>(fn), args, std::forward<Rest>(rest)...);
  }

  template <typename Fn, typename... Args, typename... Rest>
  inline auto parallel_invoke_aux(Fn&& fn, std::tuple<Args...>&& args, Rest&&... rest) {
    return parallel_invoke_with_args(std::forward<Fn>(fn), std::move(args), std::forward<Rest>(rest)...);
  }

private:
  template <typename Fn, typename ArgsTuple>
  inline auto parallel_invoke_with_args(Fn&& fn, ArgsTuple&& args_tuple) {
    using retval_t = std::invoke_result_t<decltype(std::apply<Fn, ArgsTuple>), Fn, ArgsTuple>;

    if constexpr (std::is_void_v<retval_t>) {
      std::apply(std::forward<Fn>(fn), std::forward<ArgsTuple>(args_tuple));
      return std::make_tuple(std::monostate{});

    } else {
      auto&& ret = std::apply(std::forward<Fn>(fn), std::forward<ArgsTuple>(args_tuple));
      return std::make_tuple(std::forward<decltype(ret)>(ret));
    }
  }

  template <typename Fn, typename ArgsTuple, typename... Rest>
  inline auto parallel_invoke_with_args(Fn&& fn, ArgsTuple&& args_tuple, Rest&&... rest) {
    using retval_t = std::invoke_result_t<decltype(std::apply<Fn, ArgsTuple>), Fn, ArgsTuple>;

    constexpr int n_rest_tasks = count_num_tasks<Rest...>::value;
    static_assert(n_rest_tasks > 0);

    ori::poll();

    // for immediately executing cross-worker tasks in ADWS
    // TODO: remove one of these two acquire calls?
    ito::poll([&]() { all_serialized_ = false; return ori::release_lazy(); },
              [&](ori::release_handler rh) { ori::acquire(rh); ori::acquire(rh_); });

    ito::thread<retval_t> th(
        ito::with_callback, [rh = rh_] { ori::acquire(rh); }, [] { ori::release(); },
        ito::workhint(1, n_rest_tasks),
        [fn         = std::forward<Fn>(fn),
         args_tuple = std::forward<ArgsTuple>(args_tuple)]() mutable {
          return std::apply(std::forward<decltype(fn)>(fn),
                            std::forward<decltype(args_tuple)>(args_tuple));
        });
    all_serialized_ &= th.serialized();

    auto&& ret_rest = parallel_invoke_aux(std::forward<Rest>(rest)...);

    if constexpr (std::is_void_v<retval_t>) {
      if (!th.serialized()) {
        ori::release();
      }

      th.join();
      return std::tuple_cat(std::make_tuple(std::monostate{}),
                            std::move(ret_rest));

    } else {
      if (!th.serialized()) {
        ori::release();
      }

      auto&& ret = th.join();
      return std::tuple_cat(std::make_tuple(std::forward<decltype(ret)>(ret)),
                            std::move(ret_rest));
    }
  }

  ReleaseHandler rh_;
  bool           all_serialized_ = true;
};

}

/**
 * @brief Fork parallel tasks and join them.
 *
 * @param args... Sequence of function objects and their arguments in the form of `fn1, (args1), fn2, (args2), ...`,
 *                where `fn1` is a function object and `args1` is a tuple of arguments for `fn1`, and so on.
 *                The tuples of arguments are optional if functions require no argument.
 *
 * @return A tuple collecting the results of each function invocation.
 *         For a void function, `std::monostate` is used as a placeholder.
 *
 * This function forks parallel tasks given as function objects and joins them at a time.
 * This function blocks the current thread's execution until all of the child tasks are completed.
 *
 * The executing process can be changed across this function call (due to thread migration).
 *
 * Example:
 * ```
 * ityr::parallel_invoke(
 *   []() { return 1; },                                        // no argument
 *   [](int x) { return x; }, std::make_tuple(2),               // one argument
 *   [](int x, int y) { return x * y; }, std::make_tuple(3, 4), // two arguments
 *   []() {},                                                   // no return value
 *   std::plus<int>{}, std::make_tuple(5, 6)                    // a function (not lambda)
 * );
 * // returns std::tuple(1, 2, 12, std::monospace{}, 11)
 * ```
 */
template <typename... Args>
inline auto parallel_invoke(Args&&... args) {
  auto rh = ori::release_lazy();

  ito::task_group_data tgdata;
  ito::task_group_begin(&tgdata);

  internal::parallel_invoke_state s(rh);
  auto&& ret = s.parallel_invoke_aux(std::forward<Args>(args)...);

  // No lazy release here because the suspended thread (cross-worker tasks in ADWS) is
  // always resumed by another process.
  ito::task_group_end([] { ori::release(); }, [] { ori::acquire(); });

  // TODO: avoid duplicated acquire calls
  if (!s.all_serialized()) {
    ori::acquire();
  }
  return std::move(ret);
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
      // ITYR_CHECK(std::make_tuple(std::monostate{}) == parallel_invoke([]{}));

      {
        auto ret = parallel_invoke([]{});
        ITYR_CHECK(ret == std::make_tuple(std::monostate{}));
      }

      {
        auto ret = parallel_invoke([]{}, []{});
        ITYR_CHECK(ret == std::make_tuple(std::monostate{}, std::monostate{}));
      }

      {
        auto ret = parallel_invoke([]{}, []{}, []{ return 1; });
        ITYR_CHECK(ret == std::make_tuple(std::monostate{}, std::monostate{}, 1));
      }

      {
        auto ret = parallel_invoke([]{ return 1; }, []{}, []{});
        ITYR_CHECK(ret == std::make_tuple(1, std::monostate{}, std::monostate{}));
      }

      {
        auto ret = parallel_invoke([](int){}, std::make_tuple(1), []{ return 1; }, []{});
        ITYR_CHECK(ret == std::make_tuple(std::monostate{}, 1, std::monostate{}));
      }
    });
  }

  ori::fini();
  ito::fini();
}

}
