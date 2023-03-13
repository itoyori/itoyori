#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/ito/ito.hpp"
#include "ityr/ori/ori.hpp"
#include "ityr/pattern/iterator.hpp"
#include "ityr/pattern/global_iterator.hpp"

namespace ityr {

struct serial_loop_options {
  std::size_t checkout_count = 1;
};

template <typename ForwardIterator, typename Fn>
inline void serial_for_each(ForwardIterator first,
                            ForwardIterator last,
                            Fn              fn) {
  serial_for_each(serial_loop_options{}, first, last, fn);
}

template <typename ForwardIterator, typename Fn>
inline void serial_for_each(const serial_loop_options&,
                            ForwardIterator first,
                            ForwardIterator last,
                            Fn              fn) {
  for (; first != last; ++first) {
    fn(*first);
  }
}

template <typename T, typename Mode, typename Fn>
inline void serial_for_each(const serial_loop_options& opts,
                            global_iterator<T, Mode>   first,
                            global_iterator<T, Mode>   last,
                            Fn                         fn) {
  if constexpr (global_iterator<T, Mode>::auto_checkout) {
    auto n = std::distance(first, last);
    for (std::ptrdiff_t d = 0; d < n; d += opts.checkout_count) {
      auto n_ = std::min(static_cast<std::size_t>(n - d), opts.checkout_count);
      ori::with_checkout(&*std::next(first, d), n_, Mode{}, [&](auto&& it) {
        serial_for_each(opts, it, std::next(it, n_), fn);
      });
    }
  } else {
    for (; first != last; ++first) {
      fn(*first);
    }
  }
}

struct parallel_loop_options {
  std::size_t cutoff_count   = 1;
  std::size_t checkout_count = 1;
};

inline void parallel_loop_options_assert(const parallel_loop_options& opts) {
  ITYR_CHECK(0 < opts.checkout_count);
  ITYR_CHECK(opts.checkout_count <= opts.cutoff_count);
}

template <typename ForwardIterator, typename T, typename ReduceOp>
inline T parallel_reduce(ForwardIterator       first,
                         ForwardIterator       last,
                         T                     identity,
                         ReduceOp              reduce) {
  return parallel_reduce(parallel_loop_options{}, first, last, identity, reduce);
}

template <typename ForwardIterator, typename T, typename ReduceOp>
inline T parallel_reduce(parallel_loop_options opts,
                         ForwardIterator       first,
                         ForwardIterator       last,
                         T                     identity,
                         ReduceOp              reduce) {
  auto transform = [](auto&& v) { return std::forward<decltype(v)>(v); };
  return parallel_reduce(opts, first, last, identity, reduce, transform);
}

template <typename ForwardIterator, typename T, typename ReduceOp, typename TransformOp>
inline T parallel_reduce(ForwardIterator       first,
                         ForwardIterator       last,
                         T                     identity,
                         ReduceOp              reduce,
                         TransformOp           transform) {
  return parallel_reduce(parallel_loop_options{}, first, last, identity, reduce, transform);
}

template <typename ForwardIterator, typename T, typename ReduceOp, typename TransformOp>
inline T parallel_reduce(parallel_loop_options opts,
                         ForwardIterator       first,
                         ForwardIterator       last,
                         T                     identity,
                         ReduceOp              reduce,
                         TransformOp           transform) {
  parallel_loop_options_assert(opts);

  if constexpr (is_global_iterator_v<ForwardIterator>) {
    static_assert(std::is_same_v<typename ForwardIterator::mode, ori::mode::read_t> ||
                  std::is_same_v<typename ForwardIterator::mode, ori::mode::no_access_t>);
    return parallel_reduce_aux(opts, first, last, identity, reduce, transform);

  } else if constexpr (ori::is_global_ptr_v<ForwardIterator>) {
    auto first_ = make_global_iterator(first, ori::mode::read);
    auto last_  = make_global_iterator(last , ori::mode::read);
    return parallel_reduce_aux(opts, first_, last_, identity, reduce, transform);

  } else {
    return parallel_reduce_aux(opts, first, last, identity, reduce, transform);
  }
}

template <typename ForwardIterator, typename T, typename ReduceOp, typename TransformOp>
inline T parallel_reduce_aux(parallel_loop_options opts,
                             ForwardIterator       first,
                             ForwardIterator       last,
                             T                     identity,
                             ReduceOp              reduce,
                             TransformOp           transform) {
  auto serial_fn = [=](const serial_loop_options& opts_,
                       ForwardIterator            first_,
                       ForwardIterator            last_) {
    T acc = identity;
    serial_for_each(opts_, first_, last_, [&](const auto& v) {
      acc = reduce(acc, transform(v));
    });
    return acc;
  };
  return parallel_loop_generic(opts, serial_fn, reduce, first, last);
}

template <typename SerialFn, typename CombineFn,
          typename ForwardIterator, typename... ForwardIterators>
inline auto parallel_loop_generic(parallel_loop_options opts,
                                  SerialFn              serial_fn,
                                  CombineFn             combine_fn,
                                  ForwardIterator       first,
                                  ForwardIterator       last,
                                  ForwardIterators...   firsts) {
  auto d = std::distance(first, last);
  if (static_cast<std::size_t>(d) <= opts.cutoff_count) {
    auto serial_fn_call = [&]() {
      return serial_fn(serial_loop_options{.checkout_count = opts.checkout_count},
                       first, last, firsts...);
    };
    using retval_t = std::invoke_result_t<decltype(serial_fn_call)>;
    if constexpr (std::is_void_v<retval_t>) {
      serial_fn_call();
      ori::release();
    } else {
      auto ret = serial_fn_call();
      ori::release();
      return ret;
    }
  } else {
    ori::release();

    auto mid = std::next(first, d / 2);
    auto recur_fn_left = [=]() {
      return parallel_loop_generic(opts, serial_fn, combine_fn,
                                   first, mid, firsts...);
    };
    using retval_t = std::invoke_result_t<decltype(recur_fn_left)>;

    auto recur_fn_right = [&]() {
      return parallel_loop_generic(opts, serial_fn, combine_fn,
                                   mid, last, std::next(firsts, d / 2)...);
    };

    ito::thread<retval_t> th(recur_fn_left);
    if (!th.serialized()) {
      ori::acquire();
    }

    if constexpr(std::is_void_v<retval_t>) {
      recur_fn_right();
      th.join();

      ori::acquire();
    } else {
      auto ret2 = recur_fn_right();
      auto ret1 = th.join();

      ori::acquire();

      return combine_fn(ret1, ret2);
    }
  }
}

ITYR_TEST_CASE("[ityr::pattern::parallel_loop] parallel reduce") {
  ito::init();
  ori::init();

  ITYR_SUBCASE("default cutoff") {
    int n = 10000;
    int r = ito::root_exec([=] {
      return parallel_reduce(
        ityr::count_iterator<int>(0),
        ityr::count_iterator<int>(n),
        0,
        std::plus<int>{});
    });
    ITYR_CHECK(r == n * (n - 1) / 2);
  }

  ITYR_SUBCASE("custom cutoff") {
    int n = 100000;
    int r = ito::root_exec([=] {
      return parallel_reduce(
        parallel_loop_options{.cutoff_count = 100},
        ityr::count_iterator<int>(0),
        ityr::count_iterator<int>(n),
        0,
        std::plus<int>{});
    });
    ITYR_CHECK(r == n * (n - 1) / 2);
  }

  ITYR_SUBCASE("transform") {
    int n = 100000;
    int r = ito::root_exec([=] {
      return parallel_reduce(
        parallel_loop_options{.cutoff_count = 100},
        ityr::count_iterator<int>(0),
        ityr::count_iterator<int>(n),
        0,
        std::plus<int>{},
        [](int x) { return x * x; });
    });
    ITYR_CHECK(r == n * (n - 1) * (2 * n - 1) / 6);
  }

  ori::fini();
  ito::fini();
}

ITYR_TEST_CASE("[ityr::pattern::parallel_loop] parallel reduce with global_ptr") {
  ito::init();
  ori::init();

  int n = 100000;
  ori::global_ptr<int> p = ori::malloc_coll<int>(n);

  ito::root_exec([=] {
    int count = 0;
    serial_for_each({.checkout_count = 100},
                    make_global_iterator(p    , ori::mode::write),
                    make_global_iterator(p + n, ori::mode::write),
                    [&](int& v) { v = count++; });;
  });

  ITYR_SUBCASE("default cutoff") {
    int r = ito::root_exec([=] {
      return parallel_reduce(
        p,
        p + n,
        0,
        std::plus<int>{});
    });
    ITYR_CHECK(r == n * (n - 1) / 2);
  }

  ITYR_SUBCASE("custom cutoff and checkout count") {
    int r = ito::root_exec([=] {
      return parallel_reduce(
        {.cutoff_count = 100, .checkout_count = 50},
        p,
        p + n,
        0,
        std::plus<int>{});
    });
    ITYR_CHECK(r == n * (n - 1) / 2);
  }

  ITYR_SUBCASE("without auto checkout") {
    int r = ito::root_exec([=] {
      return parallel_reduce(
        make_global_iterator(p    , ori::mode::no_access),
        make_global_iterator(p + n, ori::mode::no_access),
        0,
        std::plus<int>{},
        [](ori::global_ref<int> gref) {
          return ori::with_checkout(&gref, 1, ori::mode::read, [](const int* v) { return *v; });
        });
    });
    ITYR_CHECK(r == n * (n - 1) / 2);
  }

  ori::fini();
  ito::fini();
}

}
