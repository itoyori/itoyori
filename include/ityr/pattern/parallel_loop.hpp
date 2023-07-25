#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/ito/ito.hpp"
#include "ityr/ori/ori.hpp"
#include "ityr/pattern/iterator.hpp"
#include "ityr/pattern/global_iterator.hpp"
#include "ityr/pattern/serial_loop.hpp"

namespace ityr {

template <typename SerialFn, typename CombineFn, typename ReleaseHandler,
          typename ForwardIterator, typename... ForwardIterators>
inline std::invoke_result_t<SerialFn, ForwardIterator, ForwardIterator, ForwardIterators...>
parallel_loop_generic(const execution::parallel_policy& policy,
                      SerialFn                          serial_fn,
                      CombineFn                         combine_fn,
                      ReleaseHandler                    rh,
                      ForwardIterator                   first,
                      ForwardIterator                   last,
                      ForwardIterators...               firsts) {
  using retval_t = std::invoke_result_t<SerialFn,
                                        ForwardIterator, ForwardIterator, ForwardIterators...>;
  ori::poll();

  // for immediately executing cross-worker tasks in ADWS
  ito::poll([]() { return ori::release_lazy(); },
            [&](ori::release_handler rh_) { ori::acquire(rh); ori::acquire(rh_); });

  auto d = std::distance(first, last);
  if (static_cast<std::size_t>(d) <= policy.cutoff_count) {
    return serial_fn(first, last, firsts...);
  }

  auto mid = std::next(first, d / 2);
  auto recur_fn_left = [=]() -> retval_t {
    return parallel_loop_generic(policy, serial_fn, combine_fn, rh,
                                 first, mid, firsts...);
  };

  auto recur_fn_right = [&]() -> retval_t {
    return parallel_loop_generic(policy, serial_fn, combine_fn, rh,
                                 mid, last, std::next(firsts, d / 2)...);
  };

  auto tgdata = ito::task_group_begin();

  ito::thread<retval_t> th(ito::with_callback,
                           [=]() { ori::acquire(rh); },
                           [=]() { ori::release(); },
                           ito::with_workhint, 1, 1,
                           recur_fn_left);

  if constexpr (std::is_void_v<retval_t>) {
    recur_fn_right();

    if (!th.serialized()) {
      ori::release();
    }

    th.join();

    ito::task_group_end(tgdata,
                        []() { ori::release(); },
                        []() { ori::acquire(); });

    if (!th.serialized()) {
      ori::acquire();
    }
  } else {
    auto ret2 = recur_fn_right();

    if (!th.serialized()) {
      ori::release();
    }

    auto ret1 = th.join();

    ito::task_group_end(tgdata,
                        []() { ori::release(); },
                        []() { ori::acquire(); });

    if (!th.serialized()) {
      ori::acquire();
    }

    return combine_fn(ret1, ret2);
  }
}

template <typename SerialFn, typename CombineFn,
          typename ForwardIterator, typename... ForwardIterators>
inline auto loop_generic(const execution::sequenced_policy& policy,
                         SerialFn                           serial_fn,
                         CombineFn                          combine_fn [[maybe_unused]],
                         ForwardIterator                    first,
                         ForwardIterator                    last,
                         ForwardIterators...                firsts) {
  assert_policy(policy);
  return serial_fn(first, last, firsts...);
}

template <typename SerialFn, typename CombineFn,
          typename ForwardIterator, typename... ForwardIterators>
inline auto loop_generic(const execution::parallel_policy& policy,
                         SerialFn                          serial_fn,
                         CombineFn                         combine_fn,
                         ForwardIterator                   first,
                         ForwardIterator                   last,
                         ForwardIterators...               firsts) {
  assert_policy(policy);
  auto rh = ori::release_lazy();
  return parallel_loop_generic(policy, serial_fn, combine_fn, rh, first, last, firsts...);
}

template <typename ForwardIterator, typename Op>
inline void for_each(const execution::parallel_policy& policy,
                     ForwardIterator                   first,
                     ForwardIterator                   last,
                     Op                                op) {
  auto seq_policy = execution::to_sequenced_policy(policy);
  auto serial_fn = [=](ForwardIterator first_,
                       ForwardIterator last_) mutable {
    for_each(seq_policy, first_, last_, op);
  };
  loop_generic(policy, serial_fn, []{}, first, last);
}

template <typename ForwardIterator1, typename ForwardIterator2, typename Op>
inline void for_each(const execution::parallel_policy& policy,
                     ForwardIterator1                  first1,
                     ForwardIterator1                  last1,
                     ForwardIterator2                  first2,
                     Op                                op) {
  auto seq_policy = execution::to_sequenced_policy(policy);
  auto serial_fn = [=](ForwardIterator1 first1_,
                       ForwardIterator1 last1_,
                       ForwardIterator2 first2_) mutable {
    for_each(seq_policy, first1_, last1_, first2_, op);
  };
  loop_generic(policy, serial_fn, []{}, first1, last1, first2);
}

template <typename ForwardIterator1, typename ForwardIterator2, typename ForwardIterator3, typename Op>
inline void for_each(const execution::parallel_policy& policy,
                     ForwardIterator1                  first1,
                     ForwardIterator1                  last1,
                     ForwardIterator2                  first2,
                     ForwardIterator3                  first3,
                     Op                                op) {
  auto seq_policy = execution::to_sequenced_policy(policy);
  auto serial_fn = [=](ForwardIterator1 first1_,
                       ForwardIterator1 last1_,
                       ForwardIterator2 first2_,
                       ForwardIterator3 first3_) mutable {
    for_each(seq_policy, first1_, last1_, first2_, first3_, op);
  };
  loop_generic(policy, serial_fn, []{}, first1, last1, first2, first3);
}

ITYR_TEST_CASE("[ityr::pattern::parallel_loop] parallel for each") {
  ito::init();
  ori::init();

  int n = 100000;
  ori::global_ptr<int> p1 = ori::malloc_coll<int>(n);
  ori::global_ptr<int> p2 = ori::malloc_coll<int>(n);

  ito::root_exec([=] {
    int count = 0;
    for_each(
        execution::sequenced_policy{.checkout_count = 100},
        make_global_iterator(p1    , checkout_mode::write),
        make_global_iterator(p1 + n, checkout_mode::write),
        [&](int& v) { v = count++; });

    for_each(
        execution::par,
        make_global_iterator(p1    , checkout_mode::read),
        make_global_iterator(p1 + n, checkout_mode::read),
        ityr::count_iterator<int>(0),
        [=](int x, int i) { ITYR_CHECK(x == i); });

    for_each(
        execution::par,
        ityr::count_iterator<int>(0),
        ityr::count_iterator<int>(n),
        make_global_iterator(p1, checkout_mode::read),
        [=](int i, int x) { ITYR_CHECK(x == i); });

    for_each(
        execution::par,
        make_global_iterator(p1    , checkout_mode::read),
        make_global_iterator(p1 + n, checkout_mode::read),
        make_global_iterator(p2    , checkout_mode::write),
        [=](int x, int& y) { y = x * 2; });

    for_each(
        execution::par,
        make_global_iterator(p2    , checkout_mode::read_write),
        make_global_iterator(p2 + n, checkout_mode::read_write),
        [=](int& y) { y *= 2; });

    for_each(
        execution::par,
        ityr::count_iterator<int>(0),
        ityr::count_iterator<int>(n),
        make_global_iterator(p2, checkout_mode::read),
        [=](int i, int y) { ITYR_CHECK(y == i * 4); });
  });

  ori::free_coll(p1);
  ori::free_coll(p2);

  ori::fini();
  ito::fini();
}

template <typename ExecutionPolicy, typename ForwardIterator, typename T,
          typename BinaryReduceOp, typename UnaryTransformOp>
inline T transform_reduce(const ExecutionPolicy& policy,
                          ForwardIterator        first,
                          ForwardIterator        last,
                          T                      init,
                          BinaryReduceOp         binary_reduce_op,
                          UnaryTransformOp       unary_transform_op) {
  using it_ref = typename std::iterator_traits<ForwardIterator>::reference;
  using transformed_t = std::invoke_result_t<UnaryTransformOp, it_ref>;
  static_assert(std::is_invocable_r_v<T, BinaryReduceOp, T&, transformed_t>);
  static_assert(std::is_invocable_r_v<T, BinaryReduceOp, transformed_t, transformed_t>);

  if (first == last) {
    return init;
  }

  if constexpr (is_global_iterator_v<ForwardIterator>) {
    static_assert(std::is_same_v<typename ForwardIterator::mode, checkout_mode::read_t> ||
                  std::is_same_v<typename ForwardIterator::mode, checkout_mode::no_access_t>);

  } else if constexpr (ori::is_global_ptr_v<ForwardIterator>) {
    // automatically convert global pointers to global iterators with read-only access
    auto first_ = make_global_iterator(first, checkout_mode::read);
    auto last_  = make_global_iterator(last , checkout_mode::read);
    return transform_reduce(policy, first_, last_, init, binary_reduce_op, unary_transform_op);
  }

  auto seq_policy = execution::to_sequenced_policy(policy);
  auto serial_fn = [=](ForwardIterator first_,
                       ForwardIterator last_) mutable {
    ITYR_CHECK(std::distance(first_, last_) >= 1);
    T acc = unary_transform_op(*first_);
    for_each(seq_policy, std::next(first_), last_, [&](const auto& v) {
      acc = binary_reduce_op(acc, unary_transform_op(v));
    });
    return acc;
  };

  return binary_reduce_op(init, loop_generic(policy, serial_fn, binary_reduce_op, first, last));
}

template <typename ExecutionPolicy, typename ForwardIterator1, typename ForwardIterator2, typename T,
          typename BinaryReduceOp, typename BinaryTransformOp>
inline T transform_reduce(const ExecutionPolicy& policy,
                          ForwardIterator1       first1,
                          ForwardIterator1       last1,
                          ForwardIterator2       first2,
                          T                      init,
                          BinaryReduceOp         binary_reduce_op,
                          BinaryTransformOp      binary_transform_op) {
  using it1_ref = typename std::iterator_traits<ForwardIterator1>::reference;
  using it2_ref = typename std::iterator_traits<ForwardIterator2>::reference;
  using transformed_t = std::invoke_result_t<BinaryTransformOp, it1_ref, it2_ref>;
  static_assert(std::is_invocable_r_v<T, BinaryReduceOp, T&, transformed_t>);
  static_assert(std::is_invocable_r_v<T, BinaryReduceOp, transformed_t, transformed_t>);

  if (first1 == last1) {
    return init;
  }

  if constexpr (is_global_iterator_v<ForwardIterator1>) {
    static_assert(std::is_same_v<typename ForwardIterator1::mode, checkout_mode::read_t> ||
                  std::is_same_v<typename ForwardIterator1::mode, checkout_mode::no_access_t>);

  } else if constexpr (ori::is_global_ptr_v<ForwardIterator1>) {
    // automatically convert global pointers to global iterators with read-only access
    auto first1_ = make_global_iterator(first1, checkout_mode::read);
    auto last1_  = make_global_iterator(last1 , checkout_mode::read);
    return transform_reduce(policy, first1_, last1_, first2, init, binary_reduce_op, binary_transform_op);
  }

  if constexpr (is_global_iterator_v<ForwardIterator2>) {
    static_assert(std::is_same_v<typename ForwardIterator2::mode, checkout_mode::read_t> ||
                  std::is_same_v<typename ForwardIterator2::mode, checkout_mode::no_access_t>);

  } else if constexpr (ori::is_global_ptr_v<ForwardIterator2>) {
    // automatically convert global pointers to global iterators with read-only access
    auto first2_ = make_global_iterator(first2, checkout_mode::read);
    return transform_reduce(policy, first1, last1, first2_, init, binary_reduce_op, binary_transform_op);
  }

  auto seq_policy = execution::to_sequenced_policy(policy);
  auto serial_fn = [=](ForwardIterator1 first1_,
                       ForwardIterator1 last1_,
                       ForwardIterator2 first2_) mutable {
    ITYR_CHECK(std::distance(first1_, last1_) >= 1);
    T acc = binary_transform_op(*first1_, *first2_);
    for_each(seq_policy, std::next(first1_), last1_, std::next(first2_), [&](const auto& v1, const auto& v2) {
      acc = binary_reduce_op(acc, binary_transform_op(v1, v2));
    });
    return acc;
  };

  return binary_reduce_op(init, loop_generic(policy, serial_fn, binary_reduce_op, first1, last1, first2));
}

template <typename ExecutionPolicy, typename ForwardIterator1, typename ForwardIterator2, typename T>
inline T transform_reduce(const ExecutionPolicy& policy,
                          ForwardIterator1       first1,
                          ForwardIterator1       last1,
                          ForwardIterator2       first2,
                          T                      init) {
  return transform_reduce(policy, first1, last1, first2, init, std::plus<>{}, std::multiplies<>{});
}

template <typename ExecutionPolicy, typename ForwardIterator, typename T, typename BinaryReduceOp>
inline T reduce(const ExecutionPolicy& policy,
                ForwardIterator        first,
                ForwardIterator        last,
                T                      init,
                BinaryReduceOp         binary_reduce_op) {
  using it_ref = typename std::iterator_traits<ForwardIterator>::reference;
  static_assert(std::is_invocable_r_v<T, BinaryReduceOp, T&, it_ref>);
  static_assert(std::is_invocable_r_v<T, BinaryReduceOp, it_ref, it_ref>);

  return transform_reduce(policy, first, last, init, binary_reduce_op,
                          [](auto&& v) { return std::forward<decltype(v)>(v); });
}

template <typename ExecutionPolicy, typename ForwardIterator, typename T>
inline T reduce(const ExecutionPolicy& policy,
                ForwardIterator        first,
                ForwardIterator        last,
                T                      init) {
  return reduce(policy, first, last, init, std::plus<>{});
}

template <typename ExecutionPolicy, typename ForwardIterator>
inline typename std::iterator_traits<ForwardIterator>::value_type
reduce(const ExecutionPolicy& policy,
       ForwardIterator        first,
       ForwardIterator        last) {
  using value_type = typename std::iterator_traits<ForwardIterator>::value_type;
  return reduce(policy, first, last, value_type{});
}

template <typename ExecutionPolicy, typename ForwardIterator1,
          typename ForwardIteratorD, typename UnaryOp>
inline ForwardIteratorD transform(const ExecutionPolicy& policy,
                                  ForwardIterator1       first1,
                                  ForwardIterator1       last1,
                                  ForwardIteratorD       first_d,
                                  UnaryOp                unary_op) {
  if constexpr (is_global_iterator_v<ForwardIterator1>) {
    static_assert(std::is_same_v<typename ForwardIterator1::mode, checkout_mode::read_t> ||
                  std::is_same_v<typename ForwardIterator1::mode, checkout_mode::no_access_t>);

  } else if constexpr (ori::is_global_ptr_v<ForwardIterator1>) {
    // automatically convert global pointers to global iterators with read-only access
    auto first1_ = make_global_iterator(first1, checkout_mode::read);
    auto last1_  = make_global_iterator(last1 , checkout_mode::read);
    return transform(policy, first1_, last1_, first_d, unary_op);
  }

  // If the destination value type is trivially copyable, write-only access is possible
  using value_type_d = typename std::iterator_traits<ForwardIteratorD>::value_type;
  using checkout_mode_d = std::conditional_t<std::is_trivially_copyable_v<value_type_d>,
                                             checkout_mode::write_t,
                                             checkout_mode::read_write_t>;
  if constexpr (is_global_iterator_v<ForwardIteratorD>) {
    static_assert(std::is_same_v<typename ForwardIteratorD::mode, checkout_mode_d> ||
                  std::is_same_v<typename ForwardIteratorD::mode, checkout_mode::no_access_t>);

  } else if constexpr (ori::is_global_ptr_v<ForwardIteratorD>) {
    // automatically convert global pointers to global iterators
    auto first_d_ = make_global_iterator(first_d, checkout_mode_d{});
    return transform(policy, first1, last1, first_d_, unary_op);
  }

  auto seq_policy = execution::to_sequenced_policy(policy);
  auto serial_fn = [=](ForwardIterator1 first1_,
                       ForwardIterator1 last1_,
                       ForwardIteratorD first_d_) mutable {
    for_each(seq_policy, first1_, last1_, first_d_, [&](const auto& v1, auto&& d) {
      d = unary_op(v1);
    });
  };

  loop_generic(policy, serial_fn, []{}, first1, last1, first_d);

  return std::next(first_d, std::distance(first1, last1));
}

template <typename ExecutionPolicy, typename ForwardIterator1, typename ForwardIterator2,
          typename ForwardIteratorD, typename BinaryOp>
inline ForwardIteratorD transform(const ExecutionPolicy& policy,
                                  ForwardIterator1       first1,
                                  ForwardIterator1       last1,
                                  ForwardIterator2       first2,
                                  ForwardIteratorD       first_d,
                                  BinaryOp               binary_op) {
  if constexpr (is_global_iterator_v<ForwardIterator1>) {
    static_assert(std::is_same_v<typename ForwardIterator1::mode, checkout_mode::read_t> ||
                  std::is_same_v<typename ForwardIterator1::mode, checkout_mode::no_access_t>);

  } else if constexpr (ori::is_global_ptr_v<ForwardIterator1>) {
    // automatically convert global pointers to global iterators with read-only access
    auto first1_ = make_global_iterator(first1, checkout_mode::read);
    auto last1_  = make_global_iterator(last1 , checkout_mode::read);
    return transform(policy, first1_, last1_, first2, first_d, binary_op);
  }

  if constexpr (is_global_iterator_v<ForwardIterator2>) {
    static_assert(std::is_same_v<typename ForwardIterator2::mode, checkout_mode::read_t> ||
                  std::is_same_v<typename ForwardIterator2::mode, checkout_mode::no_access_t>);

  } else if constexpr (ori::is_global_ptr_v<ForwardIterator2>) {
    // automatically convert global pointers to global iterators with read-only access
    auto first2_ = make_global_iterator(first2, checkout_mode::read);
    return transform(policy, first1, last1, first2_, first_d, binary_op);
  }

  // If the destination value type is trivially copyable, write-only access is possible
  using value_type_d = typename std::iterator_traits<ForwardIteratorD>::value_type;
  using checkout_mode_d = std::conditional_t<std::is_trivially_copyable_v<value_type_d>,
                                             checkout_mode::write_t,
                                             checkout_mode::read_write_t>;
  if constexpr (is_global_iterator_v<ForwardIteratorD>) {
    static_assert(std::is_same_v<typename ForwardIteratorD::mode, checkout_mode_d> ||
                  std::is_same_v<typename ForwardIteratorD::mode, checkout_mode::no_access_t>);

  } else if constexpr (ori::is_global_ptr_v<ForwardIteratorD>) {
    // automatically convert global pointers to global iterators
    auto first_d_ = make_global_iterator(first_d, checkout_mode_d{});
    return transform(policy, first1, last1, first2, first_d_, binary_op);
  }

  auto seq_policy = execution::to_sequenced_policy(policy);
  auto serial_fn = [=](ForwardIterator1 first1_,
                       ForwardIterator1 last1_,
                       ForwardIterator2 first2_,
                       ForwardIteratorD first_d_) mutable {
    for_each(seq_policy, first1_, last1_, first2_, first_d_, [&](const auto& v1, const auto& v2, auto&& d) {
      d = binary_op(v1, v2);
    });
  };

  loop_generic(policy, serial_fn, []{}, first1, last1, first2, first_d);

  return std::next(first_d, std::distance(first1, last1));
}

template <typename ExecutionPolicy, typename ForwardIterator, typename T>
inline void fill(const ExecutionPolicy& policy,
                 ForwardIterator        first,
                 ForwardIterator        last,
                 const T&               value) {
  // If the value type is trivially copyable, write-only access is possible
  using checkout_mode_t = std::conditional_t<std::is_trivially_copyable_v<T>,
                                             checkout_mode::write_t,
                                             checkout_mode::read_write_t>;
  if constexpr (is_global_iterator_v<ForwardIterator>) {
    static_assert(std::is_same_v<typename ForwardIterator::mode, checkout_mode_t> ||
                  std::is_same_v<typename ForwardIterator::mode, checkout_mode::no_access_t>);

  } else if constexpr (ori::is_global_ptr_v<ForwardIterator>) {
    // automatically convert global pointers to global iterators
    auto first_ = make_global_iterator(first, checkout_mode_t{});
    auto last_  = make_global_iterator(last , checkout_mode_t{});
    fill(policy, first_, last_, value);
    return;
  }

  auto seq_policy = execution::to_sequenced_policy(policy);
  auto serial_fn = [=](ForwardIterator first_,
                       ForwardIterator last_) mutable {
    for_each(seq_policy, first_, last_, [&](auto&& d) {
      d = value;
    });
  };

  loop_generic(policy, serial_fn, []{}, first, last);
}

ITYR_TEST_CASE("[ityr::pattern::parallel_loop] reduce and transform_reduce") {
  ito::init();
  ori::init();

  ITYR_SUBCASE("default cutoff") {
    long n = 10000;
    long r = ito::root_exec([=] {
      return reduce(
          execution::par,
          ityr::count_iterator<long>(0),
          ityr::count_iterator<long>(n));
    });
    ITYR_CHECK(r == n * (n - 1) / 2);
  }

  ITYR_SUBCASE("custom cutoff") {
    long n = 100000;
    long r = ito::root_exec([=] {
      return reduce(
          execution::parallel_policy{.cutoff_count = 100},
          ityr::count_iterator<long>(0),
          ityr::count_iterator<long>(n));
    });
    ITYR_CHECK(r == n * (n - 1) / 2);
  }

  ITYR_SUBCASE("transform unary") {
    long n = 100000;
    long r = ito::root_exec([=] {
      return transform_reduce(
          execution::parallel_policy{.cutoff_count = 100},
          ityr::count_iterator<long>(0),
          ityr::count_iterator<long>(n),
          long(0),
          std::plus<long>{},
          [](long x) { return x * x; });
    });
    ITYR_CHECK(r == n * (n - 1) * (2 * n - 1) / 6);
  }

  ITYR_SUBCASE("transform binary") {
    long n = 100000;
    long r = ito::root_exec([=] {
      return transform_reduce(
          execution::parallel_policy{.cutoff_count = 100},
          ityr::count_iterator<long>(0),
          ityr::count_iterator<long>(n),
          ityr::count_iterator<long>(0),
          long(0),
          std::plus<long>{},
          [](long x, long y) { return x * y; });
    });
    ITYR_CHECK(r == n * (n - 1) * (2 * n - 1) / 6);
  }

  ITYR_SUBCASE("zero elements") {
    long r = ito::root_exec([=] {
      return reduce(
          execution::parallel_policy{.cutoff_count = 100},
          ityr::count_iterator<long>(0),
          ityr::count_iterator<long>(0),
          long(30));
    });
    ITYR_CHECK(r == 30);
  }

  ori::fini();
  ito::fini();
}

ITYR_TEST_CASE("[ityr::pattern::parallel_loop] parallel reduce with global_ptr") {
  ito::init();
  ori::init();

  long n = 100000;
  ori::global_ptr<long> p = ori::malloc_coll<long>(n);

  ito::root_exec([=] {
    long count = 0;
    for_each(
        execution::sequenced_policy{.checkout_count = 100},
        make_global_iterator(p    , checkout_mode::write),
        make_global_iterator(p + n, checkout_mode::write),
        [&](long& v) { v = count++; });
  });

  ITYR_SUBCASE("default cutoff") {
    long r = ito::root_exec([=] {
      return reduce(
          execution::par,
          p, p + n);
    });
    ITYR_CHECK(r == n * (n - 1) / 2);
  }

  ITYR_SUBCASE("custom cutoff and checkout count") {
    long r = ito::root_exec([=] {
      return reduce(
          execution::parallel_policy{.cutoff_count = 100, .checkout_count = 50},
          p, p + n);
    });
    ITYR_CHECK(r == n * (n - 1) / 2);
  }

  ITYR_SUBCASE("without auto checkout") {
    long r = ito::root_exec([=] {
      return transform_reduce(
          execution::par,
          make_global_iterator(p    , checkout_mode::no_access),
          make_global_iterator(p + n, checkout_mode::no_access),
          long(0),
          std::plus<long>{},
          [](ori::global_ref<long> gref) {
            return gref.get();
          });
    });
    ITYR_CHECK(r == n * (n - 1) / 2);
  }

  ITYR_SUBCASE("serial") {
    long r = ito::root_exec([=] {
      return reduce(
          execution::sequenced_policy{.checkout_count = 100},
          p, p + n);
    });
    ITYR_CHECK(r == n * (n - 1) / 2);
  }

  ori::free_coll(p);

  ori::fini();
  ito::fini();
}

ITYR_TEST_CASE("[ityr::pattern::parallel_loop] parallel transform with global_ptr") {
  ito::init();
  ori::init();

  long n = 100000;
  ori::global_ptr<long> p1 = ori::malloc_coll<long>(n);
  ori::global_ptr<long> p2 = ori::malloc_coll<long>(n);

  ITYR_SUBCASE("parallel") {
    ito::root_exec([=] {
      auto r = transform(
          execution::parallel_policy{.cutoff_count = 100, .checkout_count = 100},
          count_iterator<long>(0), count_iterator<long>(n), p1,
          [](long i) { return i * 2; });
      ITYR_CHECK(r == p1 + n);

      transform(
          execution::parallel_policy{.cutoff_count = 100, .checkout_count = 100},
          count_iterator<long>(0), count_iterator<long>(n), p1, p2,
          [](long i, long j) { return i * j; });

      auto sum = reduce(
          execution::parallel_policy{.cutoff_count = 100, .checkout_count = 100},
          p2, p2 + n);

      ITYR_CHECK(sum == n * (n - 1) * (2 * n - 1) / 3);
    });
  }

  ITYR_SUBCASE("serial") {
    ito::root_exec([=] {
      auto r = transform(
          execution::sequenced_policy{.checkout_count = 100},
          count_iterator<long>(0), count_iterator<long>(n), p1,
          [](long i) { return i * 2; });
      ITYR_CHECK(r == p1 + n);

      transform(
          execution::sequenced_policy{.checkout_count = 100},
          count_iterator<long>(0), count_iterator<long>(n), p1, p2,
          [](long i, long j) { return i * j; });

      auto sum = reduce(
          execution::sequenced_policy{.checkout_count = 100},
          p2, p2 + n);

      ITYR_CHECK(sum == n * (n - 1) * (2 * n - 1) / 3);
    });
  }

  ori::free_coll(p1);
  ori::free_coll(p2);

  ori::fini();
  ito::fini();
}

ITYR_TEST_CASE("[ityr::pattern::parallel_loop] parallel fill") {
  ito::init();
  ori::init();

  long n = 100000;
  ori::global_ptr<long> p = ori::malloc_coll<long>(n);

  ito::root_exec([=] {
    long val = 33;
    fill(execution::parallel_policy{.cutoff_count = 100, .checkout_count = 100},
         p, p + n, val);

    auto sum = reduce(
        execution::parallel_policy{.cutoff_count = 100, .checkout_count = 100},
        p, p + n);

    ITYR_CHECK(sum == n * val);
  });

  ori::free_coll(p);

  ori::fini();
  ito::fini();
}

}
