#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/ito/ito.hpp"
#include "ityr/ori/ori.hpp"
#include "ityr/pattern/count_iterator.hpp"
#include "ityr/pattern/global_iterator.hpp"
#include "ityr/pattern/serial_loop.hpp"

namespace ityr {

namespace internal {

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
  execution::internal::assert_policy(policy);
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
  execution::internal::assert_policy(policy);
  auto rh = ori::release_lazy();
  return parallel_loop_generic(policy, serial_fn, combine_fn, rh, first, last, firsts...);
}

}

/**
 * @brief Apply an operation to each element in a range.
 *
 * @param policy Execution policy (`ityr::execution`).
 * @param first  Begin iterator.
 * @param last   End iterator.
 * @param op     Operator for the i-th element in the range.
 *
 * This function iterates over the given range and applies the operator `op` to the i-th element.
 * The operator `op` should accept an argument of type `T`, which is the reference type of the
 * given iterator type.
 * This function resembles the standard `std::for_each()`, but it is extended to accept multiple
 * streams of iterators.
 *
 * Global pointers are not automatically checked out. If global iterators are explicitly given
 * (by `ityr::make_global_iterator`), the regions are automatically checked out with the specified
 * mode in the specified granularity (`ityr::execution::sequenced_policy::checkout_count` if serial,
 * or `ityr::execution::parallel_policy::checkout_count` if parallel).
 *
 * Example:
 * ```
 * ityr::global_vector<int> v1 = {1, 2, 3, 4, 5};
 * ityr::for_each(ityr::execution::par,
 *                ityr::make_global_iterator(v1.begin(), ityr::checkout_mode::read_write),
 *                ityr::make_global_iterator(v1.end()  , ityr::checkout_mode::read_write),
 *                [](int& x) { x++; });
 * // v1 = {2, 3, 4, 5, 6}
 * ```
 *
 * @see [std::for_each -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/for_each)
 * @see `ityr::reduce()`
 * @see `ityr::transform()`
 * @see `ityr::transform_reduce()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename ForwardIterator, typename Op>
inline void for_each(const ExecutionPolicy& policy,
                     ForwardIterator        first,
                     ForwardIterator        last,
                     Op                     op) {
  auto seq_policy = execution::internal::to_sequenced_policy(policy);
  auto serial_fn = [=](ForwardIterator first_,
                       ForwardIterator last_) mutable {
    internal::for_each_aux(seq_policy, op, first_, last_);
  };
  internal::loop_generic(policy, serial_fn, []{}, first, last);
}

/**
 * @brief Apply an operation to each element in a range.
 *
 * @param policy Execution policy (`ityr::execution`).
 * @param first1 1st begin iterator.
 * @param last1  1st end iterator.
 * @param first2 2nd begin iterator.
 * @param op     Operator for the i-th element in the 1st and 2nd iterators.
 *
 * This function iterates over the given ranges and applies the operator `op` to the i-th elements.
 * The operator `op` should accept three arguments of type `T1` and `T2`, which are the reference
 * types of the given iterators `first1` and `first2`.
 * This function resembles the standard `std::for_each()`, but it is extended to accept multiple
 * streams of iterators.
 *
 * Global pointers are not automatically checked out. If global iterators are explicitly given
 * (by `ityr::make_global_iterator`), the regions are automatically checked out with the specified
 * mode in the specified granularity (`ityr::execution::sequenced_policy::checkout_count` if serial,
 * or `ityr::execution::parallel_policy::checkout_count` if parallel).
 *
 * Example:
 * ```
 * ityr::global_vector<int> v1 = {1, 2, 3, 4, 5};
 * ityr::global_vector<int> v2 = {1, 2, 3, 4, 5};
 * ityr::for_each(ityr::execution::par,
 *                ityr::make_global_iterator(v1.begin(), ityr::checkout_mode::read),
 *                ityr::make_global_iterator(v1.end()  , ityr::checkout_mode::read),
 *                ityr::make_global_iterator(v2.begin(), ityr::checkout_mode::read_write),
 *                [](int x, int& y) { y += x; });
 * // v2 = {2, 4, 6, 8, 10}
 * ```
 *
 * @see [std::for_each -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/for_each)
 * @see `ityr::reduce()`
 * @see `ityr::transform()`
 * @see `ityr::transform_reduce()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename ForwardIterator1, typename ForwardIterator2, typename Op>
inline void for_each(const ExecutionPolicy& policy,
                     ForwardIterator1       first1,
                     ForwardIterator1       last1,
                     ForwardIterator2       first2,
                     Op                     op) {
  auto seq_policy = execution::internal::to_sequenced_policy(policy);
  auto serial_fn = [=](ForwardIterator1 first1_,
                       ForwardIterator1 last1_,
                       ForwardIterator2 first2_) mutable {
    internal::for_each_aux(seq_policy, op, first1_, last1_, first2_);
  };
  internal::loop_generic(policy, serial_fn, []{}, first1, last1, first2);
}

/**
 * @brief Apply an operation to each element in a range.
 *
 * @param policy Execution policy (`ityr::execution`).
 * @param first1 1st begin iterator.
 * @param last1  1st end iterator.
 * @param first2 2nd begin iterator.
 * @param first3 3rd begin iterator.
 * @param op     Operator for the i-th element in the 1st, 2nd, and 3rd iterators.
 *
 * This function iterates over the given ranges and applies the operator `op` to the i-th elements.
 * The operator `op` should accept three arguments of type `T1`, `T2`, and `T3`, which are the
 * reference types of the given iterators `first1`, `first2`, and `first3`.
 * This function resembles the standard `std::for_each()`, but it is extended to accept multiple
 * streams of iterators.
 *
 * Global pointers are not automatically checked out. If global iterators are explicitly given
 * (by `ityr::make_global_iterator`), the regions are automatically checked out with the specified
 * mode in the specified granularity (`ityr::execution::sequenced_policy::checkout_count` if serial,
 * or `ityr::execution::parallel_policy::checkout_count` if parallel).
 *
 * Example:
 * ```
 * ityr::global_vector<int> v1 = {1, 1, 1, 1, 1};
 * ityr::global_vector<int> v2(v1.size());
 * ityr::for_each(ityr::execution::par,
 *                ityr::count_iterator<int>(0),
 *                ityr::count_iterator<int>(5),
 *                ityr::make_global_iterator(v1.begin(), ityr::checkout_mode::read),
 *                ityr::make_global_iterator(v2.begin(), ityr::checkout_mode::write),
 *                [](int i, int x, int& y) { y = x << i; });
 * // v2 = {1, 2, 4, 8, 16}
 * ```
 *
 * @see [std::for_each -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/for_each)
 * @see `ityr::reduce()`
 * @see `ityr::transform()`
 * @see `ityr::transform_reduce()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename ForwardIterator1, typename ForwardIterator2,
          typename ForwardIterator3, typename Op>
inline void for_each(const ExecutionPolicy& policy,
                     ForwardIterator1       first1,
                     ForwardIterator1       last1,
                     ForwardIterator2       first2,
                     ForwardIterator3       first3,
                     Op                     op) {
  auto seq_policy = execution::internal::to_sequenced_policy(policy);
  auto serial_fn = [=](ForwardIterator1 first1_,
                       ForwardIterator1 last1_,
                       ForwardIterator2 first2_,
                       ForwardIterator3 first3_) mutable {
    internal::for_each_aux(seq_policy, op, first1_, last1_, first2_, first3_);
  };
  internal::loop_generic(policy, serial_fn, []{}, first1, last1, first2, first3);
}

ITYR_TEST_CASE("[ityr::pattern::serial_loop] serial for_each") {
  class move_only_t {
  public:
    move_only_t() {}
    move_only_t(const long v) : value_(v) {}

    long value() const { return value_; }

    move_only_t(const move_only_t&) = delete;
    move_only_t& operator=(const move_only_t&) = delete;

    move_only_t(move_only_t&& mo) : value_(mo.value_) {
      mo.value_ = -1;
    }
    move_only_t& operator=(move_only_t&& mo) {
      value_ = mo.value_;
      mo.value_ = -1;
      return *this;
    }

  private:
    long value_ = -1;
  };

  ori::init();

  long n = 100000;

  ITYR_SUBCASE("without global_ptr") {
    ITYR_SUBCASE("count iterator") {
      long count = 0;
      for_each(execution::seq,
               count_iterator<long>(0),
               count_iterator<long>(n),
               [&](long i) { count += i; });
      ITYR_CHECK(count == n * (n - 1) / 2);

      count = 0;
      for_each(
          execution::seq,
          count_iterator<long>(0),
          count_iterator<long>(n),
          count_iterator<long>(n),
          [&](long i, long j) { count += i + j; });
      ITYR_CHECK(count == 2 * n * (2 * n - 1) / 2);

      count = 0;
      for_each(
          execution::seq,
          count_iterator<long>(0),
          count_iterator<long>(n),
          count_iterator<long>(n),
          count_iterator<long>(2 * n),
          [&](long i, long j, long k) { count += i + j + k; });
      ITYR_CHECK(count == 3 * n * (3 * n - 1) / 2);
    }

    ITYR_SUBCASE("vector copy") {
      std::vector<long> mos1(count_iterator<long>(0),
                                    count_iterator<long>(n));

      std::vector<long> mos2;
      for_each(
          execution::seq,
          mos1.begin(), mos1.end(),
          std::back_inserter(mos2),
          [&](long i, auto&& out) { out = i; });

      long count = 0;
      for_each(
          execution::seq,
          mos2.begin(), mos2.end(),
          [&](long i) { count += i; });
      ITYR_CHECK(count == n * (n - 1) / 2);
    }

    ITYR_SUBCASE("move iterator with vector") {
      std::vector<move_only_t> mos1(count_iterator<long>(0),
                                    count_iterator<long>(n));

      std::vector<move_only_t> mos2;
      for_each(
          execution::seq,
          std::make_move_iterator(mos1.begin()),
          std::make_move_iterator(mos1.end()),
          std::back_inserter(mos2),
          [&](move_only_t&& in, auto&& out) { out = std::move(in); });

      long count = 0;
      for_each(
          execution::seq,
          mos2.begin(), mos2.end(),
          [&](move_only_t& mo) { count += mo.value(); });
      ITYR_CHECK(count == n * (n - 1) / 2);

      for_each(
          execution::seq,
          mos1.begin(), mos1.end(),
          [&](move_only_t& mo) { ITYR_CHECK(mo.value() == -1); });
    }
  }

  ITYR_SUBCASE("with global_ptr") {
    ori::global_ptr<long> gp = ori::malloc<long>(n);

    for_each(
        execution::seq,
        count_iterator<long>(0),
        count_iterator<long>(n),
        make_global_iterator(gp, checkout_mode::write),
        [&](long i, long& out) { new (&out) long(i); });

    ITYR_SUBCASE("read array without global_iterator") {
      long count = 0;
      for_each(
          execution::seq,
          gp,
          gp + n,
          [&](ori::global_ref<long> gr) { count += gr; });
      ITYR_CHECK(count == n * (n - 1) / 2);
    }

    ITYR_SUBCASE("read array with global_iterator") {
      long count = 0;
      for_each(
          execution::seq,
          make_global_iterator(gp    , checkout_mode::read),
          make_global_iterator(gp + n, checkout_mode::read),
          [&](long i) { count += i; });
      ITYR_CHECK(count == n * (n - 1) / 2);
    }

    ITYR_SUBCASE("move iterator") {
      ori::global_ptr<move_only_t> mos1 = ori::malloc<move_only_t>(n);
      ori::global_ptr<move_only_t> mos2 = ori::malloc<move_only_t>(n);

      for_each(
          execution::seq,
          make_global_iterator(gp    , checkout_mode::read),
          make_global_iterator(gp + n, checkout_mode::read),
          make_global_iterator(mos1  , checkout_mode::write),
          [&](long i, move_only_t& out) { new (&out) move_only_t(i); });

      for_each(
          execution::seq,
          make_move_iterator(mos1),
          make_move_iterator(mos1 + n),
          make_global_iterator(mos2, checkout_mode::write),
          [&](move_only_t&& in, move_only_t& out) { new (&out) move_only_t(std::move(in)); });

      long count = 0;
      for_each(
          execution::seq,
          make_global_iterator(mos2    , checkout_mode::read),
          make_global_iterator(mos2 + n, checkout_mode::read),
          [&](const move_only_t& mo) { count += mo.value(); });
      ITYR_CHECK(count == n * (n - 1) / 2);

      for_each(
          execution::seq,
          make_global_iterator(mos1    , checkout_mode::read),
          make_global_iterator(mos1 + n, checkout_mode::read),
          [&](const move_only_t& mo) { ITYR_CHECK(mo.value() == -1); });

      ori::free(mos1, n);
      ori::free(mos2, n);
    }

    ori::free(gp, n);
  }

  ori::fini();
}

ITYR_TEST_CASE("[ityr::pattern::parallel_loop] parallel for_each") {
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

/**
 * @brief Calculate reduction by transforming each element.
 *
 * @param policy             Execution policy (`ityr::execution`).
 * @param first              Begin iterator.
 * @param last               End iterator.
 * @param identity           Identity element.
 * @param binary_reduce_op   Associative binary operator.
 * @param unary_transform_op Unary operator to transform each element.
 *
 * @return The reduced result.
 *
 * This function applies `unary_transform_op` to each element in the range `[first, last)` and
 * performs reduction over them.
 *
 * If global pointers are provided as iterators, they are automatically checked out with the read-only
 * mode in the specified granularity (`ityr::execution::sequenced_policy::checkout_count` if serial,
 * or `ityr::execution::parallel_policy::checkout_count` if parallel) without explicitly passing them
 * as global iterators.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v1 = {1, 2, 3, 4, 5};
 * int r = ityr::transform_reduce(ityr::execution::par, v1.begin(), v1.end(), 0, std::plus<>{},
 *                                [](int x) { return x * x; });
 * // r = 55
 * ```
 *
 * @see [std::transform_reduce -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/transform_reduce)
 * @see `ityr::reduce()`
 * @see `ityr::transform()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename ForwardIterator, typename T,
          typename BinaryReduceOp, typename UnaryTransformOp>
inline T transform_reduce(const ExecutionPolicy& policy,
                          ForwardIterator        first,
                          ForwardIterator        last,
                          T                      identity,
                          BinaryReduceOp         binary_reduce_op,
                          UnaryTransformOp       unary_transform_op) {
  using it_ref = typename std::iterator_traits<ForwardIterator>::reference;
  using transformed_t = std::invoke_result_t<UnaryTransformOp, it_ref>;
  static_assert(std::is_invocable_r_v<T, BinaryReduceOp, T&, transformed_t>);
  static_assert(std::is_invocable_r_v<T, BinaryReduceOp, transformed_t, T&>);
  static_assert(std::is_invocable_r_v<T, BinaryReduceOp, T&, T&>);
  static_assert(std::is_invocable_r_v<T, BinaryReduceOp, transformed_t, transformed_t>);

  if constexpr (is_global_iterator_v<ForwardIterator>) {
    static_assert(std::is_same_v<typename ForwardIterator::mode, checkout_mode::read_t> ||
                  std::is_same_v<typename ForwardIterator::mode, checkout_mode::no_access_t>);

  } else if constexpr (ori::is_global_ptr_v<ForwardIterator>) {
    // automatically convert global pointers to global iterators with read-only access
    auto first_ = make_global_iterator(first, checkout_mode::read);
    auto last_  = make_global_iterator(last , checkout_mode::read);
    return transform_reduce(policy, first_, last_, identity, binary_reduce_op, unary_transform_op);
  }

  auto seq_policy = execution::internal::to_sequenced_policy(policy);
  auto serial_fn = [=](ForwardIterator first_,
                       ForwardIterator last_) mutable {
    T acc = identity;
    internal::for_each_aux(seq_policy, [&](const auto& v) {
      acc = binary_reduce_op(acc, unary_transform_op(v));
    }, first_, last_);
    return acc;
  };

  return internal::loop_generic(policy, serial_fn, binary_reduce_op, first, last);
}

/**
 * @brief Calculate reduction by transforming each element.
 *
 * @param policy              Execution policy (`ityr::execution`).
 * @param first1              1st begin iterator.
 * @param last1               1st end iterator.
 * @param first2              2nd begin iterator.
 * @param identity            Identity element.
 * @param binary_reduce_op    Associative binary operator.
 * @param binary_transform_op Binary operator to transform a pair of each element.
 *
 * @return The reduced result.
 *
 * This function applies `binary_transform_op` to a pair of each element in the range `[first1, last1)`
 * and `[first2, first2 + (last1 - first1)]` and performs reduction over them.
 *
 * If global pointers are provided as iterators, they are automatically checked out with the read-only
 * mode in the specified granularity (`ityr::execution::sequenced_policy::checkout_count` if serial,
 * or `ityr::execution::parallel_policy::checkout_count` if parallel) without explicitly passing them
 * as global iterators.
 * The specified regions can be overlapped.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v1 = {1, 2, 3, 4, 5};
 * bool r = ityr::transform_reduce(ityr::execution::par, v1.begin(), v1.end() - 1, v1.begin() + 1,
 *                                 true, std::logical_and<>{}, [](int x, int y) { return x <= y; });
 * // r = true (v1 is sorted)
 * ```
 *
 * @see [std::transform_reduce -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/transform_reduce)
 * @see `ityr::reduce()`
 * @see `ityr::transform()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename ForwardIterator1, typename ForwardIterator2, typename T,
          typename BinaryReduceOp, typename BinaryTransformOp>
inline T transform_reduce(const ExecutionPolicy& policy,
                          ForwardIterator1       first1,
                          ForwardIterator1       last1,
                          ForwardIterator2       first2,
                          T                      identity,
                          BinaryReduceOp         binary_reduce_op,
                          BinaryTransformOp      binary_transform_op) {
  using it1_ref = typename std::iterator_traits<ForwardIterator1>::reference;
  using it2_ref = typename std::iterator_traits<ForwardIterator2>::reference;
  using transformed_t = std::invoke_result_t<BinaryTransformOp, it1_ref, it2_ref>;
  static_assert(std::is_invocable_r_v<T, BinaryReduceOp, T&, transformed_t>);
  static_assert(std::is_invocable_r_v<T, BinaryReduceOp, transformed_t, T&>);
  static_assert(std::is_invocable_r_v<T, BinaryReduceOp, T&, T&>);
  static_assert(std::is_invocable_r_v<T, BinaryReduceOp, transformed_t, transformed_t>);

  if constexpr (is_global_iterator_v<ForwardIterator1>) {
    static_assert(std::is_same_v<typename ForwardIterator1::mode, checkout_mode::read_t> ||
                  std::is_same_v<typename ForwardIterator1::mode, checkout_mode::no_access_t>);

  } else if constexpr (ori::is_global_ptr_v<ForwardIterator1>) {
    // automatically convert global pointers to global iterators with read-only access
    auto first1_ = make_global_iterator(first1, checkout_mode::read);
    auto last1_  = make_global_iterator(last1 , checkout_mode::read);
    return transform_reduce(policy, first1_, last1_, first2, identity, binary_reduce_op, binary_transform_op);
  }

  if constexpr (is_global_iterator_v<ForwardIterator2>) {
    static_assert(std::is_same_v<typename ForwardIterator2::mode, checkout_mode::read_t> ||
                  std::is_same_v<typename ForwardIterator2::mode, checkout_mode::no_access_t>);

  } else if constexpr (ori::is_global_ptr_v<ForwardIterator2>) {
    // automatically convert global pointers to global iterators with read-only access
    auto first2_ = make_global_iterator(first2, checkout_mode::read);
    return transform_reduce(policy, first1, last1, first2_, identity, binary_reduce_op, binary_transform_op);
  }

  auto seq_policy = execution::internal::to_sequenced_policy(policy);
  auto serial_fn = [=](ForwardIterator1 first1_,
                       ForwardIterator1 last1_,
                       ForwardIterator2 first2_) mutable {
    T acc = identity;
    internal::for_each_aux(seq_policy, [&](const auto& v1, const auto& v2) {
      acc = binary_reduce_op(acc, binary_transform_op(v1, v2));
    }, first1_, last1_, first2_);
    return acc;
  };

  return internal::loop_generic(policy, serial_fn, binary_reduce_op, first1, last1, first2);
}

/**
 * @brief Calculate a dot product.
 *
 * @param policy   Execution policy (`ityr::execution`).
 * @param first1   1st begin iterator.
 * @param last1    1st end iterator.
 * @param first2   2nd begin iterator.
 * @param identity Identity element.
 *
 * @return The reduced result.
 *
 * Equivalent to `ityr::transform_reduce(policy, first1, last1, first2, identity, std::plus<>{}, std::multiplies<>{})`,
 * which corresponds to calculating a dot product of two vectors.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v1 = {1, 2, 3, 4, 5};
 * ityr::global_vector<int> v2 = {2, 3, 4, 5, 6};
 * int dot = ityr::transform_reduce(ityr::execution::par, v1.begin(), v1.end(), v2.begin(), 0);
 * // dot = 70
 * ```
 *
 * @see [std::transform_reduce -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/transform_reduce)
 * @see `ityr::reduce()`
 * @see `ityr::transform()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename ForwardIterator1, typename ForwardIterator2, typename T>
inline T transform_reduce(const ExecutionPolicy& policy,
                          ForwardIterator1       first1,
                          ForwardIterator1       last1,
                          ForwardIterator2       first2,
                          T                      identity) {
  return transform_reduce(policy, first1, last1, first2, identity, std::plus<>{}, std::multiplies<>{});
}

/**
 * @brief Calculate reduction.
 *
 * @param policy           Execution policy (`ityr::execution`).
 * @param first            Begin iterator.
 * @param last             End iterator.
 * @param identity         Identity element.
 * @param binary_reduce_op Associative binary operator.
 *
 * @return The reduced result.
 *
 * This function performs reduction over the elements in the given range `[first, last)`.
 *
 * If global pointers are provided as iterators, they are automatically checked out with the read-only
 * mode in the specified granularity (`ityr::execution::sequenced_policy::checkout_count` if serial,
 * or `ityr::execution::parallel_policy::checkout_count` if parallel) without explicitly passing them
 * as global iterators.
 *
 * Itoyori's reduce operation resembles the standard `std::reduce`, but it differs from the standard
 * one in the following two respects:
 *
 * - `ityr::reduce` does not require that `binary_reduce_op` be *commutative*.
 *   That is, only *associativity* is required for `binary_reduce_op`. Specifically, it must satisfy
 *   `binary_reduce_op(x, binary_reduce_op(y, z)) == binary_reduce_op(binary_reduce_op(x, y), z)`.
 * - `ityr::reduce` receives an identity element (`identity`), while `std::reduce` an initial element
 *   (`init`). In `std::reduce`, `init` is accumulated only once, while in `ityr::reduce`, `identity`
 *   can be accumulated multiple times. Therefore, the user must provide an identity element that
 *   satisfies both `binary_reduce_op(identity, x) == x` and `binary_reduce_op(x, identity) == x`.
 *
 * This means that `ityr::reduce` requires a *monoid*, which consists of an identity element and an
 * associative binary operator.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v = {1, 2, 3, 4, 5};
 * int product = ityr::reduce(ityr::execution::par, v.begin(), v.end(), 1, std::multiplies<>{});
 * // product = 120
 * ```
 *
 * @see [std::reduce -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/reduce)
 * @see `ityr::transform_reduce()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename ForwardIterator, typename T, typename BinaryReduceOp>
inline T reduce(const ExecutionPolicy& policy,
                ForwardIterator        first,
                ForwardIterator        last,
                T                      identity,
                BinaryReduceOp         binary_reduce_op) {
  using it_ref = typename std::iterator_traits<ForwardIterator>::reference;
  static_assert(std::is_invocable_r_v<T, BinaryReduceOp, T&, it_ref>);
  static_assert(std::is_invocable_r_v<T, BinaryReduceOp, it_ref, T&>);
  static_assert(std::is_invocable_r_v<T, BinaryReduceOp, T&, T&>);
  static_assert(std::is_invocable_r_v<T, BinaryReduceOp, it_ref, it_ref>);

  return transform_reduce(policy, first, last, identity, binary_reduce_op,
                          [](auto&& v) { return std::forward<decltype(v)>(v); });
}

/**
 * @brief Calculate reduction.
 *
 * @param policy   Execution policy (`ityr::execution`).
 * @param first    Begin iterator.
 * @param last     End iterator.
 * @param identity Identity element.
 *
 * @return The reduced result.
 *
 * Equivalent to `ityr::reduce(policy, first, last, identity, std::plus<>{})`.
 */
template <typename ExecutionPolicy, typename ForwardIterator, typename T>
inline T reduce(const ExecutionPolicy& policy,
                ForwardIterator        first,
                ForwardIterator        last,
                T                      identity) {
  return reduce(policy, first, last, identity, std::plus<>{});
}

/**
 * @brief Calculate reduction.
 *
 * @param policy Execution policy (`ityr::execution`).
 * @param first  Begin iterator.
 * @param last   End iterator.
 *
 * @return The reduced result.
 *
 * Equivalent to `ityr::reduce(policy, first, last, T{}, std::plus<>{})`, where type `T` is
 * the value type of given iterators (`ForwardIterator`).
 */
template <typename ExecutionPolicy, typename ForwardIterator>
inline typename std::iterator_traits<ForwardIterator>::value_type
reduce(const ExecutionPolicy& policy,
       ForwardIterator        first,
       ForwardIterator        last) {
  using value_type = typename std::iterator_traits<ForwardIterator>::value_type;
  return reduce(policy, first, last, value_type{});
}

/**
 * @brief Apply an operator to each element in a range.
 *
 * @param policy   Execution policy (`ityr::execution`).
 * @param first1   Input begin iterator.
 * @param last1    Input end iterator.
 * @param first_d  Output begin iterator.
 * @param unary_op Unary operator to transform each element.
 *
 * @return The end iterator of the output range (`first_d + (last1 - first1)`).
 *
 * This function applies `unary_op` to each element in the range `[first1, last1)` and stores the
 * result to the output range `[first_d, first_d + (last1 - first1))`. The element order is preserved,
 * and this operation corresponds to the higher-order *map* function.
 * This function resembles the standard `std::transform`.
 *
 * If the input iterators (`first1` and `last1`) are global pointers, they are automatically checked
 * out with the read-only mode in the specified granularity
 * (`ityr::execution::sequenced_policy::checkout_count` if serial,
 * or `ityr::execution::parallel_policy::checkout_count` if parallel) without explicitly passing them
 * as global iterators.
 * Similarly, if the output iterator (`first_d`) is a global pointer, the output region is checked
 * out with the write-only mode if the output type is *trivially copyable*; otherwise, it is checked
 * out with the read-write mode.
 * Overlapping regions can be specified for `first1` and `first_d`, as long as no data race occurs.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v1 = {1, 2, 3, 4, 5};
 * ityr::global_vector<int> v2(v1.size());
 * ityr::transform(ityr::execution::par, v1.begin(), v1.end(), v2.begin(),
 *                 [](int x) { return x * x; });
 * // v2 = {1, 4, 9, 16, 25}
 * ```
 *
 * @see [std::transform -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/transform)
 * @see `ityr::transform_reduce()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
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

  auto seq_policy = execution::internal::to_sequenced_policy(policy);
  auto serial_fn = [=](ForwardIterator1 first1_,
                       ForwardIterator1 last1_,
                       ForwardIteratorD first_d_) mutable {
    internal::for_each_aux(seq_policy, [&](const auto& v1, auto&& d) {
      d = unary_op(v1);
    }, first1_, last1_, first_d_);
  };

  internal::loop_generic(policy, serial_fn, []{}, first1, last1, first_d);

  return std::next(first_d, std::distance(first1, last1));
}

/**
 * @brief Apply an operator to each element in a range.
 *
 * @param policy    Execution policy (`ityr::execution`).
 * @param first1    1st input begin iterator.
 * @param last1     1st input end iterator.
 * @param first2    2nd input begin iterator.
 * @param first_d   Output begin iterator (output).
 * @param binary_op Binary operator to transform a pair of each element.
 *
 * @return The end iterator of the output range (`first_d + (last1 - first1)`).
 *
 * This function applies `binary_op` to a pair of each element in the range `[first1, last1)` and
 * `[first2, (last1 - first1))`, and the result is stored to the output range
 * `[first_d, first_d + (last1 - first1))`. The element order is preserved,
 * and this operation corresponds to the higher-order *map* function.
 * This function resembles the standard `std::transform`.
 *
 * If the input iterators (`first1`, `last1`, and `first2`) are global pointers, they are
 * automatically checked out with the read-only mode in the specified granularity
 * (`ityr::execution::sequenced_policy::checkout_count` if serial,
 * or `ityr::execution::parallel_policy::checkout_count` if parallel) without explicitly passing them
 * as global iterators.
 * Similarly, if the output iterator (`first_d`) is a global pointer, the output region is checked
 * out with the write-only mode if the output type is *trivially copyable*; otherwise, it is checked
 * out with the read-write mode.
 * Overlapping regions can be specified for `first1`, `first2`, and `first_d`, as long as no data race occurs.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v1 = {1, 2, 3, 4, 5};
 * ityr::global_vector<int> v2 = {2, 3, 4, 5, 6};
 * ityr::global_vector<int> v3(v1.size());
 * ityr::transform(ityr::execution::par, v1.begin(), v1.end(), v2.begin(), v3.begin(),
 *                 [](int x, int y) { return x * y; });
 * // v3 = {2, 6, 12, 20, 30}
 * ```
 *
 * @see [std::transform -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/transform)
 * @see `ityr::transform_reduce()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
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

  auto seq_policy = execution::internal::to_sequenced_policy(policy);
  auto serial_fn = [=](ForwardIterator1 first1_,
                       ForwardIterator1 last1_,
                       ForwardIterator2 first2_,
                       ForwardIteratorD first_d_) mutable {
    internal::for_each_aux(seq_policy, [&](const auto& v1, const auto& v2, auto&& d) {
      d = binary_op(v1, v2);
    }, first1_, last1_, first2_, first_d_);
  };

  internal::loop_generic(policy, serial_fn, []{}, first1, last1, first2, first_d);

  return std::next(first_d, std::distance(first1, last1));
}

/**
 * @brief Fill a range with a given value.
 *
 * @param policy Execution policy (`ityr::execution`).
 * @param first  Begin iterator.
 * @param last   End iterator.
 * @param value  Value to be filled with.
 *
 * This function assigns `value` to every element in the given range `[first, last)`.
 *
 * If global pointers are provided as iterators, they are automatically checked out with the read-only
 * mode in the specified granularity (`ityr::execution::sequenced_policy::checkout_count` if serial,
 * or `ityr::execution::parallel_policy::checkout_count` if parallel) without explicitly passing them
 * as global iterators.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v(5);
 * ityr::fill(ityr::execution::par, v.begin(), v.end(), 100);
 * // v = {100, 100, 100, 100, 100}
 * ```
 *
 * @see [std::reduce -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/reduce)
 * @see `ityr::transform()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
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

  auto seq_policy = execution::internal::to_sequenced_policy(policy);
  auto serial_fn = [=](ForwardIterator first_,
                       ForwardIterator last_) mutable {
    internal::for_each_aux(seq_policy, [&](auto&& d) {
      d = value;
    }, first_, last_);
  };

  internal::loop_generic(policy, serial_fn, []{}, first, last);
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
