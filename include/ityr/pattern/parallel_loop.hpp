#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/ito/ito.hpp"
#include "ityr/ori/ori.hpp"
#include "ityr/pattern/count_iterator.hpp"
#include "ityr/pattern/global_iterator.hpp"
#include "ityr/pattern/serial_loop.hpp"
#include "ityr/pattern/parallel_invoke.hpp"

namespace ityr {

namespace internal {

template <typename WorkHint, typename Op, typename ReleaseHandler,
          typename ForwardIterator, typename... ForwardIterators>
inline void parallel_loop_generic(const execution::parallel_policy<WorkHint>& policy,
                                  Op                                          op,
                                  ReleaseHandler                              rh,
                                  ForwardIterator                             first,
                                  ForwardIterator                             last,
                                  ForwardIterators...                         firsts) {
  ori::poll();

  // for immediately executing cross-worker tasks in ADWS
  ito::poll([] { return ori::release_lazy(); },
            [&](ori::release_handler rh_) { ori::acquire(rh); ori::acquire(rh_); });

  std::size_t d = std::distance(first, last);
  if (d <= policy.cutoff_count) {
    for_each_aux(
        execution::internal::to_sequenced_policy(policy),
        [&](auto&&... refs) {
          op(std::forward<decltype(refs)>(refs)...);
        },
        first, last, firsts...);
    return;
  }

  auto mid = std::next(first, d / 2);

  auto tgdata = ito::task_group_begin();

  ito::thread<void> th(
      ito::with_callback, [=] { ori::acquire(rh); }, [] { ori::release(); },
      ito::with_workhint, 1, 1,
      [=] {
        parallel_loop_generic(policy, op, rh,
                              first, mid, firsts...);
      });

  parallel_loop_generic(policy, op, rh,
                        mid, last, std::next(firsts, d / 2)...);

  if (!th.serialized()) {
    ori::release();
  }

  th.join();

  ito::task_group_end(tgdata, [] { ori::release(); }, [] { ori::acquire(); });

  // TODO: needed?
  if (!th.serialized()) {
    ori::acquire();
  }
}

template <typename Op, typename ForwardIterator, typename... ForwardIterators>
inline void loop_generic(const execution::sequenced_policy& policy,
                         Op                                 op,
                         ForwardIterator                    first,
                         ForwardIterator                    last,
                         ForwardIterators...                firsts) {
  execution::internal::assert_policy(policy);
  for_each_aux(
      execution::internal::to_sequenced_policy(policy),
      [&](auto&&... refs) {
        op(std::forward<decltype(refs)>(refs)...);
      },
      first, last, firsts...);
}

template <typename WorkHint, typename Op, typename ForwardIterator, typename... ForwardIterators>
inline void loop_generic(const execution::parallel_policy<WorkHint>& policy,
                         Op                                          op,
                         ForwardIterator                             first,
                         ForwardIterator                             last,
                         ForwardIterators...                         firsts) {
  execution::internal::assert_policy(policy);
  auto rh = ori::release_lazy();
  parallel_loop_generic(policy, op, rh, first, last, firsts...);
}

}

/**
 * @brief Apply an operator to each element in a range.
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
  internal::loop_generic(policy, op, first, last);
}

/**
 * @brief Apply an operator to each element in a range.
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
  internal::loop_generic(policy, op, first1, last1, first2);
}

/**
 * @brief Apply an operator to each element in a range.
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
  internal::loop_generic(policy, op, first1, last1, first2, first3);
}

ITYR_TEST_CASE("[ityr::pattern::serial_loop] serial for_each") {
  ori::init();

  long n = 100000;

  ITYR_SUBCASE("without global_ptr") {
    ITYR_SUBCASE("count iterator") {
      long count = 0;
      for_each(
          execution::seq,
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
      std::vector<common::move_only_t> mos1(count_iterator<long>(0),
                                            count_iterator<long>(n));

      std::vector<common::move_only_t> mos2;
      for_each(
          execution::seq,
          std::make_move_iterator(mos1.begin()),
          std::make_move_iterator(mos1.end()),
          std::back_inserter(mos2),
          [&](common::move_only_t&& in, auto&& out) { out = std::move(in); });

      long count = 0;
      for_each(
          execution::seq,
          mos2.begin(), mos2.end(),
          [&](common::move_only_t& mo) { count += mo.value(); });
      ITYR_CHECK(count == n * (n - 1) / 2);

      for_each(
          execution::seq,
          mos1.begin(), mos1.end(),
          [&](common::move_only_t& mo) { ITYR_CHECK(mo.value() == -1); });
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
      ori::global_ptr<common::move_only_t> mos1 = ori::malloc<common::move_only_t>(n);
      ori::global_ptr<common::move_only_t> mos2 = ori::malloc<common::move_only_t>(n);

      for_each(
          execution::seq,
          make_global_iterator(gp    , checkout_mode::read),
          make_global_iterator(gp + n, checkout_mode::read),
          make_global_iterator(mos1  , checkout_mode::write),
          [&](long i, common::move_only_t& out) { new (&out) common::move_only_t(i); });

      for_each(
          execution::seq,
          make_move_iterator(mos1),
          make_move_iterator(mos1 + n),
          make_global_iterator(mos2, checkout_mode::write),
          [&](common::move_only_t&& in, common::move_only_t& out) { new (&out) common::move_only_t(std::move(in)); });

      long count = 0;
      for_each(
          execution::seq,
          make_global_iterator(mos2    , checkout_mode::read),
          make_global_iterator(mos2 + n, checkout_mode::read),
          [&](const common::move_only_t& mo) { count += mo.value(); });
      ITYR_CHECK(count == n * (n - 1) / 2);

      for_each(
          execution::seq,
          make_global_iterator(mos1    , checkout_mode::read),
          make_global_iterator(mos1 + n, checkout_mode::read),
          [&](const common::move_only_t& mo) { ITYR_CHECK(mo.value() == -1); });

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
        execution::sequenced_policy(100),
        make_global_iterator(p1    , checkout_mode::write),
        make_global_iterator(p1 + n, checkout_mode::write),
        [&](int& v) { v = count++; });

    for_each(
        execution::par,
        make_global_iterator(p1    , checkout_mode::read),
        make_global_iterator(p1 + n, checkout_mode::read),
        count_iterator<int>(0),
        [=](int x, int i) { ITYR_CHECK(x == i); });

    for_each(
        execution::par,
        count_iterator<int>(0),
        count_iterator<int>(n),
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
        count_iterator<int>(0),
        count_iterator<int>(n),
        make_global_iterator(p2, checkout_mode::read),
        [=](int i, int y) { ITYR_CHECK(y == i * 4); });
  });

  ori::free_coll(p1);
  ori::free_coll(p2);

  ori::fini();
  ito::fini();
}

/**
 * @brief Transform elements in a given range and store them in another range.
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
 * If given iterators are global pointers, they are automatically checked out in the specified
 * granularity (`ityr::execution::sequenced_policy::checkout_count` if serial,
 * or `ityr::execution::parallel_policy::checkout_count` if parallel) without explicitly passing them
 * as global iterators.
 * Input global pointers (`first1` and `last1`) are automatically checked out with the read-only mode.
 * Similarly, output global iterator (`first_d`) are checked out with the write-only mode if their
 * value type is *trivially copyable*; otherwise, they are checked out with the read-write mode.
 *
 * Overlapping regions can be specified for the input and output ranges.
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
  if constexpr (ori::is_global_ptr_v<ForwardIterator1> ||
                ori::is_global_ptr_v<ForwardIteratorD>) {
    using value_type_d = typename std::iterator_traits<ForwardIteratorD>::value_type;
    return transform(
        policy,
        internal::convert_to_global_iterator(first1 , checkout_mode::read),
        internal::convert_to_global_iterator(last1  , checkout_mode::read),
        internal::convert_to_global_iterator(first_d, internal::dest_checkout_mode_t<value_type_d>{}),
        unary_op);

  } else {
    auto op = [=](const auto& r1, auto&& d) {
      d = unary_op(r1);
    };

    internal::loop_generic(policy, op, first1, last1, first_d);

    return std::next(first_d, std::distance(first1, last1));
  }
}

/**
 * @brief Transform elements in given ranges and store them in another range.
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
 * If given iterators are global pointers, they are automatically checked out in the specified
 * granularity (`ityr::execution::sequenced_policy::checkout_count` if serial,
 * or `ityr::execution::parallel_policy::checkout_count` if parallel) without explicitly passing them
 * as global iterators.
 * Input global pointers (`first1`, `last1`, and `first2`) are automatically checked out with the
 * read-only mode.
 * Similarly, output global iterator (`first_d`) are checked out with the write-only mode if their
 * value type is *trivially copyable*; otherwise, they are checked out with the read-write mode.
 *
 * Overlapping regions can be specified for the input and output ranges.
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
  if constexpr (ori::is_global_ptr_v<ForwardIterator1> ||
                ori::is_global_ptr_v<ForwardIterator2> ||
                ori::is_global_ptr_v<ForwardIteratorD>) {
    using value_type_d = typename std::iterator_traits<ForwardIteratorD>::value_type;
    return transform(
        policy,
        internal::convert_to_global_iterator(first1 , checkout_mode::read),
        internal::convert_to_global_iterator(last1  , checkout_mode::read),
        internal::convert_to_global_iterator(first2 , checkout_mode::read),
        internal::convert_to_global_iterator(first_d, internal::dest_checkout_mode_t<value_type_d>{}),
        binary_op);

  } else {
    auto op = [=](const auto& r1, const auto& r2, auto&& d) {
      d = binary_op(r1, r2);
    };

    internal::loop_generic(policy, op, first1, last1, first2, first_d);

    return std::next(first_d, std::distance(first1, last1));
  }
}

ITYR_TEST_CASE("[ityr::pattern::parallel_loop] transform") {
  ito::init();
  ori::init();

  long n = 100000;
  ori::global_ptr<long> p1 = ori::malloc_coll<long>(n);
  ori::global_ptr<long> p2 = ori::malloc_coll<long>(n);

  ITYR_SUBCASE("parallel") {
    ito::root_exec([=] {
      auto r = transform(
          execution::parallel_policy(100),
          count_iterator<long>(0), count_iterator<long>(n), p1,
          [](long i) { return i * 2; });
      ITYR_CHECK(r == p1 + n);

      transform(
          execution::parallel_policy(100),
          count_iterator<long>(0), count_iterator<long>(n), p1, p2,
          [](long i, long j) { return i * j; });

      for_each(
          execution::parallel_policy(100),
          make_global_iterator(p2    , checkout_mode::read),
          make_global_iterator(p2 + n, checkout_mode::read),
          count_iterator<long>(0),
          [=](long v, long i) { ITYR_CHECK(v == i * i * 2); });
    });
  }

  ITYR_SUBCASE("serial") {
    ito::root_exec([=] {
      auto ret = transform(
          execution::sequenced_policy(100),
          count_iterator<long>(0), count_iterator<long>(n), p1,
          [](long i) { return i * 2; });
      ITYR_CHECK(ret == p1 + n);

      transform(
          execution::sequenced_policy(100),
          count_iterator<long>(0), count_iterator<long>(n), p1, p2,
          [](long i, long j) { return i * j; });

      for_each(
          execution::sequenced_policy(100),
          make_global_iterator(p2    , checkout_mode::read),
          make_global_iterator(p2 + n, checkout_mode::read),
          count_iterator<long>(0),
          [=](long v, long i) { ITYR_CHECK(v == i * i * 2); });
    });
  }

  ori::free_coll(p1);
  ori::free_coll(p2);

  ori::fini();
  ito::fini();
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
  if constexpr (ori::is_global_ptr_v<ForwardIterator>) {
    using value_type = typename std::iterator_traits<ForwardIterator>::value_type;
    fill(
        policy,
        internal::convert_to_global_iterator(first, internal::dest_checkout_mode_t<value_type>{}),
        internal::convert_to_global_iterator(last , internal::dest_checkout_mode_t<value_type>{}),
        value);

  } else {
    auto op = [=](auto&& d) {
      d = value;
    };

    internal::loop_generic(policy, op, first, last);
  }
}

ITYR_TEST_CASE("[ityr::pattern::parallel_loop] fill") {
  ito::init();
  ori::init();

  long n = 100000;
  ori::global_ptr<long> p = ori::malloc_coll<long>(n);

  ito::root_exec([=] {
    long val = 33;
    fill(execution::parallel_policy(100), p, p + n, val);

    for_each(
        execution::parallel_policy(100),
        make_global_iterator(p    , checkout_mode::read),
        make_global_iterator(p + n, checkout_mode::read),
        [=](long v) { ITYR_CHECK(v == val); });
  });

  ori::free_coll(p);

  ori::fini();
  ito::fini();
}

/**
 * @brief Copy a range to another.
 *
 * @param policy  Execution policy (`ityr::execution`).
 * @param first1  Input begin iterator.
 * @param last1   Input end iterator.
 * @param first_d Output begin iterator.
 *
 * @return The end iterator of the output range (`first_d + (last1 - first1)`).
 *
 * This function copies the input region `[first1, last1)` to the output region
 * `[first_d, first_d + (last1 - first1))`.
 * Copy semantics is applied to each element in the ranges.
 *
 * If given iterators are global pointers, they are automatically checked out in the specified
 * granularity (`ityr::execution::sequenced_policy::checkout_count` if serial,
 * or `ityr::execution::parallel_policy::checkout_count` if parallel) without explicitly passing them
 * as global iterators.
 * Input global pointers (`first1` and `last1`) are automatically checked out with the read-only mode.
 * if their value type is *trivially copyable*; otherwise, they are checked out with the read-write
 * mode, even if they are actually not modified.
 * Similarly, output global iterator (`first_d`) are checked out with the write-only mode if their
 * value type is *trivially copyable*; otherwise, they are checked out with the read-write mode.
 *
 * The input and output regions should not be overlapped.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v1 = {1, 2, 3, 4, 5};
 * ityr::global_vector<int> v2(v1.size());
 * ityr::copy(ityr::execution::par, v1.begin(), v1.end(), v2.begin());
 * // v2 = {1, 2, 3, 4, 5}
 * ```
 *
 * @see [std::copy -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/copy)
 * @see `ityr::move`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename ForwardIterator1, typename ForwardIteratorD>
inline ForwardIteratorD copy(const ExecutionPolicy& policy,
                             ForwardIterator1       first1,
                             ForwardIterator1       last1,
                             ForwardIteratorD       first_d) {
  if constexpr (ori::is_global_ptr_v<ForwardIterator1> ||
                ori::is_global_ptr_v<ForwardIteratorD>) {
    using value_type1  = typename std::iterator_traits<ForwardIterator1>::value_type;
    using value_type_d = typename std::iterator_traits<ForwardIteratorD>::value_type;
    return copy(
        policy,
        internal::convert_to_global_iterator(first1 , internal::src_checkout_mode_t<value_type1>{}),
        internal::convert_to_global_iterator(last1  , internal::src_checkout_mode_t<value_type1>{}),
        internal::convert_to_global_iterator(first_d, internal::dest_checkout_mode_t<value_type_d>{}));

  } else {
    auto op = [=](auto&& r1, auto&& d) {
      d = std::forward<decltype(r1)>(r1);
    };

    internal::loop_generic(policy, op, first1, last1, first_d);

    return std::next(first_d, std::distance(first1, last1));
  }
}

ITYR_TEST_CASE("[ityr::pattern::parallel_loop] copy") {
  ito::init();
  ori::init();

  long n = 100000;
  ori::global_ptr<long> p1 = ori::malloc_coll<long>(n);
  ori::global_ptr<long> p2 = ori::malloc_coll<long>(n);

  ito::root_exec([=] {
    for_each(
        execution::parallel_policy(100),
        make_global_iterator(p1    , checkout_mode::write),
        make_global_iterator(p1 + n, checkout_mode::write),
        count_iterator<long>(0),
        [=](long& v, long i) { v = i * 2; });

    copy(execution::parallel_policy(100), p1, p1 + n, p2);

    for_each(
        execution::parallel_policy(100),
        make_global_iterator(p2    , checkout_mode::read),
        make_global_iterator(p2 + n, checkout_mode::read),
        count_iterator<long>(0),
        [=](long v, long i) { ITYR_CHECK(v == i * 2); });
  });

  ori::free_coll(p1);
  ori::free_coll(p2);

  ori::fini();
  ito::fini();
}

/**
 * @brief Move a range to another.
 *
 * @param policy  Execution policy (`ityr::execution`).
 * @param first1  Input begin iterator.
 * @param last1   Input end iterator.
 * @param first_d Output begin iterator.
 *
 * Equivalent to:
 * ```
 * using std::make_move_iterator;
 * ityr::copy(policy, make_move_iterator(first1), make_move_iterator(last1), first_d);
 * ```
 *
 * @see [std::move -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/move)
 * @see `ityr::copy`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename ForwardIterator1, typename ForwardIteratorD>
inline ForwardIteratorD move(const ExecutionPolicy& policy,
                             ForwardIterator1       first1,
                             ForwardIterator1       last1,
                             ForwardIteratorD       first_d) {
  using std::make_move_iterator;
  return copy(policy, make_move_iterator(first1), make_move_iterator(last1), first_d);
}

ITYR_TEST_CASE("[ityr::pattern::parallel_loop] move") {
  ito::init();
  ori::init();

  long n = 100000;
  ori::global_ptr<common::move_only_t> p1 = ori::malloc_coll<common::move_only_t>(n);
  ori::global_ptr<common::move_only_t> p2 = ori::malloc_coll<common::move_only_t>(n);

  ito::root_exec([=] {
    for_each(
        execution::parallel_policy(100),
        make_global_iterator(p1    , checkout_mode::write),
        make_global_iterator(p1 + n, checkout_mode::write),
        count_iterator<long>(0),
        [=](common::move_only_t& r, long i) { new (&r) common::move_only_t(i * 2); });

    move(execution::parallel_policy(100), p1, p1 + n, p2);

    for_each(
        execution::parallel_policy(100),
        make_global_iterator(p2    , checkout_mode::read),
        make_global_iterator(p2 + n, checkout_mode::read),
        count_iterator<long>(0),
        [=](const common::move_only_t& r, long i) { ITYR_CHECK(r.value() == i * 2); });
  });

  ori::free_coll(p1);
  ori::free_coll(p2);

  ori::fini();
  ito::fini();
}

/**
 * @brief Reverse a range.
 *
 * @param policy Execution policy (`ityr::execution`).
 * @param first  Input begin iterator.
 * @param last   Input end iterator.
 *
 * This function reverses the input region `[first1, last1)` (in-place).
 *
 * If given iterators are global pointers, they are automatically checked out in the read-write mode
 * in the specified granularity (`ityr::execution::sequenced_policy::checkout_count` if serial,
 * or `ityr::execution::parallel_policy::checkout_count` if parallel) without explicitly passing them
 * as global iterators.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v = {1, 2, 3, 4, 5};
 * ityr::reverse(ityr::execution::par, v.begin(), v.end());
 * // v = {5, 4, 3, 2, 1}
 * ```
 *
 * @see [std::reverse -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/reverse)
 * @see `ityr::reverse_copy()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename BidirectionalIterator>
inline void reverse(const ExecutionPolicy& policy,
                    BidirectionalIterator  first,
                    BidirectionalIterator  last) {
  if constexpr (ori::is_global_ptr_v<BidirectionalIterator>) {
    return reverse(
        policy,
        internal::convert_to_global_iterator(first, checkout_mode::read_write),
        internal::convert_to_global_iterator(last , checkout_mode::read_write));

  } else {
    auto op = [=](auto&& r1, auto&& r2) {
      using std::swap;
      swap(r1, r2);
    };

    using std::make_reverse_iterator;
    auto d = std::distance(first, last);
    internal::loop_generic(policy, op, first, std::next(first, d / 2), make_reverse_iterator(last));
  }
}

/**
 * @brief Copy a reversed range to another.
 *
 * @param policy  Execution policy (`ityr::execution`).
 * @param first1  Input begin iterator.
 * @param last1   Input end iterator.
 * @param first_d Output begin iterator.
 *
 * Equivalent to:
 * ```
 * using std::make_reverse_iterator;
 * ityr::copy(policy, make_reverse_iterator(last1), make_reverse_iterator(first1), first_d);
 * ```
 *
 * @see [std::reverse_copy -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/reverse_copy)
 * @see `ityr::reverse()`
 * @see `ityr::copy()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename BidirectionalIterator1, typename BidirectionalIteratorD>
inline BidirectionalIteratorD reverse_copy(const ExecutionPolicy& policy,
                                           BidirectionalIterator1 first1,
                                           BidirectionalIterator1 last1,
                                           BidirectionalIteratorD first_d) {
  using std::make_reverse_iterator;
  return copy(policy, make_reverse_iterator(last1), make_reverse_iterator(first1), first_d);
}

ITYR_TEST_CASE("[ityr::pattern::parallel_loop] reverse") {
  ito::init();
  ori::init();

  long n = 100000;
  ori::global_ptr<long> p1 = ori::malloc_coll<long>(n);
  ori::global_ptr<long> p2 = ori::malloc_coll<long>(n);

  ito::root_exec([=] {
    for_each(
        execution::parallel_policy(100),
        make_global_iterator(p1    , checkout_mode::write),
        make_global_iterator(p1 + n, checkout_mode::write),
        count_iterator<long>(0),
        [=](long& v, long i) { v = i; });

    reverse(execution::parallel_policy(100), p1, p1 + n);

    reverse_copy(execution::parallel_policy(100), p1, p1 + n, p2);

    for_each(
        execution::parallel_policy(100),
        make_global_iterator(p1    , checkout_mode::read),
        make_global_iterator(p1 + n, checkout_mode::read),
        make_global_iterator(p2    , checkout_mode::read),
        count_iterator<long>(0),
        [=](long v1, long v2, long i) {
          ITYR_CHECK(v1 == n - i - 1);
          ITYR_CHECK(v2 == i);
        });
  });

  ori::free_coll(p1);
  ori::free_coll(p2);

  ori::fini();
  ito::fini();
}

/**
 * @brief Rotate a range.
 *
 * @param policy Execution policy (`ityr::execution`).
 * @param first  Input begin iterator.
 * @param middle Input iterator to the element that should be placed at the beginning of the range.
 * @param last   Input end iterator.
 *
 * @return The iterator to the original first element (`first + (last - middle)`).
 *
 * This function performs the left rotation for the given range.
 * The elements in the range are swapped so that the range `[first, middle)` is placed before
 * the range `[middle, last)`, preserving the original order in each range.
 *
 * If given iterators are global pointers, they are automatically checked out in the read-write mode
 * in the specified granularity (`ityr::execution::sequenced_policy::checkout_count` if serial,
 * or `ityr::execution::parallel_policy::checkout_count` if parallel) without explicitly passing them
 * as global iterators.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v = {1, 2, 3, 4, 5};
 * ityr::rotate(ityr::execution::par, v.begin(), v.begin() + 2, v.end());
 * // v = {3, 4, 5, 1, 2}
 * ```
 *
 * @see [std::rotate -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/rotate)
 * @see `ityr::rotate_copy`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename BidirectionalIterator>
inline BidirectionalIterator rotate(const ExecutionPolicy& policy,
                                    BidirectionalIterator  first,
                                    BidirectionalIterator  middle,
                                    BidirectionalIterator  last) {
  // TODO: implement a version with ForwardIterator
  if (first == middle) return last;
  if (middle == last) return first;

  parallel_invoke(
      [=] { reverse(policy, first, middle); },
      [=] { reverse(policy, middle, last); });
  reverse(policy, first, last);

  return std::next(first, std::distance(middle, last));
}

/**
 * @brief Copy a rotated range to another.
 *
 * @param policy Execution policy (`ityr::execution`).
 * @param first1  Input begin iterator.
 * @param middle1 Input iterator to the element that should be placed at the beginning of the output range.
 * @param last1   Input end iterator.
 * @param first_d Output begin iterator.
 *
 * @return The end iterator of the output range (`first_d + (last1 - first1)`).
 *
 * This function copies the left rotation for the input range to the output range..
 *
 * If given iterators are global pointers, they are automatically checked out in the specified
 * granularity (`ityr::execution::sequenced_policy::checkout_count` if serial,
 * or `ityr::execution::parallel_policy::checkout_count` if parallel) without explicitly passing them
 * as global iterators.
 * Input global pointers (`first1` and `last1`) are automatically checked out with the read-only mode.
 * if their value type is *trivially copyable*; otherwise, they are checked out with the read-write
 * mode, even if they are actually not modified.
 * Similarly, output global iterator (`first_d`) are checked out with the write-only mode if their
 * value type is *trivially copyable*; otherwise, they are checked out with the read-write mode.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v1 = {1, 2, 3, 4, 5};
 * ityr::global_vector<int> v2(v1.size());
 * ityr::rotate_copy(ityr::execution::par, v1.begin(), v1.begin() + 2, v1.end(), v2.begin());
 * // v2 = {3, 4, 5, 1, 2}
 * ```
 *
 * @see [std::rotate_copy -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/rotate_copy)
 * @see `ityr::rotate`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename ForwardIterator1, typename ForwardIteratorD>
inline ForwardIteratorD rotate_copy(const ExecutionPolicy& policy,
                                    ForwardIterator1       first1,
                                    ForwardIterator1       middle1,
                                    ForwardIterator1       last1,
                                    ForwardIteratorD       first_d) {
  auto [_, last_d] = parallel_invoke(
      [=] { copy(policy, middle1, last1, first_d); },
      [=] { auto middle_d = std::next(first_d, std::distance(middle1, last1));
            return copy(policy, first1, middle1, middle_d); });
  return last_d;
}

ITYR_TEST_CASE("[ityr::pattern::parallel_loop] rotate") {
  ito::init();
  ori::init();

  long n = 100000;
  ori::global_ptr<long> p1 = ori::malloc_coll<long>(n);
  ori::global_ptr<long> p2 = ori::malloc_coll<long>(n);

  ito::root_exec([=] {
    for_each(
        execution::parallel_policy(100),
        make_global_iterator(p1    , checkout_mode::write),
        make_global_iterator(p1 + n, checkout_mode::write),
        count_iterator<long>(0),
        [=](long& v, long i) { v = i; });

    long shift = n / 3;
    rotate(execution::parallel_policy(100), p1, p1 + shift, p1 + n);

    rotate_copy(
        execution::parallel_policy(100),
        p1, p1 + (n - shift), p1 + n, p2);

    for_each(
        execution::parallel_policy(100),
        make_global_iterator(p1    , checkout_mode::read),
        make_global_iterator(p1 + n, checkout_mode::read),
        make_global_iterator(p2    , checkout_mode::read),
        count_iterator<long>(0),
        [=](long v1, long v2, long i) {
          ITYR_CHECK(v1 == (i + shift) % n);
          ITYR_CHECK(v2 == i);
        });
  });

  ori::free_coll(p1);
  ori::free_coll(p2);

  ori::fini();
  ito::fini();
}

}
