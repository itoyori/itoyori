#pragma once

#include "ityr/common/util.hpp"
#include "ityr/pattern/parallel_invoke.hpp"
#include "ityr/pattern/parallel_loop.hpp"

namespace ityr {

namespace internal {

template <typename W, typename BidirectionalIterator, typename Predicate>
inline BidirectionalIterator
stable_partition_aux(const execution::parallel_policy<W>& policy,
                     BidirectionalIterator                first,
                     BidirectionalIterator                last,
                     Predicate                            pred) {
  std::size_t d = std::distance(first, last);

  if (d <= policy.cutoff_count) {
    // TODO: consider policy.checkout_count
    ITYR_CHECK(policy.cutoff_count == policy.checkout_count);

    auto&& [css, its] = checkout_global_iterators(d, first);
    auto&& first_ = std::get<0>(its);
    auto m = std::stable_partition(first_, std::next(first_, d), pred);
    return std::next(first, std::distance(first_, m));
  }

  auto mid = std::next(first, d / 2);

  auto [m1, m2] = parallel_invoke(
      [=] { return stable_partition_aux(policy, first, mid , pred); },
      [=] { return stable_partition_aux(policy, mid  , last, pred); });

  return rotate(policy, m1, mid, m2);
}

}

/**
 * @brief Partition elements into two disjoint parts in place.
 *
 * @param policy Execution policy (`ityr::execution`).
 * @param first  Begin iterator.
 * @param last   End iterator.
 * @param pred   Predicate operator to determine a group.
 *
 * @return Iterator to the partition point (the first element of the second group).
 *
 * This function partitions the elements in the range `[first, last)` into two disjoint parts,
 * so that elements with `pred(x) == true` precede those with `pred(x) == false`.
 * This partition operation is stable, meaning that the original order of elements is preserved
 * in each group.
 *
 * If global pointers are provided as iterators, they are automatically checked out with the read-write
 * mode in the specified granularity (`ityr::execution::sequenced_policy::checkout_count` if serial,
 * or `ityr::execution::parallel_policy::checkout_count` if parallel) without explicitly passing them
 * as global iterators.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v = {1, 2, 3, 4, 5};
 * auto it = ityr::stable_partition(ityr::execution::par, v.begin(), v.end(),
 *                                  [](int x) { return x % 2 == 0; });
 * // v = {2, 4, 1, 3, 5}
 * //            ^
 * //            it
 * ```
 *
 * @see [std::stable_partition -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/stable_partition)
 * @see `ityr::partition()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename BidirectionalIterator, typename Predicate>
inline BidirectionalIterator stable_partition(const ExecutionPolicy& policy,
                                              BidirectionalIterator  first,
                                              BidirectionalIterator  last,
                                              Predicate              pred) {
  if constexpr (ori::is_global_ptr_v<BidirectionalIterator>) {
    return stable_partition(
        policy,
        internal::convert_to_global_iterator(first, checkout_mode::read_write),
        internal::convert_to_global_iterator(last , checkout_mode::read_write),
        pred);

  } else {
    return internal::stable_partition_aux(policy, first, last, pred);
  }
}

ITYR_TEST_CASE("[ityr::pattern::parallel_filter] stable_partition") {
  ito::init();
  ori::init();

  ITYR_SUBCASE("split half") {
    long n = 100000;
    ori::global_ptr<long> p = ori::malloc_coll<long>(n);

    ito::root_exec([=] {
      transform(
          execution::parallel_policy(100),
          count_iterator<long>(0), count_iterator<long>(n), p,
          [=](long i) { return i; });

      auto pp = stable_partition(
          execution::parallel_policy(100),
          p, p + n,
          [](long x) { return x % 2 == 0; });

      ITYR_CHECK(pp == p + n / 2);

      for_each(
          execution::parallel_policy(100),
          make_global_iterator(p , checkout_mode::read),
          make_global_iterator(pp, checkout_mode::read),
          count_iterator<long>(0),
          [](long x, long i) { ITYR_CHECK(x == i * 2); });

      for_each(
          execution::parallel_policy(100),
          make_global_iterator(pp   , checkout_mode::read),
          make_global_iterator(p + n, checkout_mode::read),
          count_iterator<long>(0),
          [](long x, long i) { ITYR_CHECK(x == i * 2 + 1); });
    });

    ori::free_coll(p);
  }

  ITYR_SUBCASE("split 1:2") {
    long n = 90000;
    ori::global_ptr<long> p = ori::malloc_coll<long>(n);

    ito::root_exec([=] {
      transform(
          execution::parallel_policy(100),
          count_iterator<long>(0), count_iterator<long>(n), p,
          [=](long i) { return i; });

      auto pp = stable_partition(
          execution::parallel_policy(100),
          p, p + n,
          [](long x) { return x % 3 == 0; });

      ITYR_CHECK(pp == p + n / 3);

      for_each(
          execution::parallel_policy(100),
          make_global_iterator(p , checkout_mode::read),
          make_global_iterator(pp, checkout_mode::read),
          count_iterator<long>(0),
          [](long x, long i) { ITYR_CHECK(x == i * 3); });

      for_each(
          execution::parallel_policy(100),
          make_global_iterator(pp   , checkout_mode::read),
          make_global_iterator(p + n, checkout_mode::read),
          count_iterator<long>(0),
          [](long x, long i) { ITYR_CHECK(x == (i / 2) * 3 + (i % 2) + 1); });
    });

    ori::free_coll(p);
  }

  ITYR_SUBCASE("corner cases") {
    long n = 100000;
    ori::global_ptr<long> p = ori::malloc_coll<long>(n);

    ito::root_exec([=] {
      transform(
          execution::parallel_policy(100),
          count_iterator<long>(0), count_iterator<long>(n), p,
          [=](long i) { return i; });

      auto pp1 = stable_partition(
          execution::parallel_policy(100),
          p, p + n,
          [](long) { return true; });

      ITYR_CHECK(pp1 == p + n);

      for_each(
          execution::parallel_policy(100),
          make_global_iterator(p,     checkout_mode::read),
          make_global_iterator(p + n, checkout_mode::read),
          count_iterator<long>(0),
          [](long x, long i) { ITYR_CHECK(x == i); });

      auto pp2 = stable_partition(
          execution::parallel_policy(100),
          p, p + n,
          [](long) { return false; });

      ITYR_CHECK(pp2 == p);

      for_each(
          execution::parallel_policy(100),
          make_global_iterator(p,     checkout_mode::read),
          make_global_iterator(p + n, checkout_mode::read),
          count_iterator<long>(0),
          [](long x, long i) { ITYR_CHECK(x == i); });
    });

    ori::free_coll(p);
  }

  ori::fini();
  ito::fini();
}

/**
 * @brief Partition elements into two disjoint parts in place.
 *
 * @param policy Execution policy (`ityr::execution`).
 * @param first  Begin iterator.
 * @param last   End iterator.
 * @param pred   Predicate operator to determine a group.
 *
 * @return Iterator to the partition point (the first element of the second group).
 *
 * This function partitions the elements in the range `[first, last)` into two disjoint parts,
 * so that elements with `pred(x) == true` precede those with `pred(x) == false`.
 * This partition operation may not be stable (see `ityr::stable_partition()`).
 *
 * If global pointers are provided as iterators, they are automatically checked out with the read-write
 * mode in the specified granularity (`ityr::execution::sequenced_policy::checkout_count` if serial,
 * or `ityr::execution::parallel_policy::checkout_count` if parallel) without explicitly passing them
 * as global iterators.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v = {1, 2, 3, 4, 5};
 * auto it = ityr::partition(ityr::execution::par, v.begin(), v.end(),
 *                           [](int x) { return x % 2 == 0; });
 * // v = {2, 4, 1, 3, 5}
 * //            ^
 * //            it
 * ```
 *
 * @see [std::partition -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/partition)
 * @see `ityr::stable_partition()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename BidirectionalIterator, typename Predicate>
inline BidirectionalIterator partition(const ExecutionPolicy& policy,
                                       BidirectionalIterator  first,
                                       BidirectionalIterator  last,
                                       Predicate              pred) {
  // TODO: implement faster unstable partition
  return stable_partition(policy, first, last, pred);
}

}
