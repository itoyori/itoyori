#pragma once

#include "ityr/common/util.hpp"
#include "ityr/pattern/parallel_invoke.hpp"
#include "ityr/pattern/parallel_loop.hpp"
#include "ityr/pattern/parallel_merge.hpp"

namespace ityr {

namespace internal {

template <bool Stable, typename W, typename RandomAccessIterator, typename Compare>
inline void merge_sort(const execution::parallel_policy<W>& policy,
                       RandomAccessIterator                 first,
                       RandomAccessIterator                 last,
                       Compare                              comp) {
  std::size_t d = std::distance(first, last);

  if (d <= 1) return;

  if (d <= policy.cutoff_count) {
    auto [css, its] = checkout_global_iterators(d, first);
    auto first_ = std::get<0>(its);
    std::stable_sort(first_, std::next(first_, d), comp);

  } else {
    auto middle = std::next(first, d / 2);

    parallel_invoke(
        [=] { merge_sort<Stable>(policy, first, middle, comp); },
        [=] { merge_sort<Stable>(policy, middle, last, comp); });

    internal::inplace_merge_aux<Stable>(policy, first, middle, last, comp);
  }
}

}

/**
 * @brief Stable sort for a range.
 *
 * @param policy Execution policy (`ityr::execution`).
 * @param first  Begin iterator.
 * @param last   End iterator.
 * @param comp   Binary comparison operator.
 *
 * This function sorts the given range (`[first, last)`) in place.
 * This sort is stable.
 *
 * If global pointers are provided as iterators, they are automatically checked out with the read-write
 * mode in the specified granularity (`ityr::execution::sequenced_policy::checkout_count` if serial,
 * or `ityr::execution::parallel_policy::checkout_count` if parallel) without explicitly passing them
 * as global iterators.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v = {2, 13, 4, 14, 15};
 * ityr::stable_sort(ityr::execution::par, v.begin(), v.end(),
 *                   [](int a, int b) { return a / 10 < b / 10; });
 * // v = {2, 4, 13, 14, 15}
 * ```
 *
 * @see [std::stable_sort -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/stable_sort)
 * @see `ityr::sort()`
 * @see `ityr::inplace_merge()`
 * @see `ityr::is_sorted()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename RandomAccessIterator, typename Compare>
inline void stable_sort(const ExecutionPolicy& policy,
                        RandomAccessIterator   first,
                        RandomAccessIterator   last,
                        Compare                comp) {
  if constexpr (ori::is_global_ptr_v<RandomAccessIterator>) {
    stable_sort(
        policy,
        internal::convert_to_global_iterator(first, checkout_mode::read_write),
        internal::convert_to_global_iterator(last , checkout_mode::read_write),
        comp);

  } else {
    internal::merge_sort<true>(policy, first, last, comp);
  }
}

/**
 * @brief Stable sort for a range.
 *
 * @param policy Execution policy (`ityr::execution`).
 * @param first  Begin iterator.
 * @param last   End iterator.
 *
 * Equivalent to `ityr::stable_sort(policy, first, last, std::less<>{})`.
 *
 * @see [std::stable_sort -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/stable_sort)
 * @see `ityr::sort()`
 * @see `ityr::inplace_merge()`
 * @see `ityr::is_sorted()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename RandomAccessIterator>
inline void stable_sort(const ExecutionPolicy& policy,
                        RandomAccessIterator   first,
                        RandomAccessIterator   last) {
  stable_sort(policy, first, last, std::less<>{});
}

ITYR_TEST_CASE("[ityr::pattern::parallel_sort] stable_sort") {
  ito::init();
  ori::init();

  ITYR_SUBCASE("stability test") {
    // std::pair is not trivially copyable
    struct item {
      long key;
      long val;
    };

    long n = 100000;
    long n_keys = 100;
    ori::global_ptr<item> p = ori::malloc_coll<item>(n);

    ito::root_exec([=] {
      transform(
          execution::parallel_policy(100),
          count_iterator<long>(0), count_iterator<long>(n), p,
          [=](long i) { return item{i % n_keys, (3 * i + 5) % 13}; });

      stable_sort(execution::parallel_policy(100),
                  p, p + n, [](const auto& a, const auto& b) { return a.val < b.val; });

      stable_sort(execution::parallel_policy(100),
                  p, p + n, [](const auto& a, const auto& b) { return a.key < b.key; });

      long n_values_per_key = n / n_keys;
      for (long key = 0; key < n_keys; key++) {
        bool sorted = is_sorted(execution::parallel_policy(100),
                                p + key * n_values_per_key,
                                p + (key + 1) * n_values_per_key,
                                [=](const auto& a, const auto& b) {
                                  ITYR_CHECK(a.key == key);
                                  ITYR_CHECK(b.key == key);
                                  return a.val < b.val;
                                });
        ITYR_CHECK(sorted);
      }
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

      stable_sort(execution::parallel_policy(100),
                  p, p + n, [](const auto&, const auto&) { return false; /* all equal */ });

      for_each(
          execution::parallel_policy(100),
          count_iterator<long>(0), count_iterator<long>(n),
          make_global_iterator(p, checkout_mode::read),
          [=](long i, long v) { ITYR_CHECK(i == v); });
    });

    ori::free_coll(p);
  }

  ori::fini();
  ito::fini();
}

/**
 * @brief Sort a range.
 *
 * @param policy Execution policy (`ityr::execution`).
 * @param first  Begin iterator.
 * @param last   End iterator.
 *
 * This function sorts the given range (`[first, last)`) in place.
 * This sort may not be stable.
 *
 * If global pointers are provided as iterators, they are automatically checked out with the read-write
 * mode in the specified granularity (`ityr::execution::sequenced_policy::checkout_count` if serial,
 * or `ityr::execution::parallel_policy::checkout_count` if parallel) without explicitly passing them
 * as global iterators.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v = {2, 3, 4, 1, 5};
 * ityr::sort(ityr::execution::par, v.begin(), v.end(), std::greater<>{});
 * // v = {5, 4, 3, 2, 1}
 * ```
 *
 * @see [std::sort -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/sort)
 * @see `ityr::stable_sort()`
 * @see `ityr::is_sorted()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename RandomAccessIterator, typename Compare>
inline void sort(const ExecutionPolicy& policy,
                 RandomAccessIterator   first,
                 RandomAccessIterator   last,
                 Compare                comp) {
  if constexpr (ori::is_global_ptr_v<RandomAccessIterator>) {
    sort(
        policy,
        internal::convert_to_global_iterator(first, checkout_mode::read_write),
        internal::convert_to_global_iterator(last , checkout_mode::read_write),
        comp);

  } else {
    internal::merge_sort<false>(policy, first, last, comp);
  }
}

/**
 * @brief Sort a range.
 *
 * @param policy Execution policy (`ityr::execution`).
 * @param first  Begin iterator.
 * @param last   End iterator.
 *
 * Equivalent to `ityr::sort(policy, first, last, std::less<>{})`.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v = {2, 3, 4, 1, 5};
 * ityr::sort(ityr::execution::par, v.begin(), v.end());
 * // v = {1, 2, 3, 4, 5}
 * ```
 *
 * @see [std::sort -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/sort)
 * @see `ityr::stable_sort()`
 * @see `ityr::is_sorted()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename RandomAccessIterator>
inline void sort(const ExecutionPolicy& policy,
                 RandomAccessIterator   first,
                 RandomAccessIterator   last) {
  sort(policy, first, last, std::less<>{});
}

ITYR_TEST_CASE("[ityr::pattern::parallel_sort] sort") {
  ito::init();
  ori::init();

  long n = 100000;
  ori::global_ptr<long> p = ori::malloc_coll<long>(n);

  ito::root_exec([=] {
    transform(
        execution::parallel_policy(100),
        count_iterator<long>(0), count_iterator<long>(n), p,
        [=](long i) { return (3 * i + 5) % 13; });

    ITYR_CHECK(is_sorted(execution::parallel_policy(100),
                         p, p + n) == false);

    sort(execution::parallel_policy(100), p, p + n);

    ITYR_CHECK(is_sorted(execution::parallel_policy(100),
                         p, p + n) == true);
  });

  ori::free_coll(p);

  ori::fini();
  ito::fini();
}

}
