#pragma once

#include "ityr/common/util.hpp"
#include "ityr/pattern/parallel_invoke.hpp"
#include "ityr/pattern/parallel_loop.hpp"
#include "ityr/pattern/parallel_reduce.hpp"

namespace ityr {

namespace internal {

template <typename RandomAccessIterator1, typename RandomAccessIterator2, typename Compare>
inline std::pair<RandomAccessIterator1, RandomAccessIterator2>
find_split_points_for_merge(RandomAccessIterator1 first1,
                            RandomAccessIterator1 last1,
                            RandomAccessIterator2 first2,
                            RandomAccessIterator2 last2,
                            Compare               comp) {
  std::size_t n1 = std::distance(first1, last1);
  std::size_t n2 = std::distance(first2, last2);

  if (n1 > n2) {
    // so that the size of [first1, last1) is always smaller than that of [first2, last2)
    auto [p1, p2] = find_split_points_for_merge(first2, last2, first1, last1, comp);
    return std::make_pair(p2, p1);
  }

  std::size_t m = (n1 + n2) / 2;

  if (n1 == 0) {
    return std::make_pair(first1, std::next(first2, m));
  }

  if (n1 == 1) {
    RandomAccessIterator2 it2 = std::next(first2, m);
    ITYR_CHECK(first2 <= std::prev(it2));
    auto&& [css, its] = checkout_global_iterators(1, first1, std::prev(it2));
    auto [it1r, it2l] = its;

    if (comp(*it1r, *it2l)) {
      return std::make_pair(std::next(first1), std::prev(it2));
    } else {
      return std::make_pair(first1, it2);
    }
  }

  RandomAccessIterator1 low  = first1;
  RandomAccessIterator1 high = last1;

  while (true) {
    ITYR_CHECK(low <= high);

    RandomAccessIterator1 it1 = std::next(low, std::distance(low, high) / 2);
    RandomAccessIterator2 it2 = std::next(first2, m - std::distance(first1, it1));

    if (it1 == first1 || it1 == last1) {
      return std::make_pair(it1, it2);
    }

    ITYR_CHECK(it2 != first2);
    ITYR_CHECK(it2 != last2);

    auto&& [css, its] = checkout_global_iterators(2, std::prev(it1), std::prev(it2));
    auto [it1_, it2_] = its;

    auto it1l = it1_;
    auto it1r = std::next(it1_);
    auto it2l = it2_;
    auto it2r = std::next(it2_);

    if (comp(*it2r, *it1l)) {
      ITYR_CHECK(high != std::prev(it1));
      high = std::prev(it1);

    } else if (comp(*it1r, *it2l)) {
      ITYR_CHECK(low != std::next(it1));
      low = std::next(it1);

    } else {
      return std::make_pair(it1, it2);
    }
  }
}

ITYR_TEST_CASE("[ityr::pattern::parallel_merge] find_split_points_for_merge") {
  ito::init();
  ori::init();

  auto check_fn = [](std::initializer_list<int> il1, std::initializer_list<int> il2,
                     std::size_t expected1, std::size_t expected2) {
    auto [it1, it2] = find_split_points_for_merge(il1.begin(), il1.end(), il2.begin(), il2.end(), std::less<>{});
    ITYR_CHECK(it1 == std::next(il1.begin(), expected1));
    ITYR_CHECK(it2 == std::next(il2.begin(), expected2));
  };

  check_fn({}, {}, 0, 0);
  check_fn({}, {1}, 0, 0);
  check_fn({}, {1, 2, 3, 4, 5}, 0, 2);
  check_fn({0}, {1, 2, 3, 4, 5}, 1, 2);
  check_fn({2}, {1, 2, 3, 4, 5}, 1, 2);
  check_fn({3}, {1, 2, 3, 4, 5}, 0, 3);
  check_fn({6}, {1, 2, 3, 4, 5}, 0, 3);
  check_fn({1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}, 2, 3);
  check_fn({1, 2, 3, 4, 5}, {4, 5, 6, 7, 8}, 4, 1);
  check_fn({1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, 5, 0);
  check_fn({6, 7, 8, 9, 10}, {1, 2, 3, 4, 5}, 0, 5);

  ito::fini();
  ori::fini();
}

template <typename RandomAccessIterator, typename Compare>
inline void inplace_merge_aux(const execution::parallel_policy& policy,
                              RandomAccessIterator              first,
                              RandomAccessIterator              middle,
                              RandomAccessIterator              last,
                              Compare                           comp) {
  // TODO: implement a version with BidirectionalIterator
  std::size_t d = std::distance(first, last);

  if (d <= 1) {
    return;

  } else if (d <= policy.cutoff_count) {
    auto [css, its] = checkout_global_iterators(d, first);
    auto first_ = std::get<0>(its);
    std::inplace_merge(first_,
                       std::next(first_, std::distance(first, middle)),
                       std::next(first_, d),
                       comp);

  } else {
    auto [s1, s2] = find_split_points_for_merge(first, middle, middle, last, comp);

    auto m = rotate(policy, s1, middle, s2);

    parallel_invoke(
        [=, s1 = s1] { inplace_merge_aux(policy, first, s1, m, comp); },
        [=, s2 = s2] { inplace_merge_aux(policy, m, s2, last, comp); });
  }
}

}

/**
 * @brief Merge two sorted ranges into one sorted range in place.
 *
 * @param policy Execution policy (`ityr::execution`).
 * @param first  Input begin iterator.
 * @param middle Input middle iterator that splits the two sorted ranges.
 * @param last   Input end iterator.
 * @param comp   Binary comparison operator.
 *
 * This function merges two sorted ranges (`[first, middle)` and `[middle, last)`) into one sorted
 * range (`[first, last)`) in place.
 * This merge opreration is stable.
 *
 * If global pointers are provided as iterators, they are automatically checked out with the read-write
 * mode in the specified granularity (`ityr::execution::sequenced_policy::checkout_count` if serial,
 * or `ityr::execution::parallel_policy::checkout_count` if parallel) without explicitly passing them
 * as global iterators.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v = {5, 4, 3, 2, 1, 5, 4, 3, 2, 1};
 * ityr::inplace_merge(ityr::execution::par, v.begin(), v.begin() + 5, v.end(), std::greater<>{});
 * // v = {5, 5, 4, 4, 3, 3, 2, 2, 1, 1}
 * ```
 *
 * @see [std::inplace_merge -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/inplace_merge)
 * @see `ityr::is_sorted()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename RandomAccessIterator, typename Compare>
inline void inplace_merge(const ExecutionPolicy& policy,
                          RandomAccessIterator   first,
                          RandomAccessIterator   middle,
                          RandomAccessIterator   last,
                          Compare                comp) {
  if constexpr (ori::is_global_ptr_v<RandomAccessIterator>) {
    return inplace_merge(
        policy,
        internal::convert_to_global_iterator(first , checkout_mode::read_write),
        internal::convert_to_global_iterator(middle, checkout_mode::read_write),
        internal::convert_to_global_iterator(last  , checkout_mode::read_write),
        comp);

  } else {
    internal::inplace_merge_aux(policy, first, middle, last, comp);
  }
}

/**
 * @brief Merge two sorted ranges into one sorted range in place.
 *
 * @param policy Execution policy (`ityr::execution`).
 * @param first  Input begin iterator.
 * @param middle Input middle iterator that splits the two sorted ranges.
 * @param last   Input end iterator.
 *
 * Equivalent to `ityr::inplace_merge(policy, first, middle, last, std::less<>{});`
 *
 * Example:
 * ```
 * ityr::global_vector<int> v = {1, 2, 3, 4, 5, 1, 2, 3, 4, 5};
 * ityr::inplace_merge(ityr::execution::par, v.begin(), v.begin() + 5, v.end());
 * // v = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5}
 * ```
 *
 * @see [std::inplace_merge -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/inplace_merge)
 * @see `ityr::is_sorted()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename RandomAccessIterator>
inline void inplace_merge(const ExecutionPolicy& policy,
                          RandomAccessIterator   first,
                          RandomAccessIterator   middle,
                          RandomAccessIterator   last) {
  inplace_merge(policy, first, middle, last, std::less<>{});
}

ITYR_TEST_CASE("[ityr::pattern::parallel_merge] inplace_merge") {
  ito::init();
  ori::init();

  long n = 100000;
  ori::global_ptr<long> p = ori::malloc_coll<long>(n);

  ito::root_exec([=] {
    long m = n / 3;

    transform(
        execution::parallel_policy{.cutoff_count = 100, .checkout_count = 100},
        count_iterator<long>(0), count_iterator<long>(m), p,
        [=](long i) { return i; });

    transform(
        execution::parallel_policy{.cutoff_count = 100, .checkout_count = 100},
        count_iterator<long>(0), count_iterator<long>(n - m), p + m,
        [=](long i) { return i; });

    ITYR_CHECK(is_sorted(execution::parallel_policy{.cutoff_count = 100, .checkout_count = 100},
                         p, p + m) == true);
    ITYR_CHECK(is_sorted(execution::parallel_policy{.cutoff_count = 100, .checkout_count = 100},
                         p + m, p + n) == true);
    ITYR_CHECK(is_sorted(execution::parallel_policy{.cutoff_count = 100, .checkout_count = 100},
                         p, p + n) == false);

    inplace_merge(execution::parallel_policy{.cutoff_count = 100, .checkout_count = 100},
                  p, p + m, p + n);

    ITYR_CHECK(is_sorted(execution::parallel_policy{.cutoff_count = 100, .checkout_count = 100},
                         p, p + n) == true);
  });

  ori::free_coll(p);

  ori::fini();
  ito::fini();
}

}
