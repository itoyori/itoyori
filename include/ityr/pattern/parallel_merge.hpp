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

  if (d <= 1 || first == middle || middle == last) return;

  if (d <= policy.cutoff_count) {
    auto&& [css, its] = checkout_global_iterators(d, first);
    auto first_ = std::get<0>(its);
    std::inplace_merge(first_,
                       std::next(first_, std::distance(first, middle)),
                       std::next(first_, d),
                       comp);

  } else {
    auto comp_mids = [&]{
      auto&& [css, its] = checkout_global_iterators(2, std::prev(middle));
      auto mids = std::get<0>(its);
      return !comp(*std::next(mids), *mids);
    };
    if (comp_mids()) {
      //     middle
      // ... a || b ...   where   !(a > b) <=> a <= b
      return;
    }

    auto comp_ends = [&]{
      auto&& [css, its] = checkout_global_iterators(1, std::prev(last), first);
      auto [l, f] = its;
      return comp(*l, *f);
    };
    if (comp_ends()) {
      //     middle
      // a ... || ... b   where   b < a
      // (If b == a, we shall not rotate them for stability)
      rotate(policy, first, middle, last);
      return;
    }

    auto [s1, s2] = find_split_points_for_merge(first, middle, middle, last, comp);

    // do not split between the elements with an equal value (for stable merge)
    // TODO: more efficient impl for cases where the number of equal values is small
    using value_type = typename std::iterator_traits<RandomAccessIterator>::value_type;
    if (first < s1) {
      //        s1 -------> s1     middle
      // ... x x | x x x x x | a ... || ...
      auto&& [css, its] = checkout_global_iterators(1, std::prev(s1));
      auto s1_ = std::get<0>(its);
      s1 = std::partition_point(s1, middle, [&](const value_type& r) { return !comp(*s1_, r); });
    }
    if (s2 < last) {
      //   middle     s2 <------- s2
      // ... || ... b | x x x x x | x x ...
      auto&& [css, its] = checkout_global_iterators(1, s2);
      auto s2_ = std::get<0>(its);
      s2 = std::partition_point(middle, s2, [&](const value_type& r) { return comp(r, *s2_); });
    }
    // Make sure that elements with an equal value are never swapped across the middle iterator
    if (first < s1 && s2 < last) {
      auto&& [css, its] = checkout_global_iterators(2, std::prev(s1), std::prev(s2));
      auto [s1_, s2_] = its;

      auto s1l = s1_;
      auto s1r = std::next(s1_);
      auto s2l = s2_;
      auto s2r = std::next(s2_);

      if (!comp(*s1l, *s2r)) {
        //      s1     middle     s2
        // ... x | a ... || ... b | x ...
        // (do nothing)

      } else if (!comp(*s2l, *s1r)) {
        //      s1     middle     s2
        // ... a | x ... || ... x | b ...
        // we shall not swap these two `x`s for stability, so we move `s2` to left
        s2 = std::partition_point(middle, std::prev(s2), [&](const value_type& r) { return comp(r, *s2l); });
      }
    }

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
 * @param first  Begin iterator.
 * @param middle Middle iterator that splits the two sorted ranges.
 * @param last   End iterator.
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
 * @see `ityr::sort()`
 * @see `ityr::stable_sort()`
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
    inplace_merge(
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
 * @param first  Begin iterator.
 * @param middle Middle iterator that splits the two sorted ranges.
 * @param last   End iterator.
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
 * @see `ityr::sort()`
 * @see `ityr::stable_sort()`
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

  ITYR_SUBCASE("integer") {
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
  }

  ITYR_SUBCASE("pair (stability test)") {
    long n = 100000;
    long nb = 1000;
    ori::global_ptr<std::pair<long, long>> p = ori::malloc_coll<std::pair<long, long>>(n);

    ito::root_exec([=] {
      long m = n / 2;

      transform(
          execution::parallel_policy{.cutoff_count = 100, .checkout_count = 100},
          count_iterator<long>(0), count_iterator<long>(m), p,
          [=](long i) { return std::make_pair(i / nb, i); });

      transform(
          execution::parallel_policy{.cutoff_count = 100, .checkout_count = 100},
          count_iterator<long>(0), count_iterator<long>(n - m), p + m,
          [=](long i) { return std::make_pair(i / nb, i + m); });

      auto comp_first = [](const auto& a, const auto& b) { return a.first < b.first ; };
      auto comp_second = [](const auto& a, const auto& b) { return a.second < b.second; };

      ITYR_CHECK(is_sorted(execution::parallel_policy{.cutoff_count = 100, .checkout_count = 100},
                           p, p + n, comp_second) == true);

      inplace_merge(execution::parallel_policy{.cutoff_count = 100, .checkout_count = 100},
                    p, p + m, p + n, comp_first);

      ITYR_CHECK(is_sorted(execution::parallel_policy{.cutoff_count = 100, .checkout_count = 100},
                           p, p + n, comp_first) == true);

      for (long key = 0; key < m / nb; key++) {
        bool sorted = is_sorted(execution::parallel_policy{.cutoff_count = 100, .checkout_count = 100},
                                p + key * nb * 2,
                                p + (key + 1) * nb * 2,
                                [=](const auto& a, const auto& b) {
                                  ITYR_CHECK(a.first == key);
                                  ITYR_CHECK(b.first == key);
                                  return a.second < b.second;
                                });
        ITYR_CHECK(sorted);
      }
    });

    ori::free_coll(p);
  }

  ori::fini();
  ito::fini();
}

}
