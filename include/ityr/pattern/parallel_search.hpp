#pragma once

#include "ityr/common/util.hpp"
#include "ityr/pattern/parallel_invoke.hpp"
#include "ityr/pattern/parallel_loop.hpp"

namespace ityr {

namespace internal {

template <typename W, typename LeafOp, typename SelectOp, typename ForwardIterator>
inline auto search_aux(const execution::parallel_policy<W>& policy,
                       LeafOp                               leaf_op,
                       SelectOp                             select_op,
                       ForwardIterator                      first,
                       ForwardIterator                      last) {
  if constexpr (ori::is_global_ptr_v<ForwardIterator>) {
    return search_aux(
        policy,
        leaf_op,
        select_op,
        convert_to_global_iterator(first, checkout_mode::read),
        convert_to_global_iterator(last , checkout_mode::read));

  } else {
    std::size_t d = std::distance(first, last);

    if (d <= policy.cutoff_count) {
      auto [css, its] = checkout_global_iterators(d, first);
      auto first_ = std::get<0>(its);
      // TODO: `ret` will be a sort of cache of the elements, but it requires copy operations.
      // Heavy elements (e.g., vectors) might prefer passing by reference (iterator) rather than
      // passing by value.
      auto [ret, found_its_] = leaf_op(first_, std::next(first_, d));
      auto found_its = std::apply([&](auto&&... its_) {
        return std::make_tuple((first + std::distance(first_, its_))...);
      }, found_its_);
      return std::make_tuple(ret, found_its);

    } else {
      auto mid = std::next(first, d / 2);

      auto [r1, r2] = parallel_invoke(
          [=] { return search_aux(policy, leaf_op, select_op, first, mid); },
          [=] { return search_aux(policy, leaf_op, select_op, mid, last); });

      return select_op(r1, r2);
    }
  }
}

}

/**
 * @brief Search for the minimum element in a range.
 *
 * @param policy Execution policy (`ityr::execution`).
 * @param first  Begin iterator.
 * @param last   End iterator.
 * @param comp   Binary comparison operator. Returns true if the first argument is less than
 *               the second argument.
 *
 * @return Iterator to the first minimum element in the range `[first, last)`.
 *
 * This function returns an iterator to the minimum element in the given range `[first, last)`.
 * If multiple minimum elements exist, the iterator to the first element is returned.
 *
 * If global pointers are provided as iterators, they are automatically checked out with the read-only
 * mode in the specified granularity (`ityr::execution::sequenced_policy::checkout_count` if serial,
 * or `ityr::execution::parallel_policy::checkout_count` if parallel) without explicitly passing them
 * as global iterators.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v = {2, -5, -3, 1, 5};
 * auto it = ityr::min_element(ityr::execution::par, v.begin(), v.end(),
 *                             [](int x, int y) { return std::abs(x) < std::abs(y); });
 * // it = v.begin() + 3, *it = 1
 * ```
 *
 * @see [std::min_element -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/min_element)
 * @see `ityr::max_element()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename ForwardIterator, typename Compare>
inline ForwardIterator min_element(const ExecutionPolicy& policy,
                                   ForwardIterator        first,
                                   ForwardIterator        last,
                                   Compare                comp) {
  if (std::distance(first, last) <= 1) {
    return first;
  }

  auto leaf_op = [=](const auto& f, const auto& l) {
    auto it = std::min_element(f, l, comp);
    auto val = *it;
    return std::make_tuple(val, std::make_tuple(it));
  };

  auto select_op = [=](const auto& l, const auto& r) {
    auto [val_l, its_l] = l;
    auto [val_r, its_r] = r;
    return comp(val_r, val_l) ? r : l;
  };

  auto [val, its] = internal::search_aux(policy, leaf_op, select_op, first, last);
  auto [it] = its;
  return it;
}

/**
 * @brief Search for the minimum element in a range.
 *
 * @param policy Execution policy (`ityr::execution`).
 * @param first  Begin iterator.
 * @param last   End iterator.
 *
 * @return Iterator to the first minimum element in the range `[first, last)`.
 *
 * Equivalent to `ityr::min_element(policy, first, last, std::less<>{});
 *
 * Example:
 * ```
 * ityr::global_vector<int> v = {1, 5, 3, 1, 5};
 * auto it = ityr::min_element(ityr::execution::par, v.begin(), v.end());
 * // it = v.begin(), *it = 1
 * ```
 *
 * @see [std::min_element -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/min_element)
 * @see `ityr::max_element()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename ForwardIterator>
inline ForwardIterator min_element(const ExecutionPolicy& policy,
                                   ForwardIterator        first,
                                   ForwardIterator        last) {
  return min_element(policy, first, last, std::less<>{});
}

/**
 * @brief Search for the maximum element in a range.
 *
 * @param policy Execution policy (`ityr::execution`).
 * @param first  Begin iterator.
 * @param last   End iterator.
 * @param comp   Binary comparison operator. Returns true if the first argument is less than
 *               the second argument.
 *
 * @return Iterator to the first maximum element in the range `[first, last)`.
 *
 * This function returns an iterator to the maximum element in the given range `[first, last)`.
 * If multiple maximum elements exist, the iterator to the first element is returned.
 *
 * If global pointers are provided as iterators, they are automatically checked out with the read-only
 * mode in the specified granularity (`ityr::execution::sequenced_policy::checkout_count` if serial,
 * or `ityr::execution::parallel_policy::checkout_count` if parallel) without explicitly passing them
 * as global iterators.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v = {2, -5, -3, 1, 5};
 * auto it = ityr::max_element(ityr::execution::par, v.begin(), v.end(),
 *                             [](int x, int y) { return std::abs(x) < std::abs(y); });
 * // it = v.begin() + 1, *it = -5
 * ```
 *
 * @see [std::max_element -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/max_element)
 * @see `ityr::min_element()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename ForwardIterator, typename Compare>
inline ForwardIterator max_element(const ExecutionPolicy& policy,
                                   ForwardIterator        first,
                                   ForwardIterator        last,
                                   Compare                comp) {
  if (std::distance(first, last) <= 1) {
    return first;
  }

  auto leaf_op = [=](const auto& f, const auto& l) {
    auto it = std::max_element(f, l, comp);
    auto val = *it;
    return std::make_tuple(val, std::make_tuple(it));
  };

  auto select_op = [=](const auto& l, const auto& r) {
    auto [val_l, its_l] = l;
    auto [val_r, its_r] = r;
    return comp(val_l, val_r) ? r : l;
  };

  auto [val, its] = internal::search_aux(policy, leaf_op, select_op, first, last);
  auto [it] = its;
  return it;
}

/**
 * @brief Search for the maximum element in a range.
 *
 * @param policy Execution policy (`ityr::execution`).
 * @param first  Begin iterator.
 * @param last   End iterator.
 *
 * @return Iterator to the first maximum element in the range `[first, last)`.
 *
 * Equivalent to `ityr::max_element(policy, first, last, std::less<>{});
 *
 * Example:
 * ```
 * ityr::global_vector<int> v = {2, 5, 3, 1, 5};
 * auto it = ityr::max_element(ityr::execution::par, v.begin(), v.end());
 * // it = v.begin() + 1, *it = 5
 * ```
 *
 * @see [std::max_element -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/max_element)
 * @see `ityr::min_element()`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename ForwardIterator>
inline ForwardIterator max_element(const ExecutionPolicy& policy,
                                   ForwardIterator        first,
                                   ForwardIterator        last) {
  return max_element(policy, first, last, std::less<>{});
}

ITYR_TEST_CASE("[ityr::pattern::parallel_search] min, max, minmax_element") {
  ito::init();
  ori::init();

  long n = 100000;
  ori::global_ptr<long> p = ori::malloc_coll<long>(n);

  ito::root_exec([=] {
    transform(
        execution::parallel_policy(100),
        count_iterator<long>(0), count_iterator<long>(n), p,
        [=](long i) { return (3 * i + 5) % 13; });

    long max_val = 14;
    long max_pos = n / 3;
    long max_pos_dummy = n / 3 * 2;
    p[max_pos].put(max_val);
    p[max_pos_dummy].put(max_val);

    long min_val = -1;
    long min_pos = n / 4;
    long min_pos_dummy = n - 1;
    p[min_pos].put(min_val);
    p[min_pos_dummy].put(min_val);

    auto max_it = max_element(execution::parallel_policy(100), p, p + n);
    ITYR_CHECK(max_it == p + max_pos);
    ITYR_CHECK((*max_it).get() == max_val);

    auto min_it = min_element(execution::parallel_policy(100), p, p + n);
    ITYR_CHECK(min_it == p + min_pos);
    ITYR_CHECK((*min_it).get() == min_val);
  });

  ori::free_coll(p);

  ori::fini();
  ito::fini();
}

}
