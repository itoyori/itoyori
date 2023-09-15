#pragma once

#include "ityr/common/util.hpp"
#include "ityr/ito/ito.hpp"
#include "ityr/container/workhint_view.hpp"

namespace ityr::execution {

/**
 * @brief Serial execution policy for iterator-based loop functions.
 * @see [std::execution -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t)
 * @see `ityr::execution::seq`
 * @see `ityr::execution::sequenced_policy`
 * @see `ityr::for_each()`
 */
struct sequenced_policy {
  /**
   * @brief The maximum number of elements to check out at the same time if automatic checkout is enabled.
   */
  std::size_t checkout_count = 1;

  constexpr sequenced_policy() noexcept {}

  constexpr sequenced_policy(std::size_t checkout_count) noexcept
    : checkout_count(checkout_count) {}
};

/**
 * @brief Parallel execution policy for iterator-based loop functions.
 * @see [std::execution -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t)
 * @see `ityr::execution::par`
 * @see `ityr::execution::parallel_policy`
 * @see `ityr::for_each()`
 */
template <typename W = void>
struct parallel_policy {
  constexpr parallel_policy() noexcept {}

  constexpr parallel_policy(std::size_t cutoff_count) noexcept
    : cutoff_count(cutoff_count), checkout_count(cutoff_count) {}

  constexpr parallel_policy(std::size_t cutoff_count, std::size_t checkout_count) noexcept
    : cutoff_count(cutoff_count), checkout_count(checkout_count) {}

  parallel_policy(const workhint_range<W>& workhint) noexcept
    : workhint(workhint.view()) {}

  parallel_policy(std::size_t cutoff_count, const workhint_range<W>& workhint) noexcept
    : cutoff_count(cutoff_count), checkout_count(cutoff_count), workhint(workhint.view()) {}

  parallel_policy(std::size_t cutoff_count, std::size_t checkout_count, const workhint_range<W>& workhint) noexcept
    : cutoff_count(cutoff_count), checkout_count(checkout_count), workhint(workhint.view()) {}

  parallel_policy(workhint_range_view<W> workhint) noexcept
    : workhint(workhint) {}

  parallel_policy(std::size_t cutoff_count, workhint_range_view<W> workhint) noexcept
    : cutoff_count(cutoff_count), checkout_count(cutoff_count), workhint(workhint) {}

  parallel_policy(std::size_t cutoff_count, std::size_t checkout_count, workhint_range_view<W> workhint) noexcept
    : cutoff_count(cutoff_count), checkout_count(checkout_count), workhint(workhint) {}

  /**
   * @brief The maximum number of elements to check out at the same time if automatic checkout is enabled.
   */
  std::size_t cutoff_count = 1;

  /**
   * @brief The number of elements for leaf tasks to stop parallel recursion.
   */
  std::size_t checkout_count = 1;

  /**
   * @brief Work hints for ADWS.
   */
  workhint_range_view<W> workhint;
};

/**
 * @brief Default serial execution policy for iterator-based loop functions.
 * @see `ityr::execution::sequenced_policy`
 */
inline constexpr sequenced_policy seq;

/**
 * @brief Default parallel execution policy for iterator-based loop functions.
 * @see `ityr::execution::sequenced_policy`
 */
inline constexpr parallel_policy par;

namespace internal {

inline constexpr sequenced_policy to_sequenced_policy(const sequenced_policy& policy) noexcept {
  return policy;
}

template <typename W>
inline constexpr sequenced_policy to_sequenced_policy(const parallel_policy<W>& policy) noexcept {
  return sequenced_policy(policy.checkout_count);
}

inline void assert_policy(const sequenced_policy& policy) {
  ITYR_CHECK(0 < policy.checkout_count);
}

template <typename W>
inline void assert_policy(const parallel_policy<W>& policy) {
  ITYR_CHECK(0 < policy.checkout_count);
  ITYR_CHECK(policy.checkout_count <= policy.cutoff_count);
}

template <typename W>
inline auto get_workhint(const parallel_policy<W>& policy) {
  if constexpr (std::is_void_v<W>) {
    return ito::workhint(1, 1);
  } else {
    if (policy.workhint.empty()) {
      return ito::workhint(W(1), W(1));
    } else {
      auto [w_new, w_rest] = policy.workhint.get_workhint();
      return ito::workhint(w_new, w_rest);
    }
  }
}

template <typename W>
inline auto get_child_policies(const parallel_policy<W>& policy) {
  if constexpr (std::is_void_v<W>) {
    return std::make_pair(std::cref(policy), std::cref(policy));
  } else {
    if (policy.workhint.empty()) {
      return std::make_pair(policy, policy);
    } else if (!policy.workhint.has_children()) {
      return std::make_pair(parallel_policy<W>(policy.cutoff_count, policy.checkout_count),
                            parallel_policy<W>(policy.cutoff_count, policy.checkout_count));
    } else {
      auto [c1, c2] = policy.workhint.get_children();
      return std::make_pair(parallel_policy(policy.cutoff_count, policy.checkout_count, c1),
                            parallel_policy(policy.cutoff_count, policy.checkout_count, c2));
    }
  }
}

}
}
