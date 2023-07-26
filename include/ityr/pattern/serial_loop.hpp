#pragma once

#include "ityr/common/util.hpp"
#include "ityr/ori/ori.hpp"
#include "ityr/pattern/iterator.hpp"
#include "ityr/pattern/global_iterator.hpp"
#include "ityr/container/checkout_span.hpp"

namespace ityr {

namespace execution {

struct sequenced_policy {
  std::size_t checkout_count = 1;
};

struct parallel_policy {
  std::size_t cutoff_count   = 1;
  std::size_t checkout_count = 1;
};

inline sequenced_policy to_sequenced_policy(const sequenced_policy& opts) {
  return opts;
}

inline sequenced_policy to_sequenced_policy(const parallel_policy& opts) {
  return {.checkout_count = opts.checkout_count};
}

inline void assert_policy(const sequenced_policy& opts) {
  ITYR_CHECK(0 < opts.checkout_count);
}

inline void assert_policy(const parallel_policy& opts) {
  ITYR_CHECK(0 < opts.checkout_count);
  ITYR_CHECK(opts.checkout_count <= opts.cutoff_count);
}

inline constexpr sequenced_policy seq;
inline constexpr parallel_policy  par;

}

template <typename T, typename Mode>
inline auto make_checkout_iter(global_iterator<T, Mode> it,
                               std::size_t              count) {
  auto cs = make_checkout(&*it, count, Mode{});
  return std::make_tuple(std::move(cs), cs.data());
}

template <typename T>
inline auto make_checkout_iter(global_move_iterator<T> it,
                               std::size_t             count) {
  auto cs = make_checkout(&*it, count, checkout_mode::read_write);
  return std::make_tuple(std::move(cs), std::make_move_iterator(cs.data()));
}

template <typename T>
inline auto make_checkout_iter(global_construct_iterator<T> it,
                               std::size_t                  count) {
  auto cs = make_checkout(&*it, count, checkout_mode::write);
  return std::make_tuple(std::move(cs), make_count_iterator(cs.data()));
}

template <typename T>
inline auto make_checkout_iter(global_destruct_iterator<T> it,
                               std::size_t                 count) {
  auto cs = make_checkout(&*it, count, checkout_mode::read_write);
  return std::make_tuple(std::move(cs), make_count_iterator(cs.data()));
}

inline auto checkout_global_iterators(std::size_t) {
  return std::make_tuple(std::make_tuple(), std::make_tuple());
}

template <typename ForwardIterator, typename... ForwardIterators>
inline auto checkout_global_iterators(std::size_t n, ForwardIterator it, ForwardIterators... rest) {
  if constexpr (is_global_iterator_v<ForwardIterator>) {
    if constexpr (ForwardIterator::auto_checkout) {
      auto [cs, it_] = make_checkout_iter(it, n);
      auto [css, its] = checkout_global_iterators(n, rest...);
      return std::make_tuple(std::tuple_cat(std::make_tuple(std::move(cs)), std::move(css)),
                             std::tuple_cat(std::make_tuple(it_), its));
    } else {
      auto [css, its] = checkout_global_iterators(n, rest...);
      // &*: convert global_iterator -> global_ref -> global_ptr
      return std::make_tuple(std::move(css),
                             std::tuple_cat(std::make_tuple(&*it), its));
    }
  } else {
    auto [css, its] = checkout_global_iterators(n, rest...);
    return std::make_tuple(std::move(css),
                           std::tuple_cat(std::make_tuple(it), its));
  }
}

template <typename Op, typename... ForwardIterators>
inline void apply_iterators(Op                  op,
                            std::size_t         n,
                            ForwardIterators... its) {
  for (std::size_t i = 0; i < n; (++i, ..., ++its)) {
    op(*its...);
  }
}

template <typename Op, typename ForwardIterator, typename... ForwardIterators>
inline void for_each_aux(const execution::sequenced_policy& policy,
                         Op                                 op,
                         ForwardIterator                    first,
                         ForwardIterator                    last,
                         ForwardIterators...                firsts) {
  if constexpr ((is_global_iterator_v<ForwardIterator> || ... ||
                 is_global_iterator_v<ForwardIterators>)) {
    // perform automatic checkout for global iterators
    std::size_t n = std::distance(first, last);
    std::size_t c = policy.checkout_count;

    for (std::size_t d = 0; d < n; d += c) {
      auto n_ = std::min(n - d, c);

      auto [css, its] = checkout_global_iterators(n_, first, firsts...);
      std::apply([&](auto&&... args) {
        apply_iterators(op, n_, std::forward<decltype(args)>(args)...);
      }, its);

      ((first = std::next(first, n_)), ..., (firsts = std::next(firsts, n_)));
    }

  } else {
    for (; first != last; (++first, ..., ++firsts)) {
      op(*first, *firsts...);
    }
  }
}

}
