#pragma once

#include "ityr/common/util.hpp"
#include "ityr/ori/ori.hpp"
#include "ityr/pattern/iterator.hpp"
#include "ityr/pattern/global_iterator.hpp"

namespace ityr {

struct serial_loop_options {
  std::size_t checkout_count = 1;
};

template <typename ForwardIterator, typename Fn>
inline void serial_for_each(ForwardIterator first,
                            ForwardIterator last,
                            Fn              fn) {
  serial_for_each(serial_loop_options{}, first, last, fn);
}

template <typename ForwardIterator, typename Fn>
inline void serial_for_each(const serial_loop_options&,
                            ForwardIterator first,
                            ForwardIterator last,
                            Fn              fn) {
  for (; first != last; ++first) {
    fn(*first);
  }
}

template <typename T, typename Mode, typename Fn>
inline void serial_for_each(const serial_loop_options& opts,
                            global_iterator<T, Mode>   first,
                            global_iterator<T, Mode>   last,
                            Fn                         fn) {
  if constexpr (global_iterator<T, Mode>::auto_checkout) {
    auto n = std::distance(first, last);
    for (std::ptrdiff_t d = 0; d < n; d += opts.checkout_count) {
      auto n_ = std::min(static_cast<std::size_t>(n - d), opts.checkout_count);
      ori::with_checkout(&*std::next(first, d), n_, Mode{}, [&](auto&& it) {
        serial_for_each(opts, it, std::next(it, n_), fn);
      });
    }
  } else {
    // &*: convert global_iterator -> global_ref -> global_ptr
    serial_for_each(opts, &*first, &*last, fn);
  }
}

template <typename ForwardIterator1, typename ForwardIterator2, typename Fn>
inline void serial_for_each(const serial_loop_options&,
                            ForwardIterator1 first1,
                            ForwardIterator1 last1,
                            ForwardIterator2 first2,
                            Fn               fn) {
  for (; first1 != last1; ++first1, ++first2) {
    fn(*first1, *first2);
  }
}

template <typename T1, typename Mode1, typename ForwardIterator2, typename Fn>
inline void serial_for_each(const serial_loop_options& opts,
                            global_iterator<T1, Mode1> first1,
                            global_iterator<T1, Mode1> last1,
                            ForwardIterator2           first2,
                            Fn                         fn) {
  if constexpr (global_iterator<T1, Mode1>::auto_checkout) {
    auto n = std::distance(first1, last1);
    for (std::ptrdiff_t d = 0; d < n; d += opts.checkout_count) {
      auto n_ = std::min(static_cast<std::size_t>(n - d), opts.checkout_count);
      ori::with_checkout(&*std::next(first1, d), n_, Mode1{}, [&](auto&& it1) {
        auto it2 = std::next(first2, d);
        serial_for_each(opts, it1, std::next(it1, n_), it2, fn);
      });
    }
  } else {
    // &*: convert global_iterator -> global_ref -> global_ptr
    serial_for_each(opts, &*first1, &*last1, first2, fn);
  }
}

template <typename ForwardIterator1, typename T2, typename Mode2, typename Fn,
          std::enable_if_t<not is_global_iterator_v<ForwardIterator1>, std::nullptr_t> = nullptr>
inline void serial_for_each(const serial_loop_options& opts,
                            ForwardIterator1           first1,
                            ForwardIterator1           last1,
                            global_iterator<T2, Mode2> first2,
                            Fn                         fn) {
  if constexpr (global_iterator<T2, Mode2>::auto_checkout) {
    auto n = std::distance(first1, last1);
    for (std::ptrdiff_t d = 0; d < n; d += opts.checkout_count) {
      auto n_ = std::min(static_cast<std::size_t>(n - d), opts.checkout_count);
      ori::with_checkout(&*std::next(first2, d), n_, Mode2{}, [&](auto&& it2) {
        auto it1 = std::next(first1, d);
        serial_for_each(opts, it1, std::next(it1, n_), it2, fn);
      });
    }
  } else {
    // &*: convert global_iterator -> global_ref -> global_ptr
    serial_for_each(opts, first1, last1, &*first2, fn);
  }
}

}
