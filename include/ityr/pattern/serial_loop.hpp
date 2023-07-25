#pragma once

#include "ityr/common/util.hpp"
#include "ityr/ori/ori.hpp"
#include "ityr/pattern/iterator.hpp"
#include "ityr/pattern/global_iterator.hpp"

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

template <typename T, typename Mode, typename Fn>
void with_checkout_iter(global_iterator<T, Mode> begin,
                        std::size_t              count,
                        Fn&&                     fn) {
  auto p = ori::checkout(&*begin, count, Mode{});
  std::forward<Fn>(fn)(p);
  ori::checkin(p, count, Mode{});
}

template <typename T, typename Fn>
void with_checkout_iter(global_move_iterator<T> begin,
                        std::size_t             count,
                        Fn&&                    fn) {
  auto p = ori::checkout(&*begin, count, ori::mode::read_write);
  std::forward<Fn>(fn)(std::make_move_iterator(p));
  ori::checkin(p, count, ori::mode::read_write);
}

template <typename T, typename Fn>
void with_checkout_iter(global_construct_iterator<T> begin,
                        std::size_t                  count,
                        Fn&&                         fn) {
  auto p = ori::checkout(&*begin, count, ori::mode::write);
  std::forward<Fn>(fn)(make_count_iterator(p));
  ori::checkin(p, count, ori::mode::write);
}

template <typename ForwardIterator, typename Fn>
inline void for_each(const execution::sequenced_policy& opts,
                     ForwardIterator                    first,
                     ForwardIterator                    last,
                     Fn                                 fn) {
  if constexpr (is_global_iterator_v<ForwardIterator>) {
    if constexpr (ForwardIterator::auto_checkout) {
      auto n = std::distance(first, last);
      for (std::ptrdiff_t d = 0; d < n; d += opts.checkout_count) {
        auto n_ = std::min(static_cast<std::size_t>(n - d), opts.checkout_count);
        with_checkout_iter(std::next(first, d), n_, [&](auto&& it) {
          for_each(opts, it, std::next(it, n_), fn);
        });
      }
    } else {
      // &*: convert global_iterator -> global_ref -> global_ptr
      for_each(opts, &*first, &*last, fn);
    }
  } else {
    for (; first != last; ++first) {
      fn(*first);
    }
  }
}

template <typename ForwardIterator1, typename ForwardIterator2, typename Fn>
inline void for_each(const execution::sequenced_policy& opts,
                     ForwardIterator1                   first1,
                     ForwardIterator1                   last1,
                     ForwardIterator2                   first2,
                     Fn                                 fn) {
  if constexpr (is_global_iterator_v<ForwardIterator1>) {
    if constexpr (ForwardIterator1::auto_checkout) {
      auto n = std::distance(first1, last1);
      for (std::ptrdiff_t d = 0; d < n; d += opts.checkout_count) {
        auto n_ = std::min(static_cast<std::size_t>(n - d), opts.checkout_count);
        with_checkout_iter(std::next(first1, d), n_, [&](auto&& it1) {
          auto it2 = std::next(first2, d);
          for_each(opts, it1, std::next(it1, n_), it2, fn);
        });
      }
    } else {
      // &*: convert global_iterator -> global_ref -> global_ptr
      for_each(opts, &*first1, &*last1, first2, fn);
    }
  } else if constexpr (is_global_iterator_v<ForwardIterator2>) {
    if constexpr (ForwardIterator2::auto_checkout) {
      auto n = std::distance(first1, last1);
      for (std::ptrdiff_t d = 0; d < n; d += opts.checkout_count) {
        auto n_ = std::min(static_cast<std::size_t>(n - d), opts.checkout_count);
        with_checkout_iter(std::next(first2, d), n_, [&](auto&& it2) {
          auto it1 = std::next(first1, d);
          for_each(opts, it1, std::next(it1, n_), it2, fn);
        });
      }
    } else {
      // &*: convert global_iterator -> global_ref -> global_ptr
      for_each(opts, first1, last1, &*first2, fn);
    }
  } else {
    for (; first1 != last1; ++first1, ++first2) {
      fn(*first1, *first2);
    }
  }
}

template <typename ForwardIterator1, typename ForwardIterator2, typename ForwardIterator3, typename Fn>
inline void for_each(const execution::sequenced_policy& opts,
                     ForwardIterator1                   first1,
                     ForwardIterator1                   last1,
                     ForwardIterator2                   first2,
                     ForwardIterator3                   first3,
                     Fn                                 fn) {
  if constexpr (is_global_iterator_v<ForwardIterator1>) {
    if constexpr (ForwardIterator1::auto_checkout) {
      auto n = std::distance(first1, last1);
      for (std::ptrdiff_t d = 0; d < n; d += opts.checkout_count) {
        auto n_ = std::min(static_cast<std::size_t>(n - d), opts.checkout_count);
        with_checkout_iter(std::next(first1, d), n_, [&](auto&& it1) {
          auto it2 = std::next(first2, d);
          auto it3 = std::next(first3, d);
          for_each(opts, it1, std::next(it1, n_), it2, it3, fn);
        });
      }
    } else {
      // &*: convert global_iterator -> global_ref -> global_ptr
      for_each(opts, &*first1, &*last1, first2, first3, fn);
    }
  } else if constexpr (is_global_iterator_v<ForwardIterator2>) {
    if constexpr (ForwardIterator2::auto_checkout) {
      auto n = std::distance(first1, last1);
      for (std::ptrdiff_t d = 0; d < n; d += opts.checkout_count) {
        auto n_ = std::min(static_cast<std::size_t>(n - d), opts.checkout_count);
        with_checkout_iter(std::next(first2, d), n_, [&](auto&& it2) {
          auto it1 = std::next(first1, d);
          auto it3 = std::next(first3, d);
          for_each(opts, it1, std::next(it1, n_), it2, it3, fn);
        });
      }
    } else {
      // &*: convert global_iterator -> global_ref -> global_ptr
      for_each(opts, first1, last1, &*first2, first3, fn);
    }
  } else if constexpr (is_global_iterator_v<ForwardIterator3>) {
    if constexpr (ForwardIterator3::auto_checkout) {
      auto n = std::distance(first1, last1);
      for (std::ptrdiff_t d = 0; d < n; d += opts.checkout_count) {
        auto n_ = std::min(static_cast<std::size_t>(n - d), opts.checkout_count);
        with_checkout_iter(std::next(first3, d), n_, [&](auto&& it3) {
          auto it1 = std::next(first1, d);
          auto it2 = std::next(first2, d);
          for_each(opts, it1, std::next(it1, n_), it2, it3, fn);
        });
      }
    } else {
      // &*: convert global_iterator -> global_ref -> global_ptr
      for_each(opts, first1, last1, first2, &*first3, fn);
    }
  } else {
    for (; first1 != last1; ++first1, ++first2, ++first3) {
      fn(*first1, *first2, *first3);
    }
  }
}

ITYR_TEST_CASE("[ityr::pattern::serial_loop] for_each seq") {
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

}
