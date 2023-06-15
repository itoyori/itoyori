#pragma once

#include "ityr/common/util.hpp"
#include "ityr/ori/ori.hpp"
#include "ityr/pattern/iterator.hpp"
#include "ityr/pattern/global_iterator.hpp"
#include "ityr/container/checkout_span.hpp"

namespace ityr {

template <typename T, typename Mode, typename Fn>
void with_checkout_iter(global_iterator<T, Mode> begin,
                        std::size_t              count,
                        Fn&&                     fn) {
  auto cs = make_checkout(&*begin, count, Mode{});
  std::forward<Fn>(fn)(cs.data());
}

template <typename T, typename Fn>
void with_checkout_iter(global_move_iterator<T> begin,
                        std::size_t             count,
                        Fn&&                    fn) {
  auto cs = make_checkout(&*begin, count, ori::mode::read_write);
  std::forward<Fn>(fn)(std::make_move_iterator(cs.data()));
}

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
inline void serial_for_each(const serial_loop_options& opts,
                            ForwardIterator            first,
                            ForwardIterator            last,
                            Fn                         fn) {
  if constexpr (is_global_iterator_v<ForwardIterator>) {
    if constexpr (ForwardIterator::auto_checkout) {
      auto n = std::distance(first, last);
      for (std::ptrdiff_t d = 0; d < n; d += opts.checkout_count) {
        auto n_ = std::min(static_cast<std::size_t>(n - d), opts.checkout_count);
        with_checkout_iter(std::next(first, d), n_, [&](auto&& it) {
          serial_for_each(opts, it, std::next(it, n_), fn);
        });
      }
    } else {
      // &*: convert global_iterator -> global_ref -> global_ptr
      serial_for_each(opts, &*first, &*last, fn);
    }
  } else {
    for (; first != last; ++first) {
      fn(*first);
    }
  }
}

template <typename ForwardIterator1, typename ForwardIterator2, typename Fn>
inline void serial_for_each(ForwardIterator1 first1,
                            ForwardIterator1 last1,
                            ForwardIterator2 first2,
                            Fn               fn) {
  serial_for_each(serial_loop_options{}, first1, last1, first2, fn);
}

template <typename ForwardIterator1, typename ForwardIterator2, typename Fn>
inline void serial_for_each(const serial_loop_options& opts,
                            ForwardIterator1           first1,
                            ForwardIterator1           last1,
                            ForwardIterator2           first2,
                            Fn                         fn) {
  if constexpr (is_global_iterator_v<ForwardIterator1>) {
    if constexpr (ForwardIterator1::auto_checkout) {
      auto n = std::distance(first1, last1);
      for (std::ptrdiff_t d = 0; d < n; d += opts.checkout_count) {
        auto n_ = std::min(static_cast<std::size_t>(n - d), opts.checkout_count);
        with_checkout_iter(std::next(first1, d), n_, [&](auto&& it1) {
          auto it2 = std::next(first2, d);
          serial_for_each(opts, it1, std::next(it1, n_), it2, fn);
        });
      }
    } else {
      // &*: convert global_iterator -> global_ref -> global_ptr
      serial_for_each(opts, &*first1, &*last1, first2, fn);
    }
  } else if constexpr (is_global_iterator_v<ForwardIterator2>) {
    if constexpr (ForwardIterator2::auto_checkout) {
      auto n = std::distance(first1, last1);
      for (std::ptrdiff_t d = 0; d < n; d += opts.checkout_count) {
        auto n_ = std::min(static_cast<std::size_t>(n - d), opts.checkout_count);
        with_checkout_iter(std::next(first2, d), n_, [&](auto&& it2) {
          auto it1 = std::next(first1, d);
          serial_for_each(opts, it1, std::next(it1, n_), it2, fn);
        });
      }
    } else {
      // &*: convert global_iterator -> global_ref -> global_ptr
      serial_for_each(opts, first1, last1, &*first2, fn);
    }
  } else {
    for (; first1 != last1; ++first1, ++first2) {
      fn(*first1, *first2);
    }
  }
}

template <typename ForwardIterator1, typename ForwardIterator2, typename ForwardIterator3, typename Fn>
inline void serial_for_each(ForwardIterator1 first1,
                            ForwardIterator1 last1,
                            ForwardIterator2 first2,
                            ForwardIterator3 first3,
                            Fn               fn) {
  serial_for_each(serial_loop_options{}, first1, last1, first2, first3, fn);
}

template <typename ForwardIterator1, typename ForwardIterator2, typename ForwardIterator3, typename Fn>
inline void serial_for_each(const serial_loop_options& opts,
                            ForwardIterator1           first1,
                            ForwardIterator1           last1,
                            ForwardIterator2           first2,
                            ForwardIterator3           first3,
                            Fn                         fn) {
  if constexpr (is_global_iterator_v<ForwardIterator1>) {
    if constexpr (ForwardIterator1::auto_checkout) {
      auto n = std::distance(first1, last1);
      for (std::ptrdiff_t d = 0; d < n; d += opts.checkout_count) {
        auto n_ = std::min(static_cast<std::size_t>(n - d), opts.checkout_count);
        with_checkout_iter(std::next(first1, d), n_, [&](auto&& it1) {
          auto it2 = std::next(first2, d);
          auto it3 = std::next(first3, d);
          serial_for_each(opts, it1, std::next(it1, n_), it2, it3, fn);
        });
      }
    } else {
      // &*: convert global_iterator -> global_ref -> global_ptr
      serial_for_each(opts, &*first1, &*last1, first2, first3, fn);
    }
  } else if constexpr (is_global_iterator_v<ForwardIterator2>) {
    if constexpr (ForwardIterator2::auto_checkout) {
      auto n = std::distance(first1, last1);
      for (std::ptrdiff_t d = 0; d < n; d += opts.checkout_count) {
        auto n_ = std::min(static_cast<std::size_t>(n - d), opts.checkout_count);
        with_checkout_iter(std::next(first2, d), n_, [&](auto&& it2) {
          auto it1 = std::next(first1, d);
          auto it3 = std::next(first3, d);
          serial_for_each(opts, it1, std::next(it1, n_), it2, it3, fn);
        });
      }
    } else {
      // &*: convert global_iterator -> global_ref -> global_ptr
      serial_for_each(opts, first1, last1, &*first2, first3, fn);
    }
  } else if constexpr (is_global_iterator_v<ForwardIterator3>) {
    if constexpr (ForwardIterator3::auto_checkout) {
      auto n = std::distance(first1, last1);
      for (std::ptrdiff_t d = 0; d < n; d += opts.checkout_count) {
        auto n_ = std::min(static_cast<std::size_t>(n - d), opts.checkout_count);
        with_checkout_iter(std::next(first3, d), n_, [&](auto&& it3) {
          auto it1 = std::next(first1, d);
          auto it2 = std::next(first2, d);
          serial_for_each(opts, it1, std::next(it1, n_), it2, it3, fn);
        });
      }
    } else {
      // &*: convert global_iterator -> global_ref -> global_ptr
      serial_for_each(opts, first1, last1, first2, &*first3, fn);
    }
  } else {
    for (; first1 != last1; ++first1, ++first2, ++first3) {
      fn(*first1, *first2, *first3);
    }
  }
}

ITYR_TEST_CASE("[ityr::pattern::serial_loop] serial for each") {
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
      serial_for_each(count_iterator<long>(0),
                      count_iterator<long>(n),
                      [&](long i) { count += i; });
      ITYR_CHECK(count == n * (n - 1) / 2);

      count = 0;
      serial_for_each(count_iterator<long>(0),
                      count_iterator<long>(n),
                      count_iterator<long>(n),
                      [&](long i, long j) { count += i + j; });
      ITYR_CHECK(count == 2 * n * (2 * n - 1) / 2);

      count = 0;
      serial_for_each(count_iterator<long>(0),
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
      serial_for_each(mos1.begin(), mos1.end(),
                      std::back_inserter(mos2),
                      [&](long i, auto&& out) { out = i; });

      long count = 0;
      serial_for_each(mos2.begin(), mos2.end(),
                      [&](long i) { count += i; });
      ITYR_CHECK(count == n * (n - 1) / 2);
    }

    ITYR_SUBCASE("move iterator with vector") {
      std::vector<move_only_t> mos1(count_iterator<long>(0),
                                    count_iterator<long>(n));

      std::vector<move_only_t> mos2;
      serial_for_each(std::make_move_iterator(mos1.begin()),
                      std::make_move_iterator(mos1.end()),
                      std::back_inserter(mos2),
                      [&](move_only_t&& in, auto&& out) { out = std::move(in); });

      long count = 0;
      serial_for_each(mos2.begin(), mos2.end(),
                      [&](move_only_t& mo) { count += mo.value(); });
      ITYR_CHECK(count == n * (n - 1) / 2);

      serial_for_each(mos1.begin(), mos1.end(),
                      [&](move_only_t& mo) { ITYR_CHECK(mo.value() == -1); });
    }
  }

  ITYR_SUBCASE("with global_ptr") {
    ori::global_ptr<long> gp = ori::malloc<long>(n);

    serial_for_each(count_iterator<long>(0),
                    count_iterator<long>(n),
                    make_global_iterator(gp, ori::mode::write),
                    [&](long i, long& out) { new (&out) long(i); });

    ITYR_SUBCASE("read array without global_iterator") {
      long count = 0;
      serial_for_each(gp,
                      gp + n,
                      [&](ori::global_ref<long> gr) { count += gr; });
      ITYR_CHECK(count == n * (n - 1) / 2);
    }

    ITYR_SUBCASE("read array with global_iterator") {
      long count = 0;
      serial_for_each(make_global_iterator(gp    , ori::mode::read),
                      make_global_iterator(gp + n, ori::mode::read),
                      [&](long i) { count += i; });
      ITYR_CHECK(count == n * (n - 1) / 2);
    }

    ITYR_SUBCASE("move iterator") {
      ori::global_ptr<move_only_t> mos1 = ori::malloc<move_only_t>(n);
      ori::global_ptr<move_only_t> mos2 = ori::malloc<move_only_t>(n);

      serial_for_each(make_global_iterator(gp    , ori::mode::read),
                      make_global_iterator(gp + n, ori::mode::read),
                      make_global_iterator(mos1  , ori::mode::write),
                      [&](long i, move_only_t& out) { new (&out) move_only_t(i); });

      serial_for_each(make_move_iterator(mos1),
                      make_move_iterator(mos1 + n),
                      make_global_iterator(mos2, ori::mode::write),
                      [&](move_only_t&& in, move_only_t& out) { new (&out) move_only_t(std::move(in)); });

      long count = 0;
      serial_for_each(make_global_iterator(mos2    , ori::mode::read),
                      make_global_iterator(mos2 + n, ori::mode::read),
                      [&](const move_only_t& mo) { count += mo.value(); });
      ITYR_CHECK(count == n * (n - 1) / 2);

      serial_for_each(make_global_iterator(mos1    , ori::mode::read),
                      make_global_iterator(mos1 + n, ori::mode::read),
                      [&](const move_only_t& mo) { ITYR_CHECK(mo.value() == -1); });

      ori::free(mos1, n);
      ori::free(mos2, n);
    }

    ori::free(gp, n);
  }

  ori::fini();
}

}
