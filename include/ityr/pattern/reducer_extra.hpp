#pragma once

#include "ityr/common/util.hpp"
#include "ityr/pattern/parallel_loop.hpp"
#include "ityr/container/global_vector.hpp"
#include "ityr/container/global_span.hpp"

namespace ityr::reducer {

template <typename T, typename Counter = std::size_t>
struct histogram {
  static_assert(std::is_arithmetic_v<T>);

  using value_type       = T;
  using accumulator_type = global_vector<Counter>;

  histogram(std::size_t n_bins) : n_bins_(n_bins) {}
  histogram(std::size_t n_bins, const value_type& lowest, const value_type& highest)
    : n_bins_(n_bins), lowest_(lowest), highest_(highest) {}

  void operator()(accumulator_type& acc, const value_type& x) const {
    if (lowest_ <= x && x <= highest_) {
      auto delta = (highest_ - lowest_) / n_bins_;
      std::size_t key = (x - lowest_) / delta;
      ITYR_CHECK(key < n_bins_);
      acc[key]++;
    }
  }

  void operator()(accumulator_type& acc_l, const accumulator_type& acc_r) const {
    transform(
        execution::parallel_policy(128),
        acc_l.begin(), acc_l.end(), acc_r.begin(), acc_l.begin(),
        [](const Counter& c1, const Counter& c2) { return c1 + c2; });
  }

  void operator()(const accumulator_type& acc_l, accumulator_type& acc_r) const {
    // commutative
    foldl(acc_r, acc_l);
  }

  accumulator_type operator()() const {
    return global_vector<Counter>(n_bins_, 0);
  }

private:
  std::size_t n_bins_;
  value_type  lowest_  = std::numeric_limits<value_type>::lowest();
  value_type  highest_ = std::numeric_limits<value_type>::max();
};

template <typename T>
struct vec_concat {
  using value_type       = global_vector<T>;
  using accumulator_type = global_vector<T>;

  void operator()(accumulator_type& acc_l, accumulator_type&& acc_r) const {
    acc_l.insert(acc_l.end(), make_move_iterator(acc_r.begin()), make_move_iterator(acc_r.end()));
  }

  void operator()(accumulator_type&& acc_l, accumulator_type& acc_r) const {
    acc_r.insert(acc_r.begin(), make_move_iterator(acc_l.begin()), make_move_iterator(acc_l.end()));
  }

  accumulator_type operator()() const {
    return global_vector<T>();
  }
};

template <typename T, typename BinaryOp>
struct vec_element_wise {
  using value_type       = global_vector<T>;
  using accumulator_type = global_vector<T>;

  void operator()(accumulator_type& acc_l, value_type&& acc_r) const {
    if (acc_l.empty()) {
      acc_l = std::move(acc_r);
    } else {
      ITYR_CHECK(acc_l.size() == acc_r.size());
      for_each(
          execution::sequenced_policy(acc_l.size()),
          make_global_iterator(acc_l.begin(), checkout_mode::read_write),
          make_global_iterator(acc_l.end()  , checkout_mode::read_write),
          make_global_iterator(acc_r.begin(), checkout_mode::read),
          [&](auto&& x, const auto& y) { x = bop_(x, y); });
    }
  }

  void operator()(accumulator_type&& acc_l, accumulator_type& acc_r) const {
    if (acc_r.empty()) {
      acc_r = std::move(acc_l);
    } else {
      ITYR_CHECK(acc_l.size() == acc_r.size());
      for_each(
          execution::sequenced_policy(acc_l.size()),
          make_global_iterator(acc_l.begin(), checkout_mode::read),
          make_global_iterator(acc_l.end()  , checkout_mode::read),
          make_global_iterator(acc_r.begin(), checkout_mode::read_write),
          [&](const auto& x, auto&& y) { y = bop_(x, y); });
    }
  }

  accumulator_type operator()() const {
    return global_vector<T>();
  }

private:
  static constexpr auto bop_ = BinaryOp();
};

template <typename T>
using vec_plus = vec_element_wise<T, std::plus<>>;

template <typename T>
using vec_multiplies = vec_element_wise<T, std::multiplies<>>;

template <typename T>
using vec_min = vec_element_wise<T, min_functor<>>;

template <typename T>
using vec_max = vec_element_wise<T, max_functor<>>;

ITYR_TEST_CASE("[ityr::reducer] extra reducer test") {
  ito::init();
  ori::init();

  ITYR_SUBCASE("histogram") {
    root_exec([=] {
      int n_samples = 100000;
      int n_bins = 1000;
      global_vector<double> v(global_vector_options(true, 1024), n_samples);

      transform(
          execution::parallel_policy(128),
          count_iterator<int>(0), count_iterator<int>(n_samples), v.begin(),
          [=](int i) {
            double x = (static_cast<double>(i) + 0.5) / n_bins;
            return x - static_cast<int>(x); // within [0.0, 1.0)
          });

      auto bins = reduce(
          execution::parallel_policy(128),
          v.begin(), v.end(), histogram<double>(n_bins, 0.0, 1.0));
      ITYR_CHECK(bins.size() == n_bins);

      auto count_sum = reduce(execution::par, bins.begin(), bins.end());
      ITYR_CHECK(count_sum == n_samples);

      for_each(
          execution::par,
          make_global_iterator(bins.begin(), checkout_mode::read),
          make_global_iterator(bins.end()  , checkout_mode::read),
          [=](auto count) {
            ITYR_CHECK(count == n_samples / n_bins);
          });
    });
  }

  ITYR_SUBCASE("vec_concat") {
    root_exec([=] {
      int n = 10000;

      global_vector<int> ret = transform_reduce(
          execution::parallel_policy(128),
          count_iterator<int>(0),
          count_iterator<int>(n),
          reducer::vec_concat<int>{},
          [](int i) { return global_vector<int>(1, i); });
      ITYR_CHECK(ret.size() == n);

      global_vector<int> ans(n);
      copy(
          execution::parallel_policy(128),
          count_iterator<int>(0),
          count_iterator<int>(n),
          ans.begin());

      ITYR_CHECK(ret == ans);
    });
  }

  ITYR_SUBCASE("vec_element_wise") {
    root_exec([=] {
      int n = 10000;
      int vec_size = 100;

      global_vector<int> ret = transform_reduce(
          execution::parallel_policy(128),
          count_iterator<int>(0),
          count_iterator<int>(n),
          reducer::vec_plus<int>{},
          [=](int i) { return global_vector<int>(vec_size, i); });
      ITYR_CHECK(ret.size() == vec_size);

      for_each(
          execution::par,
          make_global_iterator(ret.begin(), checkout_mode::read),
          make_global_iterator(ret.end()  , checkout_mode::read),
          [=](int x) { ITYR_CHECK(x == n * (n - 1) / 2); });
    });
  }

  ori::fini();
  ito::fini();
}

}
