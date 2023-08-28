#pragma once

#include "ityr/common/util.hpp"

namespace ityr::reducer {

template <typename T>
struct default_identity_provider {
  using value_type = T;
  operator value_type() { return T{}; }
  value_type operator()() const { return T{}; }
};

template <typename T, typename BinaryOp, typename IdentityProvider = default_identity_provider<T>>
struct monoid {
  static_assert(std::is_same_v<T, typename IdentityProvider::value_type>);

  using value_type       = T;
  using accumulator_type = T;

  void foldl(T& l, const T& r) const {
    l = bop_(l, r);
  }

  void foldr(const T& l, T& r) const {
    r = bop_(l, r);
  }

  T identity() const {
    return IdentityProvider();
  }

private:
  static constexpr auto bop_ = BinaryOp();
};

template <typename T>
struct one {
  static_assert(std::is_arithmetic_v<T>);
  using value_type = T;
  static constexpr T value = T{1};
  constexpr operator value_type() noexcept { return value; }
  constexpr value_type operator()() const noexcept { return value; }
};

template <typename T>
struct lowest {
  static_assert(std::is_arithmetic_v<T>);
  using value_type = T;
  static constexpr T value = std::numeric_limits<T>::lowest();
  constexpr operator value_type() noexcept { return value; }
  constexpr value_type operator()() const noexcept { return value; }
};

template <typename T>
struct highest {
  static_assert(std::is_arithmetic_v<T>);
  using value_type = T;
  static constexpr T value = std::numeric_limits<T>::max();
  constexpr operator value_type() noexcept { return value; }
  constexpr value_type operator()() const noexcept { return value; }
};

template <typename T = void>
struct min_functor {
  constexpr T operator()(const T& x, const T& y) const {
    return std::min(x, y);
  }
};

template <>
struct min_functor<void> {
  template <typename T, typename U>
  constexpr auto operator()(T&& v, U&& u) const {
    return std::min(std::forward<T>(v), std::forward<U>(u));
  }
};

template <typename T = void>
struct max_functor {
  constexpr T operator()(const T& x, const T& y) const {
    return std::max(x, y);
  }
};

template <>
struct max_functor<void> {
  template <typename T, typename U>
  constexpr auto operator()(T&& t, U&& u) const {
    return std::max(std::forward<T>(t), std::forward<U>(u));
  }
};

template <typename T>
using plus = monoid<T, std::plus<>>;

template <typename T>
using multiplies = monoid<T, std::multiplies<>, one<T>>;

template <typename T>
using min = monoid<T, min_functor<>, highest<T>>;

template <typename T>
using max = monoid<T, max_functor<>, lowest<T>>;

template <typename T>
struct minmax {
  using value_type       = T;
  using accumulator_type = std::pair<T, T>;

  void foldl(accumulator_type& acc, const value_type& x) const {
    acc.first = std::min(acc.first, x);
    acc.second = std::max(acc.second, x);
  }

  void foldl(accumulator_type& acc_l, const accumulator_type& acc_r) const {
    acc_l.first = std::min(acc_l.first, acc_r.first);
    acc_l.second = std::max(acc_l.second, acc_r.second);
  }

  void foldr(const accumulator_type& acc_l, accumulator_type& acc_r) const {
    // commucative
    foldl(acc_r, acc_l);
  }

  accumulator_type identity() const {
    return std::make_pair(std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest());
  }
};

using logical_and = monoid<bool, std::logical_and<>, std::true_type>;
using logical_or  = monoid<bool, std::logical_or<> , std::false_type>;

}
