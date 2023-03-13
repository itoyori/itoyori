#pragma once

#include "ityr/common/util.hpp"
#include "ityr/ori/ori.hpp"

namespace ityr {

template <bool B>
struct auto_checkout : public std::bool_constant<B> {};

template <typename T, typename Mode, bool AutoCheckout = true>
class global_iterator {
  using this_t = global_iterator<T, Mode, AutoCheckout>;

public:
  global_iterator(ori::global_ptr<T> ptr, Mode) : ptr_(ptr) {}
  global_iterator(ori::global_ptr<T> ptr, Mode, auto_checkout<AutoCheckout>) : ptr_(ptr) {}

  using mode = Mode;
  static constexpr bool auto_checkout = AutoCheckout;

  using value_type        = std::remove_cv_t<T>;
  using difference_type   = std::ptrdiff_t;
  using pointer           = ori::global_ptr<T>;
  using reference         = ori::global_ref<T>;
  using iterator_category = std::random_access_iterator_tag;

  reference operator*() const { return *ptr_; }

  reference operator[](difference_type diff) const { return ptr_ + diff; }

  this_t& operator+=(difference_type diff) { ptr_ += diff; return *this; }
  this_t& operator-=(difference_type diff) { ptr_ -= diff; return *this; }

  this_t& operator++() { ++ptr_; return *this; }
  this_t& operator--() { --ptr_; return *this; }

  this_t operator++(int) { this_t tmp(*this); ptr_++; return tmp; }
  this_t operator--(int) { this_t tmp(*this); ptr_--; return tmp; }

  this_t operator+(difference_type diff) const { return ptr_ + diff; }
  this_t operator-(difference_type diff) const { return ptr_ - diff; }
  difference_type operator-(const this_t& it) const { return ptr_ - &*it; }

private:
  ori::global_ptr<T> ptr_;
};

template <typename T, typename Mode, bool AutoCheckout>
inline bool operator==(const global_iterator<T, Mode, AutoCheckout>& it1,
                       const global_iterator<T, Mode, AutoCheckout>& it2) {
  return &*it1 == &*it2;
}

template <typename T, typename Mode, bool AutoCheckout>
inline bool operator!=(const global_iterator<T, Mode, AutoCheckout>& it1,
                       const global_iterator<T, Mode, AutoCheckout>& it2) {
  return &*it1 != &*it2;
}

template <typename T, typename Mode, bool AutoCheckout>
inline bool operator<(const global_iterator<T, Mode, AutoCheckout>& it1,
                      const global_iterator<T, Mode, AutoCheckout>& it2) {
  return &*it1 < &*it2;
}

template <typename T, typename Mode, bool AutoCheckout>
inline bool operator>(const global_iterator<T, Mode, AutoCheckout>& it1,
                      const global_iterator<T, Mode, AutoCheckout>& it2) {
  return &*it1 > &*it2;
}

template <typename T, typename Mode, bool AutoCheckout>
inline bool operator<=(const global_iterator<T, Mode, AutoCheckout>& it1,
                       const global_iterator<T, Mode, AutoCheckout>& it2) {
  return &*it1 <= &*it2;
}

template <typename T, typename Mode, bool AutoCheckout>
inline bool operator>=(const global_iterator<T, Mode, AutoCheckout>& it1,
                       const global_iterator<T, Mode, AutoCheckout>& it2) {
  return &*it1 >= &*it2;
}

template <typename>
struct is_global_iterator : public std::false_type {};

template <typename T, typename Mode, bool AutoCheckout>
struct is_global_iterator<global_iterator<T, Mode, AutoCheckout>> : public std::true_type {};

template <typename T>
inline constexpr bool is_global_iterator_v = is_global_iterator<T>::value;

static_assert(is_global_iterator_v<global_iterator<int, ori::mode::read_t>>);
static_assert(is_global_iterator_v<global_iterator<int, ori::mode::read_t, false>>);
static_assert(!is_global_iterator_v<int>);
static_assert(!is_global_iterator_v<ori::global_ptr<int>>);

template <typename T, typename Mode>
inline auto make_global_iterator(ori::global_ptr<T> ptr, Mode) {
  return global_iterator(ptr, Mode{});
}

template <typename T, typename Mode, typename AutoCheckout>
inline auto make_global_iterator(ori::global_ptr<T> ptr, Mode, AutoCheckout) {
  return global_iterator(ptr, Mode{}, AutoCheckout{});
}

static_assert(std::is_same_v<decltype(make_global_iterator(ori::global_ptr<int>{},
                                                           ori::mode::read_write))::mode,
                             ori::mode::read_write_t>);

static_assert(decltype(make_global_iterator(ori::global_ptr<int>{},
                                            ori::mode::read,
                                            auto_checkout<true>{}))::auto_checkout == true);

}
