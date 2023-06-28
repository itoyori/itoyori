#pragma once

#include "ityr/common/util.hpp"
#include "ityr/ori/ori.hpp"

namespace ityr {

namespace checkout_mode {
using read_t = ori::mode::read_t;
inline constexpr read_t read;
using write_t = ori::mode::write_t;
inline constexpr write_t write;
using read_write_t = ori::mode::read_write_t;
inline constexpr read_write_t read_write;
struct no_access_t {};
inline constexpr no_access_t no_access;
}

template <typename T, typename Mode>
class global_iterator : public ori::global_ptr<T> {
  using this_t = global_iterator<T, Mode>;
  using base_t = ori::global_ptr<T>;

public:
  global_iterator(ori::global_ptr<T> ptr, Mode) : base_t(ptr) {}

  using mode = Mode;
  static constexpr bool auto_checkout = !std::is_same_v<Mode, checkout_mode::no_access_t>;
};

template <typename T, typename Mode>
inline auto make_global_iterator(ori::global_ptr<T> ptr, Mode) {
  return global_iterator(ptr, Mode{});
}

static_assert(std::is_same_v<decltype(make_global_iterator(ori::global_ptr<int>{},
                                                           checkout_mode::read_write))::mode,
                             checkout_mode::read_write_t>);

static_assert(decltype(make_global_iterator(ori::global_ptr<int>{},
                                            checkout_mode::read))::auto_checkout == true);
static_assert(decltype(make_global_iterator(ori::global_ptr<int>{},
                                            checkout_mode::write))::auto_checkout == true);
static_assert(decltype(make_global_iterator(ori::global_ptr<int>{},
                                            checkout_mode::read_write))::auto_checkout == true);
static_assert(decltype(make_global_iterator(ori::global_ptr<int>{},
                                            checkout_mode::no_access))::auto_checkout == false);

template <typename T>
class global_move_iterator : public global_iterator<T, checkout_mode::read_write_t> {
  using base_t = global_iterator<T, checkout_mode::read_write_t>;
public:
  explicit global_move_iterator(ori::global_ptr<T> ptr)
    : base_t(ptr, checkout_mode::read_write) {}
};

template <typename T>
inline auto make_move_iterator(ori::global_ptr<T> ptr) {
  return global_move_iterator(ptr);
}

template <typename T>
class global_construct_iterator : public global_iterator<T, checkout_mode::write_t> {
  using base_t = global_iterator<T, checkout_mode::write_t>;
public:
  explicit global_construct_iterator(ori::global_ptr<T> ptr)
    : base_t(ptr, checkout_mode::write) {}
};

template <typename T>
inline auto make_construct_iterator(ori::global_ptr<T> ptr) {
  return global_construct_iterator(ptr);
}

template <typename>
struct is_global_iterator : public std::false_type {};

template <typename T, typename Mode>
struct is_global_iterator<global_iterator<T, Mode>> : public std::true_type {};

template <typename T>
struct is_global_iterator<global_move_iterator<T>> : public std::true_type {};

template <typename T>
struct is_global_iterator<global_construct_iterator<T>> : public std::true_type {};

template <typename T>
inline constexpr bool is_global_iterator_v = is_global_iterator<T>::value;

static_assert(is_global_iterator_v<global_iterator<int, checkout_mode::read_t>>);
static_assert(is_global_iterator_v<global_move_iterator<int>>);
static_assert(!is_global_iterator_v<int>);
static_assert(!is_global_iterator_v<ori::global_ptr<int>>);

}
