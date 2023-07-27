#pragma once

#include "ityr/common/util.hpp"
#include "ityr/ori/ori.hpp"

namespace ityr {

namespace checkout_mode {

/** @brief See `ityr::checkout_mode::read`. */
using read_t = ori::mode::read_t;

/**
 * @brief Read-only checkout mode.
 * @see `ityr::make_checkout()`.
 * @see `ityr::make_global_iterator()`.
 */
inline constexpr read_t read;

/** @brief See `ityr::checkout_mode::write`. */
using write_t = ori::mode::write_t;

/**
 * @brief Write-only checkout mode.
 * @see `ityr::make_checkout()`.
 * @see `ityr::make_global_iterator()`.
 */
inline constexpr write_t write;

/** @brief See `ityr::checkout_mode::read_write`. */
using read_write_t = ori::mode::read_write_t;

/**
 * @brief Read+Write checkout mode.
 * @see `ityr::make_checkout()`.
 * @see `ityr::make_global_iterator()`.
 */
inline constexpr read_write_t read_write;

/** @brief See `ityr::checkout_mode::no_access`. */
struct no_access_t {};

/**
 * @brief Checkout mode to disable automatic checkout.
 * @see `ityr::make_global_iterator()`.
 */
inline constexpr no_access_t no_access;

}

/**
 * @brief Global iterator to enable/disable automatic checkout.
 * @see `ityr::make_global_iterator()`
 */
template <typename T, typename Mode>
class global_iterator : public ori::global_ptr<T> {
  using this_t = global_iterator<T, Mode>;
  using base_t = ori::global_ptr<T>;

public:
  global_iterator(ori::global_ptr<T> ptr, Mode) : base_t(ptr) {}

  using mode = Mode;

  /**
   * @brief True if this global iterator allows automatic checkout.
   */
  static constexpr bool auto_checkout = !std::is_same_v<Mode, checkout_mode::no_access_t>;
};

/**
 * @brief Make a global iterator to enable/disable automatic checkout.
 *
 * @param ptr  Global pointer to be converted to global iterator.
 * @param mode Checkout mode (`ityr::checkout_mode`).
 *
 * @return The global iterator.
 *
 * This function converts a global pointer to a global iterator to enable/disable automatic
 * checkout. If the checkout mode is `read`, `read_write`, or `write`, these iterators are
 * automatically checked out in iterator-based loop functions (e.g., `ityr::for_each()`).
 * Some loop functions in which the access mode is clear from definition (e.g., `ityr::reduce()`)
 * automatically enable automatic checkout. If so, this explicit conversion is not needed.
 *
 * `ityr::checkout_mode::no_access` can be speficied to disable automatic checkout,
 * so that iterator-based loop functions (e.g., `ityr::transform_reduce()`) do not automatically
 * enable automatic checkout. This would be required if parallelism is nested.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v = {1, 2, 3, 4, 5};
 *
 * // vector of vectors {v, v, ..., v} (10 elements)
 * ityr::global_vector<ityr::global_vector<int>> vv(10, v);
 *
 * // Disable automatic checkout in `ityr::transform_reduce()` for nested parallelism
 * int s = ityr::transform_reduce(
 *     ityr::execution::par,
 *     ityr::make_global_iterator(vv.begin(), ityr::checkout_mode::no_access),
 *     ityr::make_global_iterator(vv.end()  , ityr::checkout_mode::no_access),
 *     0, std::plus<int>{},
 *     // A global reference is passed to the user function if automatic checkout is disabled
 *     [](ityr::ori::global_ref<ityr::global_vector<int>> v_ref) {
 *       // Checkout each vector explicitly
 *       auto cs = ityr::make_checkout(&v_ref, 1, ityr::checkout_mode::read);
 *
 *       // Get begin/end iterators to data in each vector
 *       auto v_begin = cs[0].begin();
 *       auto v_end   = cs[0].end();
 *
 *       // Explicit checkin before making parallel calls
 *       cs.checkin();
 *
 *       // Nested reduction for inner elements
 *       return ityr::reduce(ityr::execution::par, v_begin, v_end);
 *     });
 * // s = 150
 * ```
 *
 * @see `ityr::checkout_mode::read`, `ityr::checkout_mode::read_write`, `ityr::checkout_mode::write`,
 *      `ityr::checkout_mode::no_access`
 * @see `ityr::for_each()`
 */
template <typename T, typename Mode>
inline global_iterator<T, Mode> make_global_iterator(ori::global_ptr<T> ptr, Mode mode) {
  return global_iterator(ptr, mode);
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

/**
 * @brief Global iterator for moving objects.
 * @see `ityr::make_move_iterator()`
 */
template <typename T>
class global_move_iterator : public global_iterator<T, checkout_mode::read_write_t> {
  using base_t = global_iterator<T, checkout_mode::read_write_t>;
public:
  explicit global_move_iterator(ori::global_ptr<T> ptr)
    : base_t(ptr, checkout_mode::read_write) {}
};

/**
 * @brief Make a global iterator for moving objects.
 *
 * @param ptr Global pointer to be converted to global iterator.
 *
 * @return The global iterator for moving objects.
 *
 * This function converts a global pointer to a global iterator for moving objects.
 * The region will be checked out with the `ityr::checkout_mode::read_write` mode.
 * After the region is automatically checked out in generic loops, the raw pointers
 * are wrapped with `std::make_move_iterator` to force move semantics.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v = {1, 2, 3, 4, 5};
 *
 * // vector of vectors
 * ityr::global_vector<ityr::global_vector<int>> vv1(10);
 * ityr::global_vector<ityr::global_vector<int>> vv2(10);
 *
 * // `v` is copied to each element of `vv1`
 * ityr::fill(ityr::execution::par, vv1.begin(), vv1.end(), v);
 *
 * // Each element of `vv1` is moved to `vv2`
 * ityr::for_each(
 *     ityr::execution::par,
 *     ityr::make_move_iterator(vv1.begin()),
 *     ityr::make_move_iterator(vv1.end()),
 *     ityr::make_global_iterator(vv2.begin(), ityr::checkout_mode::read_write),
 *     [](ityr::global_vector<int>&& v1, ityr::global_vector<int>& v2) { v2 = std::move(v1); });
 *
 * // Now `vv2` has valid vector elements, while `vv1` does not
 * ```
 *
 * @see [std::make_move_iterator -- cppreference.com](https://en.cppreference.com/w/cpp/iterator/make_move_iterator)
 * @see `ityr::make_global_iterator()`
 */
template <typename T>
inline global_move_iterator<T> make_move_iterator(ori::global_ptr<T> ptr) {
  return global_move_iterator(ptr);
}

/**
 * @brief Global iterator for constructing objects.
 * @see `ityr::make_construct_iterator()`
 */
template <typename T>
class global_construct_iterator : public global_iterator<T, checkout_mode::write_t> {
  using base_t = global_iterator<T, checkout_mode::write_t>;
public:
  explicit global_construct_iterator(ori::global_ptr<T> ptr)
    : base_t(ptr, checkout_mode::write) {}
};

/**
 * @brief Make a global iterator for constructing objects.
 *
 * @param ptr Global pointer to be converted to global iterator.
 *
 * @return The global iterator for constructing objects.
 *
 * This function converts a global pointer to a global iterator for constructing objects.
 * The region will be checked out with the `ityr::checkout_mode::write` mode.
 * Raw pointers (not references) are returned when iterators are dereferenced.
 *
 * @see `ityr::make_global_iterator()`
 * @see `ityr::make_destruct_iterator()`
 */
template <typename T>
inline global_construct_iterator<T> make_construct_iterator(ori::global_ptr<T> ptr) {
  return global_construct_iterator(ptr);
}

/**
 * @brief Global iterator for destructing objects.
 * @see `ityr::make_destruct_iterator()`
 */
template <typename T>
class global_destruct_iterator : public global_iterator<T, checkout_mode::read_write_t> {
  using base_t = global_iterator<T, checkout_mode::read_write_t>;
public:
  explicit global_destruct_iterator(ori::global_ptr<T> ptr)
    : base_t(ptr, checkout_mode::read_write) {}
};

/**
 * @brief Make a global iterator for destructing objects.
 *
 * @param ptr Global pointer to be converted to global iterator.
 *
 * @return The global iterator for destructing objects.
 *
 * This function converts a global pointer to a global iterator for destructing objects.
 * The region will be checked out with the `ityr::checkout_mode::read_write` mode.
 * Raw pointers (not references) are returned when iterators are dereferenced.
 *
 * @see `ityr::make_global_iterator()`
 * @see `ityr::make_construct_iterator()`
 */
template <typename T>
inline global_destruct_iterator<T> make_destruct_iterator(ori::global_ptr<T> ptr) {
  return global_destruct_iterator(ptr);
}

/** @brief See `ityr::is_global_iterator_v`. */
template <typename>
struct is_global_iterator : public std::false_type {};

/** @brief See `ityr::is_global_iterator_v`. */
template <typename T, typename Mode>
struct is_global_iterator<global_iterator<T, Mode>> : public std::true_type {};

/** @brief See `ityr::is_global_iterator_v`. */
template <typename T>
struct is_global_iterator<global_move_iterator<T>> : public std::true_type {};

/** @brief See `ityr::is_global_iterator_v`. */
template <typename T>
struct is_global_iterator<global_construct_iterator<T>> : public std::true_type {};

/** @brief See `ityr::is_global_iterator_v`. */
template <typename T>
struct is_global_iterator<global_destruct_iterator<T>> : public std::true_type {};

/**
 * @brief True if `T` is a global iterator (`ityr::global_iterator`).
 * @see `ityr::make_global_iterator()`.
 */
template <typename T>
inline constexpr bool is_global_iterator_v = is_global_iterator<T>::value;

static_assert(is_global_iterator_v<global_iterator<int, checkout_mode::read_t>>);
static_assert(is_global_iterator_v<global_move_iterator<int>>);
static_assert(!is_global_iterator_v<int>);
static_assert(!is_global_iterator_v<ori::global_ptr<int>>);

}
