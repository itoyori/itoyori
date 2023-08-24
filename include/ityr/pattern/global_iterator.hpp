#pragma once

#include "ityr/common/util.hpp"
#include "ityr/ori/ori.hpp"
#include "ityr/pattern/count_iterator.hpp"
#include "ityr/container/checkout_span.hpp"

namespace ityr {

namespace internal {

template <typename GPtr, typename Mode>
using checkout_iterator_t =
  std::conditional_t<std::is_same_v<Mode, checkout_mode::no_access_t>,
                     GPtr,
                     std::conditional_t<std::is_same_v<Mode, checkout_mode::read_t>,
                                        const typename GPtr::value_type*,
                                        typename GPtr::value_type*>>;

template <typename T>
using source_checkout_mode = std::conditional_t<std::is_trivially_copyable_v<T>,
                                                checkout_mode::read_t,
                                                checkout_mode::read_write_t>;

template <typename T>
using destination_checkout_mode = std::conditional_t<std::is_trivially_copyable_v<T>,
                                                     checkout_mode::write_t,
                                                     checkout_mode::read_write_t>;

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
  using value_type        = typename base_t::value_type;
  using difference_type   = typename base_t::difference_type;
  using pointer           = typename base_t::pointer;
  using reference         = typename base_t::reference;
  using iterator_category = typename base_t::iterator_category;
  using mode              = Mode;
  using checkout_iterator = internal::checkout_iterator_t<base_t, Mode>;

  global_iterator(ori::global_ptr<T> gptr, Mode) : base_t(gptr) {}

  auto checkout_nb(std::size_t count) const {
    if constexpr(std::is_same_v<mode, checkout_mode::no_access_t>) {
      return std::make_tuple(nullptr, base_t(*this));
    } else {
      checkout_span<T, mode> cs;
      cs.checkout_nb(base_t(*this), count, mode{});
      return std::make_tuple(std::move(cs), cs.data());
    }
  }
};

/**
 * @brief Make a global iterator to enable/disable automatic checkout.
 *
 * @param gptr Global pointer to be converted to global iterator.
 * @param mode Checkout mode (`ityr::checkout_mode`).
 *
 * @return The global iterator (`ityr::global_iterator`).
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
inline global_iterator<T, Mode>
make_global_iterator(ori::global_ptr<T> gptr, Mode mode) {
  return global_iterator(gptr, mode);
}

/** @brief See `ityr::is_global_iterator_v`. */
template <typename T, typename = void>
struct is_global_iterator : public std::false_type {};

/** @brief See `ityr::is_global_iterator_v`. */
template <typename T>
struct is_global_iterator<T, std::void_t<typename T::checkout_iterator>> : public std::true_type {};

/**
 * @brief True if `T` is a global iterator (`ityr::global_iterator`).
 * @see `ityr::make_global_iterator()`.
 */
template <typename T>
inline constexpr bool is_global_iterator_v = is_global_iterator<T>::value;


/**
 * @brief Global iterator for moving objects.
 * @see `ityr::make_move_iterator()`
 */
template <typename GlobalIterator>
class global_move_iterator : public GlobalIterator {
  using base_t = GlobalIterator;

  static_assert(std::is_same_v<typename GlobalIterator::mode,
                               internal::source_checkout_mode<typename GlobalIterator::value_type>>);

public:
  using value_type        = typename base_t::value_type;
  using difference_type   = typename base_t::difference_type;
  using pointer           = typename base_t::pointer;
  using reference         = typename base_t::reference;
  using iterator_category = typename base_t::iterator_category;
  using mode              = typename GlobalIterator::mode;
  using checkout_iterator = std::move_iterator<typename GlobalIterator::checkout_iterator>;

  explicit global_move_iterator(GlobalIterator git)
    : base_t(git) {}

  GlobalIterator base() const {
    return static_cast<base_t>(*this);
  }

  auto checkout_nb(std::size_t count) const {
    auto&& [cs, it] = base_t::checkout_nb(count);
    return std::make_tuple(std::move(cs), std::make_move_iterator(it));
  }
};

/**
 * @brief Make a global iterator for moving objects.
 *
 * @param gptr Global pointer to be converted to global iterator.
 *
 * @return The global move iterator (`ityr::global_move_iterator`).
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
inline global_move_iterator<global_iterator<T, internal::source_checkout_mode<T>>>
make_move_iterator(ori::global_ptr<T> gptr) {
  return global_move_iterator(make_global_iterator(gptr, internal::source_checkout_mode<T>{}));
}

/**
 * @brief Reverse iterator for global memory.
 * @see `ityr::make_reverse_iterator()`
 */
template <typename GlobalIterator>
class global_reverse_iterator : public std::reverse_iterator<GlobalIterator> {
  using base_t = std::reverse_iterator<GlobalIterator>;

public:
  using value_type        = typename base_t::value_type;
  using difference_type   = typename base_t::difference_type;
  using pointer           = typename base_t::pointer;
  using reference         = typename base_t::reference;
  using iterator_category = typename base_t::iterator_category;
  using mode              = typename GlobalIterator::mode;
  using checkout_iterator = std::reverse_iterator<typename GlobalIterator::checkout_iterator>;

  explicit global_reverse_iterator(GlobalIterator git)
    : base_t(git) {}

  GlobalIterator base() const {
    return static_cast<base_t>(*this).base();
  }

  auto checkout_nb(std::size_t count) const {
    GlobalIterator git = base();
    auto&& [cs, it] = std::prev(git, count).checkout_nb(count);
    return std::make_tuple(std::move(cs), std::make_reverse_iterator(std::next(it, count)));
  }
};

/**
 * @brief Make a reverse iterator for global memory.
 *
 * @param gptr Global pointer to be converted to global iterator.
 * @param mode Checkout mode (`ityr::checkout_mode`).
 *
 * @return The global move iterator (`ityr::global_move_iterator`).
 *
 * This function converts a global pointer to a reverse global iterator with `mode`.
 *
 * @see [std::make_reverse_iterator -- cppreference.com](https://en.cppreference.com/w/cpp/iterator/make_reverse_iterator)
 * @see `ityr::make_global_iterator()`
 */
template <typename T, typename Mode>
inline global_reverse_iterator<global_iterator<T, Mode>>
make_reverse_iterator(ori::global_ptr<T> gptr, Mode mode) {
  return global_reverse_iterator(make_global_iterator(gptr, mode));
}

/**
 * @brief Global iterator for constructing objects.
 * @see `ityr::make_construct_iterator()`
 */
template <typename GlobalIterator>
class global_construct_iterator : public GlobalIterator {
  using base_t = GlobalIterator;

  static_assert(std::is_same_v<typename GlobalIterator::mode, checkout_mode::write_t>);

public:
  using value_type        = typename base_t::value_type;
  using difference_type   = typename base_t::difference_type;
  using pointer           = typename base_t::pointer;
  using reference         = typename base_t::reference;
  using iterator_category = typename base_t::iterator_category;
  using mode              = typename GlobalIterator::mode;
  using checkout_iterator = count_iterator<typename GlobalIterator::checkout_iterator>;

  explicit global_construct_iterator(GlobalIterator git)
    : base_t(git) {}

  GlobalIterator base() const {
    return static_cast<base_t>(*this);
  }

  auto checkout_nb(std::size_t count) const {
    auto&& [cs, it] = base_t::checkout_nb(count);
    return std::make_tuple(std::move(cs), make_count_iterator(it));
  }
};

/**
 * @brief Make a global iterator for constructing objects.
 *
 * @param git Global iterator (see `ityr::make_global_iterator`).
 *
 * @return The global iterator for constructing objects.
 *
 * @see `ityr::make_global_iterator()`
 * @see `ityr::make_destruct_iterator()`
 */
template <typename GlobalIterator>
inline global_construct_iterator<GlobalIterator>
make_construct_iterator(GlobalIterator git) {
  return global_construct_iterator(git);
}

/**
 * @brief Make a global iterator for constructing objects.
 *
 * @param gptr Global pointer to be converted to global iterator.
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
inline global_construct_iterator<global_iterator<T, checkout_mode::write_t>>
make_construct_iterator(ori::global_ptr<T> gptr) {
  return global_construct_iterator(make_global_iterator(gptr, checkout_mode::write));
}

/**
 * @brief Global iterator for destructing objects.
 * @see `ityr::make_destruct_iterator()`
 */
template <typename GlobalIterator>
class global_destruct_iterator : public GlobalIterator {
  using base_t = GlobalIterator;

  static_assert(std::is_same_v<typename GlobalIterator::mode, checkout_mode::read_write_t>);

public:
  using value_type        = typename base_t::value_type;
  using difference_type   = typename base_t::difference_type;
  using pointer           = typename base_t::pointer;
  using reference         = typename base_t::reference;
  using iterator_category = typename base_t::iterator_category;
  using mode              = typename GlobalIterator::mode;
  using checkout_iterator = count_iterator<typename GlobalIterator::checkout_iterator>;

  explicit global_destruct_iterator(GlobalIterator git)
    : base_t(git) {}

  GlobalIterator base() const {
    return static_cast<base_t>(*this);
  }

  auto checkout_nb(std::size_t count) const {
    auto&& [cs, it] = base_t::checkout_nb(count);
    return std::make_tuple(std::move(cs), make_count_iterator(it));
  }
};

/**
 * @brief Make a global iterator for destructing objects.
 *
 * @param git Global iterator (see `ityr::make_global_iterator`).
 *
 * @return The global iterator for destructing objects.
 *
 * @see `ityr::make_global_iterator()`
 * @see `ityr::make_construct_iterator()`
 */
template <typename GlobalIterator>
inline global_destruct_iterator<GlobalIterator>
make_destruct_iterator(GlobalIterator git) {
  return global_destruct_iterator(git);
}

/**
 * @brief Make a global iterator for destructing objects.
 *
 * @param gptr Global pointer to be converted to global iterator.
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
inline global_destruct_iterator<global_iterator<T, checkout_mode::read_write_t>>
make_destruct_iterator(ori::global_ptr<T> gptr) {
  return global_destruct_iterator(make_global_iterator(gptr, checkout_mode::read_write));
}

// Definitions for ADL

/**
 * @brief Make a global iterator for moving objects.
 *
 * @param git Global iterator (`ityr::global_iterator`).
 *
 * @return The global move iterator (`ityr::global_move_iterator`).
 *
 * @see [std::make_move_iterator -- cppreference.com](https://en.cppreference.com/w/cpp/iterator/make_move_iterator)
 * @see `ityr::make_global_iterator()`
 */
template <typename T, typename Mode>
inline global_move_iterator<global_iterator<T, Mode>>
make_move_iterator(global_iterator<T, Mode> git) {
  return global_move_iterator(git);
}

/**
 * @brief Make a global iterator for moving objects.
 *
 * @param git Global iterator (`ityr::global_reverse_iterator`).
 *
 * @return The global move iterator (`ityr::global_move_iterator`).
 *
 * @see [std::make_move_iterator -- cppreference.com](https://en.cppreference.com/w/cpp/iterator/make_move_iterator)
 * @see `ityr::make_global_iterator()`
 */
template <typename GlobalIterator>
inline global_move_iterator<global_reverse_iterator<GlobalIterator>>
make_move_iterator(global_reverse_iterator<GlobalIterator> git) {
  return global_move_iterator(git);
}

/**
 * @brief Make a reverse iterator for global memory.
 *
 * @param git Global iterator (`ityr::global_iterator`).
 *
 * @return The global reverse iterator (`ityr::global_reverse_iterator`).
 *
 * @see [std::make_reverse_iterator -- cppreference.com](https://en.cppreference.com/w/cpp/iterator/make_reverse_iterator)
 * @see `ityr::make_global_iterator()`
 */
template <typename T, typename Mode>
inline global_reverse_iterator<global_iterator<T, Mode>>
make_reverse_iterator(global_iterator<T, Mode> git) {
  return global_reverse_iterator(git);
}

/**
 * @brief Make a reverse iterator for global memory.
 *
 * @param git Global iterator (`ityr::global_move_iterator`).
 *
 * @return The global reverse iterator (`ityr::global_reverse_iterator`).
 *
 * @see [std::make_reverse_iterator -- cppreference.com](https://en.cppreference.com/w/cpp/iterator/make_reverse_iterator)
 * @see `ityr::make_global_iterator()`
 */
template <typename GlobalIterator>
inline global_reverse_iterator<global_move_iterator<GlobalIterator>>
make_reverse_iterator(global_move_iterator<GlobalIterator> git) {
  return global_reverse_iterator(git);
}

static_assert(is_global_iterator_v<global_iterator<int, checkout_mode::read_write_t>>);
static_assert(is_global_iterator_v<global_move_iterator<global_iterator<int, checkout_mode::read_t>>>);
static_assert(is_global_iterator_v<global_reverse_iterator<global_iterator<int, checkout_mode::read_write_t>>>);
static_assert(is_global_iterator_v<global_move_iterator<global_reverse_iterator<global_iterator<int, checkout_mode::read_t>>>>);
static_assert(!is_global_iterator_v<int>);
static_assert(!is_global_iterator_v<ori::global_ptr<int>>);

}
