#pragma once

#include <iterator>

namespace ityr {

/**
 * @brief Count iterator.
 *
 * A count iterator is a special iterator that has the same dereferenced value as the iterator value.
 *
 * This is particularly useful when used with iterator-based loop functions (e.g., `ityr::for_each()`),
 * as it can be used to represent the index of each iteration in the loop.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v = {1, 2, 3, 4, 5};
 * ityr::for_each(
 *     ityr::execution::seq,
 *     ityr::make_global_iterator(v.begin(), ityr::checkout_mode::read),
 *     ityr::make_global_iterator(v.end()  , ityr::checkout_mode::read),
 *     ityr::count_iterator<std::size_t>(0),
 *     [](int x, std::size_t i) { std::cout << "v[" << i << "] = " << x << std::endl; });
 * // Output:
 * // v[0] = 1
 * // v[1] = 2
 * // v[2] = 3
 * // v[3] = 4
 * // v[4] = 5
 * ```
 */
template <typename T>
class count_iterator {
  using this_t = count_iterator<T>;

public:
  using difference_type   = std::ptrdiff_t;
  using value_type        = T;
  using pointer           = void;
  using reference         = T;
  using iterator_category = std::random_access_iterator_tag;

  count_iterator() {}
  count_iterator(T val) : val_(val) {}

  reference operator*() const { return val_; }

  reference operator[](difference_type diff) const { return val_ + diff; }

  this_t& operator+=(difference_type diff) { val_ += diff; return *this; }
  this_t& operator-=(difference_type diff) { val_ -= diff; return *this; }

  this_t& operator++() { ++val_; return *this; }
  this_t& operator--() { --val_; return *this; }

  this_t operator++(int) { this_t tmp(*this); val_++; return tmp; }
  this_t operator--(int) { this_t tmp(*this); val_--; return tmp; }

  this_t operator+(difference_type diff) const { return val_ + diff; }
  this_t operator-(difference_type diff) const { return val_ - diff; }
  difference_type operator-(const this_t& it) const { return val_ - *it; }

private:
  T val_;
};

template <typename T>
inline bool operator==(const count_iterator<T>& it1, const count_iterator<T>& it2) {
  return *it1 == *it2;
}

template <typename T>
inline bool operator!=(const count_iterator<T>& it1, const count_iterator<T>& it2) {
  return *it1 != *it2;
}

template <typename T>
inline bool operator<(const count_iterator<T>& it1, const count_iterator<T>& it2) {
  return *it1 < *it2;
}

template <typename T>
inline bool operator>(const count_iterator<T>& it1, const count_iterator<T>& it2) {
  return *it1 > *it2;
}

template <typename T>
inline bool operator<=(const count_iterator<T>& it1, const count_iterator<T>& it2) {
  return *it1 <= *it2;
}

template <typename T>
inline bool operator>=(const count_iterator<T>& it1, const count_iterator<T>& it2) {
  return *it1 >= *it2;
}

template <typename T>
inline auto make_count_iterator(T x) {
  return count_iterator(x);
}

}
