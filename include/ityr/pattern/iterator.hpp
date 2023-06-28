#pragma once

#include <iterator>

namespace ityr {

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
