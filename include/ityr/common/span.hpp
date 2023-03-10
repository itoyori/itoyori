#pragma once

#include <cassert>
#include <optional>

#include "ityr/common/util.hpp"

namespace ityr::common {

// TODO: remove it and use std::span in C++20
template <typename T>
class span {
  using this_t = span<T>;

public:
  using element_type = T;
  using value_type   = std::remove_cv_t<T>;
  using size_type    = std::size_t;
  using pointer      = T*;
  using iterator     = pointer;
  using reference    = T&;

  span() {}
  template <typename ContiguousIterator>
  span(ContiguousIterator first, size_type n)
    : ptr_(&(*first)), n_(n) {}
  template <typename ContiguousIterator>
  span(ContiguousIterator first, ContiguousIterator last)
    : ptr_(&(*first)), n_(last - first) {}
  template <typename U>
  span(span<U> s) : ptr_(s.data()), n_(s.size() * sizeof(U) / sizeof(T)) {}

  constexpr pointer data() const noexcept { return ptr_; }
  constexpr size_type size() const noexcept { return n_; }

  constexpr iterator begin() const noexcept { return ptr_; }
  constexpr iterator end() const noexcept { return ptr_ + n_; }

  constexpr reference operator[](size_type i) const { assert(i <= n_); return ptr_[i]; }

  constexpr reference front() const { return *ptr_; }
  constexpr reference back() const { return *(ptr_ + n_ - 1); }

  constexpr bool empty() const noexcept { return n_ == 0; }

  constexpr this_t subspan(size_type offset, size_type count) const {
    assert(offset + count <= n_);
    return {ptr_ + offset, count};
  }

private:
  pointer   ptr_ = nullptr;
  size_type n_   = 0;
};

template <typename T>
inline constexpr auto data(const span<T>& s) noexcept {
  return s.data();
}

template <typename T>
inline constexpr auto size(const span<T>& s) noexcept {
  return s.size();
}

template <typename T>
inline constexpr auto begin(const span<T>& s) noexcept {
  return s.begin();
}

template <typename T>
inline constexpr auto end(const span<T>& s) noexcept {
  return s.end();
}

template <typename T>
inline std::optional<span<T>> intersection(const span<T>& s1, const span<T>& s2) {
  T* b1 = s1.data();
  T* b2 = s2.data();
  T* e1 = s1.data() + s1.size();
  T* e2 = s2.data() + s2.size();
  T* b  = std::max(b1, b2);
  T* e  = std::min(e1, e2);
  if (b < e) {
    return span<T>{b, e - b};
  } else {
    return std::nullopt;
  }
}

}
