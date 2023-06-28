#pragma once

#include "ityr/common/util.hpp"
#include "ityr/ori/ori.hpp"
#include "ityr/pattern/global_iterator.hpp"
#include "ityr/container/global_span.hpp"

namespace ityr {

template <typename T, typename Mode>
class checkout_span {
public:
  using element_type = T;
  using value_type   = std::remove_cv_t<T>;
  using size_type    = std::size_t;
  using pointer      = std::conditional_t<std::is_same_v<Mode, checkout_mode::read_t>, const T*, T*>;
  using iterator     = pointer;
  using reference    = std::conditional_t<std::is_same_v<Mode, checkout_mode::read_t>, const T&, T&>;

  checkout_span() {}
  explicit checkout_span(ori::global_ptr<T> gptr, std::size_t n, Mode)
    : ptr_(ori::checkout(gptr, n, Mode{})), n_(n) {}

  ~checkout_span() { if (ptr_) ori::checkin(ptr_, n_, Mode{}); }

  checkout_span(const checkout_span&) = delete;
  checkout_span& operator=(const checkout_span&) = delete;

  checkout_span(checkout_span&& cs) : ptr_(cs.ptr_), n_(cs.n_) { cs.ptr_ = nullptr; cs.n_ = 0; }
  checkout_span& operator=(checkout_span&& cs) {
    this->~checkout_span();
    ptr_ = cs.ptr_;
    n_   = cs.n_;
    cs.ptr_ = nullptr;
    cs.n_   = 0;
  }

  constexpr pointer data() const noexcept { return ptr_; }
  constexpr size_type size() const noexcept { return n_; }

  constexpr iterator begin() const noexcept { return ptr_; }
  constexpr iterator end() const noexcept { return ptr_ + n_; }

  constexpr reference operator[](size_type i) const { assert(i <= n_); return ptr_[i]; }

  constexpr reference front() const { return *ptr_; }
  constexpr reference back() const { return *(ptr_ + n_ - 1); }

  constexpr bool empty() const noexcept { return n_ == 0; }

  void checkin() {
    ITYR_CHECK(ptr_);
    ori::checkin(ptr_, n_, Mode{});
    ptr_ = nullptr;
    n_   = 0;
  }

private:
  pointer   ptr_ = nullptr;
  size_type n_   = 0;
};

template <typename T, typename Mode>
inline constexpr auto data(const checkout_span<T, Mode>& cs) noexcept {
  return cs.data();
}

template <typename T, typename Mode>
inline constexpr auto size(const checkout_span<T, Mode>& cs) noexcept {
  return cs.size();
}

template <typename T, typename Mode>
inline constexpr auto begin(const checkout_span<T, Mode>& cs) noexcept {
  return cs.begin();
}

template <typename T, typename Mode>
inline constexpr auto end(const checkout_span<T, Mode>& cs) noexcept {
  return cs.end();
}

template <typename T, typename Mode>
inline checkout_span<T, Mode> make_checkout(ori::global_ptr<T> gptr, std::size_t n, Mode mode) {
  return checkout_span<T, Mode>{gptr, n, mode};
}

template <typename T, typename Mode>
inline checkout_span<T, Mode> make_checkout(global_span<T> gspan, Mode mode) {
  return checkout_span<T, Mode>{gspan.data(), gspan.size(), mode};
}

}
