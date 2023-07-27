#pragma once

#include "ityr/common/util.hpp"
#include "ityr/ori/ori.hpp"
#include "ityr/pattern/global_iterator.hpp"
#include "ityr/container/global_span.hpp"

namespace ityr {

/**
 * @brief Checkout span to automatically manage the lifetime of checked-out memory.
 *
 * A global memory region can be checked out at the constructor and checked in at the destructor.
 * The checkout span can be moved but cannot be copied, in order to ensure the checkin operation is
 * performed only once.
 * The checkout span can be used as in `std::span` (C++20) to access elements in the checked-out
 * memory region.
 *
 * `ityr::make_checkout()` is a helper function to create the checkout span.
 *
 * @see `ityr::make_checkout()`.
 */
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

  /**
   * @brief Construct a checkout span by checking out a global memory region.
   */
  explicit checkout_span(ori::global_ptr<T> gptr, std::size_t n, Mode)
    : ptr_(ori::checkout(gptr, n, Mode{})), n_(n) {}

  /**
   * @brief Perform the checkin operation when destroyed.
   */
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

  /**
   * @brief Manually perform the checkout operation by checking in the previous span.
   */
  void checkout(ori::global_ptr<T> gptr, std::size_t n, Mode) {
    if (ptr_) {
      checkin();
    }
    ptr_ = ori::checkout(gptr, n, Mode{});
    n_ = n;
  }

  /**
   * @brief Manually perform the nonblocking checkout operation by checking in the previous span.
   */
  void checkout_nb(ori::global_ptr<T> gptr, std::size_t n, Mode) {
    if (ptr_) {
      checkin();
    }
    ptr_ = ori::checkout_nb(gptr, n, Mode{});
    n_ = n;
  }

  /**
   * @brief Manually perform the checkin operation by discarding the current checkout span.
   */
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

/**
 * @brief Checkout a global memory region.
 *
 * @param gptr Starting global pointer.
 * @param n    Number of elements to be checked out.
 * @param mode Checkout mode (`ityr::checkout_mode`).
 *
 * @return The checkout span `ityr::checkout_span` for the specified range.
 *
 * This function checks out the global memory range `[gptr, gptr + n)`.
 * After this call, this virtual memory region becomes directly accessible by CPU load/store
 * instructions. In programs, it is recommended to access the memory via the returned checkout span.
 *
 * The checkout mode `mode` can be either `read`, `read_write`, or `write`.
 * - If `read` or `read_write`, the checked-out region has the latest data.
 * - If `read_write` or `write`, the entire checked-out region is considered modified.
 *
 * The checkout span automatically performs a checkin operation when destroyed (e.g., when exiting
 * the scope). The lifetime of the checkout span cannot overlap with any fork/join call, because threads
 * can be migrated and a pair of checkout and checkin calls must be performed in the same process.
 *
 * Overlapping regions can be checked out by multiple processes at the same time, as long as no data
 * race occurs (i.e., all regions are checked out with `ityr::checkout_mode::read`).
 * Within each process, multiple regions can be simultaneously checked out with an arbitrary mode,
 * and the memory ordering to the checked-out region is determined to the program order (because
 * the same memory view is exposed to the process).
 *
 * Example:
 * ```
 * ityr::global_vector<int> v = {1, 2, 3, 4, 5};
 * {
 *   auto cs = ityr::make_checkout(v.data(), v.size(), ityr::checkout_mode::read);
 *   for (int i : cs) {
 *     std::cout << i << " ";
 *   }
 *   std::cout << std::endl;
 *   // automatic checkin when `cs` is destroyed
 * }
 * // Output: 1 2 3 4 5
 * ```
 *
 * @see [std::span -- cppreference.com](https://en.cppreference.com/w/cpp/container/span)
 * @see `ityr::checkout_mode::read`, `ityr::checkout_mode::read_write`, `ityr::checkout_mode::write`
 * @see `ityr::make_global_iterator()`
 */
template <typename T, typename Mode>
inline checkout_span<T, Mode> make_checkout(ori::global_ptr<T> gptr, std::size_t n, Mode mode) {
  return checkout_span<T, Mode>{gptr, n, mode};
}

/**
 * @brief Checkout a global memory region.
 *
 * @param gspan Global span to be checked out.
 * @param mode  Checkout mode (`ityr::checkout_mode`).
 *
 * @return The checkout span `ityr::checkout_span` for the specified range.
 *
 * Equivalent to `ityr::make_checkout(gspan.data(), gspan.size(), mode)`.
 */
template <typename T, typename Mode>
inline checkout_span<T, Mode> make_checkout(global_span<T> gspan, Mode mode) {
  return checkout_span<T, Mode>{gspan.data(), gspan.size(), mode};
}

namespace internal {

inline auto make_checkouts_aux() {
  return std::make_tuple();
}

template <typename T, typename Mode, typename... Rest>
inline auto make_checkouts_aux(ori::global_ptr<T> gptr, std::size_t n, Mode mode, Rest&&... rest) {
  checkout_span<T, Mode> cs;
  cs.checkout_nb(gptr, n, mode);
  return std::tuple_cat(std::make_tuple(std::move(cs)),
                        make_checkouts_aux(std::forward<Rest>(rest)...));
}

template <typename T, typename Mode, typename... Rest>
inline auto make_checkouts_aux(global_span<T> gspan, Mode mode, Rest&&... rest) {
  checkout_span<T, Mode> cs;
  cs.checkout_nb(gspan.data(), gspan.size(), mode);
  return std::tuple_cat(std::make_tuple(std::move(cs)),
                        make_checkouts_aux(std::forward<Rest>(rest)...));
}

}

/**
 * @brief Checkout multiple global memory regions.
 *
 * @param args... Sequence of checkout requests. Each checkout request should be in the form of
 *                `<global_ptr>, <num_elems>, <checkout_mode>` or `<global_span>, <checkout_mode>`.
 *
 * @return A tuple collecting the checkout spans for each checkout request.
 *
 * This function performs multiple checkout operations at the same time.
 * This may improve performance by overlapping communication to fetch remote data, compared to
 * checking out one by one.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v1 = {1, 2, 3, 4, 5};
 * ityr::global_vector<int> v2 = {2, 3, 4, 5, 6};
 * ityr::global_vector<int> v3(10);
 *
 * ityr::global_span<int> s2(v2.begin(), v2.end());
 *
 * auto [cs1, cs2, cs3] =
 *   ityr::make_checkouts(
 *       v1.data(), v1.size(), ityr::checkout_mode::read,
 *       s2, ityr::checkout_mode::read_write,
 *       v3.data() + 2, 3, ityr::checkout_mode::write);
 * ```
 *
 * @see `ityr::make_checkout()`
 */
template <typename... Args>
inline auto make_checkouts(Args&&... args) {
  auto css = internal::make_checkouts_aux(std::forward<Args>(args)...);
  ori::checkout_complete();
  return css;
}

}
