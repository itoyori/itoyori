#pragma once

#include <type_traits>
#include <iterator>
#include <cstdint>
#include <unistd.h>

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/ori/util.hpp"
#include "ityr/ori/core.hpp"

namespace ityr::ori {

template <typename T>
class global_ref;

template <typename T>
class global_ptr {
  using this_t = global_ptr<T>;

public:
  using element_type      = T;
  using value_type        = std::remove_cv_t<T>;
  using difference_type   = std::ptrdiff_t;
  using pointer           = T*;
  using reference         = global_ref<T>;
  using iterator_category = std::random_access_iterator_tag;

  global_ptr() {}
  explicit global_ptr(T* ptr) : raw_ptr_(ptr) {}

  global_ptr(const this_t&) = default;
  this_t& operator=(const this_t&) = default;

  global_ptr(std::nullptr_t) {}
  this_t& operator=(std::nullptr_t) { raw_ptr_ = nullptr; return *this; }

  T* raw_ptr() const noexcept { return raw_ptr_; }

  explicit operator bool() const noexcept { return raw_ptr_ != nullptr; }
  bool operator!() const noexcept { return raw_ptr_ == nullptr; }

  reference operator*() const noexcept {
    return *this;
  }

  reference operator[](difference_type diff) const noexcept {
    return this_t(raw_ptr_ + diff);
  }

  this_t& operator+=(difference_type diff) {
    raw_ptr_ += diff;
    return *this;
  }

  this_t& operator-=(difference_type diff) {
    raw_ptr_ -= diff;
    return *this;
  }

  this_t& operator++() { return (*this) += 1; }
  this_t& operator--() { return (*this) -= 1; }

  this_t operator++(int) { this_t tmp(*this); ++(*this); return tmp; }
  this_t operator--(int) { this_t tmp(*this); --(*this); return tmp; }

  this_t operator+(difference_type diff) const noexcept {
    return this_t(raw_ptr_ + diff);
  }

  this_t operator-(difference_type diff) const noexcept {
    return this_t(raw_ptr_ - diff);
  }

  difference_type operator-(const this_t& p) const noexcept {
    return raw_ptr_ - p.raw_ptr();
  }

  template <typename U>
  explicit operator global_ptr<U>() const noexcept {
    return global_ptr<U>(reinterpret_cast<U*>(raw_ptr_));
  }

  void swap(this_t& p) noexcept {
    std::swap(raw_ptr_, p.raw_ptr_);
  }

private:
  T* raw_ptr_ = nullptr;
};

template <typename T>
inline bool operator==(const global_ptr<T>& p1, const global_ptr<T>& p2) noexcept {
  return p1.raw_ptr() == p2.raw_ptr();
}

template <typename T>
inline bool operator==(const global_ptr<T>& p, std::nullptr_t) noexcept {
  return !p;
}

template <typename T>
inline bool operator==(std::nullptr_t, const global_ptr<T>& p) noexcept {
  return !p;
}

template <typename T>
inline bool operator!=(const global_ptr<T>& p1, const global_ptr<T>& p2) noexcept {
  return p1.raw_ptr() != p2.raw_ptr();
}

template <typename T>
inline bool operator!=(const global_ptr<T>& p, std::nullptr_t) noexcept {
  return bool(p);
}

template <typename T>
inline bool operator!=(std::nullptr_t, const global_ptr<T>& p) noexcept {
  return bool(p);
}

template <typename T>
inline bool operator<(const global_ptr<T>& p1, const global_ptr<T>& p2) noexcept {
  return p1.raw_ptr() < p2.raw_ptr();
}

template <typename T>
inline bool operator>(const global_ptr<T>& p1, const global_ptr<T>& p2) noexcept {
  return p1.raw_ptr() > p2.raw_ptr();
}

template <typename T>
inline bool operator<=(const global_ptr<T>& p1, const global_ptr<T>& p2) noexcept {
  return p1.raw_ptr() <= p2.raw_ptr();
}

template <typename T>
inline bool operator>=(const global_ptr<T>& p1, const global_ptr<T>& p2) noexcept {
  return p1.raw_ptr() >= p2.raw_ptr();
}

template <typename T>
inline void swap(global_ptr<T>& p1, global_ptr<T>& p2) noexcept {
  p1.swap(p2);
}

template <typename T>
class global_ref {
  using this_t = global_ref<T>;

public:
  global_ref(const global_ptr<T>& p) : ptr_(p) {}
  global_ref(const this_t& r) : ptr_(r.ptr_) {}

  global_ptr<T> operator&() const noexcept { return ptr_; }

  operator T() const {
    std::remove_const_t<T> ret;
    core::instance::get().get(ptr_.raw_ptr(), &ret, sizeof(T));
    return ret;
  }

  this_t& operator=(const T& v) {
    with_read_write([&](T& this_v) { this_v = v; });
    return *this;
  }

  this_t& operator=(T&& v) {
    with_read_write([&](T& this_v) { this_v = std::move(v); });
    return *this;
  }

  this_t& operator=(const this_t& r) {
    T v = r;
    return (*this = v);
  }

  this_t& operator=(this_t&& r) {
    return (*this = T(r));
  }

  this_t& operator+=(const T& v) {
    with_read_write([&](T& this_v) { this_v += v; });
    return *this;
  }

  this_t& operator-=(const T& v) {
    with_read_write([&](T& this_v) { this_v -= v; });
    return *this;
  }

  this_t& operator++() {
    with_read_write([&](T& this_v) { ++this_v; });
    return *this;
  }

  this_t& operator--() {
    with_read_write([&](T& this_v) { --this_v; });
    return *this;
  }

  T operator++(int) {
    T ret;
    with_read_write([&](T& this_v) { ret = this_v++; });
    return ret;
  }

  T operator--(int) {
    T ret;
    with_read_write([&](T& this_v) { ret = this_v--; });
    return ret;
  }

  void swap(this_t r) {
    PCAS_CHECK(&r != ptr_);
    with_read_write([&](T& this_v) {
      r.with_read_write([&](T& v) {
        using std::swap;
        swap(this_v, v);
      });
    });
  }

private:
  template <typename Fn>
  void with_read_write(Fn&& f) {
    core::instance::get().checkout(ptr_.raw_ptr(), sizeof(T), mode::read_write);
    std::forward<Fn>(f)(*ptr_.raw_ptr());
    core::instance::get().checkin(ptr_.raw_ptr(), sizeof(T), mode::read_write);
  }

  global_ptr<T> ptr_;
};

template <typename T>
inline void swap(global_ref<T> r1, global_ref<T> r2) {
  r1.swap(r2);
}

template <typename T, typename MemberT>
inline global_ref<std::remove_extent_t<MemberT>>
operator->*(global_ptr<T> ptr, MemberT T::* mp) {
  using member_t = std::remove_extent_t<MemberT>;
  member_t* member_ptr = reinterpret_cast<member_t*>(std::addressof(ptr.raw_ptr()->*mp));
  return global_ptr<member_t>(member_ptr);
}

template <typename>
struct is_global_ptr : public std::false_type {};

template <typename T>
struct is_global_ptr<global_ptr<T>> : public std::true_type {};

template <typename T>
inline constexpr bool is_global_ptr_v = is_global_ptr<T>::value;

static_assert(is_global_ptr_v<global_ptr<int>>);
template <typename T> struct test_template_type {};
static_assert(!is_global_ptr_v<test_template_type<int>>);
static_assert(!is_global_ptr_v<int>);

ITYR_TEST_CASE("[ityr::ori::global_ptr] global pointer manipulation") {
  int* a1 = reinterpret_cast<int*>(0x00100000);
  int* a2 = reinterpret_cast<int*>(0x01000000);
  int* a3 = reinterpret_cast<int*>(0x10000000);
  global_ptr<int> p1(a1);
  global_ptr<int> p2(a2);
  global_ptr<int> p3(a3);

  ITYR_SUBCASE("initialization") {
    global_ptr<int> p1_(p1);
    global_ptr<int> p2_ = p1;
    ITYR_CHECK(p1_ == p2_);
    int v = 0;
    global_ptr<int> p3_(&v);
  }

  ITYR_SUBCASE("addition and subtraction") {
    auto p = p1 + 4;
    ITYR_CHECK(p      == global_ptr<int>(a1 + 4));
    ITYR_CHECK(p - 2  == global_ptr<int>(a1 + 2));
    p++;
    ITYR_CHECK(p      == global_ptr<int>(a1 + 5));
    p--;
    ITYR_CHECK(p      == global_ptr<int>(a1 + 4));
    p += 10;
    ITYR_CHECK(p      == global_ptr<int>(a1 + 14));
    p -= 5;
    ITYR_CHECK(p      == global_ptr<int>(a1 + 9));
    ITYR_CHECK(p - p1 == 9);
    ITYR_CHECK(p1 - p == -9);
    ITYR_CHECK(p - p  == 0);
  }

  ITYR_SUBCASE("equality") {
    ITYR_CHECK(p1 == p1);
    ITYR_CHECK(p2 == p2);
    ITYR_CHECK(p3 == p3);
    ITYR_CHECK(p1 != p2);
    ITYR_CHECK(p2 != p3);
    ITYR_CHECK(p3 != p1);
    ITYR_CHECK(p1 + 1 != p1);
    ITYR_CHECK((p1 + 1) - 1 == p1);
  }

  ITYR_SUBCASE("comparison") {
    ITYR_CHECK(p1 < p1 + 1);
    ITYR_CHECK(p1 <= p1 + 1);
    ITYR_CHECK(p1 <= p1);
    ITYR_CHECK(!(p1 < p1));
    ITYR_CHECK(!(p1 + 1 < p1));
    ITYR_CHECK(!(p1 + 1 <= p1));
    ITYR_CHECK(p1 + 1 > p1);
    ITYR_CHECK(p1 + 1 >= p1);
    ITYR_CHECK(p1 >= p1);
    ITYR_CHECK(!(p1 > p1));
    ITYR_CHECK(!(p1 > p1 + 1));
    ITYR_CHECK(!(p1 >= p1 + 1));
  }

  ITYR_SUBCASE("boolean") {
    ITYR_CHECK(p1);
    ITYR_CHECK(p2);
    ITYR_CHECK(p3);
    ITYR_CHECK(!p1 == false);
    ITYR_CHECK(!!p1);
    global_ptr<void> nullp;
    ITYR_CHECK(!nullp);
    ITYR_CHECK(nullp == global_ptr<void>(nullptr));
    ITYR_CHECK(nullp == nullptr);
    ITYR_CHECK(nullptr == nullp);
    ITYR_CHECK(!(nullp != nullptr));
    ITYR_CHECK(!(nullptr != nullp));
    ITYR_CHECK(p1 != nullptr);
    ITYR_CHECK(nullptr != p1);
    ITYR_CHECK(!(p1 == nullptr));
    ITYR_CHECK(!(nullptr == p1));
  }

  ITYR_SUBCASE("dereference") {
    ITYR_CHECK(&(*p1) == p1);
    ITYR_CHECK(&p1[0] == p1);
    ITYR_CHECK(&p1[10] == p1 + 10);
    struct point1 { int x; int y; int z; };
    uintptr_t base_addr = 0x00300000;
    global_ptr<point1> px1(reinterpret_cast<point1*>(base_addr));
    ITYR_CHECK(&(px1->*(&point1::x)) == global_ptr<int>(reinterpret_cast<int*>(base_addr + offsetof(point1, x))));
    ITYR_CHECK(&(px1->*(&point1::y)) == global_ptr<int>(reinterpret_cast<int*>(base_addr + offsetof(point1, y))));
    ITYR_CHECK(&(px1->*(&point1::z)) == global_ptr<int>(reinterpret_cast<int*>(base_addr + offsetof(point1, z))));
    struct point2 { int v[3]; };
    global_ptr<point2> px2(reinterpret_cast<point2*>(base_addr));
    global_ptr<int> pv = &(px2->*(&point2::v));
    ITYR_CHECK(pv == global_ptr<int>(reinterpret_cast<int*>(base_addr)));
    ITYR_CHECK(&pv[0] == global_ptr<int>(reinterpret_cast<int*>(base_addr) + 0));
    ITYR_CHECK(&pv[1] == global_ptr<int>(reinterpret_cast<int*>(base_addr) + 1));
    ITYR_CHECK(&pv[2] == global_ptr<int>(reinterpret_cast<int*>(base_addr) + 2));
  }

  ITYR_SUBCASE("cast") {
    ITYR_CHECK(global_ptr<char>(reinterpret_cast<char*>(p1.raw_ptr())) == static_cast<global_ptr<char>>(p1));
    ITYR_CHECK(static_cast<global_ptr<char>>(p1 + 4) == static_cast<global_ptr<char>>(p1) + 4 * sizeof(int));
    global_ptr<const int> p1_const(p1);
    ITYR_CHECK(static_cast<global_ptr<const int>>(p1) == p1_const);
  }

  ITYR_SUBCASE("swap") {
    auto p1_copy = p1;
    auto p2_copy = p2;
    swap(p1, p2);
    ITYR_CHECK(p1 == p2_copy);
    ITYR_CHECK(p2 == p1_copy);
    p1.swap(p2);
    ITYR_CHECK(p1 == p1_copy);
    ITYR_CHECK(p2 == p2_copy);
  }
}

ITYR_TEST_CASE("[ityr::ori::global_ptr] global reference") {
  common::runtime_options common_opts;
  runtime_options opts;
  common::singleton_initializer<common::topology::instance> topo_;
  constexpr block_size_t bs = core::instance::instance_type::block_size;
  int n_cb = 16;
  common::singleton_initializer<core::instance> core_(n_cb * bs, bs / 4);

  auto my_rank = common::topology::my_rank();
  auto n_ranks = common::topology::n_ranks();
  auto mpicomm = common::topology::mpicomm();

  void* p_local = core::instance::get().malloc(sizeof(int));
  global_ptr<int> gp_local(reinterpret_cast<int*>(p_local));
  *gp_local = 0;

  core::instance::get().release();
  common::mpi_barrier(mpicomm);
  core::instance::get().acquire();

  void* p = p_local;
  for (int i = 0; i < n_ranks; i++) {
    core::instance::get().release();

    auto req_send = common::mpi_isend(&p, 1, (n_ranks + my_rank + 1) % n_ranks, i, mpicomm);
    auto req_recv = common::mpi_irecv(&p, 1, (n_ranks + my_rank - 1) % n_ranks, i, mpicomm);
    common::mpi_wait(req_send);
    common::mpi_wait(req_recv);

    core::instance::get().acquire();

    global_ptr<int> gp(reinterpret_cast<int*>(p));
    ITYR_CHECK(*gp == i);
    gp++;
    gp--;
    gp += 2;
    gp -= 2;
    *gp = *gp + 3;
    *gp = *gp - 3;
    *gp = (*gp * 2) / 2 - 1 + 2;
  }

  core::instance::get().release();
  common::mpi_barrier(mpicomm);
  core::instance::get().acquire();

  ITYR_CHECK(*gp_local == n_ranks);

  core::instance::get().free(p_local, sizeof(int));
}

}
