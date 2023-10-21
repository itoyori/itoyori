#pragma once

#include "ityr/common/util.hpp"
#include "ityr/ito/ito.hpp"
#include "ityr/ori/ori.hpp"
#include "ityr/pattern/root_exec.hpp"
#include "ityr/pattern/serial_loop.hpp"
#include "ityr/pattern/parallel_loop.hpp"
#include "ityr/pattern/parallel_reduce.hpp"
#include "ityr/container/checkout_span.hpp"

namespace ityr {

/**
 * @brief Options for `ityr::global_vector`.
 * @see `ityr::global_vector`
 */
struct global_vector_options {
  /**
   * @brief A collective global vector is initialized if true.
   */
  bool collective : 1;

  /**
   * @brief Construction of vector elements is parallelized if true.
   */
  bool parallel_construct : 1;

  /**
   * @brief Destruction of vector elements is parallelized if true.
   */
  bool parallel_destruct : 1;

  /**
   * @brief The number of elements for leaf tasks to stop parallel recursion in construction and destruction.
   */
  int cutoff_count : 29;

  constexpr static int default_cutoff_count = 1024;

  global_vector_options()
    : collective(false),
      parallel_construct(false),
      parallel_destruct(false),
      cutoff_count(default_cutoff_count) {}

  explicit global_vector_options(bool collective)
    : collective(collective),
      parallel_construct(collective),
      parallel_destruct(collective),
      cutoff_count(default_cutoff_count) {}

  global_vector_options(bool collective,
                        int  cutoff_count)
    : collective(collective),
      parallel_construct(collective),
      parallel_destruct(collective),
      cutoff_count(cutoff_count) {}

  global_vector_options(bool collective,
                        bool parallel_construct,
                        bool parallel_destruct,
                        int  cutoff_count = default_cutoff_count)
    : collective(collective),
      parallel_construct(parallel_construct),
      parallel_destruct(parallel_destruct),
      cutoff_count(cutoff_count) {}
};

// should be 32 bit long
static_assert(sizeof(global_vector_options) == 4);

/**
 * @brief Global vector to manage a global memory region.
 *
 * A global vector is a container for managing a contiguous global memory region.
 * This resembles the standard `std::vector` container and has some extensions for global memory.
 *
 * As a global vector manages global memory, its elements cannot be directly accessed. Access to
 * its elements must be granted by checkout/checkin operations (e.g., `ityr::make_checkout()`).
 *
 * A global vector can accept `ityr::global_vector_options` as the first argument when initialized.
 * Global vectors have two types (collective or noncollective), which can be configured with the
 * `ityr::global_vector_options::collective` option.
 *
 * - A collective global vector must be allocated and deallocated by all processes collectively,
 *   either in the SPMD region or in the root thread. Its global memory is distributed to the
 *   processes by following the memory distribution policy. Some operations that modify the vector
 *   capacity (e.g., `push_back()`) are not permitted in the fork-join region (except for the root
 *   thread) for collective global vectors.
 * - A noncollective global vector can be independently allocated and deallocated in each process.
 *   Its memory is allocated in the local process and can be deallocated from any other processes.
 *
 * Once allocated, both global vectors can be uniformly accessed by global iterators, for example.
 *
 * Example:
 * ```
 * assert(ityr::is_spmd());
 *
 * // Collective global vector's memory is distributed to all processes
 * // (Note: This example vector is too small to be distributed to multiple processes.)
 * ityr::global_vector<int> v_coll({.collective = true}, {1, 2, 3, 4, 5});
 *
 * // Create a global span to prevent copying the global vector
 * ityr::global_span<int> s_coll(v_coll);
 *
 * ityr::root_exec([=] {
 *   // Noncollective global vector's memory is allocated in the local process
 *   ityr::global_vector<int> v_noncoll = {2, 3, 4, 5, 6};
 *
 *   // Calculate a dot product of the collective and noncollective vectors in parallel
 *   int dot = ityr::transform_reduce(ityr::execution::par,
 *                                    s_coll.begin(), s_coll.end(), v_noncoll.begin(), 0);
 *   // dot = 70
 * });
 * ```
 *
 * In addition, the construction and destruction of vector elements can also be parallelized by
 * setting the `ityr::global_vector_options::parallel_construct` and
 * `ityr::global_vector_options::parallel_destruct` options. The cutoff count for leaf tasks can
 * be configured by the `ityr::global_vector_options::cutoff_count` option.
 * Destruction for elements may be skipped if `T` is trivially destructive.
 *
 * @see [std::vector -- cppreference.com](https://en.cppreference.com/w/cpp/container/vector)
 * @see `ityr::global_vector_options`
 * @see `ityr::global_span`.
 */
template <typename T>
class global_vector {
  using this_t = global_vector;

public:
  using element_type           = T;
  using value_type             = std::remove_cv_t<element_type>;
  using size_type              = std::size_t;
  using pointer                = ori::global_ptr<element_type>;
  using const_pointer          = ori::global_ptr<std::add_const_t<element_type>>;
  using difference_type        = typename std::iterator_traits<pointer>::difference_type;
  using reference              = typename std::iterator_traits<pointer>::reference;
  using const_reference        = typename std::iterator_traits<const_pointer>::reference;
  using iterator               = pointer;
  using const_iterator         = const_pointer;
  using reverse_iterator       = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  global_vector() noexcept
    : global_vector(global_vector_options()) {}

  explicit global_vector(size_type count)
    : global_vector(global_vector_options(), count) {}

  explicit global_vector(size_type count, const value_type& value)
    : global_vector(global_vector_options(), count, value) {}

  template <typename InputIterator>
  global_vector(InputIterator first, InputIterator last)
    : global_vector(global_vector_options(), first, last) {}

  global_vector(std::initializer_list<T> il)
    : global_vector(global_vector_options(), il) {}

  /* with options */

  explicit global_vector(const global_vector_options& opts) noexcept : opts_(opts) {}

  explicit global_vector(const global_vector_options& opts, size_type count) : opts_(opts) {
    initialize_uniform(count);
  }

  explicit global_vector(const global_vector_options& opts, size_type count, const T& value) : opts_(opts) {
    initialize_uniform(count, value);
  }

  template <typename InputIterator,
            typename = std::enable_if_t<std::is_convertible_v<typename std::iterator_traits<InputIterator>::iterator_category,
                                                              std::input_iterator_tag>>>
  global_vector(const global_vector_options& opts, InputIterator first, InputIterator last) : opts_(opts) {
    // TODO: automatic checkout by making global iterators?
    initialize_from_iter(
        first, last,
        typename std::iterator_traits<InputIterator>::iterator_category{});
  }

  global_vector(const global_vector_options& opts, std::initializer_list<T> il) : opts_(opts) {
    initialize_from_iter(il.begin(), il.end(), std::random_access_iterator_tag{});
  }

  ~global_vector() {
    if (begin() != nullptr) {
      destruct_elems(begin(), end());
      free_mem(begin(), capacity());
    }
  }

  global_vector(const this_t& other) : opts_(other.options()) {
    initialize_from_iter(
        make_global_iterator(other.cbegin(), checkout_mode::read),
        make_global_iterator(other.cend()  , checkout_mode::read),
        std::random_access_iterator_tag{});
  }
  this_t& operator=(const this_t& other) {
    // TODO: skip freeing memory and reuse it when it has enough amount of memory
    this->~global_vector();
    // should we copy options?
    opts_ = other.options();
    initialize_from_iter(
        make_global_iterator(other.cbegin(), checkout_mode::read),
        make_global_iterator(other.cend()  , checkout_mode::read),
        std::random_access_iterator_tag{});
    return *this;
  }

  global_vector(this_t&& other) noexcept
    : opts_(other.opts_),
      begin_(other.begin_),
      end_(other.end_),
      reserved_end_(other.reserved_end_) {
    other.begin_ = other.end_ = other.reserved_end_ = nullptr;
  }
  this_t& operator=(this_t&& other) noexcept {
    this->~global_vector();
    opts_         = other.opts_;
    begin_        = other.begin_;
    end_          = other.end_;
    reserved_end_ = other.reserved_end_;
    other.begin_ = other.end_ = other.reserved_end_ = nullptr;
    return *this;
  }

  pointer data() noexcept { return begin_; }
  const_pointer data() const noexcept { return begin_; }

  size_type size() const noexcept { return end_ - begin_; }
  size_type capacity() const noexcept { return reserved_end_ - begin_; }

  global_vector_options options() const noexcept { return opts_; }

  iterator begin() noexcept { return begin_; }
  iterator end() noexcept { return end_; }
  const_iterator begin() const noexcept { return begin_; }
  const_iterator end() const noexcept { return end_; }

  const_iterator cbegin() const noexcept { return ori::const_pointer_cast<std::add_const_t<T>>(begin_); }
  const_iterator cend() const noexcept { return ori::const_pointer_cast<std::add_const_t<T>>(end_); }

  reverse_iterator rbegin() const noexcept { return std::make_reverse_iterator(end()); }
  reverse_iterator rend() const noexcept { return std::make_reverse_iterator(begin()); }
  const_reverse_iterator rbegin() noexcept { return std::make_reverse_iterator(end()); }
  const_reverse_iterator rend() noexcept { return std::make_reverse_iterator(begin()); }

  const_reverse_iterator crbegin() const noexcept { return std::make_reverse_iterator(cend()); }
  const_reverse_iterator crend() const noexcept { return std::make_reverse_iterator(cbegin()); }

  reference operator[](size_type i) {
    ITYR_CHECK(i <= size());
    return *(begin() + i);
  }

  const_reference operator[](size_type i) const {
    ITYR_CHECK(i <= size());
    return *(begin() + i);
  }

  reference at(size_type i) {
    check_range(i);
    return (*this)[i];
  }

  const_reference at(size_type i) const {
    check_range(i);
    return (*this)[i];
  }

  reference front() { ITYR_CHECK(!empty()); return *begin(); }
  reference back() { ITYR_CHECK(!empty()); return *(end() - 1); }
  const_reference front() const { ITYR_CHECK(!empty()); return *begin(); }
  const_reference back() const { ITYR_CHECK(!empty()); return *(end() - 1); }

  bool empty() const noexcept { return size() == 0; }

  void swap(this_t& other) noexcept {
    using std::swap;
    swap(opts_        , other.opts_        );
    swap(begin_       , other.begin_       );
    swap(end_         , other.end_         );
    swap(reserved_end_, other.reserved_end_);
  }

  void clear() {
    if (!empty()) {
      destruct_elems(begin(), end());
      end_ = begin();
    }
  }

  void reserve(size_type new_cap) {
    if (capacity() == 0 && new_cap > 0) {
      begin_        = allocate_mem(new_cap);
      end_          = begin_;
      reserved_end_ = begin_ + new_cap;

    } else if (new_cap > capacity()) {
      realloc_mem(new_cap);
    }
  }

  void resize(size_type count) {
    resize_impl(count);
  }

  void resize(size_type count, const value_type& value) {
    resize_impl(count, value);
  }

  void push_back(const value_type& value) {
    push_back_impl(value);
  }

  void push_back(value_type&& value) {
    push_back_impl(std::move(value));
  }

  template <typename... Args>
  reference emplace_back(Args&&... args) {
    push_back_impl(std::forward<Args>(args)...);
    return back();
  }

  void pop_back() {
    ITYR_CHECK(size() > 0);
    root_exec_if_coll([&] {
      auto cs = make_checkout(end() - 1, 1, checkout_mode::read_write);
      std::destroy_at(&cs[0]);
    });
    --end_;
  }

  iterator insert(const_iterator position, const T& x) {
    return insert_one(position - cbegin(), x);
  }

  iterator insert(const_iterator position, T&& x) {
    return insert_one(position - cbegin(), std::move(x));
  }

  iterator insert(const_iterator position, size_type n, const T& x) {
    return insert_n(position - cbegin(), n, x);
  }

  template <typename InputIterator,
            typename = std::enable_if_t<std::is_convertible_v<typename std::iterator_traits<InputIterator>::iterator_category,
                                                              std::input_iterator_tag>>>
  iterator insert(const_iterator position, InputIterator first, InputIterator last) {
    return insert_iter(position - cbegin(), first, last,
                       typename std::iterator_traits<InputIterator>::iterator_category{});
  }

  iterator insert(const_iterator position, std::initializer_list<T> il) {
    return insert_iter(position - cbegin(), il.begin(), il.end(),
                       std::random_access_iterator_tag{});
  }

  template <typename... Args>
  iterator emplace(const_iterator position, Args&&... args) {
    return insert_iter(position - begin(), std::forward<Args>(args)...);
  }

private:
  void check_range(size_type i) const {
    if (i >= size()) {
      std::stringstream ss;
      ss << "Global vector: Index " << i << " is out of range [0, " << size() << ").";
      throw std::out_of_range(ss.str());
    }
  }

  size_type next_size(size_type least) const {
    return std::max(least, size() * 2);
  }

  pointer allocate_mem(size_type count) const {
    if (opts_.collective) {
      return coll_exec_if_coll([=] {
        return ori::malloc_coll<T>(count);
      });
    } else {
      return ori::malloc<T>(count);
    }
  }

  void free_mem(pointer p, size_type count) const {
    if (opts_.collective) {
      coll_exec_if_coll([=] {
        ori::free_coll<T>(p);
      });
    } else {
      ori::free<T>(p, count);
    }
  }

  template <typename Fn, typename... Args>
  auto root_exec_if_coll(Fn&& fn, Args&&... args) const {
    if (opts_.collective) {
      if (ito::is_spmd()) {
        return root_exec(std::forward<Fn>(fn), std::forward<Args>(args)...);
      } else {
        return std::forward<Fn>(fn)(std::forward<Args>(args)...);
      }
    } else {
      return std::forward<Fn>(fn)(std::forward<Args>(args)...);
    }
  }

  template <typename Fn, typename... Args>
  auto coll_exec_if_coll(Fn&& fn, Args&&... args) const {
    if (opts_.collective) {
      if (ito::is_spmd()) {
        return std::forward<Fn>(fn)(std::forward<Args>(args)...);
      } else if (ito::is_root()) {
        return ito::coll_exec(std::forward<Fn>(fn), std::forward<Args>(args)...);
      } else {
        common::die("Collective operations for ityr::global_vector must be executed on the root thread or SPMD region.");
      }
    } else {
      return std::forward<Fn>(fn)(std::forward<Args>(args)...);
    }
  }

  template <typename... Args>
  void initialize_uniform(size_type count, const Args&... args) {
    begin_        = allocate_mem(count);
    end_          = begin_ + count;
    reserved_end_ = begin_ + count;

    construct_elems(begin(), end(), args...);
  }

  template <typename InputIterator>
  void initialize_from_iter(InputIterator first, InputIterator last, std::input_iterator_tag) {
    ITYR_CHECK(!opts_.collective);
    ITYR_CHECK(!opts_.parallel_construct);

    for (; first != last; ++first) {
      push_back_impl(*first);
    }
  }

  template <typename ForwardIterator>
  void initialize_from_iter(ForwardIterator first, ForwardIterator last, std::forward_iterator_tag) {
    auto d = std::distance(first, last);

    if (d > 0) {
      begin_        = allocate_mem(d);
      end_          = begin_ + d;
      reserved_end_ = begin_ + d;

      construct_elems_from_iter(first, last, begin());

    } else {
      begin_ = end_ = reserved_end_ = nullptr;
    }
  }

  template <typename... Args>
  void construct_elems(pointer b, pointer e, const Args&... args) const {
    root_exec_if_coll([=, opts = opts_]() {
      if (opts.parallel_construct) {
        for_each(
            execution::parallel_policy(opts.cutoff_count),
            make_construct_iterator(b),
            make_construct_iterator(e),
            [=](T* p) { new (p) T(args...); });
      } else {
        for_each(
            execution::sequenced_policy(opts.cutoff_count),
            make_construct_iterator(b),
            make_construct_iterator(e),
            [&](T* p) { new (p) T(args...); });
      }
    });
  }

  template <typename ForwardIterator>
  void construct_elems_from_iter(ForwardIterator first, ForwardIterator last, pointer b) const {
    root_exec_if_coll([=, opts = opts_]() {
      if (opts.parallel_construct) {
        for_each(
            execution::parallel_policy(opts.cutoff_count),
            first,
            last,
            make_construct_iterator(b),
            [](auto&& src, T* p) { new (p) T(std::forward<decltype(src)>(src)); });
      } else {
        for_each(
            execution::sequenced_policy(opts.cutoff_count),
            first,
            last,
            make_construct_iterator(b),
            [](auto&& src, T* p) { new (p) T(std::forward<decltype(src)>(src)); });
      }
    });
  }

  void destruct_elems(pointer b, pointer e) const {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      root_exec_if_coll([=, opts = opts_]() {
        if (opts.parallel_destruct) {
          for_each(
              execution::parallel_policy(opts.cutoff_count),
              make_destruct_iterator(b),
              make_destruct_iterator(e),
              [](T* p) { std::destroy_at(p); });
        } else {
          for_each(
              execution::sequenced_policy(opts.cutoff_count),
              make_destruct_iterator(b),
              make_destruct_iterator(e),
              [](T* p) { std::destroy_at(p); });
        }
      });
    }
  }

  void realloc_mem(size_type count) {
    pointer   old_begin    = begin_;
    pointer   old_end      = end_;
    size_type old_capacity = capacity();

    begin_        = allocate_mem(count);
    end_          = begin_ + (old_end - old_begin);
    reserved_end_ = begin_ + count;

    if (old_end - old_begin > 0) {
      construct_elems_from_iter(
          make_move_iterator(old_begin),
          make_move_iterator(old_end),
          begin());

      destruct_elems(old_begin, old_end);
    }

    if (old_capacity > 0) {
      free_mem(old_begin, old_capacity);
    }
  }

  template <typename... Args>
  void resize_impl(size_type count, const Args&... args) {
    if (count > size()) {
      if (count > capacity()) {
        size_type new_cap = next_size(count);
        realloc_mem(new_cap);
      }
      construct_elems(end(), begin() + count, args...);
      end_ = begin() + count;

    } else if (count < size()) {
      destruct_elems(begin() + count, end());
      end_ = begin() + count;
    }
  }

  template <typename... Args>
  void push_back_impl(Args&&... args) {
    if (size() + 1 > capacity()) {
      size_type new_cap = next_size(size() + 1);
      realloc_mem(new_cap);
    }

    root_exec_if_coll([&] {
      auto cs = make_checkout(end(), 1, checkout_mode::write);
      new (&cs[0]) T(std::forward<Args>(args)...);
    });

    ++end_;
  }

  void make_space_for_insertion(size_type i, size_type n) {
    ITYR_CHECK(i <= size());
    ITYR_CHECK(n > 0);

    if (size() + n > capacity()) {
      size_type new_cap = next_size(size() + n);
      realloc_mem(new_cap);
    }

    construct_elems(end(), end() + n);

    move_backward(
        execution::sequenced_policy(opts_.cutoff_count),
        begin() + i, end(), end() + n);
  }

  template <typename... Args>
  iterator insert_one(size_type i, Args&&... args) {
    if (i == size()) {
      push_back_impl(std::forward<Args>(args)...);
      return begin() + i;
    }

    make_space_for_insertion(i, 1);

    root_exec_if_coll([&] {
      auto cs = make_checkout(begin() + i, 1, internal::dest_checkout_mode_t<T>{});
      cs[0] = T(std::forward<Args>(args)...);
    });

    ++end_;
    return begin() + i;
  }

  iterator insert_n(size_type i, size_type n, const value_type& value) {
    if (n == 0) {
      return begin() + i;
    }

    make_space_for_insertion(i, n);

    root_exec_if_coll([&] {
      fill(execution::sequenced_policy(opts_.cutoff_count),
           begin() + i, begin() + i + n, value);
    });

    end_ += n;
    return begin() + i;
  }

  template <typename InputIterator>
  iterator insert_iter(size_type i, InputIterator first, InputIterator last, std::input_iterator_tag) {
    size_type pos = i;
    for (; first != last; ++first) {
      insert_one(pos++, *first);
    }
    return begin() + i;
  }

  template <typename ForwardIterator>
  iterator insert_iter(size_type i, ForwardIterator first, ForwardIterator last, std::forward_iterator_tag) {
    if (first == last) {
      return begin() + i;
    }

    size_type n = std::distance(first, last);
    make_space_for_insertion(i, n);

    root_exec_if_coll([&] {
      copy(execution::sequenced_policy(opts_.cutoff_count),
           first, last, begin() + i);
    });

    end_ += n;
    return begin() + i;
  }

  global_vector_options opts_;
  pointer               begin_        = nullptr;
  pointer               end_          = nullptr;
  pointer               reserved_end_ = nullptr;
};

template <typename T>
inline void swap(global_vector<T>& v1, global_vector<T>& v2) noexcept {
  v1.swap(v2);
}

template <typename T>
bool operator==(const global_vector<T>& x, const global_vector<T>& y) {
  return equal(
      execution::parallel_policy(x.options().cutoff_count),
      x.begin(), x.end(), y.begin(), y.end());
}

template <typename T>
bool operator!=(const global_vector<T>& x, const global_vector<T>& y) {
  return !(x == y);
}

ITYR_TEST_CASE("[ityr::container::global_vector] test") {
  ito::init();
  ori::init();

  auto my_rank = common::topology::my_rank();
  auto n_ranks = common::topology::n_ranks();

  long n = 10000;

  ITYR_SUBCASE("collective") {
    global_vector<long> gv1(global_vector_options{true, 256},
                            count_iterator<long>(0),
                            count_iterator<long>(n));
    ITYR_CHECK(!gv1.empty());
    ITYR_CHECK(gv1.size() == std::size_t(n));
    ITYR_CHECK(gv1.capacity() >= std::size_t(n));
    root_exec([&] {
      long count = reduce(
          execution::parallel_policy(128),
          gv1.begin(), gv1.end());
      ITYR_CHECK(count == n * (n - 1) / 2);
    });

    ITYR_SUBCASE("copy") {
      global_vector<long> gv2 = gv1;
      root_exec([&] {
        for_each(
            execution::parallel_policy(128),
            make_global_iterator(gv2.begin(), checkout_mode::read_write),
            make_global_iterator(gv2.end()  , checkout_mode::read_write),
            [](long& i) { i *= 2; });

        long count1 = reduce(
            execution::parallel_policy(128),
            gv1.begin(), gv1.end());
        ITYR_CHECK(count1 == n * (n - 1) / 2);

        long count2 = reduce(
            execution::parallel_policy(128),
            gv2.begin(), gv2.end());
        ITYR_CHECK(count2 == n * (n - 1));

        // collective allocation on the root thread
        global_vector<long> gv3 = gv1;

        for_each(
            execution::parallel_policy(128),
            make_global_iterator(gv1.begin(), checkout_mode::read),
            make_global_iterator(gv1.end()  , checkout_mode::read),
            make_global_iterator(gv3.begin(), checkout_mode::read),
            [](long i, long j) { ITYR_CHECK(i == j); });
      });
    }

    ITYR_SUBCASE("move") {
      global_vector<long> gv2 = std::move(gv1);
      ITYR_CHECK(gv1.empty());
      ITYR_CHECK(gv1.capacity() == 0);
      root_exec([&] {
        long count = reduce(
            execution::parallel_policy(128),
            gv2.begin(), gv2.end());
        ITYR_CHECK(count == n * (n - 1) / 2);
      });
    }

    ITYR_SUBCASE("resize") {
      gv1.resize(n * 10, 3);
      root_exec([&] {
        long count = reduce(
            execution::parallel_policy(128),
            gv1.begin(), gv1.end());
        ITYR_CHECK(count == n * (n - 1) / 2 + (n * 9) * 3);
      });
      gv1.resize(n * 5);
      root_exec([&] {
        long count = reduce(
            execution::parallel_policy(128),
            gv1.begin(), gv1.end());
        ITYR_CHECK(count == n * (n - 1) / 2 + (n * 4) * 3);
      });
    }

    ITYR_SUBCASE("clear") {
      gv1.clear();
      ITYR_CHECK(gv1.empty());
      ITYR_CHECK(gv1.capacity() >= std::size_t(n));
    }

    ITYR_SUBCASE("move-only elems") {
      global_vector<common::move_only_t> gv2(global_vector_options{true, 256},
                                             gv1.begin(),
                                             gv1.end());
      long next_size = gv2.capacity() * 2;
      gv2.resize(next_size);
      root_exec([&] {
        long count = transform_reduce(
            execution::parallel_policy(128),
            gv2.begin(),
            gv2.end(),
            reducer::plus<long>{},
            [](const common::move_only_t& mo) { return mo.value(); });

        ITYR_CHECK(count == n * (n - 1) / 2);
      });
    }
  }

  ITYR_SUBCASE("noncollective") {
    global_vector<global_vector<long>> gvs(global_vector_options{true, false, false});

    gvs.resize(n_ranks);

    global_vector<long> gv1;

    for (long i = 0; i < n; i++) {
      gv1.push_back(i);
    }

    gvs[my_rank] = std::move(gv1);

    root_exec([&]() {
      auto check_sum = [&](long ans) {
        long count = transform_reduce(
            execution::par,
            make_global_iterator(gvs.begin(), checkout_mode::no_access),
            make_global_iterator(gvs.end()  , checkout_mode::no_access),
            reducer::plus<long>{},
            [&](auto&& gv_ref) {
              auto cs = make_checkout(&gv_ref, 1, checkout_mode::read_write);
              auto gv_begin = cs[0].begin();
              auto gv_end   = cs[0].end();
              cs.checkin();
              return reduce(execution::parallel_policy(128),
                            gv_begin, gv_end);
            });

        ITYR_CHECK(count == ans);
      };

      check_sum(n * (n - 1) / 2 * n_ranks);

      for_each(
          execution::par,
          make_global_iterator(gvs.begin(), checkout_mode::read_write),
          make_global_iterator(gvs.end()  , checkout_mode::read_write),
          [&](global_vector<long>& gv) {
            for (long i = 0; i < 100; i++) {
              gv.push_back(i);
            }
            for (long i = 0; i < 100; i++) {
              gv.pop_back();
            }
            gv.resize(2 * n);
            for_each(
                execution::sequenced_policy(128),
                count_iterator<long>(n),
                count_iterator<long>(2 * n),
                make_global_iterator(gv.begin() + n, checkout_mode::write),
                [](long i, long& x) { x = i; });
          });

      check_sum((2 * n) * (2 * n - 1) / 2 * n_ranks);
    });
  }

  ITYR_SUBCASE("insert") {
    root_exec([&]() {
      global_vector<int> v1 = {1, 2, 3, 4, 5};
      global_vector<int> v2 = {10, 20, 30};

      v1.insert(v1.begin() + 2, 0);
      ITYR_CHECK(v1 == global_vector<int>({1, 2, 0, 3, 4, 5}));

      v1.insert(v1.end() - 1, 3, -1);
      ITYR_CHECK(v1 == global_vector<int>({1, 2, 0, 3, 4, -1, -1, -1, 5}));

      v1.insert(v1.begin(), v2.begin(), v2.end());
      ITYR_CHECK(v1 == global_vector<int>({10, 20, 30, 1, 2, 0, 3, 4, -1, -1, -1, 5}));
    });
  }

  ITYR_SUBCASE("initializer list") {
    root_exec([&]() {
      global_vector<int> v = {1, 2, 3, 4, 5};

      int product = reduce(execution::par, v.begin(), v.end(), reducer::multiplies<int>{});
      ITYR_CHECK(product == 120);

      v.insert(v.end(), {6, 7, 8});
      ITYR_CHECK(v == global_vector<int>({1, 2, 3, 4, 5, 6, 7, 8}));
    });
  }

  ori::fini();
  ito::fini();
}

}
