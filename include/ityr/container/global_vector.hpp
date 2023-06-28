#include "ityr/common/util.hpp"
#include "ityr/ito/ito.hpp"
#include "ityr/ori/ori.hpp"
#include "ityr/pattern/root_exec.hpp"
#include "ityr/pattern/parallel_loop.hpp"
#include "ityr/container/checkout_span.hpp"

namespace ityr {

struct global_vector_options {
  bool        collective         = false;
  bool        parallel_construct = false;
  bool        parallel_destruct  = false;
  std::size_t cutoff_count       = 1024;
};

template <typename T>
class global_vector {
  using this_t = global_vector;

public:
  using value_type      = T;
  using size_type       = std::size_t;
  using pointer         = ori::global_ptr<T>;
  using const_pointer   = ori::global_ptr<std::add_const_t<T>>;
  using iterator        = pointer;
  using const_iterator  = const_pointer;
  using difference_type = typename std::iterator_traits<pointer>::difference_type;
  using reference       = typename std::iterator_traits<pointer>::reference;
  using const_reference = typename std::iterator_traits<const_pointer>::reference;

  global_vector() : global_vector(global_vector_options()) {}
  explicit global_vector(size_type count) : global_vector(global_vector_options(), count) {}
  explicit global_vector(size_type count, const T& value) : global_vector(global_vector_options(), count, value) {}
  template <typename InputIterator>
  global_vector(InputIterator first, InputIterator last) : global_vector(global_vector_options(), first, last) {}

  explicit global_vector(const global_vector_options& opts) : opts_(opts) {}

  explicit global_vector(const global_vector_options& opts, size_type count) : opts_(opts) {
    initialize_uniform(count);
  }

  explicit global_vector(const global_vector_options& opts, size_type count, const T& value) : opts_(opts) {
    initialize_uniform(count, value);
  }

  template <typename InputIterator>
  global_vector(const global_vector_options& opts, InputIterator first, InputIterator last) : opts_(opts) {
    initialize_from_iter(first, last,
                         typename std::iterator_traits<InputIterator>::iterator_category());
  }

  ~global_vector() {
    if (begin() != nullptr) {
      destruct_elems(begin(), end());
      free_mem(begin(), capacity());
    }
  }

  global_vector(const this_t& other) : opts_(other.options()) {
    initialize_from_iter(other.cbegin(), other.cend(), std::random_access_iterator_tag{});
  }
  this_t& operator=(const this_t& other) {
    // TODO: skip freeing memory and reuse it when it has enough amount of memory
    this->~global_vector();
    // should we copy options?
    opts_ = other.options();
    initialize_from_iter(other.cbegin(), other.cend(), std::random_access_iterator_tag{});
    return *this;
  }

  global_vector(this_t&& other)
    : opts_(other.opts_),
      begin_(other.begin_),
      end_(other.end_),
      reserved_end_(other.reserved_end_) {
    other.begin_ = other.end_ = other.reserved_end_ = nullptr;
  }
  this_t& operator=(this_t&& other) {
    this->~global_vector();
    opts_         = other.opts_;
    begin_        = other.begin_;
    end_          = other.end_;
    reserved_end_ = other.reserved_end_;
    other.begin_ = other.end_ = other.reserved_end_ = nullptr;
    return *this;
  }

  pointer data() const noexcept { return begin_; }
  size_type size() const noexcept { return end_ - begin_; }
  size_type capacity() const noexcept { return reserved_end_ - begin_; }

  global_vector_options options() const noexcept { return opts_; }

  iterator begin() const noexcept { return begin_; }
  iterator end() const noexcept { return end_; }

  const_iterator cbegin() const noexcept { return ori::const_pointer_cast<std::add_const_t<T>>(begin_); }
  const_iterator cend() const noexcept { return ori::const_pointer_cast<std::add_const_t<T>>(end_); }

  reference operator[](size_type i) const {
    ITYR_CHECK(i <= size());
    return *(begin() + i);
  }
  reference at(size_type i) const {
    if (i >= size()) {
      std::stringstream ss;
      ss << "Global vector: Index " << i << " is out of range [0, " << size() << ").";
      throw std::out_of_range(ss.str());
    }
    return (*this)[i];
  }

  reference front() const { return *begin(); }
  reference back() const { return *(end() - 1); }

  bool empty() const noexcept { return size() == 0; }

  void swap(this_t& other) noexcept {
    using std::swap;
    swap(opts_        , other.opts_        );
    swap(begin_       , other.begin_       );
    swap(end_         , other.end_         );
    swap(reserved_end_, other.reserved_end_);
  }

  void clear() {
    destruct_elems(begin(), end());
    end_ = begin();
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
    ITYR_CHECK(!opts_.collective);
    ITYR_CHECK(size() > 0);
    auto cs = make_checkout(end() - 1, 1, checkout_mode::read_write);
    std::destroy_at(&cs[0]);
    --end_;
  }

private:
  size_type next_size(size_type least) const {
    return std::max(least, size() * 2);
  }

  pointer allocate_mem(size_type count) const {
    if (opts_.collective) {
      ITYR_CHECK(ito::is_spmd());
      return ori::malloc_coll<T>(count);
    } else {
      return ori::malloc<T>(count);
    }
  }

  void free_mem(pointer p, size_type count) const {
    if (opts_.collective) {
      ITYR_CHECK(ito::is_spmd());
      ori::free_coll<T>(p);
    } else {
      ori::free<T>(p, count);
    }
  }

  template <typename Fn, typename... Args>
  auto root_exec_if_coll(Fn&& f, Args&&... args) const {
    if (opts_.collective) {
      ITYR_CHECK(ito::is_spmd());
      return root_exec(std::forward<Fn>(f), std::forward<Args>(args)...);
    } else {
      return std::forward<Fn>(f)(std::forward<Args>(args)...);
    }
  }

  template <typename... Args>
  void initialize_uniform(size_type count, Args&&... args) {
    begin_        = allocate_mem(count);
    end_          = begin_ + count;
    reserved_end_ = begin_ + count;

    construct_elems(begin(), end(), std::forward<Args>(args)...);
  }

  template <typename InputIterator>
  void initialize_from_iter(InputIterator first, InputIterator last, std::input_iterator_tag) {
    ITYR_CHECK(!opts_.collective);
    ITYR_CHECK(!opts_.parallel_construct);

    for (; first != last; ++first) {
      emplace_back(*first);
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
  void construct_elems(pointer b, pointer e, Args&&... args) const {
    root_exec_if_coll([=, opts = opts_]() {
      if (opts.parallel_construct) {
        parallel_for_each({.cutoff_count = opts.cutoff_count, .checkout_count = opts.cutoff_count},
                          make_global_iterator(b, checkout_mode::write),
                          make_global_iterator(e, checkout_mode::write),
                          [=](T& x) { new (&x) T(args...); });
      } else {
        serial_for_each({.checkout_count = opts.cutoff_count},
                        make_global_iterator(b, checkout_mode::write),
                        make_global_iterator(e, checkout_mode::write),
                        [&](T& x) { new (&x) T(args...); });
      }
    });
  }

  template <typename ForwardIterator>
  void construct_elems_from_iter(ForwardIterator first, ForwardIterator last, pointer b) const {
    root_exec_if_coll([=, opts = opts_]() {
      if (opts.parallel_construct) {
        parallel_for_each({.cutoff_count = opts.cutoff_count, .checkout_count = opts.cutoff_count},
                          first,
                          last,
                          make_global_iterator(b, checkout_mode::write),
                          [](auto&& src, T& x) { new (&x) T(std::forward<decltype(src)>(src)); });
      } else {
        serial_for_each({.checkout_count = opts.cutoff_count},
                        first,
                        last,
                        make_global_iterator(b, checkout_mode::write),
                        [](auto&& src, T& x) { new (&x) T(std::forward<decltype(src)>(src)); });
      }
    });
  }

  void destruct_elems(pointer b, pointer e) const {
    if constexpr (!std::is_trivially_destructible_v<T>) {
      root_exec_if_coll([=, opts = opts_]() {
        if (opts.parallel_destruct) {
          parallel_for_each({.cutoff_count = opts.cutoff_count, .checkout_count = opts.cutoff_count},
                            make_global_iterator(b, checkout_mode::read_write),
                            make_global_iterator(e, checkout_mode::read_write),
                            [](T& x) { std::destroy_at(&x); });
        } else {
          serial_for_each({.checkout_count = opts.cutoff_count},
                          make_global_iterator(b, checkout_mode::read_write),
                          make_global_iterator(e, checkout_mode::read_write),
                          [](T& x) { std::destroy_at(&x); });
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
      construct_elems_from_iter(make_move_iterator(old_begin),
                                make_move_iterator(old_end),
                                begin());

      destruct_elems(old_begin, old_end);
    }

    if (old_capacity > 0) {
      free_mem(old_begin, old_capacity);
    }
  }

  template <typename... Args>
  void resize_impl(size_type count, Args&&... args) {
    if (count > size()) {
      if (count > capacity()) {
        size_type new_cap = next_size(count);
        realloc_mem(new_cap);
      }
      construct_elems(end(), begin() + count, std::forward<Args>(args)...);
      end_ = begin() + count;

    } else if (count < size()) {
      destruct_elems(begin() + count, end());
      end_ = begin() + count;
    }
  }

  template <typename... Args>
  void push_back_impl(Args&&... args) {
    ITYR_CHECK(!opts_.collective);
    if (size() == capacity()) {
      size_type new_cap = next_size(size() + 1);
      realloc_mem(new_cap);
    }
    auto cs = make_checkout(end(), 1, checkout_mode::write);
    new (&cs[0]) T(std::forward<Args>(args)...);
    ++end_;
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

ITYR_TEST_CASE("[ityr::container::global_vector] test") {
  class move_only_t {
  public:
    move_only_t() {}
    move_only_t(const long v) : value_(v) {}

    long value() const { return value_; }

    move_only_t(const move_only_t&) = delete;
    move_only_t& operator=(const move_only_t&) = delete;

    move_only_t(move_only_t&& mo) : value_(mo.value_) {
      mo.value_ = -1;
    }
    move_only_t& operator=(move_only_t&& mo) {
      value_ = mo.value_;
      mo.value_ = -1;
      return *this;
    }

  private:
    long value_ = -1;
  };

  ito::init();
  ori::init();

  auto my_rank = common::topology::my_rank();
  auto n_ranks = common::topology::n_ranks();

  long n = 10000;

  ITYR_SUBCASE("collective") {
    global_vector<long> gv1({.collective         = true,
                             .parallel_construct = true,
                             .parallel_destruct  = true,
                             .cutoff_count       = 256},
                            count_iterator<long>(0),
                            count_iterator<long>(n));
    ITYR_CHECK(!gv1.empty());
    ITYR_CHECK(gv1.size() == std::size_t(n));
    ITYR_CHECK(gv1.capacity() >= std::size_t(n));
    root_exec([&] {
      long count = parallel_reduce({.cutoff_count   = 128,
                                    .checkout_count = 128},
                                   gv1.begin(),
                                   gv1.end(),
                                   long(0),
                                   std::plus<long>{});
      ITYR_CHECK(count == n * (n - 1) / 2);
    });

    ITYR_SUBCASE("copy") {
      global_vector<long> gv2 = gv1;
      root_exec([&] {
        parallel_for_each({.cutoff_count   = 128,
                           .checkout_count = 128},
                          make_global_iterator(gv2.begin(), checkout_mode::read_write),
                          make_global_iterator(gv2.end()  , checkout_mode::read_write),
                          [](long& i) { i *= 2; });

        long count1 = parallel_reduce({.cutoff_count   = 128,
                                       .checkout_count = 128},
                                      gv1.begin(),
                                      gv1.end(),
                                      long(0),
                                      std::plus<long>{});
        ITYR_CHECK(count1 == n * (n - 1) / 2);

        long count2 = parallel_reduce({.cutoff_count   = 128,
                                       .checkout_count = 128},
                                      gv2.begin(),
                                      gv2.end(),
                                      long(0),
                                      std::plus<long>{});
        ITYR_CHECK(count2 == n * (n - 1));
      });
    }

    ITYR_SUBCASE("move") {
      global_vector<long> gv2 = std::move(gv1);
      ITYR_CHECK(gv1.empty());
      ITYR_CHECK(gv1.capacity() == 0);
      root_exec([&] {
        long count = parallel_reduce({.cutoff_count   = 128,
                                      .checkout_count = 128},
                                     gv2.begin(),
                                     gv2.end(),
                                     long(0),
                                     std::plus<long>{});
        ITYR_CHECK(count == n * (n - 1) / 2);
      });
    }

    ITYR_SUBCASE("resize") {
      gv1.resize(n * 10, 3);
      root_exec([&] {
        long count = parallel_reduce({.cutoff_count   = 128,
                                      .checkout_count = 128},
                                     gv1.begin(),
                                     gv1.end(),
                                     long(0),
                                     std::plus<long>{});
        ITYR_CHECK(count == n * (n - 1) / 2 + (n * 9) * 3);
      });
      gv1.resize(n * 5);
      root_exec([&] {
        long count = parallel_reduce({.cutoff_count   = 128,
                                      .checkout_count = 128},
                                     gv1.begin(),
                                     gv1.end(),
                                     long(0),
                                     std::plus<long>{});
        ITYR_CHECK(count == n * (n - 1) / 2 + (n * 4) * 3);
      });
    }

    ITYR_SUBCASE("clear") {
      gv1.clear();
      ITYR_CHECK(gv1.empty());
      ITYR_CHECK(gv1.capacity() >= std::size_t(n));
    }

    ITYR_SUBCASE("move-only elems") {
      global_vector<move_only_t> gv2({.collective         = true,
                                      .parallel_construct = true,
                                      .parallel_destruct  = true,
                                      .cutoff_count       = 256},
                                     gv1.begin(),
                                     gv1.end());
      long next_size = gv2.capacity() * 2;
      gv2.resize(next_size);
      root_exec([&] {
        long count = parallel_reduce({.cutoff_count   = 128,
                                      .checkout_count = 128},
                                     gv2.begin(),
                                     gv2.end(),
                                     long(0),
                                     std::plus<long>{},
                                     [](const move_only_t& mo) { return mo.value(); });
        ITYR_CHECK(count == n * (n - 1) / 2 - (next_size - n));
      });
    }
  }

  ITYR_SUBCASE("noncollective") {
    global_vector<global_vector<long>> gvs({.collective         = true,
                                            .parallel_construct = false,
                                            .parallel_destruct  = false});

    gvs.resize(n_ranks);

    global_vector<long> gv1({.collective         = false,
                             .parallel_construct = false,
                             .parallel_destruct  = false});

    for (long i = 0; i < n; i++) {
      gv1.push_back(i);
    }

    gvs[my_rank] = std::move(gv1);

    root_exec([&]() {
      auto check_sum = [&](long ans) {
        long count =
          parallel_reduce(make_global_iterator(gvs.begin(), checkout_mode::no_access),
                          make_global_iterator(gvs.end()  , checkout_mode::no_access),
                          long(0),
                          std::plus<long>{},
                          [&](auto&& gv_ref) {
            auto cs = make_checkout(&gv_ref, 1, checkout_mode::read_write);
            auto gv_begin = cs[0].begin();
            auto gv_end   = cs[0].end();
            cs.checkin();
            return parallel_reduce({.cutoff_count   = 128,
                                    .checkout_count = 128},
                                   gv_begin,
                                   gv_end,
                                   long(0),
                                   std::plus<long>{});
          });
        ITYR_CHECK(count == ans);
      };

      check_sum(n * (n - 1) / 2 * n_ranks);

      parallel_for_each(make_global_iterator(gvs.begin(), checkout_mode::read_write),
                        make_global_iterator(gvs.end()  , checkout_mode::read_write),
                        [&](global_vector<long>& gv) {
        for (long i = 0; i < 100; i++) {
          gv.push_back(i);
        }
        for (long i = 0; i < 100; i++) {
          gv.pop_back();
        }
        gv.resize(2 * n);
        serial_for_each({.checkout_count = 128},
                        count_iterator<long>(n),
                        count_iterator<long>(2 * n),
                        make_global_iterator(gv.begin() + n, checkout_mode::write),
                        [](long i, long& x) { x = i; });
      });

      check_sum((2 * n) * (2 * n - 1) / 2 * n_ranks);
    });
  }

  ori::fini();
  ito::fini();
}

}
