#pragma once

#include "ityr/common/util.hpp"
#include "ityr/ito/ito.hpp"
#include "ityr/ori/ori.hpp"
#include "ityr/pattern/parallel_loop.hpp"
#include "ityr/container/workhint_view.hpp"

namespace ityr {

template <typename W>
class workhint_range {
  static_assert(std::is_trivial_v<W>);

public:
  using value_type = W;

  workhint_range() {}

  explicit workhint_range(std::size_t n_leaves)
    : n_leaves_(n_leaves),
      bin_tree_(mem_alloc(size())) {
    ITYR_CHECK(common::is_pow2(n_leaves));
  }

  ~workhint_range() {
    if (bin_tree_) {
      mem_free(bin_tree_);
    }
  }

  workhint_range(const workhint_range&) = delete;
  workhint_range& operator=(const workhint_range&) = delete;

  workhint_range(workhint_range&& r)
    : n_leaves_(r.n_leaves_), bin_tree_(r.bin_tree_) {
    r.bin_tree_ = nullptr;
  }
  workhint_range& operator=(workhint_range&& r) {
    n_leaves_ = r.n_leaves_;
    bin_tree_ = r.bin_tree_;
    r.bin_tree_ = nullptr;
  }

  std::size_t size() const {
    return n_leaves_ * 2 - 1;
  }

  workhint_range_view<W> view() const {
    return workhint_range_view<W>(global_span<std::pair<value_type, value_type>>(bin_tree_, size()));
  }

private:
  ori::global_ptr<std::pair<value_type, value_type>> mem_alloc(std::size_t size) {
    if (ito::is_spmd()) {
      return ori::malloc_coll<std::pair<value_type, value_type>>(size);
    } else if (ito::is_root()) {
      return ito::coll_exec([=] { return ori::malloc_coll<std::pair<value_type, value_type>>(size); });
    } else {
      common::die("workhint_range must be created on the root thread or SPMD region.");
    }
  }

  void mem_free(ori::global_ptr<std::pair<value_type, value_type>> p) {
    if (ito::is_spmd()) {
      return ori::free_coll<std::pair<value_type, value_type>>(p);
    } else if (ito::is_root()) {
      return ito::coll_exec([=] { ori::free_coll(p); });
    } else {
      common::die("workhint_range must be destroyed on the root thread or SPMD region.");
    }
  }

  std::size_t                                        n_leaves_;
  ori::global_ptr<std::pair<value_type, value_type>> bin_tree_;
};

template <>
class workhint_range<void> {};

namespace internal {

template <typename W, typename Op, typename ReleaseHandler,
          typename ForwardIterator, typename... ForwardIterators>
inline W create_workhint_range_aux(workhint_range_view<W> target_wh,
                                   std::size_t            checkout_count,
                                   Op                     op,
                                   ReleaseHandler         rh,
                                   ForwardIterator        first,
                                   ForwardIterator        last,
                                   ForwardIterators...    firsts) {
  ori::poll();

  // for immediately executing cross-worker tasks in ADWS
  ito::poll([] { return ori::release_lazy(); },
            [&](ori::release_handler rh_) { ori::acquire(rh); ori::acquire(rh_); });

  if (target_wh.empty()) {
    W w {};
    for_each_aux(
        execution::sequenced_policy(checkout_count),
        [&](auto&&... refs) {
          w += op(std::forward<decltype(refs)>(refs)...);
        },
        first, last, firsts...);
    return w;
  }

  std::size_t d = std::distance(first, last);
  auto mid = std::next(first, d / 2);

  auto tgdata = ito::task_group_begin();

  workhint_range_view<W> c1, c2;
  if (target_wh.has_children()) {
    auto children = target_wh.get_children();
    c1 = children.first;
    c2 = children.second;
  }

  ito::thread<W> th(
      ito::with_callback, [=] { ori::acquire(rh); }, [] { ori::release(); },
      ito::workhint(1, 1),
      [=] {
        return create_workhint_range_aux(c1, checkout_count, op, rh,
                                         first, mid, firsts...);
      });

  W w2 = create_workhint_range_aux(c2, checkout_count, op, rh,
                                   mid, last, std::next(firsts, d / 2)...);

  if (!th.serialized()) {
    ori::release();
  }

  W w1 = th.join();

  ito::task_group_end(tgdata, [] { ori::release(); }, [] { ori::acquire(); });

  if (!th.serialized()) {
    ori::acquire();
  }

  target_wh.set_workhint(w1, w2);

  return w1 + w2;
}

}

template <typename ExecutionPolicy, typename ForwardIterator, typename Op>
inline auto create_workhint_range(const ExecutionPolicy& policy,
                                  ForwardIterator        first,
                                  ForwardIterator        last,
                                  Op                     op,
                                  std::size_t            n_leaves) {
  ITYR_REQUIRE_MESSAGE(common::is_pow2(n_leaves),
                       "The number of leaves for workhint_range must be a power of two.");

  if constexpr (ori::is_global_ptr_v<ForwardIterator>) {
    return create_workhint_range(
        policy,
        internal::convert_to_global_iterator(first, checkout_mode::read),
        internal::convert_to_global_iterator(last , checkout_mode::read),
        op,
        n_leaves);

  } else {
    using ref_t = typename std::iterator_traits<ForwardIterator>::reference;
    using workhint_t = std::invoke_result_t<Op, ref_t>;
    workhint_range<workhint_t> workhint(n_leaves);

    auto rh = ori::release_lazy();
    internal::create_workhint_range_aux(workhint.view(), policy.checkout_count, op, rh, first, last);
    return workhint;
  }
}

template <typename ExecutionPolicy, typename ForwardIterator, typename Op>
inline auto create_workhint_range(const ExecutionPolicy& policy,
                                  ForwardIterator        first,
                                  ForwardIterator        last,
                                  Op                     op) {
  return create_workhint_range(policy, first, last, op,
                               common::next_pow2(std::distance(first, last)));
}

ITYR_TEST_CASE("[ityr::workhint] workhint range test") {
  ito::init();
  ori::init();

  int n = 100000;
  ori::global_ptr<long> p = ori::malloc_coll<long>(n);

  root_exec([=] {
    for_each(
        execution::parallel_policy(100),
        count_iterator<long>(0),
        count_iterator<long>(n),
        make_global_iterator(p, checkout_mode::write),
        [](long i, long& x) { x = i; });

    auto workhint = create_workhint_range(
        execution::parallel_policy(100),
        p, p + n,
        [](long x) { return x; });

    for_each(
        execution::parallel_policy(100, workhint),
        count_iterator<long>(0),
        count_iterator<long>(n),
        make_global_iterator(p, checkout_mode::read),
        [](long i, long x) { ITYR_CHECK(i == x); });
  });

  ori::free_coll(p);

  ori::fini();
  ito::fini();
}

}
