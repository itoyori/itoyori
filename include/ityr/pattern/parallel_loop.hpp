#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/ito/ito.hpp"
#include "ityr/pattern/iterator.hpp"

namespace ityr {

using cutoff_t = std::size_t;

struct parallel_loop_options {
  cutoff_t cutoff = 1;
};

template <typename ForwardIterator, typename Fn>
inline void for_each_serial(ForwardIterator first,
                            ForwardIterator last,
                            Fn&&            fn) {
  for (; first != last; ++first) {
    std::forward<Fn>(fn)(*first);
  }
}

template <typename ForwardIterator, typename T, typename ReduceOp>
inline T parallel_reduce(ForwardIterator       first,
                         ForwardIterator       last,
                         T                     init,
                         ReduceOp              reduce) {
  return parallel_reduce(parallel_loop_options{}, first, last, init, reduce);
}

template <typename ForwardIterator, typename T, typename ReduceOp>
inline T parallel_reduce(parallel_loop_options options,
                         ForwardIterator       first,
                         ForwardIterator       last,
                         T                     init,
                         ReduceOp              reduce) {
  auto transform = [](auto&& v) { return std::forward<decltype(v)>(v); };
  return parallel_reduce(options, first, last, init, reduce, transform);
}

template <typename ForwardIterator, typename T, typename ReduceOp, typename TransformOp>
inline T parallel_reduce(ForwardIterator       first,
                         ForwardIterator       last,
                         T                     init,
                         ReduceOp              reduce,
                         TransformOp           transform) {
  return parallel_reduce(parallel_loop_options{}, first, last, init, reduce, transform);
}

template <typename ForwardIterator, typename T, typename ReduceOp, typename TransformOp>
inline T parallel_reduce(parallel_loop_options options,
                         ForwardIterator       first,
                         ForwardIterator       last,
                         T                     init,
                         ReduceOp              reduce,
                         TransformOp           transform) {
  return parallel_reduce_aux(options, first, last, init, reduce, transform);
}

template <typename ForwardIterator, typename T, typename ReduceOp, typename TransformOp>
inline T parallel_reduce_aux(parallel_loop_options options,
                             ForwardIterator       first,
                             ForwardIterator       last,
                             T                     init,
                             ReduceOp              reduce,
                             TransformOp           transform) {
  auto d = std::distance(first, last);
  if (static_cast<cutoff_t>(d) <= options.cutoff) {
    T acc = init;
    for_each_serial(first, last, [&](const auto& v) {
      acc = reduce(acc, transform(v));
    });
    return acc;
  } else {
    auto mid = std::next(first, d / 2);
    ito::thread<T> th(parallel_reduce_aux<ForwardIterator, T, ReduceOp, TransformOp>,
                      options, first, mid, init, reduce, transform);
    T acc2 = parallel_reduce_aux(options, mid, last, init, reduce, transform);
    auto acc1 = th.join();
    return reduce(acc1, acc2);
  }
}

ITYR_TEST_CASE("[ityr::pattern::parallel_loop] parallel reduce") {
  common::topology topo;
  ito::init(topo);

  ITYR_SUBCASE("default cutoff") {
    int n = 10000;
    int r = ito::root_exec([=] {
      return parallel_reduce(
        ityr::count_iterator<int>(0),
        ityr::count_iterator<int>(n),
        0,
        std::plus<int>{});
    });
    ITYR_CHECK(r == n * (n - 1) / 2);
  }

  ITYR_SUBCASE("custom cutoff") {
    int n = 100000;
    int r = ito::root_exec([=] {
      return parallel_reduce(
        parallel_loop_options{.cutoff = 100},
        ityr::count_iterator<int>(0),
        ityr::count_iterator<int>(n),
        0,
        std::plus<int>{});
    });
    ITYR_CHECK(r == n * (n - 1) / 2);
  }

  ITYR_SUBCASE("transform") {
    int n = 100000;
    int r = ito::root_exec([=] {
      return parallel_reduce(
        parallel_loop_options{.cutoff = 100},
        ityr::count_iterator<int>(0),
        ityr::count_iterator<int>(n),
        0,
        std::plus<int>{},
        [](int x) { return x * x; });
    });
    ITYR_CHECK(r == n * (n - 1) * (2 * n - 1) / 6);
  }

  ito::fini();
}

}
