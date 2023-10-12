#pragma once

#include "ityr/common/util.hpp"
#include "ityr/pattern/parallel_invoke.hpp"
#include "ityr/pattern/parallel_loop.hpp"
#include "ityr/pattern/parallel_reduce.hpp"
#include "ityr/pattern/parallel_sort.hpp"
#include "ityr/pattern/random.hpp"

namespace ityr {

namespace internal {

template <typename W, typename RandomAccessIterator,
          typename SplittableUniformRandomBitGenerator>
inline RandomAccessIterator
random_partition(const execution::parallel_policy<W>&  policy,
                 RandomAccessIterator                  first,
                 RandomAccessIterator                  last,
                 SplittableUniformRandomBitGenerator&& urbg) {
  std::size_t d = std::distance(first, last);

  if (d <= policy.cutoff_count) {
    // TODO: consider policy.checkout_count
    ITYR_CHECK(policy.cutoff_count == policy.checkout_count);

    auto&& [css, its] = checkout_global_iterators(d, first);
    auto&& first_ = std::get<0>(its);
    auto m = std::stable_partition(first_, std::next(first_, d),
                                   [&](auto&&) { return urbg() % 2 == 0; });
    return std::next(first, std::distance(first_, m));
  }

  auto mid = std::next(first, d / 2);

  auto child_urbg1 = urbg.split();
  auto child_urbg2 = urbg.split();

  auto [m1, m2] = parallel_invoke(
      [=]() mutable { return random_partition(policy, first, mid , child_urbg1); },
      [=]() mutable { return random_partition(policy, mid  , last, child_urbg2); });

  // TODO: use swap_ranges; stability is not needed
  return rotate(policy, m1, mid, m2);
}

template <typename W, typename RandomAccessIterator,
          typename SplittableUniformRandomBitGenerator>
inline void shuffle(const execution::parallel_policy<W>&  policy,
                    RandomAccessIterator                  first,
                    RandomAccessIterator                  last,
                    SplittableUniformRandomBitGenerator&& urbg) {
  std::size_t d = std::distance(first, last);

  if (d <= 1) return;

  if (d <= policy.cutoff_count) {
    auto [css, its] = checkout_global_iterators(d, first);
    auto first_ = std::get<0>(its);
    std::shuffle(first_, std::next(first_, d), urbg);

  } else {
    auto mid = random_partition(policy, first, last, urbg);

    auto child_urbg1 = urbg.split();
    auto child_urbg2 = urbg.split();

    parallel_invoke(
        [=]() mutable { shuffle(policy, first, mid , child_urbg1); },
        [=]() mutable { shuffle(policy, mid  , last, child_urbg2); });
  }
}

}

/**
 * @brief Randomly shuffle elements in a range.
 *
 * @param policy Execution policy (`ityr::execution`).
 * @param first  Begin iterator.
 * @param last   End iterator.
 * @param urbg   Uniform random bit generator. It needs to be *splittable* if parallel.
 *
 * This function randomly shuffles the elements in the input range.
 * Randomness is given by the random number generator `urbg`.
 *
 * Although the standard `std::shuffle()` does not have a parallel variant, `ityr::shuffle()` can be
 * computed in parallel if a *splittable* random number generator is provided.
 * A *splittable* random number generator has a member `split()` to spawn an apparently independent
 * child stream of random numbers. *LXM* is a representative splittable random number generator,
 * which was presented in the following paper.
 *
 * [Guy L. Steele Jr. and Sebastiano Vigna. "LXM: better splittable pseudorandom number generators (and almost as fast)" in ACM OOPSLA '21.](https://doi.org/10.1145/3485525)
 *
 * For a C++ implementation, see [s417-lama/lxm_random](https://github.com/s417-lama/lxm_random).
 * The default random number generator in Itoyori (`ityr::default_random_engine`) uses LXM.
 *
 * Example:
 * ```
 * ityr::global_vector<int> v = {1, 2, 3, 4, 5};
 * ityr::default_random_engine rng(42); // seed = 42
 * ityr::shuffle(ityr::execution::par, v.begin(), v.end(), rng);
 * // v = {1, 2, 5, 4, 3}
 * ```
 *
 * @see [std::shuffle -- cppreference.com](https://en.cppreference.com/w/cpp/algorithm/shuffle)
 * @see `ityr::default_random_engine`
 * @see `ityr::execution::sequenced_policy`, `ityr::execution::seq`,
 *      `ityr::execution::parallel_policy`, `ityr::execution::par`
 */
template <typename ExecutionPolicy, typename RandomAccessIterator,
          typename UniformRandomBitGenerator>
inline void shuffle(const ExecutionPolicy&      policy,
                    RandomAccessIterator        first,
                    RandomAccessIterator        last,
                    UniformRandomBitGenerator&& urbg) {
  if constexpr (ori::is_global_ptr_v<RandomAccessIterator>) {
    shuffle(
        policy,
        internal::convert_to_global_iterator(first, checkout_mode::read_write),
        internal::convert_to_global_iterator(last , checkout_mode::read_write),
        std::forward<UniformRandomBitGenerator>(urbg));

  } else {
    internal::shuffle(policy, first, last, std::forward<UniformRandomBitGenerator>(urbg));
  }
}

ITYR_TEST_CASE("[ityr::pattern::parallel_shuffle] shuffle") {
  ito::init();
  ori::init();

  long n = 100000;
  ori::global_ptr<long> p1 = ori::malloc_coll<long>(n);
  ori::global_ptr<long> p2 = ori::malloc_coll<long>(n);

  ito::root_exec([=] {
    transform(
        execution::parallel_policy(100),
        count_iterator<long>(0), count_iterator<long>(n), p1,
        [=](long i) { return i; });

    transform(
        execution::parallel_policy(100),
        count_iterator<long>(0), count_iterator<long>(n), p2,
        [=](long i) { return i; });

    ITYR_CHECK(equal(execution::parallel_policy(100),
                     p1, p1 + n, p2) == true);
  });

  ITYR_SUBCASE("should not lose values") {
    ito::root_exec([=] {
      shuffle(execution::parallel_policy(100), p1, p1 + n,
              default_random_engine{});

      ITYR_CHECK(equal(execution::parallel_policy(100),
                       p1, p1 + n, p2) == false);

      ITYR_CHECK(reduce(execution::parallel_policy(100),
                        p1, p1 + n) == n * (n - 1) / 2);

      sort(execution::parallel_policy(100), p1, p1 + n);

      ITYR_CHECK(equal(execution::parallel_policy(100),
                       p1, p1 + n, p2) == true);
    });
  }

  ITYR_SUBCASE("same RNG, same result") {
    ito::root_exec([=] {
      uint64_t seed = 42;
      default_random_engine rng(seed);

      auto rng_copy = rng;

      shuffle(execution::parallel_policy(100), p1, p1 + n, rng);
      shuffle(execution::parallel_policy(100), p2, p2 + n, rng_copy);

      ITYR_CHECK(equal(execution::parallel_policy(100),
                       p1, p1 + n, p2) == true);
    });
  }

  ITYR_SUBCASE("differente RNG, different result") {
    ito::root_exec([=] {
      uint64_t seed1 = 42;
      default_random_engine rng1(seed1);

      uint64_t seed2 = 417;
      default_random_engine rng2(seed2);

      shuffle(execution::parallel_policy(100), p1, p1 + n, rng1);
      shuffle(execution::parallel_policy(100), p2, p2 + n, rng2);

      ITYR_CHECK(equal(execution::parallel_policy(100),
                       p1, p1 + n, p2) == false);
    });
  }

  ori::free_coll(p1);
  ori::free_coll(p2);

  ori::fini();
  ito::fini();
}

}
