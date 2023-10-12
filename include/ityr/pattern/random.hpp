#pragma once

#include "ityr/common/util.hpp"

#if __has_include(<lxm_random/lxm_random.hpp>)
#include <lxm_random/lxm_random.hpp>
namespace ityr {
/*
 * @brief Default *splittable* random number generator.
 *
 * A *splittable* random number generator must have a member `split()` to spawn an independent
 * child stream of random numbers. By default, Itoyori uses *LXM*, a representative splittable
 * random number generator presented in the following paper.
 *
 * [Guy L. Steele Jr. and Sebastiano Vigna. "LXM: better splittable pseudorandom number generators (and almost as fast)" in ACM OOPSLA '21.](https://doi.org/10.1145/3485525)
 *
 * In order to use `ityr::default_random_engine`, the header file `<lxm_random/lxm_random.hpp>`
 * must be located under the include path at compilation time.
 * The header file for LXM can be found at [s417-lama/lxm_random](https://github.com/s417-lama/lxm_random).
 *
 * @see `ityr::shuffle()`
 */
using default_random_engine = lxm_random::lxm_random;
}
#else
namespace ityr {
namespace internal {
class random_engine_dummy {
  void report_error() { common::die("<lxm_random/lxm_random.hpp> was not loaded but a ityr::default_random_engine is used."); }
public:
  template <typename... Args>
  random_engine_dummy(Args&&...) { report_error(); }
  uint64_t operator()() { report_error(); return {}; }
  template <typename... Args>
  random_engine_dummy split(Args&&...) { report_error(); return {}; }
};
}
using default_random_engine = internal::random_engine_dummy;
}
#endif
