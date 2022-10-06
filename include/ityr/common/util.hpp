#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <sstream>

#define ITYR_CONCAT_(x, y) x##y
#define ITYR_CONCAT(x, y) ITYR_CONCAT_(x, y)

#ifdef DOCTEST_LIBRARY_INCLUDED

#define ITYR_TEST_CASE(name)                 DOCTEST_TEST_CASE(name)
#define ITYR_SUBCASE(name)                   DOCTEST_SUBCASE(name)
#define ITYR_CHECK(cond)                     DOCTEST_CHECK(cond)
#define ITYR_CHECK_MESSAGE(cond, ...)        DOCTEST_CHECK_MESSAGE(cond, __VA_ARGS__)
#define ITYR_REQUIRE(cond)                   DOCTEST_REQUIRE(cond)
#define ITYR_REQUIRE_MESSAGE(cond, ...)      DOCTEST_REQUIRE_MESSAGE(cond, __VA_ARGS__)
#define ITYR_CHECK_THROWS_AS(exp, exception) DOCTEST_CHECK_THROWS_AS(exp, exception)

#else

#ifdef __COUNTER__
#define ITYR_ANON_NAME(x) ITYR_CONCAT(x, __COUNTER__)
#else
#define ITYR_ANON_NAME(x) ITYR_CONCAT(x, __LINE__)
#endif

#define ITYR_TEST_CASE(name)                 [[maybe_unused]] static inline void ITYR_ANON_NAME(__ityr_test_anon_fn)()
#define ITYR_SUBCASE(name)
#define ITYR_CHECK(cond)                     ITYR_ASSERT(cond)
#define ITYR_CHECK_MESSAGE(cond, ...)        ITYR_ASSERT(cond)
#define ITYR_REQUIRE(cond)                   ITYR_ASSERT(cond)
#define ITYR_REQUIRE_MESSAGE(cond, ...)      ITYR_ASSERT(cond)
#define ITYR_CHECK_THROWS_AS(exp, exception) exp

#endif

#ifdef NDEBUG
#define ITYR_ASSERT(cond) do { (void)sizeof(cond); } while (0)
#else
#include <cassert>
#define ITYR_ASSERT(cond) assert(cond)
#endif

namespace ityr {
namespace common {

__attribute__((noinline))
inline void die(const char* fmt, ...) {
  constexpr int slen = 128;
  char msg[slen];

  va_list args;
  va_start(args, fmt);
  vsnprintf(msg, slen, fmt, args);
  va_end(args);

  fprintf(stderr, "\x1b[31m%s\x1b[39m\n", msg);
  fflush(stderr);

  abort();
}

template <typename T>
inline T get_env_(const char* env_var, T default_val) {
  if (const char* val_str = std::getenv(env_var)) {
    T val;
    std::stringstream ss(val_str);
    ss >> val;
    if (ss.fail()) {
      fprintf(stderr, "Environment variable '%s' is invalid.\n", env_var);
      exit(1);
    }
    return val;
  } else {
    return default_val;
  }
}

template <typename T>
inline T get_env(const char* env_var, T default_val, int rank) {
  static bool print_env = get_env_("ITYR_PRINT_ENV", false);

  T val = get_env_(env_var, default_val);
  if (print_env && rank == 0) {
    std::cout << env_var << " = " << val << std::endl;
  }
  return val;
}

}
}
