#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstdarg>
#include <ctime>
#include <unistd.h>
#include <iostream>
#include <sstream>
#include <optional>

#include "ityr/common/options.hpp"

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

#define ITYR_ANON_VAL ITYR_CONCAT(anon_, __LINE__)

namespace ityr::common {

inline uint64_t clock_gettime_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000 + (uint64_t)ts.tv_nsec;
}

inline constexpr int max_verbose_level = ITYR_MAX_VERBOSE_LEVEL;

template <int Level = 1>
inline void verbose(const char* fmt, ...) {
  if (Level <= max_verbose_level) {
    constexpr int slen = 256;
    static char msg[slen];

    va_list args;
    va_start(args, fmt);
    vsnprintf(msg, slen, fmt, args);
    va_end(args);

    fprintf(stderr, "%ld: %s\n", clock_gettime_ns(), msg);
  }
}

[[noreturn]] __attribute__((noinline))
inline void die(const char* fmt, ...) {
  constexpr int slen = 256;
  static char msg[slen];

  va_list args;
  va_start(args, fmt);
  vsnprintf(msg, slen, fmt, args);
  va_end(args);

  fprintf(stderr, "\x1b[31m%s\x1b[39m\n", msg);
  fflush(stderr);

  std::abort();
}

template <typename T>
inline T getenv_with_default(const char* env_var, T default_val) {
  if (const char* val_str = std::getenv(env_var)) {
    T val;
    std::stringstream ss(val_str);
    ss >> val;
    if (ss.fail()) {
      die("Environment variable '%s' is invalid.\n", env_var);
    }
    return val;
  } else {
    return default_val;
  }
}

inline uint64_t next_pow2(uint64_t x) {
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  x |= x >> 32;
  return x + 1;
}

ITYR_TEST_CASE("[ityr::common::util] next_pow2") {
  ITYR_CHECK(next_pow2(0) == 0);
  ITYR_CHECK(next_pow2(1) == 1);
  ITYR_CHECK(next_pow2(2) == 2);
  ITYR_CHECK(next_pow2(3) == 4);
  ITYR_CHECK(next_pow2(4) == 4);
  ITYR_CHECK(next_pow2(5) == 8);
  ITYR_CHECK(next_pow2(15) == 16);
  ITYR_CHECK(next_pow2((uint64_t(1) << 38) - 100) == uint64_t(1) << 38);
}

template <typename T>
inline bool is_pow2(T x) {
  return !(x & (x - 1));
}

template <typename T>
inline T round_down_pow2(T x, T alignment) {
  ITYR_CHECK(is_pow2(alignment));
  return x & ~(alignment - 1);
}

template <typename T>
inline T* round_down_pow2(T* x, uintptr_t alignment) {
  ITYR_CHECK(is_pow2(alignment));
  return reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(x) & ~(alignment - 1));
}

template <typename T>
inline T round_up_pow2(T x, T alignment) {
  ITYR_CHECK(is_pow2(alignment));
  return (x + alignment - 1) & ~(alignment - 1);
}

template <typename T>
inline T* round_up_pow2(T* x, uintptr_t alignment) {
  ITYR_CHECK(is_pow2(alignment));
  return reinterpret_cast<T*>((reinterpret_cast<uintptr_t>(x) + alignment - 1) & ~(alignment - 1));
}

ITYR_TEST_CASE("[ityr::common::util] round up/down for integers") {
  ITYR_CHECK(is_pow2(128));
  ITYR_CHECK(!is_pow2(129));
  ITYR_CHECK(round_down_pow2(1100, 128) == 1024);
  ITYR_CHECK(round_down_pow2(128, 128) == 128);
  ITYR_CHECK(round_down_pow2(129, 128) == 128);
  ITYR_CHECK(round_down_pow2(255, 128) == 128);
  ITYR_CHECK(round_down_pow2(73, 128) == 0);
  ITYR_CHECK(round_down_pow2(0, 128) == 0);
  ITYR_CHECK(round_up_pow2(1100, 128) == 1152);
  ITYR_CHECK(round_up_pow2(128, 128) == 128);
  ITYR_CHECK(round_up_pow2(129, 128) == 256);
  ITYR_CHECK(round_up_pow2(255, 128) == 256);
  ITYR_CHECK(round_up_pow2(73, 128) == 128);
  ITYR_CHECK(round_up_pow2(0, 128) == 0);
}

inline std::size_t get_page_size() {
  static std::size_t pagesize = sysconf(_SC_PAGE_SIZE);
  return pagesize;
}

template <typename T>
class singleton {
public:
  static auto& get() {
    ITYR_CHECK(initialized());
    return *get_optional();
  }

  static bool initialized() {
    return get_optional().has_value();
  }

  template <typename... Args>
  static void init(Args&&... args) {
    ITYR_CHECK(!initialized());
    get_optional().emplace(std::forward<Args>(args)...);
  }

  static void fini() {
    ITYR_CHECK(initialized());
    get_optional().reset();
  }

private:
  static auto& get_optional() {
    static std::optional<T> instance;
    return instance;
  }
};

template <typename Singleton>
class singleton_initializer {
public:
  template <typename... Args>
  singleton_initializer(Args&&... args) {
    if (!Singleton::initialized()) {
      Singleton::init(std::forward<Args>(args)...);
      should_finalize_ = true;
    }
  }

  ~singleton_initializer() {
    if (should_finalize_) {
      Singleton::fini();
    }
  }

  singleton_initializer(const singleton_initializer&) = delete;
  singleton_initializer& operator=(const singleton_initializer&) = delete;

  singleton_initializer(singleton_initializer&&) = delete;
  singleton_initializer& operator=(singleton_initializer&&) = delete;

private:
  bool should_finalize_ = false;
};

}
