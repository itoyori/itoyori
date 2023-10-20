#pragma once

#include <cstdio>
#include <vector>
#include <algorithm>

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"

#define ITYR_STR_EXPAND(x) #x
#define ITYR_STR(x) ITYR_STR_EXPAND(x)
#define ITYR_PRINT_MACRO(x) printf(#x "=" ITYR_STR_EXPAND(x) "\n")

namespace ityr::common {

inline void print_compile_options() {
#ifndef ITYR_MAX_VERBOSE_LEVEL
#define ITYR_MAX_VERBOSE_LEVEL 0
#endif
  ITYR_PRINT_MACRO(ITYR_MAX_VERBOSE_LEVEL);

#ifndef ITYR_PROFILER_MODE
#define ITYR_PROFILER_MODE disabled
#endif
  ITYR_PRINT_MACRO(ITYR_PROFILER_MODE);

#ifndef ITYR_RMA_IMPL
#if __has_include(<utofu.h>)
#define ITYR_RMA_IMPL utofu
#else
#define ITYR_RMA_IMPL mpi
#endif
#endif
  ITYR_PRINT_MACRO(ITYR_RMA_IMPL);

#ifndef ITYR_ALLOCATOR_USE_BOOST
#define ITYR_ALLOCATOR_USE_BOOST 0
#endif
  ITYR_PRINT_MACRO(ITYR_ALLOCATOR_USE_BOOST);

#ifndef ITYR_ALLOCATOR_USE_DYNAMIC_WIN
#define ITYR_ALLOCATOR_USE_DYNAMIC_WIN false
#endif
  ITYR_PRINT_MACRO(ITYR_ALLOCATOR_USE_DYNAMIC_WIN);
}

class option_base {
public:
  virtual ~option_base() = default;
  virtual void print() const = 0;
};

template <typename Derived, typename T>
class option : public singleton<Derived>, public option_base {
  using base_t = singleton<Derived>;

public:
  using value_type = T;

  option(value_type val) : val_(val) {}

  static value_type value() {
    ITYR_CHECK(base_t::initialized());
    return base_t::get().val_;
  }

  static void set(value_type val) {
    base_t::init(val);
  }

  static void unset() {
    ITYR_CHECK(base_t::initialized());
    base_t::fini();
  }

  void print() const override {
    std::cout << Derived::name() << "=" << val_ << std::endl;
  }

protected:
  value_type val_;
};

inline std::vector<option_base*>& get_options() {
  static std::vector<option_base*> opts;
  return opts;
}

template <typename Option>
class option_initializer {
public:
  option_initializer()
    : init_(getenv_coll(Option::name(), Option::default_value())) {
    auto& opts = get_options();
    option_base* opt = &Option::get();
    if (std::find(opts.begin(), opts.end(), opt) == opts.end()) {
      opts.push_back(&Option::get());
      should_pop_ = true;
    }
  }
  ~option_initializer() {
    if (should_pop_) {
      get_options().pop_back();
    }
  }
private:
  singleton_initializer<Option> init_;
  bool                          should_pop_ = false;
};

inline void print_runtime_options() {
  for (option_base* opt : get_options()) {
    opt->print();
  }
}

struct enable_shared_memory_option : public option<enable_shared_memory_option, bool> {
  using option::option;
  static std::string name() { return "ITYR_ENABLE_SHARED_MEMORY"; }
  static bool default_value() { return true; }
};

struct global_clock_sync_round_trips_option : public option<global_clock_sync_round_trips_option, std::size_t> {
  using option::option;
  static std::string name() { return "ITYR_GLOBAL_CLOCK_SYNC_ROUND_TRIPS"; }
  static std::size_t default_value() { return 100; }
};

struct prof_output_per_rank_option : public option<prof_output_per_rank_option, bool> {
  using option::option;
  static std::string name() { return "ITYR_PROF_OUTPUT_PER_RANK"; }
  static bool default_value() { return false; }
};

struct rma_use_mpi_win_allocate : public option<rma_use_mpi_win_allocate, bool> {
  using option::option;
  static std::string name() { return "ITYR_RMA_USE_MPI_WIN_ALLOCATE"; }
  static bool default_value() { return false; }
};

struct allocator_block_size_option : public option<allocator_block_size_option, std::size_t> {
  using option::option;
  static std::string name() { return "ITYR_ALLOCATOR_BLOCK_SIZE"; }
  static std::size_t default_value() { return std::size_t(2) * 1024 * 1024; }
};

struct allocator_max_unflushed_free_objs_option : public option<allocator_max_unflushed_free_objs_option, std::size_t> {
  using option::option;
  static std::string name() { return "ITYR_ALLOCATOR_MAX_UNFLUSHED_FREE_OBJS"; }
  static std::size_t default_value() { return 10; }
};

struct runtime_options {
  option_initializer<enable_shared_memory_option>              ITYR_ANON_VAR;
  option_initializer<global_clock_sync_round_trips_option>     ITYR_ANON_VAR;
  option_initializer<prof_output_per_rank_option>              ITYR_ANON_VAR;
  option_initializer<rma_use_mpi_win_allocate>                 ITYR_ANON_VAR;
  option_initializer<allocator_block_size_option>              ITYR_ANON_VAR;
  option_initializer<allocator_max_unflushed_free_objs_option> ITYR_ANON_VAR;
};

}
