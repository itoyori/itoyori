#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/options.hpp"

namespace ityr::ori {

inline void print_compile_options() {
#ifndef ITYR_ORI_BLOCK_SIZE
#define ITYR_ORI_BLOCK_SIZE 65536
#endif
  ITYR_PRINT_MACRO(ITYR_ORI_BLOCK_SIZE);

#ifndef ITYR_ORI_DEFAULT_MEM_MAPPER
#define ITYR_ORI_DEFAULT_MEM_MAPPER cyclic
#endif
  ITYR_PRINT_MACRO(ITYR_ORI_DEFAULT_MEM_MAPPER);

#ifndef ITYR_ORI_ENABLE_WRITE_THROUGH
#define ITYR_ORI_ENABLE_WRITE_THROUGH false
#endif
  ITYR_PRINT_MACRO(ITYR_ORI_ENABLE_WRITE_THROUGH);
}

struct cache_size_option : public common::option<cache_size_option, std::size_t> {
  using option::option;
  static std::string name() { return "ITYR_ORI_CACHE_SIZE"; };
  static std::size_t default_value() { return std::size_t(16) * 1024 * 1024; };
};

struct sub_block_size_option : public common::option<sub_block_size_option, std::size_t> {
  using option::option;
  static std::string name() { return "ITYR_ORI_SUB_BLOCK_SIZE"; };
  static std::size_t default_value() { return 4096; };
};

struct max_dirty_cache_size_option : public common::option<max_dirty_cache_size_option, std::size_t> {
  using option::option;
  static std::string name() { return "ITYR_ORI_MAX_DIRTY_CACHE_SIZE"; };
  static std::size_t default_value() { return cache_size_option::value() / 2; };
};

struct noncoll_allocator_size_option : public common::option<noncoll_allocator_size_option, std::size_t> {
  using option::option;
  static std::string name() { return "ITYR_ORI_NONCOLL_ALLOCATOR_SIZE"; };
  static std::size_t default_value() { return std::size_t(4) * 1024 * 1024; };
};

struct lazy_release_check_interval_option : public common::option<lazy_release_check_interval_option, int> {
  using option::option;
  static std::string name() { return "ITYR_ORI_LAZY_RELEASE_CHECK_INTERVAL"; };
  static int default_value() { return 10; };
};

struct lazy_release_make_mpi_progress_option : public common::option<lazy_release_make_mpi_progress_option, bool> {
  using option::option;
  static std::string name() { return "ITYR_ORI_LAZY_RELEASE_MAKE_MPI_PROGRESS"; };
  static bool default_value() { return true; };
};

struct runtime_options {
  common::option_initializer<cache_size_option>                     ITYR_ANON_VAL;
  common::option_initializer<sub_block_size_option>                 ITYR_ANON_VAL;
  common::option_initializer<max_dirty_cache_size_option>           ITYR_ANON_VAL;
  common::option_initializer<noncoll_allocator_size_option>         ITYR_ANON_VAL;
  common::option_initializer<lazy_release_check_interval_option>    ITYR_ANON_VAL;
  common::option_initializer<lazy_release_make_mpi_progress_option> ITYR_ANON_VAL;
};

}
