#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/options.hpp"

namespace ityr::ito {

inline void print_compile_options() {
#ifndef ITYR_ITO_SCHEDULER
#define ITYR_ITO_SCHEDULER ws_workfirst
#endif
  ITYR_PRINT_MACRO(ITYR_ITO_SCHEDULER);
}

struct stack_size_option : public common::option<stack_size_option, std::size_t> {
  using option::option;
  static std::string name() { return "ITYR_ITO_STACK_SIZE"; }
  static std::size_t default_value() { return std::size_t(2) * 1024 * 1024; }
};

struct wsqueue_capacity_option : public common::option<wsqueue_capacity_option, std::size_t> {
  using option::option;
  static std::string name() { return "ITYR_ITO_WSQUEUE_CAPACITY"; }
  static std::size_t default_value() { return 1024; }
};

struct thread_state_allocator_size_option : public common::option<thread_state_allocator_size_option, std::size_t> {
  using option::option;
  static std::string name() { return "ITYR_ITO_THREAD_STATE_ALLOCATOR_SIZE"; }
  static std::size_t default_value() { return std::size_t(2) * 1024 * 1024; }
};

struct suspended_thread_allocator_size_option : public common::option<suspended_thread_allocator_size_option, std::size_t> {
  using option::option;
  static std::string name() { return "ITYR_ITO_SUSPENDED_THREAD_ALLOCATOR_SIZE"; }
  static std::size_t default_value() { return std::size_t(2) * 1024 * 1024; }
};

struct runtime_options {
  common::option_initializer<stack_size_option>                      ITYR_ANON_VAR;
  common::option_initializer<wsqueue_capacity_option>                ITYR_ANON_VAR;
  common::option_initializer<thread_state_allocator_size_option>     ITYR_ANON_VAR;
  common::option_initializer<suspended_thread_allocator_size_option> ITYR_ANON_VAR;
};

}
