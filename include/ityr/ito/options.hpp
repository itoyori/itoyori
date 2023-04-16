#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/options.hpp"

namespace ityr::ito {

inline void print_compile_options() {
#ifndef ITYR_ITO_SCHEDULER
#define ITYR_ITO_SCHEDULER randws
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

struct adws_wsqueue_capacity_option : public common::option<adws_wsqueue_capacity_option, std::size_t> {
  using option::option;
  static std::string name() { return "ITYR_ITO_ADWS_WSQUEUE_CAPACITY"; }
  static std::size_t default_value() { return 256; }
};

struct adws_max_num_queue_option : public common::option<adws_max_num_queue_option, int> {
  using option::option;
  static std::string name() { return "ITYR_ITO_ADWS_MAX_NUM_QUEUE"; }
  static int default_value() { return 50; }
};

struct adws_max_dist_tree_depth_option : public common::option<adws_max_dist_tree_depth_option, int> {
  using option::option;
  static std::string name() { return "ITYR_ITO_ADWS_MAX_DIST_TREE_DEPTH"; }
  static int default_value() { return 50; }
};

struct runtime_options {
  common::option_initializer<stack_size_option>                      ITYR_ANON_VAR;
  common::option_initializer<wsqueue_capacity_option>                ITYR_ANON_VAR;
  common::option_initializer<thread_state_allocator_size_option>     ITYR_ANON_VAR;
  common::option_initializer<suspended_thread_allocator_size_option> ITYR_ANON_VAR;
  common::option_initializer<adws_wsqueue_capacity_option>           ITYR_ANON_VAR;
  common::option_initializer<adws_max_num_queue_option>              ITYR_ANON_VAR;
  common::option_initializer<adws_max_dist_tree_depth_option>        ITYR_ANON_VAR;
};

}
