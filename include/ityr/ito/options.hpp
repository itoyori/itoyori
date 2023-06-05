#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/options.hpp"

namespace ityr::ito {

inline void print_compile_options() {
#ifndef ITYR_ITO_SCHEDULER
#define ITYR_ITO_SCHEDULER randws
#endif
  ITYR_PRINT_MACRO(ITYR_ITO_SCHEDULER);

#ifndef ITYR_ITO_DAG_PROF
#define ITYR_ITO_DAG_PROF disabled
#endif
  ITYR_PRINT_MACRO(ITYR_ITO_DAG_PROF);
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

struct sched_loop_make_mpi_progress_option : public common::option<sched_loop_make_mpi_progress_option, bool> {
  using option::option;
  static std::string name() { return "ITYR_ITO_SCHED_LOOP_MAKE_MPI_PROGRESS"; }
  static bool default_value() { return true; }
};

struct adws_enable_steal_option : public common::option<adws_enable_steal_option, bool> {
  using option::option;
  static std::string name() { return "ITYR_ITO_ADWS_ENABLE_STEAL"; }
  static bool default_value() { return true; }
};

struct adws_wsqueue_capacity_option : public common::option<adws_wsqueue_capacity_option, std::size_t> {
  using option::option;
  static std::string name() { return "ITYR_ITO_ADWS_WSQUEUE_CAPACITY"; }
  static std::size_t default_value() { return 256; }
};

struct adws_max_depth_option : public common::option<adws_max_depth_option, int> {
  using option::option;
  static std::string name() { return "ITYR_ITO_ADWS_MAX_DEPTH"; }
  static int default_value() { return 20; }
};

struct adws_max_dtree_reuse_option : public common::option<adws_max_dtree_reuse_option, int> {
  using option::option;
  static std::string name() { return "ITYR_ITO_ADWS_MAX_DTREE_REUSE"; }
  static int default_value() { return 10; }
};

struct adws_min_drange_size_option : public common::option<adws_min_drange_size_option, double> {
  using option::option;
  static std::string name() { return "ITYR_ITO_ADWS_MIN_DRANGE_SIZE"; }
  static double default_value() { return 0.01; }
};

struct runtime_options {
  common::option_initializer<stack_size_option>                      ITYR_ANON_VAR;
  common::option_initializer<wsqueue_capacity_option>                ITYR_ANON_VAR;
  common::option_initializer<thread_state_allocator_size_option>     ITYR_ANON_VAR;
  common::option_initializer<suspended_thread_allocator_size_option> ITYR_ANON_VAR;
  common::option_initializer<sched_loop_make_mpi_progress_option>    ITYR_ANON_VAR;
  common::option_initializer<adws_enable_steal_option>               ITYR_ANON_VAR;
  common::option_initializer<adws_wsqueue_capacity_option>           ITYR_ANON_VAR;
  common::option_initializer<adws_max_depth_option>                  ITYR_ANON_VAR;
  common::option_initializer<adws_max_dtree_reuse_option>            ITYR_ANON_VAR;
  common::option_initializer<adws_min_drange_size_option>            ITYR_ANON_VAR;
};

}
