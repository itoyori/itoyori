/**********************************************************************************************/
/*  This program is part of the Barcelona OpenMP Tasks Suite                                  */
/*  Copyright (C) 2009 Barcelona Supercomputing Center - Centro Nacional de Supercomputacion  */
/*  Copyright (C) 2009 Universitat Politecnica de Catalunya                                   */
/*                                                                                            */
/*  This program is free software; you can redistribute it and/or modify                      */
/*  it under the terms of the GNU General Public License as published by                      */
/*  the Free Software Foundation; either version 2 of the License, or                         */
/*  (at your option) any later version.                                                       */
/*                                                                                            */
/*  This program is distributed in the hope that it will be useful,                           */
/*  but WITHOUT ANY WARRANTY; without even the implied warranty of                            */
/*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                             */
/*  GNU General Public License for more details.                                              */
/*                                                                                            */
/*  You should have received a copy of the GNU General Public License                         */
/*  along with this program; if not, write to the Free Software                               */
/*  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA            */
/**********************************************************************************************/

/*
 * Original code from the Cilk project (by Keith Randall)
 *
 * Copyright (c) 2000 Massachusetts Institute of Technology
 * Copyright (c) 2000 Matteo Frigo
 */

#include "ityr/ityr.hpp"

using result_t = uint64_t;
using state_t = char;

int  n_input       = 10;
int  n_repeats     = 10;
bool verify_result = true;

/* Checking information */
constexpr result_t solutions[] = {
  1,
  0,
  0,
  2,
  10, /* 5 */
  4,
  40,
  92,
  352,
  724, /* 10 */
  2680,
  14200,
  73712,
  365596,
  2279184, /* 15 */
  14772512,
  95815104,
  666090624,
  4968057848,
  39029188884, /* 20 */
};

constexpr int max_solutions = sizeof(solutions) / sizeof(solutions[0]);

/*
 * <a> contains array of <n> queen positions.  Returns 1
 * if none of the queens conflict, and returns 0 otherwise.
 */
bool ok(int n, state_t* a) {
  for (int i = 0; i < n; i++) {
    state_t p = a[i];
    for (int j = i + 1; j < n; j++) {
      state_t q = a[j];
      if (q == p || q == p - (j - i) || q == p + (j - i))
        return false;
    }
  }
  return true;
}

struct board {
  state_t array[max_solutions];
};

result_t nqueens(int n, int j, board b, int depth) {
  if (n == j) {
    /* good solution, count it */
    return 1;
  } else {
    /* try each possible position for queen <j> */
    return ityr::parallel_reduce(
      ityr::count_iterator<int>(0),
      ityr::count_iterator<int>(n),
      0,
      std::plus<result_t>(),
      [=](int i) mutable -> result_t {
        b.array[j] = static_cast<state_t>(i);
        if (ok(j + 1, b.array)) {
          return nqueens(n, j + 1, b, depth + 1);
        } else {
          return 0;
        }
      });
  }
}

void show_help_and_exit(int argc [[maybe_unused]], char** argv) {
  if (ityr::is_master()) {
    printf("Usage: %s [options]\n"
           "  options:\n"
           "    -n : input size (int)\n"
           "    -r : # of repeats (int)\n"
           "    -v : verify the result (int)\n", argv[0]);
  }
  exit(1);
}

int main(int argc, char** argv) {
  int opt;
  while ((opt = getopt(argc, argv, "n:r:v:h")) != EOF) {
    switch (opt) {
      case 'n':
        n_input = atoi(optarg);
        break;
      case 'r':
        n_repeats = atoi(optarg);
        break;
      case 'v':
        verify_result = atoi(optarg);
        break;
      case 'h':
      default:
        show_help_and_exit(argc, argv);
    }
  }

  if (n_input >= max_solutions) {
    if (ityr::is_master()) {
      printf("N=%d (-n) cannot be greater than %d\n", n_input, max_solutions);
    }
    exit(1);
  }

  ityr::init();

  if (ityr::is_master()) {
    setlocale(LC_NUMERIC, "en_US.UTF-8");
    printf("=============================================================\n"
           "[NQueens]\n"
           "N:                            %d\n"
           "# of processes:               %d\n"
           "# of repeats:                 %d\n"
           "Verify result:                %d\n"
           "=============================================================\n\n",
           n_input, ityr::n_ranks(), n_repeats, verify_result);
    fflush(stdout);
  }

  for (int r = 0; r < n_repeats; r++) {
    ityr::profiler_begin();

    auto t0 = ityr::common::clock_gettime_ns();

    result_t result = ityr::ito::root_exec([]{
      return nqueens(n_input, 0, board{}, 0);
    });

    auto t1 = ityr::common::clock_gettime_ns();

    ityr::profiler_end();

    if (ityr::is_master()) {
      printf("[%d] %ld ns\n", r, t1 - t0);

      if (verify_result) {
        result_t answer = solutions[n_input - 1];
        if (result == answer) {
          printf("Result verified: nqueens(%d) = %ld\n", n_input, result);
        } else {
          printf("Wrong answer: nqueens(%d) should be %ld but got %ld\n",
                 n_input, answer, result);
        }
      }

      fflush(stdout);
    }

    ityr::profiler_flush();
  }

  ityr::fini();
  return 0;
}
