#include "ityr/ityr.hpp"

using result_t = uint64_t;

int  n_input       = 20;
int  n_repeats     = 10;
bool verify_result = true;

result_t fib_fast(int n) {
  result_t px[2] = {1, 1};
  for (int i = 2; i < n; i++) {
    result_t x = px[0] + px[1];
    px[0] = px[1];
    px[1] = x;
  }
  return px[0] + px[1];
}

result_t fib_rec(int n) {
  if (n <= 1) {
    return 1;
  } else {
    auto [x, y] =
      ityr::parallel_invoke(
        [=] { return fib_rec(n - 1); },
        [=] { return fib_rec(n - 2); }
      );
    return x + y;
  }
}

void run() {
  for (int r = 0; r < n_repeats; r++) {
    ityr::profiler_begin();

    auto t0 = ityr::gettime_ns();

    result_t result = ityr::root_exec([]{
      return fib_rec(n_input);
    });

    auto t1 = ityr::gettime_ns();

    ityr::profiler_end();

    if (ityr::is_master()) {
      printf("[%d] %'ld ns", r, t1 - t0);

      if (verify_result) {
        result_t answer = fib_fast(n_input);
        if (result == answer) {
          printf(" - Result verified: fib(%d) = %ld", n_input, result);
        } else {
          printf(" - Wrong result: fib(%d) should be %ld but got %ld",
                 n_input, answer, result);
        }
      }

      printf("\n");
      fflush(stdout);
    }

    ityr::profiler_flush();
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
  ityr::init();

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

  if (ityr::is_master()) {
    setlocale(LC_NUMERIC, "en_US.UTF-8");
    printf("=============================================================\n"
           "[Fibonacci]\n"
           "# of processes:               %d\n"
           "N:                            %d\n"
           "# of repeats:                 %d\n"
           "Verify result:                %d\n"
           "-------------------------------------------------------------\n",
           ityr::n_ranks(), n_input, n_repeats, verify_result);

    printf("[Compile Options]\n");
    ityr::print_compile_options();
    printf("-------------------------------------------------------------\n");
    printf("[Runtime Options]\n");
    ityr::print_runtime_options();
    printf("=============================================================\n\n");
    fflush(stdout);
  }

  run();

  ityr::fini();
  return 0;
}
