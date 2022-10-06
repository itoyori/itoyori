#define DOCTEST_CONFIG_IMPLEMENT
#define DOCTEST_CONFIG_NO_SHORT_MACRO_NAMES
#include "doctest/doctest.h"

#include "ityr/ityr.hpp"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  doctest::Context ctx;
  ctx.setOption("abort-after", 5);
  ctx.setOption("force-colors", true);
  ctx.applyCommandLine(argc, argv);

  int test_result = ctx.run();

  MPI_Finalize();

  return test_result;
}
