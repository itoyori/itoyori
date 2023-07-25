#define DOCTEST_CONFIG_IMPLEMENT
#define DOCTEST_CONFIG_NO_SHORT_MACRO_NAMES
#include "doctest/doctest.h"

#define ITYR_ITO_SCHEDULER adws
#include "ityr/ityr.hpp"

// from: https://github.com/doctest/doctest/blob/master/examples/all_features/asserts_used_outside_of_tests.cpp
void assert_handler(const doctest::AssertData& ad) {
  using namespace doctest;

  std::cout << Color::LightGrey << skipPathFromFilename(ad.m_file) << "(" << ad.m_line << "): ";
  std::cout << Color::Red << failureString(ad.m_at) << ": ";

  // handling only normal (comparison and unary) asserts - exceptions-related asserts have been skipped
  if (ad.m_at & assertType::is_normal) {
    std::cout << Color::Cyan << assertString(ad.m_at) << "( " << ad.m_expr << " ) ";
    std::cout << Color::None << (ad.m_threw ? "THREW exception: " : "is NOT correct!\n");
    if (ad.m_threw) {
      std::cout << ad.m_exception;
    } else {
      std::cout << "  values: " << assertString(ad.m_at) << "( " << ad.m_decomp << " )";
    }
  } else {
    std::cout << Color::None << "an assert dealing with exceptions has failed!";
  }

  std::cout << std::endl;
  std::abort();
}

void adws_test() {
  int n_tasks_per_rank = 1024;
  int n_tasks = ityr::n_ranks() * n_tasks_per_rank;

  ityr::ito::adws_enable_steal_option::set(false);

  ityr::root_exec([=] {
    int n_repeats = 3;
    for (int i = 0; i < n_repeats; i++) {
      ityr::for_each(
          ityr::execution::par,
          ityr::count_iterator<int>(0),
          ityr::count_iterator<int>(n_tasks),
          [=](int i) {
            /* printf("%d\n", i); */
            ITYR_CHECK(i / n_tasks_per_rank == (ityr::n_ranks() - ityr::my_rank() - 1));
          });
    }
  });

  ityr::ito::adws_enable_steal_option::set(true);

  ityr::root_exec([=] {
    int n_repeats = 3;
    for (int i = 0; i < n_repeats; i++) {
      int n_migrated =
        ityr::transform_reduce(
            ityr::execution::par,
            ityr::count_iterator<int>(0),
            ityr::count_iterator<int>(n_tasks),
            0,
            std::plus<int>{},
            [=](int i) {
              usleep(i);
              ityr::common::mpi_make_progress();
              /* printf("%d\n", i); */
              if (i / n_tasks_per_rank != (ityr::n_ranks() - ityr::my_rank() - 1)) {
                return 1;
              } else {
                return 0;
              }
            });
      ITYR_CHECK(n_migrated > 0);
      printf("Migrated: %d/%d\n", n_migrated, n_tasks);
    }
  });
}

int main(int argc, char** argv) {
  doctest::Context ctx(argc, argv);
  ctx.setAssertHandler(assert_handler);
  ctx.setAsDefaultForAssertsOutOfTestCases();

  ityr::init();

  adws_test();

  ityr::fini();
  return 0;
}
