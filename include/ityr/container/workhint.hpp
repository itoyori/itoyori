#pragma once

#include "ityr/common/util.hpp"
#include "ityr/container/workhint_view.hpp"
#include "ityr/container/global_vector.hpp"

namespace ityr {

template <typename WorkHint>
class workhint_range {
  using value_type = WorkHint;

public:
  workhint_range(int n_leaves)
    : btree_(n_leaves * 2 - 1) {}

  operator workhint_range_view<WorkHint>() const {
    return workhint_range_view<WorkHint>(global_span(btree_));
  }

private:
  global_vector<std::pair<value_type, value_type>> btree_;
};

}
