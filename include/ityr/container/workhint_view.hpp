#pragma once

#include "ityr/common/util.hpp"
#include "ityr/container/global_span.hpp"

namespace ityr {

template <typename WorkHint>
class workhint_range_view {
  using value_type = WorkHint;

public:
  constexpr workhint_range_view() noexcept {}

  constexpr explicit workhint_range_view(global_span<std::pair<value_type, value_type>> s)
    : btree_view_(s) {}

  std::pair<value_type, value_type> get_workhint() const {
    return btree_view_[0].get();
  }

  void set_workhint(const value_type& v1, const value_type& v2) const {
    return btree_view_[0].set(std::make_pair(v1, v2));
  }

  std::pair<workhint_range_view, workhint_range_view> get_children() const {
    auto n = btree_view_.size() - 1;
    ITYR_CHECK(n > 1);
    ITYR_CHECK(n % 2 == 0);
    return std::make_pair(workhint_range_view(btree_view_.subspan(1        , n / 2)),
                          workhint_range_view(btree_view_.subspan(n / 2 + 1, n / 2)));
  }

  bool has_children() const { return btree_view_.size() > 2; }

private:
  global_span<std::pair<value_type, value_type>> btree_view_;
};

}
