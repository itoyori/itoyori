#pragma once

#include "ityr/common/util.hpp"
#include "ityr/container/global_span.hpp"

namespace ityr {

template <typename W>
class workhint_range;

template <typename W>
class workhint_range_view {
  using value_type = W;

public:
  constexpr workhint_range_view() noexcept {}

  constexpr explicit workhint_range_view(global_span<std::pair<value_type, value_type>> s)
    : bin_tree_view_(s) {}

  std::pair<value_type, value_type> get_workhint() const {
    return bin_tree_view_[0].get();
  }

  void set_workhint(const value_type& v1, const value_type& v2) {
    bin_tree_view_[0].put(std::make_pair(v1, v2));
  }

  std::pair<workhint_range_view, workhint_range_view> get_children() const {
    auto n = bin_tree_view_.size() - 1;
    ITYR_CHECK(n > 1);
    ITYR_CHECK(n % 2 == 0);
    return std::make_pair(workhint_range_view(bin_tree_view_.subspan(1        , n / 2)),
                          workhint_range_view(bin_tree_view_.subspan(n / 2 + 1, n / 2)));
  }

  bool empty() const { return bin_tree_view_.empty(); }

  bool has_children() const { return bin_tree_view_.size() > 2; }

private:
  global_span<std::pair<value_type, value_type>> bin_tree_view_;
};

template <>
class workhint_range_view<void> {};

}
