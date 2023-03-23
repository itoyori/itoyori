#pragma once

#include <limits>
#include <forward_list>

#include "ityr/common/util.hpp"
#include "ityr/ori/util.hpp"

namespace ityr::ori {

struct block_region {
  template <typename T1, typename T2>
  block_region(T1 b, T2 e) {
    ITYR_CHECK(0 <= b);
    ITYR_CHECK(static_cast<uint64_t>(b) < static_cast<uint64_t>(std::numeric_limits<block_size_t>::max()));
    ITYR_CHECK(0 <= e);
    ITYR_CHECK(static_cast<uint64_t>(e) < static_cast<uint64_t>(std::numeric_limits<block_size_t>::max()));
    begin = static_cast<block_size_t>(b);
    end   = static_cast<block_size_t>(e);
  }

  block_size_t begin;
  block_size_t end;
};

inline bool operator==(const block_region& br1, const block_region& br2) noexcept {
  return br1.begin == br2.begin && br1.end == br2.end;
}

inline bool operator!=(const block_region& br1, const block_region& br2) noexcept {
  return !(br1 == br2);
}

inline bool overlap(const block_region& br1, const block_region& br2) {
  return br1.begin < br2.end && br2.begin < br1.end;
}

inline bool contiguous(const block_region& br1, const block_region& br2) {
  return br1.begin == br2.end || br1.end == br2.begin;
}

inline block_region get_union(const block_region& br1, const block_region& br2) {
  ITYR_CHECK((overlap(br1, br2) || contiguous(br1, br2)));
  return {std::min(br1.begin, br2.begin), std::max(br1.end, br2.end)};
}

inline block_region get_intersection(const block_region& br1, const block_region& br2) {
  ITYR_CHECK(overlap(br1, br2));
  return {std::max(br1.begin, br2.begin), std::min(br1.end, br2.end)};
}

class block_regions {
public:
  block_regions() {}
  block_regions(std::initializer_list<block_region> regions)
    : regions_(regions) {}

  auto& get() { return regions_; }

  bool empty() const {
    return regions_.empty();
  }

  void clear() {
    regions_.clear();
  }

  void add(block_region br) {
    auto it = regions_.before_begin();

    // skip until it overlaps br (or br < it)
    while (std::next(it) != regions_.end() &&
           std::next(it)->end < br.begin) it++;

    if (std::next(it) == regions_.end() ||
        br.end < std::next(it)->begin) {
      // no overlap
      regions_.insert_after(it, br);
    } else {
      // at least two sections are overlapping -> merge
      it++;
      *it = get_union(*it, br);

      while (std::next(it) != regions_.end() &&
             it->end >= std::next(it)->begin) {
        *it = get_union(*it, *std::next(it));
        regions_.erase_after(it);
      }
    }
  }

  void remove(block_region br) {
    auto it = regions_.before_begin();

    while (std::next(it) != regions_.end()) {
      if (br.end <= std::next(it)->begin) break;

      if (std::next(it)->end <= br.begin) {
        // no overlap
        it++;
      } else if (br.begin <= std::next(it)->begin && std::next(it)->end <= br.end) {
        // br contains std::next(it)
        regions_.erase_after(it);
        // do not increment it
      } else if (br.begin <= std::next(it)->begin && br.end <= std::next(it)->end) {
        // the left end of std::next(it) is overlaped
        std::next(it)->begin = br.end;
        it++;
      } else if (std::next(it)->begin <= br.begin && std::next(it)->end <= br.end) {
        // the right end of std::next(it) is overlaped
        std::next(it)->end = br.begin;
        it++;
      } else if (std::next(it)->begin <= br.begin && br.end <= std::next(it)->end) {
        // std::next(it) contains br
        block_region new_br = {std::next(it)->begin, br.begin};
        std::next(it)->begin = br.end;
        regions_.insert_after(it, new_br);
        it++;
      } else {
        common::die("Something is wrong in block_regions::remove()\n");
      }
    }
  }

  bool include(block_region br) {
    for (const auto& br_ : regions_) {
      if (br.begin < br_.begin) break;
      if (br.end <= br_.end) return true;
    }
    return false;
  }

  auto inverse(block_region br) {
    std::forward_list<block_region> ret;
    auto it = ret.before_begin();
    for (auto [b, e] : regions_) {
      if (br.begin < b) {
        it = ret.insert_after(it, {br.begin, std::min(b, br.end)});
      }
      if (br.begin < e) {
        br.begin = e;
        if (br.begin >= br.end) break;
      }
    }
    if (br.begin < br.end) {
      ret.insert_after(it, br);
    }
    return ret;
  }

private:
  std::forward_list<block_region> regions_;
};

ITYR_TEST_CASE("[ityr::ori::block_regions] add") {
  block_regions brs;
  auto check_equal = [&](std::forward_list<block_region> ans) {
    ITYR_CHECK(brs.get() == ans);
  };
  brs.add({2, 5});
  check_equal({{2, 5}});
  brs.add({11, 20});
  check_equal({{2, 5}, {11, 20}});
  brs.add({20, 21});
  check_equal({{2, 5}, {11, 21}});
  brs.add({15, 23});
  check_equal({{2, 5}, {11, 23}});
  brs.add({8, 23});
  check_equal({{2, 5}, {8, 23}});
  brs.add({7, 25});
  check_equal({{2, 5}, {7, 25}});
  brs.add({0, 7});
  check_equal({{0, 25}});
  brs.add({30, 50});
  check_equal({{0, 25}, {30, 50}});
  brs.add({30, 50});
  check_equal({{0, 25}, {30, 50}});
  brs.add({35, 45});
  check_equal({{0, 25}, {30, 50}});
  brs.add({60, 100});
  check_equal({{0, 25}, {30, 50}, {60, 100}});
  brs.add({0, 120});
  check_equal({{0, 120}});
  brs.add({200, 300});
  check_equal({{0, 120}, {200, 300}});
  brs.add({600, 700});
  check_equal({{0, 120}, {200, 300}, {600, 700}});
  brs.add({400, 500});
  check_equal({{0, 120}, {200, 300}, {400, 500}, {600, 700}});
  brs.add({300, 600});
  check_equal({{0, 120}, {200, 700}});
  brs.add({50, 600});
  check_equal({{0, 700}});
}

ITYR_TEST_CASE("[ityr::ori::block_regions] remove") {
  block_regions brs{{2, 5}, {6, 9}, {11, 20}, {50, 100}};
  auto check_equal = [&](std::forward_list<block_region> ans) {
    ITYR_CHECK(brs.get() == ans);
  };
  brs.remove({6, 9});
  check_equal({{2, 5}, {11, 20}, {50, 100}});
  brs.remove({4, 10});
  check_equal({{2, 4}, {11, 20}, {50, 100}});
  brs.remove({70, 80});
  check_equal({{2, 4}, {11, 20}, {50, 70}, {80, 100}});
  brs.remove({18, 55});
  check_equal({{2, 4}, {11, 18}, {55, 70}, {80, 100}});
  brs.remove({10, 110});
  check_equal({{2, 4}});
  brs.remove({2, 4});
  check_equal({});
  brs.remove({2, 4});
  check_equal({});
}

ITYR_TEST_CASE("[ityr::ori::block_regions] include") {
  block_regions brs{{2, 5}, {6, 9}, {11, 20}, {50, 100}};
  ITYR_CHECK(brs.include({2, 5}));
  ITYR_CHECK(brs.include({3, 5}));
  ITYR_CHECK(brs.include({2, 4}));
  ITYR_CHECK(brs.include({3, 4}));
  ITYR_CHECK(brs.include({7, 9}));
  ITYR_CHECK(brs.include({50, 100}));
  ITYR_CHECK(!brs.include({9, 11}));
  ITYR_CHECK(!brs.include({3, 6}));
  ITYR_CHECK(!brs.include({7, 10}));
  ITYR_CHECK(!brs.include({2, 100}));
  ITYR_CHECK(!brs.include({0, 3}));
}

ITYR_TEST_CASE("[ityr::ori::block_regions] inverse") {
  block_regions brs{{2, 5}, {6, 9}, {11, 20}, {50, 100}};
  ITYR_CHECK(brs.inverse({0, 120}) == block_regions({{0, 2}, {5, 6}, {9, 11}, {20, 50}, {100, 120}}).get());
  ITYR_CHECK(brs.inverse({0, 100}) == block_regions({{0, 2}, {5, 6}, {9, 11}, {20, 50}}).get());
  ITYR_CHECK(brs.inverse({0, 25}) == block_regions({{0, 2}, {5, 6}, {9, 11}, {20, 25}}).get());
  ITYR_CHECK(brs.inverse({8, 15}) == block_regions({{9, 11}}).get());
  ITYR_CHECK(brs.inverse({30, 40}) == block_regions({{30, 40}}).get());
  ITYR_CHECK(brs.inverse({50, 100}) == block_regions({}).get());
  ITYR_CHECK(brs.inverse({60, 90}) == block_regions({}).get());
  ITYR_CHECK(brs.inverse({2, 5}) == block_regions({}).get());
  ITYR_CHECK(brs.inverse({2, 6}) == block_regions({{5, 6}}).get());
  block_regions brs_empty{};
  ITYR_CHECK(brs_empty.inverse({0, 100}) == block_regions({{0, 100}}).get());
}

}
