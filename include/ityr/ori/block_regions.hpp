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

  std::size_t size() const { return end - begin; }

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
  const auto& get() const { return regions_; }

  auto begin() { return regions_.begin(); }
  auto end() { return regions_.end(); }

  auto begin() const { return regions_.begin(); }
  auto end() const { return regions_.end(); }

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

  bool include(block_region br) const {
    for (const auto& br_ : regions_) {
      if (br.begin < br_.begin) break;
      if (br.end <= br_.end) return true;
    }
    return false;
  }

  block_regions inverse(block_region br) const {
    block_regions ret;
    std::forward_list<block_region>& regs = ret.regions_;

    auto it = regs.before_begin();
    for (auto [b, e] : regions_) {
      if (br.begin < b) {
        it = regs.insert_after(it, {br.begin, std::min(b, br.end)});
      }
      if (br.begin < e) {
        br.begin = e;
        if (br.begin >= br.end) break;
      }
    }
    if (br.begin < br.end) {
      regs.insert_after(it, br);
    }
    return ret;
  }

  std::size_t size() const {
    std::size_t ret = 0;
    for (auto&& reg : regions_) {
      ret += reg.size();
    }
    return ret;
  }

private:
  std::forward_list<block_region> regions_;
};

inline bool operator==(const block_regions& brs1, const block_regions& brs2) noexcept {
  return brs1.get() == brs2.get();
}

inline bool operator!=(const block_regions& brs1, const block_regions& brs2) noexcept {
  return !(brs1 == brs2);
}

inline block_regions get_intersection(const block_regions& brs1, const block_regions& brs2) {
  block_regions ret;
  std::forward_list<block_region>& regs = ret.get();

  auto it_ret = regs.before_begin();
  auto it1 = brs1.begin();
  auto it2 = brs2.begin();

  while (it1 != brs1.end() && it2 != brs2.end()) {
    if (it1->end <= it2->begin) {
      it1++;
      continue;
    }

    if (it2->end <= it1->begin) {
      it2++;
      continue;
    }

    it_ret = regs.insert_after(it_ret, get_intersection(*it1, *it2));

    if (it1->end <= it2->end) {
      it1++;
    } else {
      it2++;
    }
  }

  return ret;
}

ITYR_TEST_CASE("[ityr::ori::block_regions] add") {
  block_regions brs;
  brs.add({2, 5});
  ITYR_CHECK(brs == (block_regions{{2, 5}}));
  brs.add({11, 20});
  ITYR_CHECK(brs == (block_regions{{2, 5}, {11, 20}}));
  brs.add({20, 21});
  ITYR_CHECK(brs == (block_regions{{2, 5}, {11, 21}}));
  brs.add({15, 23});
  ITYR_CHECK(brs == (block_regions{{2, 5}, {11, 23}}));
  brs.add({8, 23});
  ITYR_CHECK(brs == (block_regions{{2, 5}, {8, 23}}));
  brs.add({7, 25});
  ITYR_CHECK(brs == (block_regions{{2, 5}, {7, 25}}));
  brs.add({0, 7});
  ITYR_CHECK(brs == (block_regions{{0, 25}}));
  brs.add({30, 50});
  ITYR_CHECK(brs == (block_regions{{0, 25}, {30, 50}}));
  brs.add({30, 50});
  ITYR_CHECK(brs == (block_regions{{0, 25}, {30, 50}}));
  brs.add({35, 45});
  ITYR_CHECK(brs == (block_regions{{0, 25}, {30, 50}}));
  brs.add({60, 100});
  ITYR_CHECK(brs == (block_regions{{0, 25}, {30, 50}, {60, 100}}));
  brs.add({0, 120});
  ITYR_CHECK(brs == (block_regions{{0, 120}}));
  brs.add({200, 300});
  ITYR_CHECK(brs == (block_regions{{0, 120}, {200, 300}}));
  brs.add({600, 700});
  ITYR_CHECK(brs == (block_regions{{0, 120}, {200, 300}, {600, 700}}));
  brs.add({400, 500});
  ITYR_CHECK(brs == (block_regions{{0, 120}, {200, 300}, {400, 500}, {600, 700}}));
  brs.add({300, 600});
  ITYR_CHECK(brs == (block_regions{{0, 120}, {200, 700}}));
  brs.add({50, 600});
  ITYR_CHECK(brs == (block_regions{{0, 700}}));
}

ITYR_TEST_CASE("[ityr::ori::block_regions] remove") {
  block_regions brs{{2, 5}, {6, 9}, {11, 20}, {50, 100}};
  brs.remove({6, 9});
  ITYR_CHECK(brs == (block_regions{{2, 5}, {11, 20}, {50, 100}}));
  brs.remove({4, 10});
  ITYR_CHECK(brs == (block_regions{{2, 4}, {11, 20}, {50, 100}}));
  brs.remove({70, 80});
  ITYR_CHECK(brs == (block_regions{{2, 4}, {11, 20}, {50, 70}, {80, 100}}));
  brs.remove({18, 55});
  ITYR_CHECK(brs == (block_regions{{2, 4}, {11, 18}, {55, 70}, {80, 100}}));
  brs.remove({10, 110});
  ITYR_CHECK(brs == (block_regions{{2, 4}}));
  brs.remove({2, 4});
  ITYR_CHECK(brs == (block_regions{}));
  brs.remove({2, 4});
  ITYR_CHECK(brs == (block_regions{}));
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
  ITYR_CHECK(brs.inverse({0, 120}) == block_regions({{0, 2}, {5, 6}, {9, 11}, {20, 50}, {100, 120}}));
  ITYR_CHECK(brs.inverse({0, 100}) == block_regions({{0, 2}, {5, 6}, {9, 11}, {20, 50}}));
  ITYR_CHECK(brs.inverse({0, 25}) == block_regions({{0, 2}, {5, 6}, {9, 11}, {20, 25}}));
  ITYR_CHECK(brs.inverse({8, 15}) == block_regions({{9, 11}}));
  ITYR_CHECK(brs.inverse({30, 40}) == block_regions({{30, 40}}));
  ITYR_CHECK(brs.inverse({50, 100}) == block_regions({}));
  ITYR_CHECK(brs.inverse({60, 90}) == block_regions({}));
  ITYR_CHECK(brs.inverse({2, 5}) == block_regions({}));
  ITYR_CHECK(brs.inverse({2, 6}) == block_regions({{5, 6}}));
  block_regions brs_empty{};
  ITYR_CHECK(brs_empty.inverse({0, 100}) == block_regions({{0, 100}}));
}

ITYR_TEST_CASE("[ityr::ori::block_regions] intersection") {
  ITYR_CHECK((get_intersection(block_regions{{2, 5}, {11, 20}, {25, 27}, {50, 100}},
                               block_regions{{3, 4}, {9, 15}, {16, 19}, {50, 100}})) ==
             (block_regions{{3, 4}, {11, 15}, {16, 19}, {50, 100}}));
  ITYR_CHECK((get_intersection(block_regions{},
                               block_regions{{3, 4}, {9, 15}, {16, 19}, {50, 100}})) ==
             (block_regions{}));
  ITYR_CHECK((get_intersection(block_regions{{2, 5}, {11, 20}, {25, 27}, {50, 100}},
                               block_regions{})) ==
             (block_regions{}));
  ITYR_CHECK((get_intersection(block_regions{},
                               block_regions{})) ==
             (block_regions{}));
}

}
