#pragma once

#include <limits>
#include <forward_list>

#include "ityr/common/util.hpp"
#include "ityr/ori/util.hpp"

namespace ityr::ori {

template <typename T>
struct region {
  template <typename T1, typename T2>
  region(T1 b, T2 e) {
    ITYR_CHECK(0 <= b);
    ITYR_CHECK(static_cast<uint64_t>(b) <= static_cast<uint64_t>(std::numeric_limits<T>::max()));
    ITYR_CHECK(0 <= e);
    ITYR_CHECK(static_cast<uint64_t>(e) <= static_cast<uint64_t>(std::numeric_limits<T>::max()));
    begin = static_cast<T>(b);
    end   = static_cast<T>(e);
  }

  std::size_t size() const { return end - begin; }

  T begin;
  T end;
};

template <typename T>
inline bool operator==(const region<T>& r1, const region<T>& r2) noexcept {
  return r1.begin == r2.begin && r1.end == r2.end;
}

template <typename T>
inline bool operator!=(const region<T>& r1, const region<T>& r2) noexcept {
  return !(r1 == r2);
}

template <typename T>
inline bool overlap(const region<T>& r1, const region<T>& r2) {
  return r1.begin < r2.end && r2.begin < r1.end;
}

template <typename T>
inline bool contiguous(const region<T>& r1, const region<T>& r2) {
  return r1.begin == r2.end || r1.end == r2.begin;
}

template <typename T>
inline region<T> get_union(const region<T>& r1, const region<T>& r2) {
  ITYR_CHECK((overlap(r1, r2) || contiguous(r1, r2)));
  return {std::min(r1.begin, r2.begin), std::max(r1.end, r2.end)};
}

template <typename T>
inline region<T> get_intersection(const region<T>& r1, const region<T>& r2) {
  ITYR_CHECK(overlap(r1, r2));
  return {std::max(r1.begin, r2.begin), std::min(r1.end, r2.end)};
}

template <typename T>
class region_set {
public:
  using iterator       = typename std::forward_list<region<T>>::iterator;
  using const_iterator = typename std::forward_list<region<T>>::const_iterator;

  region_set() {}
  region_set(std::initializer_list<region<T>> regions)
    : regions_(regions) {}

  auto& get() { return regions_; }
  const auto& get() const { return regions_; }

  iterator before_begin() { return regions_.before_begin(); }
  iterator begin() { return regions_.begin(); }
  iterator end() { return regions_.end(); }

  const_iterator before_begin() const { return regions_.cbefore_begin(); }
  const_iterator begin() const { return regions_.cbegin(); }
  const_iterator end() const { return regions_.cend(); }

  bool empty() const {
    return regions_.empty();
  }

  void clear() {
    regions_.clear();
  }

  iterator add(const region<T>& r, iterator begin_it) {
    auto it = begin_it;

    // skip until it overlaps r (or r < it)
    while (std::next(it) != regions_.end() &&
           std::next(it)->end < r.begin) it++;

    if (std::next(it) == regions_.end() ||
        r.end < std::next(it)->begin) {
      // no overlap
      it = regions_.insert_after(it, r);
    } else {
      // at least two regions are overlapping -> merge
      it++;
      *it = get_union(*it, r);

      while (std::next(it) != regions_.end() &&
             it->end >= std::next(it)->begin) {
        *it = get_union(*it, *std::next(it));
        regions_.erase_after(it);
      }
    }

    // return an iterator to the added element
    return it;
  }

  iterator add(const region<T>& r) {
    return add(r, regions_.before_begin());
  }

  void remove(const region<T>& r) {
    auto it = regions_.before_begin();

    while (std::next(it) != regions_.end()) {
      if (r.end <= std::next(it)->begin) break;

      if (std::next(it)->end <= r.begin) {
        // no overlap
        it++;
      } else if (r.begin <= std::next(it)->begin && std::next(it)->end <= r.end) {
        // r contains std::next(it)
        regions_.erase_after(it);
        // do not increment it
      } else if (r.begin <= std::next(it)->begin && r.end <= std::next(it)->end) {
        // the left end of std::next(it) is overlaped
        std::next(it)->begin = r.end;
        it++;
      } else if (std::next(it)->begin <= r.begin && std::next(it)->end <= r.end) {
        // the right end of std::next(it) is overlaped
        std::next(it)->end = r.begin;
        it++;
      } else if (std::next(it)->begin <= r.begin && r.end <= std::next(it)->end) {
        // std::next(it) contains r
        region<T> new_r = {std::next(it)->begin, r.begin};
        std::next(it)->begin = r.end;
        regions_.insert_after(it, new_r);
        it++;
      } else {
        common::die("Something is wrong in region<T>s::remove()\n");
      }
    }
  }

  bool include(const region<T>& r) const {
    for (const auto& r_ : regions_) {
      if (r.begin < r_.begin) break;
      if (r.end <= r_.end) return true;
    }
    return false;
  }

  region_set<T> complement(region<T> r) const {
    region_set<T> ret;
    std::forward_list<region<T>>& regs = ret.regions_;

    auto it = regs.before_begin();
    for (auto [b, e] : regions_) {
      if (r.begin < b) {
        it = regs.insert_after(it, {r.begin, std::min(b, r.end)});
      }
      if (r.begin < e) {
        r.begin = e;
        if (r.begin >= r.end) break;
      }
    }
    if (r.begin < r.end) {
      regs.insert_after(it, r);
    }
    return ret;
  }

  region_set<T> intersection(const region<T>& r) const {
    region_set<T> ret;
    std::forward_list<region<T>>& regs = ret.get();

    auto it_ret = regs.before_begin();
    auto it = regions_.begin();

    while (it != regions_.end()) {
      if (it->end <= r.begin) {
        it++;
        continue;
      }

      if (r.end <= it->begin) {
        break;
      }

      it_ret = regs.insert_after(it_ret, get_intersection(*it, r));

      if (r.end < it->end) {
        break;
      }

      it++;
    }

    return ret;
  }

  region_set<T> intersection(const region_set<T>& rs) const {
    region_set<T> ret;
    std::forward_list<region<T>>& regs = ret.get();

    auto it_ret = regs.before_begin();
    auto it1 = regions_.begin();
    auto it2 = rs.begin();

    while (it1 != regions_.end() && it2 != rs.end()) {
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

  std::size_t size() const {
    std::size_t ret = 0;
    for (auto&& reg : regions_) {
      ret += reg.size();
    }
    return ret;
  }

private:
  std::forward_list<region<T>> regions_;
};

template <typename T>
inline bool operator==(const region_set<T>& rs1, const region_set<T>& rs2) noexcept {
  return rs1.get() == rs2.get();
}

template <typename T>
inline bool operator!=(const region_set<T>& rs1, const region_set<T>& rs2) noexcept {
  return !(rs1 == rs2);
}

template <typename T>
inline region_set<T> get_complement(const region_set<T>& rs, const region<T>& r) {
  return rs.complement(r);
}

template <typename T>
inline region_set<T> get_intersection(const region_set<T>& rs, const region<T>& r) {
  return rs.intersection(r);
}

template <typename T>
inline region_set<T> get_intersection(const region_set<T>& rs1, const region_set<T>& rs2) {
  return rs1.intersection(rs2);
}

using block_region = region<block_size_t>;
using block_region_set = region_set<block_size_t>;

ITYR_TEST_CASE("[ityr::ori::block_region_set] add") {
  block_region_set brs;
  brs.add({2, 5});
  ITYR_CHECK(brs == (block_region_set{{2, 5}}));
  brs.add({11, 20});
  ITYR_CHECK(brs == (block_region_set{{2, 5}, {11, 20}}));
  brs.add({20, 21});
  ITYR_CHECK(brs == (block_region_set{{2, 5}, {11, 21}}));
  brs.add({15, 23});
  ITYR_CHECK(brs == (block_region_set{{2, 5}, {11, 23}}));
  brs.add({8, 23});
  ITYR_CHECK(brs == (block_region_set{{2, 5}, {8, 23}}));
  brs.add({7, 25});
  ITYR_CHECK(brs == (block_region_set{{2, 5}, {7, 25}}));
  brs.add({0, 7});
  ITYR_CHECK(brs == (block_region_set{{0, 25}}));
  brs.add({30, 50});
  ITYR_CHECK(brs == (block_region_set{{0, 25}, {30, 50}}));
  brs.add({30, 50});
  ITYR_CHECK(brs == (block_region_set{{0, 25}, {30, 50}}));
  brs.add({35, 45});
  ITYR_CHECK(brs == (block_region_set{{0, 25}, {30, 50}}));
  brs.add({60, 100});
  ITYR_CHECK(brs == (block_region_set{{0, 25}, {30, 50}, {60, 100}}));
  brs.add({0, 120});
  ITYR_CHECK(brs == (block_region_set{{0, 120}}));
  brs.add({200, 300});
  ITYR_CHECK(brs == (block_region_set{{0, 120}, {200, 300}}));
  brs.add({600, 700});
  ITYR_CHECK(brs == (block_region_set{{0, 120}, {200, 300}, {600, 700}}));
  brs.add({400, 500});
  ITYR_CHECK(brs == (block_region_set{{0, 120}, {200, 300}, {400, 500}, {600, 700}}));
  brs.add({300, 600});
  ITYR_CHECK(brs == (block_region_set{{0, 120}, {200, 700}}));
  brs.add({50, 600});
  ITYR_CHECK(brs == (block_region_set{{0, 700}}));
}

ITYR_TEST_CASE("[ityr::ori::block_region_set] remove") {
  block_region_set brs{{2, 5}, {6, 9}, {11, 20}, {50, 100}};
  brs.remove({6, 9});
  ITYR_CHECK(brs == (block_region_set{{2, 5}, {11, 20}, {50, 100}}));
  brs.remove({4, 10});
  ITYR_CHECK(brs == (block_region_set{{2, 4}, {11, 20}, {50, 100}}));
  brs.remove({70, 80});
  ITYR_CHECK(brs == (block_region_set{{2, 4}, {11, 20}, {50, 70}, {80, 100}}));
  brs.remove({18, 55});
  ITYR_CHECK(brs == (block_region_set{{2, 4}, {11, 18}, {55, 70}, {80, 100}}));
  brs.remove({10, 110});
  ITYR_CHECK(brs == (block_region_set{{2, 4}}));
  brs.remove({2, 4});
  ITYR_CHECK(brs == (block_region_set{}));
  brs.remove({2, 4});
  ITYR_CHECK(brs == (block_region_set{}));
}

ITYR_TEST_CASE("[ityr::ori::block_region_set] include") {
  block_region_set brs{{2, 5}, {6, 9}, {11, 20}, {50, 100}};
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

ITYR_TEST_CASE("[ityr::ori::block_region_set] complement") {
  block_region_set brs{{2, 5}, {6, 9}, {11, 20}, {50, 100}};
  ITYR_CHECK(brs.complement({0, 120}) == block_region_set({{0, 2}, {5, 6}, {9, 11}, {20, 50}, {100, 120}}));
  ITYR_CHECK(brs.complement({0, 100}) == block_region_set({{0, 2}, {5, 6}, {9, 11}, {20, 50}}));
  ITYR_CHECK(brs.complement({0, 25}) == block_region_set({{0, 2}, {5, 6}, {9, 11}, {20, 25}}));
  ITYR_CHECK(brs.complement({8, 15}) == block_region_set({{9, 11}}));
  ITYR_CHECK(brs.complement({30, 40}) == block_region_set({{30, 40}}));
  ITYR_CHECK(brs.complement({50, 100}) == block_region_set({}));
  ITYR_CHECK(brs.complement({60, 90}) == block_region_set({}));
  ITYR_CHECK(brs.complement({2, 5}) == block_region_set({}));
  ITYR_CHECK(brs.complement({2, 6}) == block_region_set({{5, 6}}));
  block_region_set brs_empty{};
  ITYR_CHECK(brs_empty.complement({0, 100}) == block_region_set({{0, 100}}));
}

ITYR_TEST_CASE("[ityr::ori::block_region_set] intersection") {
  ITYR_CHECK((get_intersection(block_region_set{{2, 5}, {11, 20}, {25, 27}, {50, 100}},
                               block_region_set{{3, 4}, {9, 15}, {16, 19}, {50, 100}})) ==
             (block_region_set{{3, 4}, {11, 15}, {16, 19}, {50, 100}}));
  ITYR_CHECK((get_intersection(block_region_set{},
                               block_region_set{{3, 4}, {9, 15}, {16, 19}, {50, 100}})) ==
             (block_region_set{}));
  ITYR_CHECK((get_intersection(block_region_set{{2, 5}, {11, 20}, {25, 27}, {50, 100}},
                               block_region_set{})) ==
             (block_region_set{}));
  ITYR_CHECK((get_intersection(block_region_set{},
                               block_region_set{})) ==
             (block_region_set{}));
}

}
