#pragma once

#include <array>
#include <optional>
#include <algorithm>

#include "ityr/common/util.hpp"
#include "ityr/ori/util.hpp"

namespace ityr::ori {

template <typename Key, typename Entry, int NEntries = 3>
class tlb {
public:
  tlb() {}

  void add(const Key& key, const Entry& entry) {
    // FIXME: the same entry can be duplicated?
    for (auto&& te : entries_) {
      if (!te.has_value()) {
        te.emplace(key, entry, timestamp_++);
        return;
      }
    }

    // TLB is full, so evict the element with the smallest timestamp
    auto it = std::min_element(entries_.begin(), entries_.end(), [](const auto& te1, const auto& te2) {
      return te1->timestamp < te2->timestamp;
    });
    std::optional<tlb_entry>& victim = *it;
    victim.emplace(key, entry, timestamp_++);
  }

  std::optional<Entry> get(const Key& key) {
    return get([&](const Key& k) { return k == key; });
  }

  template <typename Fn>
  std::optional<Entry> get(Fn fn) {
    for (auto&& te : entries_) {
      if (te.has_value() && fn(te->key)) {
        te->timestamp = timestamp_++;
        return te->entry;
      }
    }
    return std::nullopt;
  }

  void clear() {
    for (auto&& te : entries_) {
      te.reset();
    }
    timestamp_ = 0;
  }

private:
  using timestamp_t = int;

  struct tlb_entry {
    Key         key;
    Entry       entry;
    timestamp_t timestamp;

    tlb_entry(Key k, Entry e, timestamp_t t)
      : key(k), entry(e), timestamp(t) {}
  };

  std::array<std::optional<tlb_entry>, NEntries> entries_;
  timestamp_t                                    timestamp_ = 0;
};

ITYR_TEST_CASE("[ityr::ori::tlb] test TLB") {
  using key_t = int;
  using element_t = int;
  constexpr int n_elements = 5;
  tlb<key_t, element_t, n_elements> tlb_;

  int n = 100;
  for (int i = 0; i < n; i++) {
    tlb_.add(i, i * 2);
  }

  for (int i = 0; i < n; i++) {
    auto&& ret = tlb_.get(i);
    if (i < n - n_elements) {
      ITYR_CHECK(!ret.has_value());
    } else {
      ITYR_CHECK(ret.has_value());
      ITYR_CHECK(*ret == i * 2);
    }
  }

  tlb_.clear();

  for (int i = 0; i < n; i++) {
    auto&& ret = tlb_.get(i);
    ITYR_CHECK(!ret.has_value());
  }
}

}
