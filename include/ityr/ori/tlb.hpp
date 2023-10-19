#pragma once

#include <array>
#include <algorithm>

#include "ityr/common/util.hpp"
#include "ityr/ori/util.hpp"

namespace ityr::ori {

template <typename Key, typename Entry, int NEntries = 3>
class tlb {
public:
  tlb() : tlb(Key{}, Entry{}) {}
  tlb(Key invalid_key, Entry invalid_entry) {
    entries_.fill({invalid_key, invalid_entry, 0});
  }

  void add(const Key& key, const Entry& entry) {
    // FIXME: the same entry can be duplicated?
    Key invalid_key = entries_[0].key;
    for (int i = 1; i <= NEntries; i++) {
      if (entries_[i].key == invalid_key) {
        entries_[i] = {key, entry, timestamp_++};
        return;
      }
    }

    // TLB is full, so evict the element with the smallest timestamp
    tlb_entry& victim = *std::min_element(
        entries_.begin() + 1, entries_.end(),
        [](const auto& te1, const auto& te2) {
          return te1.timestamp < te2.timestamp;
        });
    victim = {key, entry, timestamp_++};
  }

  Entry get(const Key& key) {
    return get([&](const Key& k) { return k == key; });
  }

  template <typename Fn>
  Entry get(Fn fn) {
    int found_index = 0;
    for (int i = 1; i <= NEntries; i++) {
      if (fn(entries_[i].key)) {
        // Do not immediately return here for branch-less execution (using CMOV etc.)
        found_index = i;
      }
    }
    entries_[found_index].timestamp = timestamp_++;
    return entries_[found_index].entry;
  }

  void clear() {
    tlb_entry invalid_te = entries_[0];
    entries_.fill(invalid_te);
    timestamp_ = 0;
  }

private:
  using timestamp_t = int;

  struct tlb_entry {
    Key         key;
    Entry       entry;
    timestamp_t timestamp;
  };

  std::array<tlb_entry, NEntries + 1> entries_;
  timestamp_t                         timestamp_ = 0;
};

ITYR_TEST_CASE("[ityr::ori::tlb] test TLB") {
  using key_t = int;
  using element_t = int;
  constexpr int n_elements = 5;

  key_t invalid_key = -1;
  element_t invalid_element = -1;

  tlb<key_t, element_t, n_elements> tlb_(invalid_key, invalid_element);

  int n = 100;
  for (int i = 0; i < n; i++) {
    tlb_.add(i, i * 2);
  }

  for (int i = 0; i < n; i++) {
    element_t ret = tlb_.get(i);
    if (i < n - n_elements) {
      ITYR_CHECK(ret == invalid_element);
    } else {
      ITYR_CHECK(ret == i * 2);
    }
  }

  tlb_.clear();

  for (int i = 0; i < n; i++) {
    element_t ret = tlb_.get(i);
    ITYR_CHECK(ret == invalid_element);
  }
}

}
