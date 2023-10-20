#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cstdint>
#include <vector>
#include <list>
#include <limits>
#include <iterator>

#include "ityr/common/util.hpp"

#if __has_include(<ankerl/unordered_dense.h>)
#include <ankerl/unordered_dense.h>
namespace ityr::ori {
template <typename Key, typename Value>
using unordered_map = ankerl::unordered_dense::map<Key, Value>;
}
#else
#include <unordered_map>
namespace ityr::ori {
template <typename Key, typename Value>
using unordered_map = std::unordered_map<Key, Value>;
}
#endif

namespace ityr::ori {

using cache_entry_idx_t = int;

class cache_full_exception : public std::exception {};

template <typename Key, typename Entry>
class cache_system {
public:
  cache_system(cache_entry_idx_t nentries) : cache_system(nentries, Entry{}) {}
  cache_system(cache_entry_idx_t nentries, const Entry& e)
    : nentries_(nentries),
      entry_initial_state_(e),
      entries_(init_entries()),
      lru_(init_lru()),
      table_(init_table()) {}

  cache_entry_idx_t num_entries() const { return nentries_; }

  bool is_cached(Key key) const {
    return table_.find(key) != table_.end();
  }

  template <bool UpdateLRU = true>
  Entry& ensure_cached(Key key) {
    auto it = table_.find(key);
    if (it == table_.end()) {
      cache_entry_idx_t idx = get_empty_slot();
      cache_entry& ce = entries_[idx];

      ce.entry.on_cache_map(idx);

      ce.allocated = true;
      ce.key = key;
      table_[key] = idx;
      if constexpr (UpdateLRU) {
        move_to_back_lru(ce);
      }
      return ce.entry;
    } else {
      cache_entry_idx_t idx = it->second;
      cache_entry& ce = entries_[idx];
      if constexpr (UpdateLRU) {
        move_to_back_lru(ce);
      }
      return ce.entry;
    }
  }

  void ensure_evicted(Key key) {
    auto it = table_.find(key);
    if (it != table_.end()) {
      cache_entry_idx_t idx = it->second;
      cache_entry& ce = entries_[idx];
      ITYR_CHECK(ce.entry.is_evictable());
      ce.entry.on_evict();
      ce.key = {};
      ce.entry = entry_initial_state_;
      table_.erase(key);
      ce.allocated = false;
    }
  }

  template <typename Func>
  void for_each_entry(Func&& f) {
    for (auto& ce : entries_) {
      if (ce.allocated) {
        f(ce.entry);
      }
    }
  }

private:
  struct cache_entry {
    bool                                            allocated;
    Key                                             key;
    Entry                                           entry;
    cache_entry_idx_t                               idx = std::numeric_limits<cache_entry_idx_t>::max();
    typename std::list<cache_entry_idx_t>::iterator lru_it;

    cache_entry(const Entry& e) : entry(e) {}
  };

  std::vector<cache_entry> init_entries() {
    std::vector<cache_entry> entries;
    for (cache_entry_idx_t idx = 0; idx < nentries_; idx++) {
      cache_entry& ce = entries.emplace_back(entry_initial_state_);
      ce.allocated = false;
      ce.idx = idx;
    }
    return entries;
  }

  std::list<cache_entry_idx_t> init_lru() {
    std::list<cache_entry_idx_t> lru;
    for (auto& ce : entries_) {
      lru.push_back(ce.idx);
      ce.lru_it = std::prev(lru.end());
      ITYR_CHECK(*ce.lru_it == ce.idx);
    }
    return lru;
  }

  unordered_map<Key, cache_entry_idx_t> init_table() {
    unordered_map<Key, cache_entry_idx_t> table;
    // To improve performance of the hash table
    table.reserve(nentries_);
    return table;
  }

  void move_to_back_lru(cache_entry& ce) {
    lru_.splice(lru_.end(), lru_, ce.lru_it);
    ITYR_CHECK(std::prev(lru_.end()) == ce.lru_it);
    ITYR_CHECK(*ce.lru_it == ce.idx);
  }

  cache_entry_idx_t get_empty_slot() {
    // FIXME: Performance issue?
    for (const auto& idx : lru_) {
      cache_entry& ce = entries_[idx];
      if (!ce.allocated) {
        return ce.idx;
      }
      if (ce.entry.is_evictable()) {
        Key prev_key = ce.key;
        table_.erase(prev_key);
        ce.entry.on_evict();
        ce.allocated = false;
        return ce.idx;
      }
    }
    throw cache_full_exception{};
  }

  cache_entry_idx_t                     nentries_;
  Entry                                 entry_initial_state_;
  std::vector<cache_entry>              entries_; // index (cache_entry_idx_t) -> entry (cache_entry)
  std::list<cache_entry_idx_t>          lru_; // front (oldest) <----> back (newest)
  unordered_map<Key, cache_entry_idx_t> table_; // hash table (Key -> cache_entry_idx_t)
};

ITYR_TEST_CASE("[ityr::ori::cache_system] testing cache system") {
  using key_t = int;
  struct test_entry {
    bool              evictable = true;
    cache_entry_idx_t entry_idx = std::numeric_limits<cache_entry_idx_t>::max();

    bool is_evictable() const { return evictable; }
    void on_evict() {}
    void on_cache_map(cache_entry_idx_t idx) { entry_idx = idx; }
  };

  int nelems = 100;
  cache_system<key_t, test_entry> cs(nelems);

  int nkey = 1000;
  std::vector<key_t> keys;
  for (int i = 0; i < nkey; i++) {
    keys.push_back(i);
  }

  ITYR_SUBCASE("basic test") {
    for (key_t k : keys) {
      test_entry& e = cs.ensure_cached(k);
      ITYR_CHECK(cs.is_cached(k));
      for (int i = 0; i < 10; i++) {
        test_entry& e2 = cs.ensure_cached(k);
        ITYR_CHECK(e.entry_idx == e2.entry_idx);
      }
    }
  }

  ITYR_SUBCASE("all entries should be cached when the number of entries is small enough") {
    for (int i = 0; i < nelems; i++) {
      cs.ensure_cached(keys[i]);
      ITYR_CHECK(cs.is_cached(keys[i]));
    }
    for (int i = 0; i < nelems; i++) {
      cs.ensure_cached(keys[i]);
      ITYR_CHECK(cs.is_cached(keys[i]));
      for (int j = 0; j < nelems; j++) {
        ITYR_CHECK(cs.is_cached(keys[j]));
      }
    }
  }

  ITYR_SUBCASE("nonevictable entries should not be evicted") {
    int nrem = 50;
    for (int i = 0; i < nrem; i++) {
      test_entry& e = cs.ensure_cached(keys[i]);
      ITYR_CHECK(cs.is_cached(keys[i]));
      e.evictable = false;
    }
    for (key_t k : keys) {
      cs.ensure_cached(k);
      ITYR_CHECK(cs.is_cached(k));
      for (int j = 0; j < nrem; j++) {
        ITYR_CHECK(cs.is_cached(keys[j]));
      }
    }
    for (int i = 0; i < nrem; i++) {
      test_entry& e = cs.ensure_cached(keys[i]);
      ITYR_CHECK(cs.is_cached(keys[i]));
      e.evictable = true;
    }
  }

  ITYR_SUBCASE("should throw exception if cache is full") {
    for (int i = 0; i < nelems; i++) {
      test_entry& e = cs.ensure_cached(keys[i]);
      ITYR_CHECK(cs.is_cached(keys[i]));
      e.evictable = false;
    }
    ITYR_CHECK_THROWS_AS(cs.ensure_cached(keys[nelems]), cache_full_exception);
    for (int i = 0; i < nelems; i++) {
      test_entry& e = cs.ensure_cached(keys[i]);
      ITYR_CHECK(cs.is_cached(keys[i]));
      e.evictable = true;
    }
    cs.ensure_cached(keys[nelems]);
    ITYR_CHECK(cs.is_cached(keys[nelems]));
  }

  ITYR_SUBCASE("LRU eviction") {
    for (int i = 0; i < nkey; i++) {
      cs.ensure_cached(keys[i]);
      ITYR_CHECK(cs.is_cached(keys[i]));
      for (int j = 0; j <= i - nelems; j++) {
        ITYR_CHECK(!cs.is_cached(keys[j]));
      }
      for (int j = std::max(0, i - nelems + 1); j < i; j++) {
        ITYR_CHECK(cs.is_cached(keys[j]));
      }
    }
  }

  for (key_t k : keys) {
    cs.ensure_evicted(k);
  }
}

}
