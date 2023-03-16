#pragma once

#include <list>
#include <vector>
#include <optional>
#include <random>
#include <algorithm>

#include "ityr/common/util.hpp"

namespace ityr::common {

class freelist {
public:
  freelist() {}
  freelist(uintptr_t addr, std::size_t size) : fl_(1, entry{addr, size}) {}

  std::optional<uintptr_t> get(std::size_t size) {
    auto it = fl_.begin();
    while (it != fl_.end()) {
      if (it->size == size) {
        auto ret = it->addr;
        fl_.erase(it);
        return ret;
      } else if (it->size > size) {
        auto ret = it->addr;
        it->addr += size;
        it->size -= size;
        return ret;
      }
      it = std::next(it);
    }
    return std::nullopt;
  }

  std::optional<uintptr_t> get(std::size_t size, std::size_t alignment) {
    // TODO: consider better implementation
    auto s = get(size);
    if (!s.has_value()) {
      return std::nullopt;
    }

    if (*s % alignment == 0) {
      return *s;
    }

    add(*s, size);

    std::size_t req_size = size + alignment;

    s = get(req_size);
    if (!s.has_value()) {
      return std::nullopt;
    }

    auto addr_got = *s;
    auto addr_ret = round_up_pow2(addr_got, alignment);

    ITYR_CHECK(addr_ret >= addr_got);
    ITYR_CHECK(addr_got + req_size >= addr_ret + size);

    add(addr_got, addr_ret - addr_got);
    add(addr_ret + size, (addr_got + req_size) - (addr_ret + size));

    return addr_ret;
  }

  void add(uintptr_t addr, std::size_t size) {
    if (size == 0) return;

    auto it = fl_.begin();
    while (it != fl_.end()) {
      if (addr + size == it->addr) {
        it->addr = addr;
        it->size += size;
        return;
      } else if (addr + size < it->addr) {
        fl_.insert(it, entry{addr, size});
        return;
      } else if (addr == it->addr + it->size) {
        it->size += size;
        auto next_it = std::next(it);
        if (next_it != fl_.end() &&
            next_it->addr == it->addr + it->size) {
          it->size += next_it->size;
          fl_.erase(next_it);
        }
        return;
      }
      it = std::next(it);
    }
    fl_.insert(it, entry{addr, size});
  }

  std::size_t count() const { return fl_.size(); }

private:
  struct entry {
    uintptr_t   addr;
    std::size_t size;
  };

  std::list<entry> fl_;
};

ITYR_TEST_CASE("[ityr::common::freelist] freelist management") {
  uintptr_t addr = 100;
  std::size_t size = 920;
  freelist fl(addr, size);

  std::vector<uintptr_t> got;

  std::size_t n = 100;
  for (std::size_t i = 0; i < size / n; i++) {
    auto s = fl.get(n);
    ITYR_CHECK(s.has_value());
    got.push_back(*s);
  }
  ITYR_CHECK(!fl.get(n).has_value());

  // check for no overlap
  for (std::size_t i = 0; i < got.size(); i++) {
    for (std::size_t j = 0; j < got.size(); j++) {
      if (i != j) {
        ITYR_CHECK((got[i] + n <= got[j] ||
                    got[j] + n <= got[i]));
      }
    }
  }

  // random shuffle
  std::random_device seed_gen;
  std::mt19937 engine(seed_gen());
  std::shuffle(got.begin(), got.end(), engine);

  for (auto&& s : got) {
    fl.add(s, n);
  }

  ITYR_CHECK(fl.count() == 1);
  ITYR_CHECK(*fl.get(size) == addr);
}

}
