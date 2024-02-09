#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/options.hpp"

#if __has_include(<numa.h>)
#include <numa.h>

namespace ityr::common::numa {

inline bool enabled() {
  return numa_available() >= 0;
}

using node_t = int;

class node_bitmask {
public:
  node_bitmask()
    : bitmask_(alloc_node_bitmask()) {}

  struct bitmask* get() const { return bitmask_->get(); }

  void setbit(node_t node) {
    if (!enabled()) return;
    ITYR_CHECK(bitmask_.has_value());
    ITYR_CHECK(0 <= node);
    ITYR_CHECK(node <= numa_max_node());
    numa_bitmask_setbit(get(), node);
  }

  void clear() {
    if (!enabled()) return;
    ITYR_CHECK(bitmask_.has_value());
    numa_bitmask_clearall(get());
  }

private:
  using bitmask_holder_t = std::unique_ptr<struct bitmask, void(*)(struct bitmask*)>;

  std::optional<bitmask_holder_t> alloc_node_bitmask() const {
    if (!enabled()) return std::nullopt;
    return bitmask_holder_t{numa_allocate_nodemask(), numa_free_nodemask};
  }

  std::optional<bitmask_holder_t> bitmask_;
};

inline node_t get_current_node() {
  if (!enabled()) return 0;
  return numa_node_of_cpu(sched_getcpu());
}

inline void bind_to(void* addr, std::size_t size, node_t node) {
  if (!enabled()) return;
  ITYR_CHECK(0 <= node);
  ITYR_CHECK(node <= numa_max_node());
  ITYR_CHECK(reinterpret_cast<uintptr_t>(addr) % numa_pagesize() == 0);
  ITYR_CHECK(size % get_page_size() == 0);
  numa_tonode_memory(addr, size, node);
}

inline void interleave(void* addr, std::size_t size, const node_bitmask& nodemask) {
  if (!enabled()) return;
  ITYR_CHECK(reinterpret_cast<uintptr_t>(addr) % numa_pagesize() == 0);
  ITYR_CHECK(size % numa_pagesize() == 0);
  numa_interleave_memory(addr, size, nodemask.get());
}

}

#else

namespace ityr::common::numa {

using node_t = int;

class node_bitmask {
public:
  node_bitmask() {}
  void* get() const { return nullptr; }
  void setbit(node_t) {}
  void clear() {}
};

inline bool enabled() { return false; }
inline node_t get_current_node() { return 0; }
inline void bind_to(void*, std::size_t, node_t) {}
inline void interleave(void*, std::size_t, const node_bitmask&) {}

}

#endif
