#pragma once

#include <optional>
#include <algorithm>

#include "ityr/common/util.hpp"
#include "ityr/ori/cache_system.hpp"
#include "ityr/ori/coll_mem.hpp"
#include "ityr/ori/options.hpp"

namespace ityr::ori {

enum class access_mode {
  read,
  write,
  read_write,
};

class core {
public:
  static constexpr std::size_t block_size = ITYR_ORI_BLOCK_SIZE;

  core() {}

  void* malloc_coll(std::size_t size) { return malloc_coll<default_mem_mapper>(size); }

  template <template <std::size_t> typename MemMapper, typename... MemMapperArgs>
  void* malloc_coll(std::size_t size, MemMapperArgs... mmargs) {
    if (size == 0) {
      common::die("Memory allocation size cannot be 0");
    }

    auto mmapper = std::make_unique<MemMapper<block_size>>(size, common::topology::n_ranks(), mmargs...);
    coll_mem& cm = coll_mem_create(size, std::move(mmapper));
    return cm.vm().addr();
  }

  void free_coll(void* addr) {
    if (!addr) {
      common::die("Null pointer was passed to free_coll()");
    }

    coll_mem& cm = coll_mem_get(addr);
    ITYR_CHECK(addr == cm.vm().addr());

    coll_mem_destroy(cm);
  }

  template <access_mode Mode>
  void checkout(void* addr, std::size_t size);

  template <access_mode Mode>
  void checkin(void* addr, std::size_t size);

private:
  coll_mem& coll_mem_get(void* addr) {
    for (auto [addr_begin, addr_end, id] : coll_mem_ids_) {
      if (addr_begin <= addr && addr < addr_end) {
        return *coll_mems_[id];
      }
    }
    common::die("Address %p was passed but not allocated by Itoyori", addr);
  }

  coll_mem& coll_mem_create(std::size_t size, std::unique_ptr<mem_mapper::base> mmapper) {
    coll_mem_id_t id = coll_mems_.size();

    coll_mem& cm = *coll_mems_.emplace_back(std::in_place, size, id, std::move(mmapper));
    std::byte* raw_ptr = reinterpret_cast<std::byte*>(cm.vm().addr());

    coll_mem_ids_.emplace_back(std::make_tuple(raw_ptr, raw_ptr + size, id));

    coll_mem_writing_back_.emplace_back(false);
    ITYR_CHECK(coll_mems_.size() == coll_mem_writing_back_.size());

    return cm;
  }

  void coll_mem_destroy(coll_mem& cm) {
    std::byte* p = reinterpret_cast<std::byte*>(cm.vm().addr());
    auto it = std::find(coll_mem_ids_.begin(), coll_mem_ids_.end(),
                        std::make_tuple(p, p + cm.size(), cm.id()));
    ITYR_CHECK(it != coll_mem_ids_.end());
    coll_mem_ids_.erase(it);

    coll_mems_[cm.id()].reset();

    coll_mem_writing_back_[cm.id()] = false;
  }

  template <std::size_t BlockSize>
  using default_mem_mapper = mem_mapper::ITYR_ORI_DEFAULT_MEM_MAPPER<BlockSize>;

  std::vector<std::optional<coll_mem>>                 coll_mems_;
  std::vector<std::tuple<void*, void*, coll_mem_id_t>> coll_mem_ids_;
  std::vector<bool>                                    coll_mem_writing_back_;
};

ITYR_TEST_CASE("[ityr::ori::core] malloc and free with block policy") {
  common::singleton_initializer<common::topology::instance> topo;
  core c;

  ITYR_SUBCASE("free immediately") {
    int n = 10;
    for (int i = 1; i < n; i++) {
      auto p = c.malloc_coll<mem_mapper::block>(i * 1234);
      c.free_coll(p);
    }
  }

  ITYR_SUBCASE("free after accumulation") {
    int n = 10;
    void* ptrs[n];
    for (int i = 1; i < n; i++) {
      ptrs[i] = c.malloc_coll<mem_mapper::block>(i * 2743);
    }
    for (int i = 1; i < n; i++) {
      c.free_coll(ptrs[i]);
    }
  }
}

ITYR_TEST_CASE("[pcas::pcas] malloc and free with cyclic policy") {
  common::singleton_initializer<common::topology::instance> topo;
  core c;

  ITYR_SUBCASE("free immediately") {
    int n = 10;
    for (int i = 1; i < n; i++) {
      auto p = c.malloc_coll<mem_mapper::cyclic>(i * 123456);
      c.free_coll(p);
    }
  }

  ITYR_SUBCASE("free after accumulation") {
    int n = 10;
    void* ptrs[n];
    for (int i = 1; i < n; i++) {
      ptrs[i] = c.malloc_coll<mem_mapper::cyclic>(i * 27438, core::block_size * i);
    }
    for (int i = 1; i < n; i++) {
      c.free_coll(ptrs[i]);
    }
  }
}

}
