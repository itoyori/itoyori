#pragma once

#include "ityr/common/util.hpp"
#include "ityr/ori/util.hpp"
#include "ityr/ori/mem_mapper.hpp"
#include "ityr/ori/coll_mem.hpp"

namespace ityr::ori {

class coll_mem_manager {
public:
  coll_mem_manager() {}

  coll_mem& get(void* addr) {
    for (auto [addr_begin, addr_end, id] : coll_mem_ids_) {
      if (addr_begin <= addr && addr < addr_end) {
        return *coll_mems_[id];
      }
    }
    common::die("Address %p was passed but not allocated by Itoyori", addr);
  }

  coll_mem& create(std::size_t size, std::unique_ptr<mem_mapper::base> mmapper) {
    coll_mem_id_t id = coll_mems_.size();

    coll_mem& cm = *coll_mems_.emplace_back(std::in_place, size, id, std::move(mmapper));
    std::byte* raw_ptr = reinterpret_cast<std::byte*>(cm.vm().addr());

    coll_mem_ids_.emplace_back(std::make_tuple(raw_ptr, raw_ptr + size, id));

    return cm;
  }

  void destroy(coll_mem& cm) {
    std::byte* p = reinterpret_cast<std::byte*>(cm.vm().addr());
    auto it = std::find(coll_mem_ids_.begin(), coll_mem_ids_.end(),
                        std::make_tuple(p, p + cm.size(), cm.id()));
    ITYR_CHECK(it != coll_mem_ids_.end());
    coll_mem_ids_.erase(it);

    coll_mems_[cm.id()].reset();
  }

private:
  std::vector<std::optional<coll_mem>>                 coll_mems_;
  std::vector<std::tuple<void*, void*, coll_mem_id_t>> coll_mem_ids_;
};

}
