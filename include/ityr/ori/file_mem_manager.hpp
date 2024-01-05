#pragma once

#include <unordered_map>

#include "ityr/common/util.hpp"
#include "ityr/ori/util.hpp"
#include "ityr/ori/file_mem.hpp"

namespace ityr::ori::file_mem_manager {

class file_mem_manager {
public:
  file_mem& get(void* addr) {
    auto it = file_mem_entries_.find(addr);
    if (it != file_mem_entries_.end()) {
      return it->second.fm;
    } else {
      common::die("File pointer %p was passed but not allocated by Itoyori", addr);
    }
  }

  void* create(const std::string& fpath, bool mlock) {
    entry e(fpath, mlock);

    ITYR_CHECK(file_mem_entries_.find(e.vm.addr()) == file_mem_entries_.end());
    void* addr = e.vm.addr();

    file_mem_entries_[addr] = std::move(e);
    return addr;
  }

  void destroy(void* addr) {
    auto it = file_mem_entries_.find(addr);
    if (it != file_mem_entries_.end()) {
      file_mem_entries_.erase(it);
    } else {
      common::die("File pointer %p was passed but not allocated by Itoyori", addr);
    }
  }

private:
  struct entry {
    entry() {}
    entry(const std::string& fpath, bool mlock)
      : fm(fpath),
        vm(common::reserve_same_vm_coll(fm.size(), common::get_page_size())) {
      fm.map_to_vm(vm.addr(), vm.size(), 0);
      if (mlock) {
        if (::mlock2(vm.addr(), vm.size(), MLOCK_ONFAULT) == -1) {
          perror("mlock2");
          common::die("mlock2() failed");
        }
      }
    }

    file_mem            fm;
    common::virtual_mem vm;
  };

  std::unordered_map<void*, entry> file_mem_entries_;
};

using instance = common::singleton<file_mem_manager>;

}
