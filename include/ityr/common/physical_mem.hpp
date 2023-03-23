#pragma once

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <sstream>

#include "ityr/common/util.hpp"
#include "ityr/common/virtual_mem.hpp"

namespace ityr::common {

class physical_mem {
public:
  physical_mem() {}
  physical_mem(const std::string& shm_name, std::size_t size, bool own)
    : shm_name_(shm_name), size_(size), own_(own), fd_(init_shmem_fd()) {}

  ~physical_mem() {
    if (fd_ != -1) {
      close(fd_);
      if (own_ && shm_unlink(shm_name_.c_str()) == -1) {
        perror("shm_unlink");
        die("[ityr::common::physical_mem] shm_unlink() failed");
      }
    }
  }

  physical_mem(const physical_mem&) = delete;
  physical_mem& operator=(const physical_mem&) = delete;

  physical_mem(physical_mem&& pm)
    : shm_name_(std::move(pm.shm_name_)), size_(pm.size_), own_(pm.own_), fd_(pm.fd_) { pm.fd_ = -1; }
  physical_mem& operator=(physical_mem&& pm) {
    this->~physical_mem();
    shm_name_ = std::move(pm.shm_name_);
    size_     = pm.size_;
    own_      = pm.own_;
    fd_       = pm.fd_;
    pm.fd_ = -1;
    return *this;
  }

  std::size_t size() const { return size_; }

  void map_to_vm(void* addr, std::size_t size, std::size_t offset) const {
    ITYR_CHECK(addr != nullptr);
    ITYR_CHECK(reinterpret_cast<uintptr_t>(addr) % get_page_size() == 0);
    ITYR_CHECK(offset % get_page_size() == 0);
    ITYR_CHECK(offset + size <= size_);
    // MAP_FIXED_NOREPLACE is never set here, as this map method is used to
    // map to physical memory a given virtual address, which is already reserved by mmap.
    void* ret = mmap(addr, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd_, offset);
    if (ret == MAP_FAILED) {
      perror("mmap");
      die("[ityr::common::physical_mem] mmap(%p, %lu, ...) failed", addr, size);
    }
  }

private:
  int init_shmem_fd() const {
    int oflag = O_RDWR;
    if (own_) oflag |= O_CREAT | O_TRUNC;

    int fd = shm_open(shm_name_.c_str(), oflag, S_IRUSR | S_IWUSR);
    if (fd == -1) {
      perror("shm_open");
      die("[ityr::common::physical_mem] shm_open() failed");
    }

    if (own_ && ftruncate(fd, size_) == -1) {
      perror("ftruncate");
      die("[ityr::common::physical_mem] ftruncate(%d, %lu) failed", fd, size_);
    }

    return fd;
  }

  std::string shm_name_;
  std::size_t size_;
  bool        own_;
  int         fd_ = -1;
};

ITYR_TEST_CASE("[ityr::common::physical_mem] map physical memory to two different virtual addresses") {
  runtime_options opts;
  singleton_initializer<topology::instance> topo;

  std::size_t pagesize = get_page_size();

  std::stringstream ss;
  ss << "/ityr_test_" << topology::my_rank();

  std::size_t alloc_size = 16 * pagesize;

  physical_mem pm(ss.str(), alloc_size, true);

  virtual_mem vm1(alloc_size);
  virtual_mem vm2(alloc_size);
  ITYR_CHECK(vm1.addr() != vm2.addr());

  int* b1;
  int* b2;

  ITYR_SUBCASE("map whole memory") {
    b1 = reinterpret_cast<int*>(vm1.addr());
    b2 = reinterpret_cast<int*>(vm2.addr());
    pm.map_to_vm(b1, alloc_size, 0);
    pm.map_to_vm(b2, alloc_size, 0);
  }

  ITYR_SUBCASE("map partial memory") {
    std::size_t offset = 3 * pagesize;
    b1 = reinterpret_cast<int*>(reinterpret_cast<std::byte*>(vm1.addr()) + offset);
    b2 = reinterpret_cast<int*>(reinterpret_cast<std::byte*>(vm2.addr()) + offset);
    pm.map_to_vm(b1, pagesize, offset);
    pm.map_to_vm(b2, pagesize, offset);
  }

  ITYR_CHECK(b1 != b2);
  ITYR_CHECK(b1[0] == 0);
  ITYR_CHECK(b2[0] == 0);
  b1[0] = 417;
  ITYR_CHECK(b1[0] == 417);
  ITYR_CHECK(b2[0] == 417);
}

}
