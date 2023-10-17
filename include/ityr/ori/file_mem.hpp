#pragma once

#include <fcntl.h>
#include <sys/stat.h>

#include "ityr/common/util.hpp"
#include "ityr/common/virtual_mem.hpp"
#include "ityr/ori/util.hpp"

namespace ityr::ori {

class file_mem {
public:
  file_mem() {}

  explicit file_mem(const std::string& fpath)
    : fd_(file_open(fpath)),
      size_(file_size(fd_)) {}

  ~file_mem() {
    if (fd_ != -1) {
      file_close(fd_);
    }
  }

  file_mem(const file_mem&) = delete;
  file_mem& operator=(const file_mem&) = delete;

  file_mem(file_mem&& fm)
    : fd_(fm.fd_), size_(fm.size_) { fm.fd_ = -1; }
  file_mem& operator=(file_mem&& fm) {
    this->~file_mem();
    fd_    = fm.fd_;
    size_  = fm.size_;
    fm.fd_ = -1;
    return *this;
  }

  std::size_t size() const { return size_; }

  void map_to_vm(void* addr, std::size_t size, std::size_t offset) const {
    ITYR_CHECK(addr != nullptr);
    ITYR_CHECK(reinterpret_cast<uintptr_t>(addr) % common::get_page_size() == 0);
    ITYR_CHECK(offset % common::get_page_size() == 0);
    ITYR_CHECK(offset + size <= size_);

    void* ret = mmap(addr, size, PROT_READ, MAP_PRIVATE | MAP_FIXED, fd_, offset);
    if (ret == MAP_FAILED) {
      perror("mmap");
      common::die("[ityr::ori::file_mem] mmap(%p, %lu, ...) failed", addr, size);
    }
  }

private:
  static int file_open(const std::string& fpath) {
    int fd = open(fpath.c_str(), O_RDONLY);
    if (fd == -1) {
      perror("open");
      common::die("[ityr::ori::file_mem] open() failed");
    }
    return fd;
  }

  static void file_close(int fd) {
    if (close(fd) == -1) {
      perror("close");
      common::die("[ityr::ori::file_mem] close() failed");
    }
  }

  static std::size_t file_size(int fd) {
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
      perror("fstat");
      abort();
    }
    return sb.st_size;
  }

  int         fd_ = -1;
  std::size_t size_;
};

}
