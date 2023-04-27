#pragma once

#include "ityr/common/util.hpp"
#include "ityr/ori/util.hpp"

namespace ityr::ori::mem_mapper {

struct segment {
  int         owner;
  std::size_t offset_b;
  std::size_t offset_e;
  std::size_t pm_offset;

  bool operator==(const segment& b) const {
    return owner == b.owner && offset_b == b.offset_b && offset_e == b.offset_e && pm_offset == b.pm_offset;
  }
  bool operator!=(const segment& b) const {
    return !(*this == b);
  }
};

class base {
public:
  base(std::size_t size, int n_ranks)
    : size_(size), n_ranks_(n_ranks) {}

  virtual ~base() = default;

  virtual std::size_t block_size() const = 0;

  virtual std::size_t local_size(int rank) const = 0;

  virtual std::size_t effective_size() const = 0;

  // Returns the segment info that specifies the owner and the range [offset_b, offset_e)
  // of the block that contains the given offset.
  // pm_offset is the offset from the beginning of the owner's local physical memory for the block.
  virtual segment get_segment(std::size_t offset) const = 0;

protected:
  std::size_t size_;
  int         n_ranks_;
};

template <block_size_t BlockSize>
class block : public base {
public:
  using base::base;

  std::size_t block_size() const override { return BlockSize; }

  std::size_t local_size(int rank [[maybe_unused]]) const override {
    return local_size_impl();
  }

  std::size_t effective_size() const override {
    return local_size_impl() * n_ranks_;
  }

  segment get_segment(std::size_t offset) const override {
    ITYR_CHECK(offset < effective_size());
    std::size_t size_l   = local_size_impl();
    int         owner    = offset / size_l;
    std::size_t offset_b = owner * size_l;
    std::size_t offset_e = std::min((owner + 1) * size_l, size_);
    return segment{owner, offset_b, offset_e, 0};
  }

private:
  // non-virtual common part
  std::size_t local_size_impl() const {
    std::size_t nblock_g = (size_ + BlockSize - 1) / BlockSize;
    std::size_t nblock_l = (nblock_g + n_ranks_ - 1) / n_ranks_;
    return nblock_l * BlockSize;
  }
};

ITYR_TEST_CASE("[ityr::ori::mem_mapper::block] calculate local block size") {
  constexpr block_size_t bs = 65536;
  auto local_block_size = [](std::size_t size, int n_ranks) -> std::size_t {
    return block<bs>(size, n_ranks).local_size(0);
  };
  ITYR_CHECK(local_block_size(bs * 4     , 4) == bs    );
  ITYR_CHECK(local_block_size(bs * 12    , 4) == bs * 3);
  ITYR_CHECK(local_block_size(bs * 13    , 4) == bs * 4);
  ITYR_CHECK(local_block_size(bs * 12 + 1, 4) == bs * 4);
  ITYR_CHECK(local_block_size(bs * 12 - 1, 4) == bs * 3);
  ITYR_CHECK(local_block_size(1          , 4) == bs    );
  ITYR_CHECK(local_block_size(1          , 1) == bs    );
  ITYR_CHECK(local_block_size(bs * 3     , 1) == bs * 3);
}

ITYR_TEST_CASE("[ityr::ori::mem_mapper::block] get block information at specified offset") {
  constexpr block_size_t bs = 65536;
  auto get_segment = [](std::size_t offset, std::size_t size, int n_ranks) -> segment {
    return block<bs>(size, n_ranks).get_segment(offset);
  };
  ITYR_CHECK(get_segment(0         , bs * 4     , 4) == (segment{0, 0     , bs         , 0}));
  ITYR_CHECK(get_segment(bs        , bs * 4     , 4) == (segment{1, bs    , bs * 2     , 0}));
  ITYR_CHECK(get_segment(bs * 2    , bs * 4     , 4) == (segment{2, bs * 2, bs * 3     , 0}));
  ITYR_CHECK(get_segment(bs * 3    , bs * 4     , 4) == (segment{3, bs * 3, bs * 4     , 0}));
  ITYR_CHECK(get_segment(bs * 4 - 1, bs * 4     , 4) == (segment{3, bs * 3, bs * 4     , 0}));
  ITYR_CHECK(get_segment(0         , bs * 12    , 4) == (segment{0, 0     , bs * 3     , 0}));
  ITYR_CHECK(get_segment(bs        , bs * 12    , 4) == (segment{0, 0     , bs * 3     , 0}));
  ITYR_CHECK(get_segment(bs * 3    , bs * 12    , 4) == (segment{1, bs * 3, bs * 6     , 0}));
  ITYR_CHECK(get_segment(bs * 11   , bs * 12 - 1, 4) == (segment{3, bs * 9, bs * 12 - 1, 0}));
}

template <block_size_t BlockSize>
class cyclic : public base {
public:
  cyclic(std::size_t size, int n_ranks, std::size_t segment_size = BlockSize)
    : base(size, n_ranks),
      segment_size_(segment_size) {
    ITYR_CHECK(segment_size >= BlockSize);
    ITYR_CHECK(segment_size % BlockSize == 0);
  }

  std::size_t block_size() const override { return BlockSize; }

  std::size_t local_size(int rank [[maybe_unused]]) const override {
    return local_size_impl();
  }

  std::size_t effective_size() const override {
    return local_size_impl() * n_ranks_;
  }

  segment get_segment(std::size_t offset) const override {
    ITYR_CHECK(offset < effective_size());
    std::size_t block_num_g = offset / segment_size_;
    std::size_t block_num_l = block_num_g / n_ranks_;
    int         owner       = block_num_g % n_ranks_;
    std::size_t offset_b    = block_num_g * segment_size_;
    std::size_t offset_e    = std::min((block_num_g + 1) * segment_size_, size_);
    std::size_t pm_offset   = block_num_l * segment_size_;
    return segment{owner, offset_b, offset_e, pm_offset};
  }

private:
  // non-virtual common part
  std::size_t local_size_impl() const {
    std::size_t nblock_g = (size_ + segment_size_ - 1) / segment_size_;
    std::size_t nblock_l = (nblock_g + n_ranks_ - 1) / n_ranks_;
    return nblock_l * segment_size_;
  }

  std::size_t segment_size_;
};

ITYR_TEST_CASE("[ityr::ori::mem_mapper::cyclic] calculate local block size") {
  constexpr block_size_t bs = 65536;
  std::size_t ss = bs * 2;
  auto local_block_size = [=](std::size_t size, int n_ranks) -> std::size_t {
    return cyclic<bs>(size, n_ranks, ss).local_size(0);
  };
  ITYR_CHECK(local_block_size(ss * 4     , 4) == ss    );
  ITYR_CHECK(local_block_size(ss * 12    , 4) == ss * 3);
  ITYR_CHECK(local_block_size(ss * 13    , 4) == ss * 4);
  ITYR_CHECK(local_block_size(ss * 12 + 1, 4) == ss * 4);
  ITYR_CHECK(local_block_size(ss * 12 - 1, 4) == ss * 3);
  ITYR_CHECK(local_block_size(1          , 4) == ss    );
  ITYR_CHECK(local_block_size(1          , 1) == ss    );
  ITYR_CHECK(local_block_size(ss * 3     , 1) == ss * 3);
}

ITYR_TEST_CASE("[ityr::ori::mem_mapper::cyclic] get block information at specified offset") {
  constexpr block_size_t bs = 65536;
  std::size_t ss = bs * 2;
  auto get_segment = [=](std::size_t offset, std::size_t size, int n_ranks) -> segment {
    return cyclic<bs>(size, n_ranks, ss).get_segment(offset);
  };
  ITYR_CHECK(get_segment(0         , ss * 4     , 4) == (segment{0, 0      , ss         , 0     }));
  ITYR_CHECK(get_segment(ss        , ss * 4     , 4) == (segment{1, ss     , ss * 2     , 0     }));
  ITYR_CHECK(get_segment(ss * 2    , ss * 4     , 4) == (segment{2, ss * 2 , ss * 3     , 0     }));
  ITYR_CHECK(get_segment(ss * 3    , ss * 4     , 4) == (segment{3, ss * 3 , ss * 4     , 0     }));
  ITYR_CHECK(get_segment(ss * 4 - 1, ss * 4     , 4) == (segment{3, ss * 3 , ss * 4     , 0     }));
  ITYR_CHECK(get_segment(0         , ss * 12    , 4) == (segment{0, 0      , ss         , 0     }));
  ITYR_CHECK(get_segment(ss        , ss * 12    , 4) == (segment{1, ss     , ss * 2     , 0     }));
  ITYR_CHECK(get_segment(ss * 3    , ss * 12    , 4) == (segment{3, ss * 3 , ss * 4     , 0     }));
  ITYR_CHECK(get_segment(ss * 5 + 2, ss * 12    , 4) == (segment{1, ss * 5 , ss * 6     , ss    }));
  ITYR_CHECK(get_segment(ss * 11   , ss * 12 - 1, 4) == (segment{3, ss * 11, ss * 12 - 1, ss * 2}));
}

template <block_size_t BlockSize>
class block_adws : public base {
public:
  using base::base;

  std::size_t block_size() const override { return BlockSize; }

  std::size_t local_size(int rank [[maybe_unused]]) const override {
    return local_size_impl();
  }

  std::size_t effective_size() const override {
    return local_size_impl() * n_ranks_;
  }

  segment get_segment(std::size_t offset) const override {
    ITYR_CHECK(offset < effective_size());
    std::size_t size_l   = local_size_impl();
    int         seg_idx  = offset / size_l;
    std::size_t offset_b = seg_idx * size_l;
    std::size_t offset_e = std::min((seg_idx + 1) * size_l, size_);
    return segment{n_ranks_ - seg_idx - 1, offset_b, offset_e, 0};
  }

private:
  // non-virtual common part
  std::size_t local_size_impl() const {
    std::size_t nblock_g = (size_ + BlockSize - 1) / BlockSize;
    std::size_t nblock_l = (nblock_g + n_ranks_ - 1) / n_ranks_;
    return nblock_l * BlockSize;
  }
};

}
