#pragma once

#include "ityr/common/util.hpp"

namespace ityr::ori::mem_mapper {

struct block_info {
  int         owner;
  std::size_t offset_b;
  std::size_t offset_e;
  std::size_t pm_offset;

  bool operator==(const block_info& b) const {
    return owner == b.owner && offset_b == b.offset_b && offset_e == b.offset_e && pm_offset == b.pm_offset;
  }
  bool operator!=(const block_info& b) const {
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

  // Returns the block info that specifies the owner and the range [offset_b, offset_e)
  // of the block that contains the given offset.
  // pm_offset is the offset from the beginning of the owner's local physical memory for the block.
  virtual block_info get_block_info(std::size_t offset) const = 0;

protected:
  std::size_t size_;
  int         n_ranks_;
};

template <std::size_t BlockSize>
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

  block_info get_block_info(std::size_t offset) const override {
    ITYR_CHECK(offset < effective_size());
    std::size_t size_l   = local_size_impl();
    int         owner    = offset / size_l;
    std::size_t offset_b = owner * size_l;
    std::size_t offset_e = std::min((owner + 1) * size_l, size_);
    return block_info{owner, offset_b, offset_e, 0};
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
  constexpr std::size_t bs = 65536;
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
  constexpr std::size_t bs = 65536;
  auto block_index_info = [](std::size_t offset, std::size_t size, int n_ranks) -> block_info {
    return block<bs>(size, n_ranks).get_block_info(offset);
  };
  ITYR_CHECK(block_index_info(0         , bs * 4     , 4) == (block_info{0, 0     , bs         , 0}));
  ITYR_CHECK(block_index_info(bs        , bs * 4     , 4) == (block_info{1, bs    , bs * 2     , 0}));
  ITYR_CHECK(block_index_info(bs * 2    , bs * 4     , 4) == (block_info{2, bs * 2, bs * 3     , 0}));
  ITYR_CHECK(block_index_info(bs * 3    , bs * 4     , 4) == (block_info{3, bs * 3, bs * 4     , 0}));
  ITYR_CHECK(block_index_info(bs * 4 - 1, bs * 4     , 4) == (block_info{3, bs * 3, bs * 4     , 0}));
  ITYR_CHECK(block_index_info(0         , bs * 12    , 4) == (block_info{0, 0     , bs * 3     , 0}));
  ITYR_CHECK(block_index_info(bs        , bs * 12    , 4) == (block_info{0, 0     , bs * 3     , 0}));
  ITYR_CHECK(block_index_info(bs * 3    , bs * 12    , 4) == (block_info{1, bs * 3, bs * 6     , 0}));
  ITYR_CHECK(block_index_info(bs * 11   , bs * 12 - 1, 4) == (block_info{3, bs * 9, bs * 12 - 1, 0}));
}

template <std::size_t BlockSize>
class cyclic : public base {
public:
  cyclic(std::size_t size, int n_ranks, std::size_t granularity = BlockSize)
    : base(size, n_ranks),
      granularity_(granularity) {
    ITYR_CHECK(granularity >= BlockSize);
    ITYR_CHECK(granularity % BlockSize == 0);
  }

  std::size_t block_size() const override { return BlockSize; }

  std::size_t local_size(int rank [[maybe_unused]]) const override {
    return local_size_impl();
  }

  std::size_t effective_size() const override {
    return local_size_impl() * n_ranks_;
  }

  block_info get_block_info(std::size_t offset) const override {
    ITYR_CHECK(offset < effective_size());
    std::size_t block_num_g = offset / granularity_;
    std::size_t block_num_l = block_num_g / n_ranks_;
    int         owner       = block_num_g % n_ranks_;
    std::size_t offset_b    = block_num_g * granularity_;
    std::size_t offset_e    = std::min((block_num_g + 1) * granularity_, size_);
    std::size_t pm_offset   = block_num_l * granularity_;
    return block_info{owner, offset_b, offset_e, pm_offset};
  }

private:
  // non-virtual common part
  std::size_t local_size_impl() const {
    std::size_t nblock_g = (size_ + granularity_ - 1) / granularity_;
    std::size_t nblock_l = (nblock_g + n_ranks_ - 1) / n_ranks_;
    return nblock_l * granularity_;
  }

  std::size_t granularity_;
};

ITYR_TEST_CASE("[ityr::ori::mem_mapper::cyclic] calculate local block size") {
  constexpr std::size_t mb = 65536;
  std::size_t bs = mb * 2;
  auto local_block_size = [=](std::size_t size, int n_ranks) -> std::size_t {
    return cyclic<mb>(size, n_ranks, bs).local_size(0);
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

ITYR_TEST_CASE("[ityr::ori::mem_mapper::cyclic] get block information at specified offset") {
  constexpr std::size_t mb = 65536;
  std::size_t bs = mb * 2;
  auto block_index_info = [=](std::size_t offset, std::size_t size, int n_ranks) -> block_info {
    return cyclic<mb>(size, n_ranks, bs).get_block_info(offset);
  };
  ITYR_CHECK(block_index_info(0         , bs * 4     , 4) == (block_info{0, 0      , bs         , 0     }));
  ITYR_CHECK(block_index_info(bs        , bs * 4     , 4) == (block_info{1, bs     , bs * 2     , 0     }));
  ITYR_CHECK(block_index_info(bs * 2    , bs * 4     , 4) == (block_info{2, bs * 2 , bs * 3     , 0     }));
  ITYR_CHECK(block_index_info(bs * 3    , bs * 4     , 4) == (block_info{3, bs * 3 , bs * 4     , 0     }));
  ITYR_CHECK(block_index_info(bs * 4 - 1, bs * 4     , 4) == (block_info{3, bs * 3 , bs * 4     , 0     }));
  ITYR_CHECK(block_index_info(0         , bs * 12    , 4) == (block_info{0, 0      , bs         , 0     }));
  ITYR_CHECK(block_index_info(bs        , bs * 12    , 4) == (block_info{1, bs     , bs * 2     , 0     }));
  ITYR_CHECK(block_index_info(bs * 3    , bs * 12    , 4) == (block_info{3, bs * 3 , bs * 4     , 0     }));
  ITYR_CHECK(block_index_info(bs * 5 + 2, bs * 12    , 4) == (block_info{1, bs * 5 , bs * 6     , bs    }));
  ITYR_CHECK(block_index_info(bs * 11   , bs * 12 - 1, 4) == (block_info{3, bs * 11, bs * 12 - 1, bs * 2}));
}

}
