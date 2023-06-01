#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/ori/util.hpp"
#include "ityr/ori/options.hpp"
#include "ityr/ori/cache_system.hpp"
#include "ityr/ori/block_regions.hpp"

namespace ityr::ori {

class cache_profiler_disabled {
public:
  cache_profiler_disabled(cache_entry_idx_t) {}
  void record(cache_entry_idx_t, block_region, const block_regions&) {}
  void record_writeonly(cache_entry_idx_t, block_region, const block_regions&) {}
  void start() {}
  void stop() {}
  void print() const {}
};

class cache_profiler_stats {
public:
  cache_profiler_stats(cache_entry_idx_t n_blocks)
    : n_blocks_(n_blocks),
      blocks_(n_blocks_) {}

  void record(cache_entry_idx_t    block_idx,
              block_region         requested_region,
              const block_regions& fetched_regions) {
    if (!enabled_) return;

    ITYR_CHECK(0 <= block_idx);
    ITYR_CHECK(block_idx < n_blocks_);
    cache_block& blk = blocks_[block_idx];

    requested_bytes_ += requested_region.size();
    fetched_bytes_   += fetched_regions.size();

    block_regions hit_regions          = fetched_regions.inverse(requested_region);
    block_regions temporal_hit_regions = get_intersection(hit_regions, blk.requested_regions);

    std::size_t temporal_hit_size = temporal_hit_regions.size();
    std::size_t spatial_hit_size  = hit_regions.size() - temporal_hit_size;

    temporal_hit_bytes_ += temporal_hit_size;
    spatial_hit_bytes_  += spatial_hit_size;

    blk.requested_regions.add(requested_region);

    if (fetched_regions.empty()) {
      block_hit_count_++;
    } else {
      block_miss_count_++;
    }
  }

  void record_writeonly(cache_entry_idx_t    block_idx,
                        block_region         requested_region,
                        const block_regions& valid_regions) {
    if (!enabled_) return;

    ITYR_CHECK(0 <= block_idx);
    ITYR_CHECK(block_idx < n_blocks_);
    cache_block& blk = blocks_[block_idx];

    requested_bytes_ += requested_region.size();

    block_regions skip_fetch_hit_regions = valid_regions.inverse(requested_region);

    block_regions hit_regions          = skip_fetch_hit_regions.inverse(requested_region);
    block_regions temporal_hit_regions = get_intersection(hit_regions, blk.requested_regions);

    std::size_t temporal_hit_size = temporal_hit_regions.size();
    std::size_t spatial_hit_size  = hit_regions.size() - temporal_hit_size;

    temporal_hit_bytes_ += temporal_hit_size;
    spatial_hit_bytes_  += spatial_hit_size;

    skip_fetch_hit_bytes_ += skip_fetch_hit_regions.size();

    blk.requested_regions.add(requested_region);

    block_hit_count_++;
  }

  void clear(cache_entry_idx_t block_idx) {
    ITYR_CHECK(0 <= block_idx);
    ITYR_CHECK(block_idx < n_blocks_);
    cache_block& blk = blocks_[block_idx];

    blk.requested_regions.clear();
  }

  void clear_all() {
    for (auto&& blk : blocks_) {
      blk.requested_regions.clear();
    }
  }

  void start() {
    requested_bytes_      = 0;
    fetched_bytes_        = 0;
    temporal_hit_bytes_   = 0;
    spatial_hit_bytes_    = 0;
    skip_fetch_hit_bytes_ = 0;
    block_hit_count_      = 0;
    block_miss_count_     = 0;

    enabled_ = true;
  }

  void stop() {
    enabled_ = false;
  }

  void print() const {
    auto requested_bytes_all      = common::mpi_reduce_value(requested_bytes_     , 0, common::topology::mpicomm());
    auto fetched_bytes_all        = common::mpi_reduce_value(fetched_bytes_       , 0, common::topology::mpicomm());
    auto temporal_hit_bytes_all   = common::mpi_reduce_value(temporal_hit_bytes_  , 0, common::topology::mpicomm());
    auto spatial_hit_bytes_all    = common::mpi_reduce_value(spatial_hit_bytes_   , 0, common::topology::mpicomm());
    auto skip_fetch_hit_bytes_all = common::mpi_reduce_value(skip_fetch_hit_bytes_, 0, common::topology::mpicomm());

    if (common::topology::my_rank() == 0) {
      printf("Cache requested: %15ld bytes\n"  , requested_bytes_all);
      printf("Remote fetched:  %15ld bytes\n"  , fetched_bytes_all);
      printf("Temporal hit:    %15ld bytes\n"  , temporal_hit_bytes_all);
      printf("Spatial hit:     %15ld bytes\n"  , spatial_hit_bytes_all);
      printf("Skip-fetch hit:  %15ld bytes\n"  , skip_fetch_hit_bytes_all);
      printf("Cache hit:       %15ld / block\n", block_hit_count_);
      printf("Cache miss:      %15ld / block\n", block_miss_count_);
      printf("\n");
      fflush(stdout);
    }
  }

private:
  struct cache_block {
    block_regions requested_regions;
  };

  cache_entry_idx_t        n_blocks_;
  std::vector<cache_block> blocks_;

  std::size_t              requested_bytes_      = 0;
  std::size_t              fetched_bytes_        = 0;
  std::size_t              temporal_hit_bytes_   = 0;
  std::size_t              spatial_hit_bytes_    = 0;
  std::size_t              skip_fetch_hit_bytes_ = 0;
  std::size_t              block_hit_count_      = 0;
  std::size_t              block_miss_count_     = 0;

  bool                     enabled_ = false;
};

using cache_profiler = ITYR_CONCAT(cache_profiler_, ITYR_ORI_CACHE_PROF);

}
