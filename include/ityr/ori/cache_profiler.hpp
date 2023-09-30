#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/ori/util.hpp"
#include "ityr/ori/options.hpp"
#include "ityr/ori/cache_system.hpp"
#include "ityr/ori/block_region_set.hpp"

namespace ityr::ori {

class cache_profiler_disabled {
public:
  cache_profiler_disabled(cache_entry_idx_t) {}
  void record(cache_entry_idx_t, block_region, const block_region_set&) {}
  void record_writeonly(cache_entry_idx_t, block_region, const block_region_set&) {}
  void invalidate(cache_entry_idx_t, const block_region_set&) {}
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
              const block_region_set& fetched_regions) {
    ITYR_CHECK(0 <= block_idx);
    ITYR_CHECK(block_idx < n_blocks_);
    cache_block& blk = blocks_[block_idx];

    if (enabled_) {
      requested_bytes_ += requested_region.size();
      fetched_bytes_   += fetched_regions.size();

      block_region_set hit_regions          = fetched_regions.complement(requested_region);
      block_region_set temporal_hit_regions = get_intersection(hit_regions, blk.requested_regions);

      std::size_t temporal_hit_size = temporal_hit_regions.size();
      std::size_t spatial_hit_size  = hit_regions.size() - temporal_hit_size;

      temporal_hit_bytes_ += temporal_hit_size;
      spatial_hit_bytes_  += spatial_hit_size;

      if (fetched_regions.empty()) {
        block_hit_count_++;
      } else {
        block_miss_count_++;
      }
    }

    blk.requested_regions.add(requested_region);
  }

  void record_writeonly(cache_entry_idx_t    block_idx,
                        block_region         requested_region,
                        const block_region_set& valid_regions) {
    ITYR_CHECK(0 <= block_idx);
    ITYR_CHECK(block_idx < n_blocks_);
    cache_block& blk = blocks_[block_idx];

    if (enabled_) {
      requested_bytes_ += requested_region.size();

      block_region_set skip_fetch_hit_regions = valid_regions.complement(requested_region);

      block_region_set hit_regions          = skip_fetch_hit_regions.complement(requested_region);
      block_region_set temporal_hit_regions = get_intersection(hit_regions, blk.requested_regions);

      std::size_t temporal_hit_size = temporal_hit_regions.size();
      std::size_t spatial_hit_size  = hit_regions.size() - temporal_hit_size;

      temporal_hit_bytes_ += temporal_hit_size;
      spatial_hit_bytes_  += spatial_hit_size;

      skip_fetch_hit_bytes_ += skip_fetch_hit_regions.size();

      block_hit_count_++;
    }

    blk.requested_regions.add(requested_region);
  }

  void invalidate(cache_entry_idx_t block_idx, const block_region_set& valid_regions) {
    ITYR_CHECK(0 <= block_idx);
    ITYR_CHECK(block_idx < n_blocks_);
    cache_block& blk = blocks_[block_idx];

    if (enabled_) {
      wasted_fetched_bytes_ += valid_regions.size() - blk.requested_regions.size();
    }

    blk.requested_regions.clear();
  }

  void start() {
    requested_bytes_      = 0;
    fetched_bytes_        = 0;
    wasted_fetched_bytes_ = 0;
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
    auto wasted_fetched_bytes_all = common::mpi_reduce_value(wasted_fetched_bytes_, 0, common::topology::mpicomm());
    auto temporal_hit_bytes_all   = common::mpi_reduce_value(temporal_hit_bytes_  , 0, common::topology::mpicomm());
    auto spatial_hit_bytes_all    = common::mpi_reduce_value(spatial_hit_bytes_   , 0, common::topology::mpicomm());
    auto skip_fetch_hit_bytes_all = common::mpi_reduce_value(skip_fetch_hit_bytes_, 0, common::topology::mpicomm());
    auto block_hit_count_all      = common::mpi_reduce_value(block_hit_count_     , 0, common::topology::mpicomm());
    auto block_miss_count_all     = common::mpi_reduce_value(block_miss_count_    , 0, common::topology::mpicomm());

    if (common::topology::my_rank() == 0) {
      printf("[Cache blocks]\n");
      printf("  User requested:   %18ld bytes\n" , requested_bytes_all);
      printf("  Fetched:          %18ld bytes\n" , fetched_bytes_all);
      printf("  Fetched (wasted): %18ld bytes\n" , wasted_fetched_bytes_all);
      printf("  Temporal hit:     %18ld bytes\n" , temporal_hit_bytes_all);
      printf("  Spatial hit:      %18ld bytes\n" , spatial_hit_bytes_all);
      printf("  Skip-fetch hit:   %18ld bytes\n" , skip_fetch_hit_bytes_all);
      printf("  Hit count:        %18ld blocks\n", block_hit_count_all);
      printf("  Miss count:       %18ld blocks\n", block_miss_count_all);
      printf("\n");
      fflush(stdout);
    }
  }

private:
  struct cache_block {
    block_region_set requested_regions;
  };

  cache_entry_idx_t        n_blocks_;
  std::vector<cache_block> blocks_;

  std::size_t              requested_bytes_      = 0; // requested by the user (through checkout calls)
  std::size_t              fetched_bytes_        = 0; // fetched from remote processes
  std::size_t              wasted_fetched_bytes_ = 0; // fetched but not requested by the user
  std::size_t              temporal_hit_bytes_   = 0; // cache hit for data requested again by the user
  std::size_t              spatial_hit_bytes_    = 0; // cache hit for data not previously requested by the user
  std::size_t              skip_fetch_hit_bytes_ = 0; // cache hit for write-only data (skipping remote fetch)
  std::size_t              block_hit_count_      = 0; // Cache hits counted for each block
  std::size_t              block_miss_count_     = 0; // Cache misses counted for each block

  bool                     enabled_ = false;
};

using cache_profiler = ITYR_CONCAT(cache_profiler_, ITYR_ORI_CACHE_PROF);

}
