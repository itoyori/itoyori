#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/mpi_util.hpp"

#if ITYR_RMA_IMPL == utofu && __has_include(<utofu.h>)

#include <utofu.h>

namespace ityr::common::rma {

class utofu {
public:
  utofu()
    : vcq_hdl_(init_vcq_hdl()),
      vcq_ids_(init_vcq_ids()) {}

  ~utofu() {
    mpi_barrier(topology::mpicomm());
    utofu_free_vcq(vcq_hdl_);
  }

  class win {
  public:
    win(utofu_vcq_hdl_t vcq_hdl, void* baseptr, std::size_t bytes)
      : vcq_hdl_(vcq_hdl),
        baseptr_(reinterpret_cast<std::byte*>(baseptr)),
        bytes_(bytes),
        stadds_(init_stadds()) {}

    ~win() { destroy(); }

    win(const win&) = delete;
    win& operator=(const win&) = delete;

    win(win&& w) : vcq_hdl_(w.vcq_hdl_), baseptr_(w.baseptr_), bytes_(w.bytes_), stadds_(std::move(w.stadds_)) {
      w.stadds_ = {};
    }
    win& operator=(win&& w) {
      destroy();
      vcq_hdl_  = w.vcq_hdl_;
      baseptr_  = w.baseptr_;
      bytes_    = w.bytes_;
      stadds_   = std::move(w.stadds_);
      w.stadds_ = {};
      return *this;
    }

    utofu_stadd_t stadd(topology::rank_t target_rank) const {
      ITYR_CHECK(0 <= target_rank);
      ITYR_CHECK(target_rank < topology::n_ranks());
      ITYR_CHECK(stadds_.size() == topology::n_ranks());
      return stadds_[target_rank];
    }

    utofu_stadd_t my_stadd() const {
      return stadd(topology::my_rank());
    }

    utofu_stadd_t my_stadd(const void* ptr) const {
      std::size_t offset = reinterpret_cast<const std::byte*>(ptr) - baseptr_;
      return stadd(topology::my_rank()) + offset;
    }

  private:
    void destroy() {
      if (stadds_.size() == topology::n_ranks()) {
        mpi_barrier(topology::mpicomm());
        utofu_dereg_mem(vcq_hdl_, my_stadd(), 0);
      }
    }

    std::vector<utofu_stadd_t> init_stadds() {
      utofu_stadd_t my_stadd;
      int r = utofu_reg_mem(vcq_hdl_, baseptr_, bytes_, 0, &my_stadd);
      ITYR_CHECK_MESSAGE(r == UTOFU_SUCCESS, "utofu_reg_mem() error: %d", r);

      std::vector<utofu_stadd_t> stadds(topology::n_ranks());
      MPI_Allgather(&my_stadd, 1, MPI_UINT64_T, stadds.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);
      return stadds;
    }

    utofu_vcq_hdl_t            vcq_hdl_;
    std::byte*                 baseptr_;
    std::size_t                bytes_;
    std::vector<utofu_stadd_t> stadds_;
  };

  win create_win(void* baseptr, std::size_t bytes) {
    return win(vcq_hdl_, baseptr, bytes);
  }

  void get_nb(const win&  origin_win,
              std::byte*  origin_addr,
              std::size_t bytes,
              const win&  target_win,
              int         target_rank,
              std::size_t target_disp) {
    constexpr unsigned long int post_flags = UTOFU_ONESIDED_FLAG_TCQ_NOTICE |
                                             UTOFU_ONESIDED_FLAG_LOCAL_MRQ_NOTICE;
    while (true) {
      int r = utofu_get(vcq_hdl_, vcq_ids_[target_rank], origin_win.my_stadd(origin_addr),
                        target_win.stadd(target_rank) + target_disp, bytes, 0, post_flags, 0);
      if (r == UTOFU_ERR_BUSY) {
        poll_tcq_until_empty();
      } else {
        ITYR_CHECK_MESSAGE(r == UTOFU_SUCCESS, "utofu_get() error: %d", r);
        break;
      }
    }

    n_ongoing_tcq_reqs_++;
    n_ongoing_mrq_reqs_++;
  }

  void get_nb(std::byte*, std::size_t, const win&, int, std::size_t) {
    common::die("utofu rma layer is not supported for get/put (nocache) interface");
  }

  void put_nb(const win&       origin_win,
              const std::byte* origin_addr,
              std::size_t      bytes,
              const win&       target_win,
              int              target_rank,
              std::size_t      target_disp) {
    constexpr unsigned long int post_flags = UTOFU_ONESIDED_FLAG_TCQ_NOTICE |
                                             UTOFU_ONESIDED_FLAG_LOCAL_MRQ_NOTICE;
    while (true) {
      int r = utofu_put(vcq_hdl_, vcq_ids_[target_rank], origin_win.my_stadd(origin_addr),
                        target_win.stadd(target_rank) + target_disp, bytes, 0, post_flags, 0);
      if (r == UTOFU_ERR_BUSY) {
        poll_tcq_until_empty();
      } else {
        ITYR_CHECK_MESSAGE(r == UTOFU_SUCCESS, "utofu_put() error: %d", r);
        break;
      }
    }

    n_ongoing_tcq_reqs_++;
    n_ongoing_mrq_reqs_++;
  }

  void put_nb(const std::byte*, std::size_t, const win&, int, std::size_t) {
    common::die("utofu rma layer is not supported for get/put (nocache) interface");
  }

  void flush(const win&) {
    // TODO: flush for each win
    for (int i = 0; i < n_ongoing_tcq_reqs_; i++) {
      void* cbdata;
      while (true) {
        int r = utofu_poll_tcq(vcq_hdl_, 0, &cbdata);
        if (r != UTOFU_ERR_NOT_FOUND) {
          ITYR_CHECK_MESSAGE(r == UTOFU_SUCCESS, "utofu_poll_tcq() error: %d", r);
          break;
        }
      }
    }
    n_ongoing_tcq_reqs_ = 0;

    for (int i = 0; i < n_ongoing_mrq_reqs_; i++) {
      struct utofu_mrq_notice notice;
      while (true) {
        int r = utofu_poll_mrq(vcq_hdl_, 0, &notice);
        if (r != UTOFU_ERR_NOT_FOUND) {
          ITYR_CHECK_MESSAGE(r == UTOFU_SUCCESS, "utofu_poll_mrq() error: %d", r);
          break;
        }
      }
    }
    n_ongoing_mrq_reqs_ = 0;
  }

private:
  utofu_vcq_hdl_t init_vcq_hdl() {
    std::size_t num_tnis;
    utofu_tni_id_t* tni_ids;
    utofu_get_onesided_tnis(&tni_ids, &num_tnis);

    utofu_tni_id_t tni_id = tni_ids[0];
    free(tni_ids);

    utofu_vcq_hdl_t vcq_hdl;
    int r = utofu_create_vcq(tni_id, 0, &vcq_hdl);
    ITYR_CHECK_MESSAGE(r == UTOFU_SUCCESS, "utofu_create_vcq() error: %d", r);

    return vcq_hdl;
  }

  std::vector<utofu_vcq_id_t> init_vcq_ids() {
    utofu_vcq_id_t my_vcq_id;
    int r = utofu_query_vcq_id(vcq_hdl_, &my_vcq_id);
    ITYR_CHECK_MESSAGE(r == UTOFU_SUCCESS, "utofu_query_vcq_id() error: %d", r);

    std::vector<utofu_vcq_id_t> vcq_ids(topology::n_ranks());
    MPI_Allgather(&my_vcq_id, 1, MPI_UINT64_T, vcq_ids.data(), 1, MPI_UINT64_T, MPI_COMM_WORLD);

    return vcq_ids;
  }

  void poll_tcq_until_empty() {
    void* cbdata;
    while (true) {
      int r = utofu_poll_tcq(vcq_hdl_, 0, &cbdata);
      if (r == UTOFU_SUCCESS) {
        n_ongoing_tcq_reqs_--;
      } else {
        ITYR_CHECK_MESSAGE(r == UTOFU_ERR_NOT_FOUND, "utofu_poll_tcq() error: %d", r);
        break;
      }
    }
  }

  utofu_vcq_hdl_t             vcq_hdl_;
  std::vector<utofu_vcq_id_t> vcq_ids_;
  int                         n_ongoing_tcq_reqs_ = 0;
  int                         n_ongoing_mrq_reqs_ = 0;
};

}

#endif
