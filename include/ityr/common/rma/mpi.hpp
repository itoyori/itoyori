#pragma once

#include "ityr/common/util.hpp"
#include "ityr/common/topology.hpp"
#include "ityr/common/mpi_util.hpp"
#include "ityr/common/mpi_rma.hpp"

namespace ityr::common::rma {

class mpi {
public:
  using win = mpi_win_manager<void>;

  win create_win(void* baseptr, std::size_t bytes) {
    return win(topology::mpicomm(), baseptr, bytes);
  }

  void get_nb(const win&,
              std::byte*  origin_addr,
              std::size_t bytes,
              const win&  target_win,
              int         target_rank,
              std::size_t target_disp) {
    mpi_get_nb(origin_addr, bytes, target_rank, target_disp, target_win.win());
  }

  void get_nb(std::byte*  origin_addr,
              std::size_t bytes,
              const win&  target_win,
              int         target_rank,
              std::size_t target_disp) {
    mpi_get_nb(origin_addr, bytes, target_rank, target_disp, target_win.win());
  }

  void put_nb(const win&,
              const std::byte* origin_addr,
              std::size_t      bytes,
              const win&       target_win,
              int              target_rank,
              std::size_t      target_disp) {
    mpi_put_nb(origin_addr, bytes, target_rank, target_disp, target_win.win());
  }

  void put_nb(const std::byte* origin_addr,
              std::size_t      bytes,
              const win&       target_win,
              int              target_rank,
              std::size_t      target_disp) {
    mpi_put_nb(origin_addr, bytes, target_rank, target_disp, target_win.win());
  }

  void flush(const win& win) {
    MPI_Win_flush_all(win.win());
  }
};

}

