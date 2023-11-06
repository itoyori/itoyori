# Welcome to Itoyori

Itoyori is a distributed multithreading runtime system for global-view fork-join task parallelism.
It is implemented as a C++17 header-only library over MPI (which must have a full support of MPI-3 RMA).

This README explains the basic usage of Itoyori for running example programs.
For more information, please see [publications](#publications).

- Tutorial: https://itoyori.github.io/md_01_tutorial.html
- API Documentation: https://itoyori.github.io/namespaceityr.html

## Features

Itoyori offers a simple, unified programming model for both shared-memory and distributed-memory computers.

- **Distributed Multithreading:**
  Programmers can dynamically create a massive number of tasks (user-level threads) by calling fork/join APIs.
  The threads are automatically scheduled even across different computing nodes for global load balancing.
- **Global Address Space:**
  Itoyori also provides a view of shared memory over distributed memory, often referred to as Partitioned Global Address Space (PGAS).
  Global memory can be uniformly accessed from any computing nodes by calling checkout/checkin APIs.
- **High-Level C++ Parallel Loops:**
  Higher-level parallel loops, including map/reduce patterns, are built on top of lower-level fork/join APIs and checkout/checkin APIs.
  They resemble the C++17 parallel STL but have slight differences (e.g., Itoyori supports more general *reducers* for parallel reduction).
- **Task Schedulers:**
  The default task (thread) scheduler is the randomized work-stealing scheduler with the child-first (work-first) policy.
  Another option is *Almost Deterministic Work Stealing (ADWS)*, which offers good data locality on deep memory hierarchies.

## Supported Architectures

- x86_64
- aarch64

## Getting Started

```sh
git clone https://github.com/itoyori/itoyori.git --recursive
```

Itoyori is a C++17 header-only library located at `include/` dir.
This repository contains some CMake settings to build tests and examples.

To build tests and examples:
```sh
cmake .
make -j
```

To run test:
```sh
make test
```

Examples (at `examples/` dir) include Fib, NQueens, and Cilksort.
Cilksort involves global memory accesses (over PGAS), while Fib and NQueens do not.

To run Cilksort:
```sh
mpirun setarch $(uname -m) --addr-no-randomize ./examples/cilksort.out
```

The command `setarch $(uname -m) --addr-no-randomize` is needed for Itoyori to disable address space layout randomization (ASLR).
The argument `$(uname -m)` might not be needed depending on the `setarch` version.

Please see the example programs for usage (e.g., [cilksort.cpp](./examples/cilksort.cpp)).

## Profiling

Profiler-enabled versions are also built by CMake.

To show event statistics:
```sh
mpirun setarch $(uname -m) --addr-no-randomize ./examples/cilksort_prof_stats.out
```

To record execution traces:
```sh
mpirun setarch $(uname -m) --addr-no-randomize ./examples/cilksort_prof_trace.out
```

The output trace can be visualized by using [MassiveLogger](https://github.com/massivethreads/massivelogger) viewer:
```sh
./massivelogger/run_viewer.bash ityr_log*
```

## Program Structure

The `include/ityr/` dir includes following sub directories:
- `common`: common utils for the following layers
- `ito`: low-level threading layer (fork-join primitives)
- `ori`: low-level PGAS layer (checkout/checkin APIs)
- `pattern`: parallel patterns (e.g., `for_each()`, `reduce()`)
- `container`: containers for global memory objects (e.g., `global_vector`, `global_span`)

The `ito` and `ori` layers are loosely coupled, so that each layer runs independently.
These two low-level layers are integrated into high-level parallel patterns and containers, by appropriately inserting global memory fences to fork-join calls, for example.
Thus, it is highly recommended to use these high-level interfaces (under `ityr::` namespace), rather than low-level ones (under `ityr::ito` or `ityr::ori` namespace).

Git submodules are used for the following purposes, but not required to run Itoyori:
- `doctest`: used for testing
- `massivelogger`: needed to collect execution traces

## Guidelines for Using MPI

As Itoyori heavily uses MPI-3 RMA (one-sided communication) for communication between workers, the performance can be significantly affected by the MPI implementation being used.
Itoyori assumes **truly one-sided** communication of MPI-3 RMA, and preferably, RMA calls should be offloaded to RDMA.

Truly one-sided communication implies that an RMA operation can be completed without the involvement of the target process.
In other words, an RMA operation should make progress even if the target process is busy executing tasks without calling any MPI calls.

You can check if RMA calls of your MPI installation are truly one-sided by running `<example>_prof_stats.out` programs, in which statistics profiling is enabled.
For instance, you can check a profile of Cilksort by running:
```sh
mpirun -n <N_PROC> setarch $(uname -m) --addr-no-randomize ./examples/cilksort_prof_stats.out -n <N_INPUT>
```

Example output of Cilksort with 2 workers on the same computer:
```
[0] 80,218,887 ns - Result verified
  rma_get                :   0.001338 % (            2147 ns /       160438053 ns ) count:          2 ave:     1073 ns max:     1663 ns
  rma_put                :   0.000000 % (               0 ns /       160438053 ns ) count:          0 ave:        0 ns max:        0 ns
  rma_atomic_faa         :   0.000000 % (               0 ns /       160438053 ns ) count:          0 ave:        0 ns max:        0 ns
  rma_atomic_cas         :   0.000000 % (               0 ns /       160438053 ns ) count:          0 ave:        0 ns max:        0 ns
  rma_atomic_get         :   0.000000 % (               0 ns /       160438053 ns ) count:          0 ave:        0 ns max:        0 ns
  rma_atomic_put         :   0.000000 % (               0 ns /       160438053 ns ) count:          0 ave:        0 ns max:        0 ns
  rma_flush              :  49.994422 % (        80210078 ns /       160438053 ns ) count:          2 ave: 40105039 ns max: 80205858 ns
  global_lock_trylock    :   0.000000 % (               0 ns /       160438053 ns ) count:          0 ave:        0 ns max:        0 ns
...
  wsqueue_pass           :   0.000000 % (               0 ns /       160438053 ns ) count:          0 ave:        0 ns max:        0 ns
  wsqueue_empty          :  49.996029 % (        80212656 ns /       160438053 ns ) count:          2 ave: 40106328 ns max: 80206456 ns
  wsqueue_empty_batch    :   0.000000 % (               0 ns /       160438053 ns ) count:          0 ave:        0 ns max:        0 ns
  P_sched_loop           :  50.008906 % (        80233315 ns /       160438053 ns ) count:          2 ave: 40116657 ns max: 80211849 ns
  P_sched_fork           :   0.385890 % (          619114 ns /       160438053 ns ) count:       4463 ave:      138 ns max:     8279 ns
...
  P_cb_post_suspend      :   0.000000 % (               0 ns /       160438053 ns ) count:          0 ave:        0 ns max:        0 ns
  P_thread               :  48.992382 % (        78602423 ns /       160438053 ns ) count:      13387 ave:     5871 ns max:    41718 ns
  P_spmd                 :   0.009036 % (           14497 ns /       160438053 ns ) count:          4 ave:     3624 ns max:     6439 ns
...
```

This result implies that RMA calls are not truly one-sided.
What is happening is that one worker is continuously executing tasks without any MPI calls, while the other tries work stealing but ends up being blocked due to lack of progress.
This situation leads to all tasks being executed by just one worker.

The above result was obtained with MPICH (v4.0), which seems not supporting truly one-sided communication.
Nevertheless, you can emulate truly one-sided communication by launching asynchronous progress threads by setting:
```sh
export MPICH_ASYNC_PROGRESS=1
```

Then, we will get the following result:
```
[0] 47,116,446 ns - Result verified
  rma_get                :   0.200830 % (          189249 ns /        94233227 ns ) count:        150 ave:     1261 ns max:    32485 ns
  rma_put                :   0.000000 % (               0 ns /        94233227 ns ) count:          0 ave:        0 ns max:        0 ns
  rma_atomic_faa         :   0.262830 % (          247673 ns /        94233227 ns ) count:        148 ave:     1673 ns max:    24942 ns
  rma_atomic_cas         :   0.051618 % (           48641 ns /        94233227 ns ) count:         29 ave:     1677 ns max:     7781 ns
  rma_atomic_get         :   0.028849 % (           27185 ns /        94233227 ns ) count:         59 ave:      460 ns max:    13337 ns
  rma_atomic_put         :   0.060474 % (           56987 ns /        94233227 ns ) count:         18 ave:     3165 ns max:    12509 ns
  rma_flush              :   1.596035 % (         1503995 ns /        94233227 ns ) count:        388 ave:     3876 ns max:    52504 ns
  global_lock_trylock    :   0.150568 % (          141885 ns /        94233227 ns ) count:         29 ave:     4892 ns max:    15343 ns
...
  wsqueue_pass           :   0.000000 % (               0 ns /        94233227 ns ) count:          0 ave:        0 ns max:        0 ns
  wsqueue_empty          :   0.443419 % (          417848 ns /        94233227 ns ) count:         65 ave:     6428 ns max:    33481 ns
  wsqueue_empty_batch    :   0.000000 % (               0 ns /        94233227 ns ) count:          0 ave:        0 ns max:        0 ns
  P_sched_loop           :   1.734392 % (         1634374 ns /        94233227 ns ) count:         30 ave:    54479 ns max:    96559 ns
  P_sched_fork           :   0.806280 % (          759784 ns /        94233227 ns ) count:       4463 ave:      170 ns max:     5371 ns
...
  P_cb_post_suspend      :   0.000000 % (               0 ns /        94233227 ns ) count:          0 ave:        0 ns max:        0 ns
  P_thread               :  95.385054 % (        89884414 ns /        94233227 ns ) count:      13387 ave:     6714 ns max:    66788 ns
  P_spmd                 :   0.062008 % (           58432 ns /        94233227 ns ) count:          4 ave:    14608 ns max:    24406 ns
...
```

However, MPICH's approach is based on *active messages* to simulate one-sided communication by asynchronous two-sided communication, which does not take full advantage of RDMA.
From our experience, Open MPI better offloads RMA operations to RDMA.

We confirmed that Itoyori worked well on RDMA-capable interconnects with the following MPI configurations:
- Open MPI v5.0.0rc11 with UCX v1.14.0 (which requires MLNX_OFED >= 5.0) over InfiniBand
- Fujitsu MPI v4.0.1 (based on Open MPI) over Tofu-D Interconnect

Note that actual MPI behaviors will depend on actual hardware configurations and driver versions.

Open MPI v5.0.x enables the use of an MCA parameter `osc_ucx_acc_single_intrinsic`, which accelerates network atomic operations being heavily used in Itoyori.
A recommended way to run Itoyori with Open MPI is:
```sh
mpirun --mca osc ucx --mca osc_ucx_acc_single_intrinsic true ...
```

However, local executions with Open MPI (without high-performance network cards) may degrade to an implementation that is not truly one-sided.
In addition, there seems no option to launch asynchronous progress threads in Open MPI.
Therefore, for the debugging purpose on local machines, we recommend to use MPICH-based MPI implementations with `MPICH_ASYNC_PROGRESS=1`.

## Publications

Overview of the Itoyori runtime system and its PGAS implementation:
- [Shumpei Shiina and Kenjiro Taura. *Itoyori: Reconciling Global Address Space and Global Fork-Join Task Parallelism.* in SC '23](https://dl.acm.org/doi/abs/10.1145/3581784.3607049)

About the threading layer (formarly called *uni-address threads*):
- [Shumpei Shiina and Kenjiro Taura. *Distributed Continuation Stealing is More Scalable than You Might Think.* in Cluster '22](https://sshiina.gitlab.io/papers/cluster22.pdf)
- [Shigeki Akiyama and Kenjiro Taura. *Scalable Work Stealing of Native Threads on an x86-64 Infiniband Cluster.* in JIP 2016](https://doi.org/10.2197/ipsjjip.24.583)
- [Shigeki Akiyama and Kenjiro Taura. *Uni-Address Threads: Scalable Thread Management for RDMA-Based Work Stealing.* in HPDC '15](https://dl.acm.org/doi/abs/10.1145/2749246.2749272)

About the task scheduler *Almost Deterministic Work Stealing (ADWS)*:
- [Shumpei Shiina and Kenjiro Taura. *Improving Cache Utilization of Nested Parallel Programs by Almost Deterministic Work Stealing.* in TPDS 2022](https://doi.org/10.1109/CLUSTER51413.2022.00027)
- [Shumpei Shiina and Kenjiro Taura. *Almost Deterministic Work Stealing.* in SC '19](https://sshiina.gitlab.io/papers/sc19.pdf)

## About the Name of Itoyori

Itoyori is named after the fish *thread*fin breams ("糸撚魚" in Japanese).
