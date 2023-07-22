# Tutorial

This tutorial explains the basics of writing Itoyori programs.

## Execution Model

An Itoyori program begins with the Single Program Multiple Data (SPMD) execution model, as it is launched by MPI.
Later, it can be switched to the task-parallel execution mode by spawning the root thread.

A sample program:
```cpp
#include "ityr/ityr.hpp"

int main() {
  ityr::init();

  auto my_rank = ityr::my_rank();
  auto n_ranks = ityr::n_ranks();

  // SPMD region
  printf("MPI Process %d/%d: begin\n", my_rank, n_ranks);

  ityr::root_exec([=] {
    // Only one root thread is spawned globally
    printf("Root thread started by %d\n", my_rank);
  });
  // returns when the root thread is completed

  // SPMD region
  printf("MPI Process %d/%d: end\n", my_rank, n_ranks);

  ityr::fini();
}
```

Notes:
- `ityr` is short for Itoyori
- `ityr::init()` and `ityr::fini()` must be called to initialize/finalize the Itoyori runtime system
- `ityr::my_rank()` and `ityr::n_ranks()` correspond to its MPI rank and size
- `ityr::root_exec()` must be collectively called by all processes to spawn the root thread
    - Each process stays in a scheduler loop until the root thread (and its descendant threads) is completed

Suppose that the above program is saved as `sample.cpp`, it can be compiled with:
```sh
mpicxx -I/path/to/ityr/include/dir -std=c++17 -fno-stack-protector sample.cpp
```

Notes:
- Itoyori is a C++17 header-only library
- `-fno-stack-protector` option is necessary for dynamic thread migration preserving virtual addresses

Example output with 4 MPI processes:
```console
$ mpirun -n 4 setarch $(uname -m) --addr-no-randomize ./a.out
MPI Process 0/4: begin
Root thread executed by 0
MPI Process 1/4: begin
MPI Process 2/4: begin
MPI Process 3/4: begin
MPI Process 0/4: end
MPI Process 1/4: end
MPI Process 2/4: end
MPI Process 3/4: end
```

The output indicates that the SPMD region is executed by all processes, while the root thread is executed once.

Notes:
- The output order across MPI processes is not guaranteed
- "Threads" refers to user-level threads implemented in Itoyori
    - Hereafter, "threads" denote "user-level threads," unless explicitly stated as "kernel-level threads"
- Itoyori APIs are not thread-safe, in terms of kernel-level threads in each MPI process
    - Itoyori assumes that each MPI process is mapped to a single core (so called "flat MPI")
    - This means that the user should not manually parallelize the code with Pthreads or OpenMP
- Threads (including the root thread) can be migrated to other MPI processes during execution (e.g., at fork/join calls)

## Recursive Fork/Join Parallelism

After the root thread is spawned, the user can arbitrarily spawn a bunch of lightweight threads.
This section explains how to express recursive fork/join parallelism in Itoyori.

Let's take the Fibonacci sequence as an example.
The Fibonacci sequence is defined as follows.

$$
\mathit{fib}(n) = \mathit{fib}(n-1) + \mathit{fib}(n-2),\;
\mathit{fib}(0) = \mathit{fib}(1) = 1
$$

The following program calculates the n-th Fibonacci number in a stupid way, by recursively solving subproblems in parallel (divide-and-conquer).

Fibonacci program:
```cpp
#include "ityr/ityr.hpp"

long fib(int n) {
  if (n <= 1) {
    return 1;
  } else {
    auto [x, y] =
      ityr::parallel_invoke(
        [=] { return fib(n - 1); },
        [=] { return fib(n - 2); }
      );
    return x + y;
  }
}

int main() {
  ityr::init();

  int n = 15;
  long result =
    ityr::root_exec([=] {
      return fib(n);
    });

  if (ityr::is_master()) {
    printf("fib(%d) = %ld\n", n, result);
  }

  ityr::fini();
}
```

Notes:
- `ityr::parallel_invoke()` forks parallel tasks as child threads and joins them at a time
    - It returns a tuple that consists of the return values of each parallel task
    - The notation `parallel_invoke` is also used in shared-memory fork/join libraries such as oneTBB (formarly Intel TBB) and Microsoft PPL
- `ityr::root_exec()` can also return a value, which is shared by all processes
- `ityr::is_master()` is equivalent to `ityr::my_rank() == 0`

Lambda expressions for `ityr::parallel_invoke()` should capture values by copy.
This is because threads cannot have any raw pointer or reference to other threads, including their parents (see [Pitfalls](./02_pitfalls.md) for details).

Alternatively, arguments can be passed as tuples without lambdas:
```cpp
auto [x, y] =
  ityr::parallel_invoke(
    fib, std::make_tuple(n - 1),
    fib, std::make_tuple(n - 2)
  );
```

Also, `ityr::parallel_invoke()` can accept an arbitrary number of parallel tasks, as shown in the below Tribonacci example (an extention to Fibonacci).

Tribonacci example:
```cpp
long trib(int n) {
  if (n <= 1) {
    return 1;
  } else if (n == 2) {
    return 2;
  } else {
    auto [x, y, z] =
      ityr::parallel_invoke(
        [=] { return trib(n - 1); },
        [=] { return trib(n - 2); },
        [=] { return trib(n - 3); }
      );
    return x + y + z;
  }
}
```

The Tribonacci sequence is defined as follows.

$$
\mathit{trib}(n) = \mathit{trib}(n-1) + \mathit{trib}(n-2) + \mathit{trib}(n-3),\;
\mathit{trib}(0) = \mathit{trib}(1) = 1,\;
\mathit{trib}(2) = 2
$$

## Global Memory Access

Unlike the above Fibonacci example, practical real-world applications would need global memory.
Itoyori supports global address space through **checkout/checkin APIs**.
In Itoyori, global addresses are represented as raw virtual addresses, which can be directly accessed with CPU load/store instructions, but access to the virtual memory region must be granted through explicit checkout/checkin calls.

To make checkout/checkin calls:
```cpp
ityr::ori::global_ptr<int> gp = /* ... */;
{
  // checkout 10 elements starting at a global address `gp` with read+write mode
  auto cs = ityr::make_checkout(gp, 10, ityr::checkout_mode::read_write);

  // Global memory `[gp, gp + 10)` can now be accessed locally
  cs[0] = 0;
  /* ... */
  cs[9] = 9;

  // checkin when `cs` is destroyed
}
```

Notes:
- Although a global pointer can be expressed as just a raw pointer (`int*`), it is recommended to use a wrapper class `ityr::ori::global_ptr<int>` to prevent dereferencing without checkout
- The global pointer type `ityr::ori::global_ptr` is prefixed with `ityr::ori`, which is the namespace of the low-level global address space layer
    - This low-level layer is not intended to be directly used by the user unless absolutely necessary
    - Instead, it is recommended to use higher-level primitives such as `ityr::global_span` for safety
- The user must specify a checkout mode (`ityr::checkout_mode`) for `ityr::make_checkout()`
    - The mode is either `read`, `read_write`, or `write`, as explained later
- `cs` is of type `ityr::checkout_span`, which automatically performs a checkin operation when destroyed

About the checkout mode:
- If `read` or `read_write`, the checked-out region has valid data after the checkout call
    - If `write`, the region may have indeterminate values by skipping fetching data from remote nodes, which can be useful for write-only access (e.g., initialization)
- If `read_write` or `write`, the entire checked-out region is treated as modified
    - Internally, the cache for this region is considered *dirty*

In the following, we explain how to use checkout/checkin APIs through an example of parallel mergesort, in which the input array is divided into two subarrays and sorted recursively (divide-and-conquer).

Parallel mergesort example:
```cpp
void msort(ityr::global_span<int> a) {
  if (a.size() < cutoff) {
    // switch to serial sort when the array is sufficiently small
    auto ac = ityr::make_checkout(a.data(), a.size(), ityr::checkout_mode::read_write);
    std::sort(ac.begin(), ac.end());

  } else {
    std::size_t m = a.size() / 2;

    // recursively sort two subarrays (divide-and-conquer)
    ityr::parallel_invoke(
      [=] { msort(a.subspan(0, m           )); },
      [=] { msort(a.subspan(m, a.size() - m)); }
    );

    // merge two sorted subarrays
    auto ac = ityr::make_checkout(a.data(), a.size(), ityr::checkout_mode::read_write);
    std::inplace_merge(ac.begin(), ac.begin() + m, ac.end());
  }
}
```

The parallel mergesort example is written in a completely data-race-free manner.
In fact, Itoyori does not allow any data race; i.e., the same region can be concurrently checked out by multiple processes with the `ityr::checkout_mode::read` mode only.

As Itoyori performs software caching for global memory accesses, the user can expect both temporal and spatial locality is exploited by the system.
This means that, even if the same or close memory regions are checked out multiple times, the cache prevents redundant and fine-grained communication.

Full mergesort program:
```cpp
#include "ityr/ityr.hpp"

std::size_t cutoff = 128;

void msort(ityr::global_span<int> a) {
  if (a.size() < cutoff) {
    auto ac = ityr::make_checkout(a.data(), a.size(), ityr::checkout_mode::read_write);
    std::sort(ac.begin(), ac.end());
  } else {
    std::size_t m = a.size() / 2;
    ityr::parallel_invoke(
      [=] { msort(a.subspan(0, m           )); },
      [=] { msort(a.subspan(m, a.size() - m)); }
    );
    auto ac = ityr::make_checkout(a.data(), a.size(), ityr::checkout_mode::read_write);
    std::inplace_merge(ac.begin(), ac.begin() + m, ac.end());
  }
}

int main() {
  ityr::init();
  {
    int n = 16384;
    ityr::global_vector<int> a_vec({.collective = true}, n);
    ityr::global_span<int> a(a_vec.begin(), a_vec.end());

    if (ityr::is_master()) {
      // initialize the array with random numbers
      auto ac = ityr::make_checkout(a.data(), a.size(), ityr::checkout_mode::write);
      for (auto&& x : ac) {
        x = rand();
      }
    }

    // perform parallel mergesort
    ityr::root_exec([=] {
      msort(a);
    });

    if (ityr::is_master()) {
      // check if the entire array is sorted
      auto ac = ityr::make_checkout(a.data(), a.size(), ityr::checkout_mode::read);
      bool sorted = std::is_sorted(ac.begin(), ac.end());
      std::cout << "is_sorted: " << std::boolalpha << sorted << std::endl;
    }
  }
  ityr::fini();
}
```

Notes:
- `ityr::global_vector` is used to allocate global memory
    - The first (optional) argument is `ityr::global_vector_options`
    - `.collective = true` means the global memory should be collectively allocated by all processes (this must be performed outside `ityr::root_exec()`)
    - If `.collective = false`, the global memory is allocated in local memory of each process (noncollective)
- `ityr::global_span` is used for task-parallel execution, instead of `ityr::global_vector`, in order to avoid unnecessary vector copy
    - See [Pitfalls](./02_pitfalls.md)
- `ityr::checkout_mode::write` is specified for array initialization in order to skip fetching unnecessary data
- `ityr::checkout_mode::read` is specified for checking the result, in which the array is never modified
- It is allowed to access the global variable `cutoff` without checkout, but it is on the user's responsibility to guarantee that global variables have the same values across all processes
    - Global variables are assumed to have the same virtual addresses across all processes

Also note that the above example does not work with an array larger than each process's local cache.
The following runtime error suggests that checkout requests are too large.

```
cache is exhausted (too much checked-out memory)
```

To avoid this, checkout must be performed in a sufficiently coarse-grained manner.
For example, the merge operation should be parallelized to decompose checkout/checkin operations into smaller ones, and also to increase parallelism.
See Itoyori's [Cilksort example](https://github.com/itoyori/itoyori/blob/master/examples/cilksort.cpp) for a parallelized merge implementation.
For more regular patterns, high-level parallel loops can be used as explained in the next section.

## Parallel Loops and Global Iterators

When computing on an array that is much larger than each process's local cache, checkout calls have to be made in sufficiently small granularity.
This results in the following complicated code to express a simple *for* loop:
```cpp
ityr::global_vector<int> v(/* ... */);
/* ... */
std::size_t block_size = /* ... */;
for (std::size_t i = 0; i < v.size(); i += block_size) {
  auto begin = v.begin() + i;
  auto end   = v.begin() + std::min(i + block_size, v.size());

  auto cs = ityr::make_checkout(begin, end - begin, ityr::checkout_mode::read_write);
  for (auto&& x : cs) {
    x = /* ... */;
  }
}
```

It makes each checkout call with a size no larger than `block_size` to prevent too large checkout requests.

By using a higher-order function `ityr::serial_for_each` with *global iterators*, the same goal can be achieved:
```cpp
ityr::global_vector<int> v(/* ... */);
/* ... */
std::size_t block_size = /* ... */;
ityr::serial_for_each(
    {.checkout_count = block_size},
    ityr::make_global_iterator(v.begin(), ityr::checkout_mode::read_write),
    ityr::make_global_iterator(v.end()  , ityr::checkout_mode::read_write),
    [=](int& x) {
  x = /* ... */;
});
```

`ityr::serial_for_each` receives a user-defined function (lambda) that operates on each element, resulting in more concise and structured code.

Notes:
- Itoyori's iterator-based functions such as `ityr::serial_for_each()` resemble C++ STL algorithms
- The first (optional) argument `ityr::serial_loop_options` specifies the number of elements that are internally checked out at the same time
    - By default, the checkout count is 1
- If global iterators made with `ityr::make_global_iterator()` are passed, they are automatically checked out in the specified granularity
    - Global pointers can be treated as iterators, but they are not automatically checked out; instead `ityr::ori::global_ref` is passed to user functions
    - If the checkout mode is `read`, a const reference is passed to the user function; otherwise, a nonconst reference is passed

This can be easily translated into a parallel *for* loop:
```cpp
ityr::global_vector<int> v(/* ... */);
/* ... */
std::size_t block_size = /* ... */;
ityr::parallel_for_each(
    {.cutoff_count = block_size, .checkout_count = block_size},
    ityr::make_global_iterator(v.begin(), ityr::checkout_mode::read_write),
    ityr::make_global_iterator(v.end()  , ityr::checkout_mode::read_write),
    [=](int& x) {
  x = /* ... */;
});
```

Notes:
- Internally, `ityr::parallel_for_each()` recursively divides the index space into two parts and runs them in parallel
- The first (optional) argument `ityr::parallel_loop_options` accepts `cutoff_count` option, which specifies the cutoff count for the leaf tasks
    - Usually, the same value should be specified to both `cutoff_count` and `checkout_count`

In addition, `ityr::parallel_for_each()` can accept multiple iterators:
```cpp
ityr::global_vector<int> v(/* ... */);
/* ... */
std::size_t block_size = /* ... */;
ityr::parallel_for_each(
    {.cutoff_count = block_size, .checkout_count = block_size},
    ityr::count_iterator<int>(0),
    ityr::count_iterator<int>(v.size()),
    ityr::make_global_iterator(v.begin(), ityr::checkout_mode::write),
    [=](int i, int& x) {
  x = i;
});
```

`ityr::count_iterator` is a special iterator that counts up its value when incremented.
Using it with `ityr::parallel_for_each` corresponds to parallelizing a loop that looks like `for (int i = 0; i < n; i++)`.
Thus, the elements of `v` in the above code will be 0, 1, 2, ... , n.

AXPY is an example that can be concisely written with `ityr::parallel_for_each`.
AXPY computes a scalar-vector product and adds the result to another vector:

$$
y \gets a \times x + y
$$

where *a* is a scalar and *x* and *y* are vectors.

AXPY example:
```cpp
void axpy(double a, ityr::global_span<double> xs, ityr::global_span<double> ys) {
  ityr::parallel_for_each(
      ityr::make_global_iterator(xs.begin(), ityr::checkout_mode::read),
      ityr::make_global_iterator(xs.end()  , ityr::checkout_mode::read),
      ityr::make_global_iterator(ys.begin(), ityr::checkout_mode::read_write),
      [=](const double& x, double& y) {
    y += a * x;
  });
}
```

Similarly, the sum of vector elements can be computed in parallel with `ityr::parallel_reduce`.

Calculate sum:
```cpp
double sum(ityr::global_span<double> xs) {
  return
    ityr::parallel_reduce(
        xs.begin(), xs.end(),
        double(0), std::plus<double>{});
}
```

Note that `ityr::make_global_iterator()` is not needed here because the checkout mode is automatically inferred to `ityr::checkout_mode::read`.
The user can also provide a user-defined function (lambda) to process each element before summation.

For example, to calculate L2 norm:
```cpp
double norm(ityr::global_span<double> xs) {
  double s2 =
    ityr::parallel_reduce(
        xs.begin(), xs.end(),
        double(0), std::plus<double>{},
        [](double x) { return x * x; });
  return std::sqrt(s2);
}
```

In the above code, `x * x` is applied to each element before summed up.

`ityr::parallel_reduce` supports a general reduction operation more than just summation.
The user can also change the identity value and the combine operator, as long as the operator is *associative* (*commutativity* is not required in Itoyori).

Full code example to calculate AXPY and show the result's L2 norm:
```cpp
#include "ityr/ityr.hpp"

void axpy(double a, ityr::global_span<double> xs, ityr::global_span<double> ys) {
  ityr::parallel_for_each(
      ityr::make_global_iterator(xs.begin(), ityr::checkout_mode::read),
      ityr::make_global_iterator(xs.end()  , ityr::checkout_mode::read),
      ityr::make_global_iterator(ys.begin(), ityr::checkout_mode::read_write),
      [=](const double& x, double& y) {
    y += a * x;
  });
}

double norm(ityr::global_span<double> xs) {
  double s2 =
    ityr::parallel_reduce(
        xs.begin(), xs.end(),
        double(0), std::plus<double>{},
        [](double x) { return x * x; });
  return std::sqrt(s2);
}

int main() {
  ityr::init();
  {
    double a = 0.1;
    std::size_t n = 10000;

    ityr::global_vector<double> x_vec({.collective = true}, n, 1.0);
    ityr::global_vector<double> y_vec({.collective = true}, n, 1.0);

    ityr::global_span<double> x(x_vec.begin(), x_vec.end());
    ityr::global_span<double> y(y_vec.begin(), y_vec.end());

    double result =
      ityr::root_exec([=] {
        axpy(a, x, y);
        return norm(y);
      });

    if (ityr::is_master()) {
      std::cout << result << std::endl;
    }
  }
  ityr::fini();
}
```

## What's Next

It is recommended to read [Pitfalls](./02_pitfalls.md) for writing Itoyori programs.
