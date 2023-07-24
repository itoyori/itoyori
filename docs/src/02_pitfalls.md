# Pitfalls

Itoyori enables concise and straightforward expression for parallel algorithms, but Itoyori-style programming has some (perhaps nonintuitive) pitfalls.
If you encounter segmentation faults or weird behavior with Itoyori, it is recommended to consult this document to check if your program does not violate any of the following rules.

[TOC]

## Lifetime of Global Objects

Global memory must be freed before finalization.

Bad example:
```cpp
int main() {
  ityr::init();
  ityr::global_vector<int> gv(100);
  /* ... */
  ityr::fini();
  // The global vector is destroyed after finalization!
}
```

Good example:
```cpp
int main() {
  ityr::init();
  {
    ityr::global_vector<int> gv(100);
    /* ... */
  }
  ityr::fini();
}
```

## Captures in Lambda Expressions for Forking

In Itoyori, threads cannot have raw pointers/references to any other threads, including their parents.
Therefore, function arguments should be passed to child threads by values, not by references.
This means lambda expressions for fork operations should capture variables by copy.

However, this will enforce copy semantics to captured objects.
For example, this can be problematic when `ityr::global_vector` is used for parallel execution.

Bad example:
```cpp
ityr::global_vector<int> gv({.collective = true}, 100);
/* ... */
ityr::root_exec([=] { // A copy of the entire global vector is created
  ityr::parallel_for_each(
      ityr::make_global_iterator(gv.begin(), ityr::checkout_mode::read),
      ityr::make_global_iterator(gv.end()  , ityr::checkout_mode::read),
      [=](int x) { /* ... */ });
});
```

The above program will copy the whole data stored in the global vector across `ityr::root_exec()`.
To prevent this unnecessary copy, the user can instead use `ityr::global_span`, which does not hold ownership for the memory region.

Good example:
```cpp
ityr::global_vector<int> gv({.collective = true}, 100);
ityr::global_span<int> gs(gv.begin(), gv.end());
/* ... */
ityr::root_exec([=] {
  ityr::parallel_for_each(
    ityr::make_global_iterator(gs.begin(), ityr::checkout_mode::read),
    ityr::make_global_iterator(gs.end()  , ityr::checkout_mode::read),
    [=](int x) { /* ... */ });
});
```

Another pitfall exists in using lambda expressions inside a class/struct.
To demonstrate this problem, suppose that an additional parameter `cutoff` is added to the Fibonacci example, so that sufficiently small leaf computations run sequentially.

Bad example:
```cpp
struct fib {
  fib(int c) : cutoff(c) {}

  long calc(int n) const {
    if (n <= cutoff) {
      return calc_serial(n);
    } else {
      auto [x, y] =
        ityr::parallel_invoke(
          [=] { return calc(n - 1); },
          [=] { return calc(n - 2); }
        );
      return x + y;
    }
  }

  long calc_serial(int n) const { /* ... */ }

  int cutoff;
};
```

Until C++20, `this` is implicitly captured by reference, even if the default capture `=` is specified.
This means that `*this` object in the parent is referred by children, which is not allowed in Itoyori.
To prevent that, `*this` objects must be explicitly copy-captured (e.g., `[=, *this]`) when making fork/join calls inside a class/struct.

Good example:
```cpp
struct fib {
  fib(int c) : cutoff(c) {}

  long calc(int n) const {
    if (n <= cutoff) {
      return calc_serial(n);
    } else {
      auto [x, y] =
        ityr::parallel_invoke(
          [=, *this] { return calc(n - 1); },
          [=, *this] { return calc(n - 2); }
        );
      return x + y;
    }
  }

  long calc_serial(int n) const { /* ... */ }

  int cutoff;
};
```

Nevertheless, copy semantics will apply to the class object with this fix, which may not be desired in some cases.
One option is to move the `cutoff` parameter outside the class by globalizing it.

## Usage of Heap Memory Across Thread Migration

As threads can be dynamically migrated to other processes in Itoyori, allocating objects in normal heap memory is not recommended.
For example, standard containers such as `std::vector` will cause heap memory allocation.

Bad example:
```cpp
ityr::root_exec([=] {
  std::vector<int> v(100);

  ityr::parallel_invoke(
    /* ... */
  );

  // The executing process can be different from the previous one
  for (auto&& x : v) {
    x = /* ... */;
  }
});
```

In the above example, the root thread allocates `std::vector` in the local process's heap memory, after which the thread forks child threads.
However, at fork/join calls, the thread can be dynamically migrated to another process that does not have access to the previous process's heap memory.
In order to keep heap-allocated objects across fork/join calls, they must be allocated in global heaps (e.g., by using `std::global_vector`) and accessed with checkout/checkin calls.

Good example:
```cpp
ityr::root_exec([=] {
  ityr::global_vector<int> v(100);

  ityr::parallel_invoke(
    /* ... */
  );

  auto vc = ityr::make_checkout(v.begin(), v.end(), ityr::checkout_mode::read_write);
  for (auto&& x : vc) {
    x = /* ... */;
  }
});
```

## Checkout/Checkin Across Thread Migration

Similar to the previous pitfall, checkout/checkin operations cannot go across fork/join calls.

Bad example:
```cpp
ityr::root_exec([=] {
  ityr::global_vector<int> v(100);

  auto vc = ityr::make_checkout(v.begin(), v.end(), ityr::checkout_mode::read_write);
  /* ... */

  ityr::parallel_invoke(
    /* ... */
  );

  // The checkin operation occurs here
});
```

With `ityr::make_checkout()`, a checkin operation is automatically performed when its lifetime is over, but if fork/join calls are in the middle, the thread can be migrated to other processes.
As performing a pair of checkout/checkin operation in different processes is not allowed, checked-out memory must be returned to the system before fork/join calls.

Good example:
```cpp
ityr::root_exec([=] {
  ityr::global_vector<int> v(100);

  auto vc = ityr::make_checkout(v.begin(), v.end(), ityr::checkout_mode::read_write);
  /* ... */
  vc.checkin(); // explicit checkin

  ityr::parallel_invoke(
    /* ... */
  );

  // checkout again after fork/join if needed
  vc = ityr::make_checkout(v.begin(), v.end(), ityr::checkout_mode::read_write);
  /* ... */
});
```

Note that, if the thread is eventually executed by the same process, the global memory is likely to be cached in the local process.

## Nested Parallelism with Global Iterators

Global iterators are convenient to automatically checkout global memory with high-level parallel patterns, but unfortunately, they are incompatible with nested parallelism.

Bad example:
```cpp
ityr::root_exec([=] {
  ityr::global_vector<int> gv(100);
  /* ... */
  ityr::parallel_reduce(
      gv.begin(), gv.end(),
      0, std::plus<int>{},
      [=](int x) {
    /* ... */
    ityr::parallel_invoke(
      /* ... */
    });
    return x + /* ... */;
  });
});
```

In the above example, `ityr::parallel_reduce()` internally checks out memory for `gv` at some granularity, but performing fork/join operations at each iteration causes the above-mentioned issue of checkout/checkin across thread migration.
If there is nested parallelism, automatic checkout should be disabled by specifing `ityr::checkout_mode::no_access` mode.

Good example:
```cpp
ityr::root_exec([=] {
  ityr::global_vector<int> gv(100);
  /* ... */
  ityr::parallel_reduce(
      ityr::make_global_iterator(gv.begin(), ityr::checkout_mode::no_access),
      ityr::make_global_iterator(gv.end()  , ityr::checkout_mode::no_access),
      0, std::plus<int>{},
      [=](auto&& x_ref) {
    /* ... */
    ityr::parallel_invoke(
      /* ... */
    });
    return x_ref.get() + /* ... */;
  });
});
```

With `ityr::checkout_mode::no_access` mode, global references (`ityr::ori::global_ref`) are passed as arguments to the user-provided function.
Calling checkout/checkin operations is then on the user's responsibility.
In the above case, global reference `x_ref` is used to get the global value with `ityr::ori::global_ref::get()`.

## Unattended Data Races

Itoyori's checkout mode (`ityr::checkout_mode`) must be specified so that no data race occurs.
Even if the program does not actually modify the checked-out data, the runtime system treats the checked-out region as *dirty* if `read_write` or `write` mode is specified.
This means that the checkout mode is different from access privilege, and specifying `ityr::checkout_mode::read_write` is not a conservative approach.

Bad example:
```cpp
ityr::global_span<int> a(/* ... */);

ityr::parallel_invoke(
  [=] {
    auto cs = ityr::make_checkout(a.data(), a.size(), ityr::checkout_mode::read_write);
    /* read-only access for `cs` */
  },
  [=] {
    auto cs = ityr::make_checkout(a.data(), a.size(), ityr::checkout_mode::read_write);
    /* read-only access for `cs` */
  }
);
```

The above program concurrently checks out the same region with the `read_write` mode.
Even if the program does not actually write to the region, this is not allowed in Itoyori because it can lead to unattended data update.

Good example:
```cpp
ityr::global_span<int> a(/* ... */);

ityr::parallel_invoke(
  [=] {
    auto cs = ityr::make_checkout(a.data(), a.size(), ityr::checkout_mode::read);
    /* read-only access for `cs` */
  },
  [=] {
    auto cs = ityr::make_checkout(a.data(), a.size(), ityr::checkout_mode::read);
    /* read-only access for `cs` */
  }
);
```

The user should precisely specify the checkout mode for each checkout call.
