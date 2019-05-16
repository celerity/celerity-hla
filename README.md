## Project Aim

Provide a user-friendly interface using C++ Standard Library paradigms and concepts

As a library user, I want:

- interop with C++ Standard Library types especially with containers std::vector and std::array
- standard implementations of common algorithms and selected C++ Standard Library algorithms
- extensive support for multi-dimensional buffers including multi-dimensional versions of selected standard algorithms
- a C++20 ranges-like interface including kernel composition for exposing multiple, dependent tasks to the runtime

As a library user, it would be nice to have

- a C++20 module library

From a technical point of view, it should:

- impose zero overhead (ideally)
- use an iterator-based algorithms interface (preferably using sentinels for end iterators, requires C++17 for support in range-based for loops)
- work with all major compilers (clang, gcc, msvc, icc)
- be extensible and configurable

## Components

### C++ Standard Library Interop

- `device_vector` for wrapping celerity buffers avoid intrusive changes of the public inteface of the celerity core.
- iterators for `device_vector` for providing a std like algorithm interface
  - `one_to_one_iterator`
  - `neighbour_iterator`
  - `clamping_neighbour_iterator`
- `copy`, `copy_if`, `copy_n`, `transform` for copying data from/to STD containers
- STD-like constructors for celerity buffers (using ranges or iterator-pairs)

### Algorithms

- `begin(device_vector)`, `end(device_vector)` to enable range-based for loops on master
- use execution policies akin to STD execution policies to decide where to run the algorithm (on the master or some node)

#### STD Algorithms

- `copy`, `copy_if`, `copy_n`
- `count`, `count_if`
- `for_each`, `for_each_n`
- `transform`
- `fill`, `fill_n`
- `generate`, `generate_n`
- `min`, `max`, `minmax`
- `iota`
- `reduce`
- `inner_product`
- `adjacent_difference`
- `partial_sum`
- `exclusive_scan`
- `inclusive_scan`

#### Common distributed algorithms

-

### Multi-dimensional Buffer Support

- multi-dimensional `one_to_one_iterator`
- multi-dimensional `neighbour_iterator` maps to neighbour accessor
- multi-dimensional `clamping_neighbour_iterator`
- multi-dimensional `slice_iterator` maps to slice accessor
- multi-dimensional `n_dim_iterator` for STD containers

#### Multi-dimensional Algorithms

- `copy`, `copy_if`, `copy_n`
- `count`, `count_if`
- `for_each`, `for_each_n`
- `transform`
- `fill`, `fill_n`
- `generate`, `generate_n`
- `min`, `max`, `minmax`
- `iota` ?
- `reduce` ?
- `inner_product` ?
- `adjacent_difference` ?
- `partial_sum` ?
- `exclusive_scan` ?
- `inclusive_scan` ?

### Ranges

- C++20 ranges for expressing (sub-) regions
- Range adaptors/actions
- `ContiguousIterator` concept