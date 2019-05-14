## Project Aim

Provide a user-friendly interface using C++ Standard Library paradigms and concepts

As a library user, I want:

- helpers to improve readability (e.g. when creating buffer accessors)
- interop with C++ Standard Library types especially with containers std::vector and std::array
- standard implementations of common algorithms and selected C++ Standard Library algorithms
- extensive support for multi-dimensional buffers including multi-dimensional versions of selected standard algorithms
- an async interface using futures
- a C++20 ranges-like interface including kernel composition for exposing multiple, dependent tasks to the runtime

As a library user, it would be nice to have

- a C++20 module library

From a technical point of view, it should:

- impose zero overhead (ideally)
- use an iterator-based algorithms interface (preferably using sentinels for end iterators, requires C++17 for support in range-based for loops)
- work with all major compilers (clang, gcc, msvc, icc)
- be extensible and configurable

## Components

## Helpers

- `clamping_iterator` 

### C++ Standard Library interop

- `buffer_iterator` for providing a std like algorithm interface for celerity buffers
- `begin(celerity::buffer<>)`, `end(celerity::buffer<>)` to enable range-based for loops on master
- `copy`, `copy_if`, `copy_n` for copying data from/to a std container

### Algorithms

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

- multi-dimensional `buffer_iterator`
- multi-dimensional `filter_iterator` maps to neighbour accessor
- `slice_iterator` maps to slice accessor
- `n_dim_iterator` for STD containers

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