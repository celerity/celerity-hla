## Project Aim

Provide a user-friendly interface using C++ Standard Library paradigms and concepts.

#### Goals

 - ease first steps for users without experience with SYCL
 - provide standard facilities for common tasks

#### As a library user, I want:

- [ ] interop with C++ Standard Library types especially with containers std::vector and std::array - **_in progress_**
- [ ] standard implementations of common algorithms and selected C++ Standard Library algorithms  - **_in progress_**
- [ ] extensive support for multi-dimensional buffers including multi-dimensional versions of selected standard algorithms - **_partially done_**
- [x] a C++20 ranges-like interface including kernel composition for exposing task sequences to the runtime
- [ ] CMake integration

#### As a library user, it would be nice to have

- [ ] a Conan package
- [ ] a vcpkg package
- [ ] C++20 modules

#### From a technical point of view, it should:

- impose zero runtime overhead (ideally)
- use an iterator-based algorithms interface (preferably using sentinels for end iterators, requires C++17 for support in range-based for loops)
- work with all major compilers 
  - [x] clang
  - [ ] gcc
  - [ ] msvc
  - [ ] icc

## Components

### C++ Standard Library Interop

- [x] ~~`device_vector` for wrapping celerity buffers avoid intrusive changes of the public interface of the celerity core.~~
  instead global functions in combination with ADL
- [ ] ~~iterators for `device_vector` for providing a std like algorithm interface~~
  Iterators do only specify the range of iteratiuon but not which elements are 
  accessible. Range-access is controlled by accepting different accessor types
  in the callback function (see [Multi-dimensional Buffer Support](#multi-dimensional-buffer-support)).
- [ ] `copy`, `copy_if`, `copy_n`, `transform` for copying data from/to STD containers
- [ ] ~~STD-like constructors for celerity buffers (using ranges or iterator-pairs)~~
-     would affect the public interface of celerity buffers. Maybe a dedicated buffer type will be implemented to support this (as originally planned)

### Algorithms

- [x] `begin(buffer)`, `end(buffer)` to enable range-based for loops on master - _only inside of `on_master(...)`_
- [x] use execution policies akin to STD execution policies to decide where to run the algorithm (distributed or master-only) - _range adaptors/actions require distributed execution_ 

#### STD Algorithms

- [x] `copy` - rudimentary
- [ ] `copy_if`
- [ ] `copy_n`
- [ ] `count`
- [ ] `count_if`
- [x] `for_each` - master-only as device kernels can (typically) not have side-effects
- [ ] `for_each_n`
- [x] `transform` - no support for STD containers, no in-place transformation
- [x] `fill`
- [x] `fill_n` - only available as building block with unspecified output iterator
- [x] `generate`
- [x] `generate_n` - only available as building block with unspecified output iterator
- [ ] `min`, `max`, `minmax`
- [ ] `iota`
- [ ] `reduce`
- [ ] `inner_product`
- [ ] `adjacent_difference`
- [ ] `partial_sum`
- [ ] `exclusive_scan`
- [ ] `inclusive_scan`

#### Common distributed algorithms

- [ ] pool - akin to the convolution neural network layer, reduce input range to smaller range using pooling operation (e.g. max-pooling)

### Multi-dimensional Buffer Support

- [x] multi-dimensional ~~`one_to_one_iterator`~~ single-element accessor
- [x] multi-dimensional ~~`neighbour_iterator`~~ chunk<...> accessor
- [x] ~~multi-dimensional `clamping_neighbour_iterator`~~ the chunk<> accessor 
  - `chunk<>` detects whether is lies on the borders and computation may branch accordingly 
- [x] multi-dimensional `slice_iterator` slice accessor
- [ ] multi-dimensional `n_dim_iterator` for STD containers

#### Multi-dimensional Algorithms

- [x] `copy` - rudimentary
- [ ] `copy_if`
- [ ] `copy_n`
- [ ] `count`
- [ ] `count_if`
- [x] `for_each` - master-only as device kernels can (typically) not have side-effects
- [ ] `for_each_n`
- [x] `transform`
- [x] `fill`
- [x] `fill_n` - only available as building block with unspecified output iterator
- [x] `generate`
- [x] `generate_n` - only available as building block with unspecified output iterator
- [ ] `min`
- [ ] `max`
- [ ] `minmax`
- [ ] `iota` ?
- [ ] `reduce` ?
- [ ] `inner_product` ?
- [ ] `adjacent_difference` ?
- [ ] `partial_sum` ?
- [ ] `exclusive_scan` ?
- [ ] `inclusive_scan` ?

### Ranges

- [ ] C++20 ranges for expressing (sub-) regions
- [x] Range adaptors/actions for composing task graph
    - [ ] adaptor/action for custom kernels using the traditional celerity programming model
    - [x] explore possibility to fuse compatible kernels
       - fusion of
         - [x] kernels with one input and one output with single element access (i.e. not `chunk<>`, `slice<>` or `all<>`)
         - [ ] kernels with two inputs and one output with single element access (i.e. not `chunk<>`, `slice<>` or `all<>`)
         - [ ] `fill` kernels and single `chunk<>` access kernels by serializing filling of chunk
         - [ ] non `fill` kernels and 'chunk<>' access kernels by serializing kernel (requires cl::sycl::detail::make_item)
- [ ] ~~`ContiguousIterator` concept~~ will be reformulated soon