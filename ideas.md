# Project Aim

Provide a user-friendly interface using C++ Standard Library paradigms and concepts

As a library user, I want:

- helpers to improve readability (e.g. when creating buffer accessors)
- interop with C++ Standard Library types especially with containers std::vector and std::array
- standard implementations of common algorithms and selected C++ Standard Library algorithms
- extensive support for multi-dimensional buffers including multi-dimensional versions of selected standard algorithms
- an async interface using futures

As a library user, it would be nice to have

- a C++20 ranges-like interface including kernel composition
- a C++20 module library

From a technical point of view, it should:

- impose zero overhead (ideally)
- use an iterator-based algorithms interface
- work with all major compilers (clang, gcc, msvc, icc)
- be extensible and configurable