# Celerity HLA

A high-level API on top of [Celerity](https://github.com/celerity/celerity-runtime) to support functional, C++20 range-adaptors-like patterns.

## Overview

```cpp
buffer<int, 2> in_a{{16, 16}}, 
buffer<int, 2> in_b{{16, 16}};

constexpr float alpha = 2.0f;

// matrix-matrix multiplication
const auto multiply = [](const hla::slice<int, 1>& a, 
                         const hla::slice<int, 0>& b) { 
    return std::inner_product(begin(a), end(a), begin(b), 0);
};

// out = alpha*(A*B)
auto out = in_a | hla::transform<class _1>(multiply) << in_b
                | hla::transform<class _2>([=](int i){ return alpha*i; })
                | hla::submit_to(q);
```

## Dependencies

 - Celerity >= v0.2.1
 - CMake (3.5.1 or newer)
 - A C++17 compiler
