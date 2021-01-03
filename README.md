# MultiFloats.jl

#### Copyright © 2019-2021 by David K. Zhang. Released under the [MIT License](https://github.com/dzhang314/MultiFloats.jl/blob/master/LICENSE).

**MultiFloats.jl** is a Julia package for extended-precision arithmetic using 100 – 400 bits (≈30 – 120 digits). In this range, it is the fastest extended-precision library that I am aware of. At 100-bit precision, **MultiFloats.jl** is roughly **40x faster than [`BigFloat`](https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers/#Arbitrary-Precision-Arithmetic)** and **2x faster than [DoubleFloats.jl](https://github.com/JuliaMath/DoubleFloats.jl)**.

**MultiFloats.jl** achieves speed by using native, vectorizable `Float64` operations and immutable data structures that do not dynamically allocate memory. In many cases, `MultiFloat` arithmetic can be performed entirely in CPU registers, eliminating memory access altogether. In contrast, [`BigFloat`](https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers/#Arbitrary-Precision-Arithmetic) allocates memory with every single arithmetic operation, requiring frequent pauses for garbage collection.

**MultiFloats.jl** currently provides basic arithmetic operations (`+`, `-`, `*`, `/`, `sqrt`), comparison operators (`==`, `!=`, `<`, `>`, `<=`, `>=`), and floating-point introspection methods (`isfinite`, `eps`, `minfloat`, etc.). Work on trigonometric functions, exponentials, and logarithms is currently in progress.

**MultiFloats.jl** stores extended-precision numbers in a generalized form of [double-double representation](https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format#Double-double_arithmetic) which supports arbitrary number of components. This idea takes inspiration from Jonathan Shewchuk's work on [adaptive-precision floating-point arithmetic](http://dx.doi.org/10.1007/pl00009321) and Yozo Hida, Xiaoye Li, and David Bailey's [algorithms for quad-double arithmetic](https://doi.org/10.1109/ARITH.2001.930115), combined in a novel fashion with Julia's unique JIT architecture and metaprogramming capabilities.

## Usage

**MultiFloats.jl** provides the types `Float64x2`, `Float64x3`, ..., `Float64x8`, which represent extended-precision numbers with 2x, 3x, ..., 8x the precision of `Float64`. These are all instances of the parametric type `MultiFloat{T,N}`, where `T = Float64` and <code>N&nbsp;=&nbsp;2,&nbsp;3,&nbsp;...,&nbsp;8</code>.

Instances of `Float64x2`, `Float64x3`, ..., `Float64x8` are convertible to and from `Float64` and `BigFloat`, as shown in the following example.

```julia
julia> using MultiFloats

julia> x = Float64x4(2.0)

julia> y = sqrt(x)
1.41421356237309504880168872420969807856967187537694807317667973799

julia> y * y - x
-1.1566582006914837e-66
```

A comparison with `sqrt(BigFloat(2))` reveals that all displayed digits are correct in this example.

<sup>Note: **MultiFloats.jl** also provides a `Float64x1` type that has the same precision as `Float64`, but behaves like `Float64x2`–`Float64x8` in terms of supported operations. This is occasionally useful for testing, since any code that works for `Float64x1` should also work for `Float64x2`–`Float64x8` and vice versa.</sup>

## Benchmarks

Two basic linear algebra tasks are used below to compare the performance of extended-precision floating-point libraries:

* QR factorization of a random 400×400 matrix
* Computing the pseudoinverse of a random 400×250 matrix (using **[GenericSVD.jl](https://github.com/JuliaLinearAlgebra/GenericSVD.jl)**)

[See benchmark code here.](https://gist.github.com/dzhang314/3e10463843f4ab5f5a4a2206c877771b) The timings reported below are averages of 10 runs performed under identical conditions on an Intel Core i7-8650U (Surface Book 2 13.5").

|                 | MultiFloats `Float64x2` | Julia Base `BigFloat`        | ArbNumerics `ArbFloat`  | Decimals `Decimal` | DecFP `Dec128`        | DoubleFloats `Double64` | Quadmath `Float128`   |
|-----------------|---------------------------|--------------------------|---------------------------|----------------------|-------------------------|---------------------------|-------------------------|
| 400×400 `qr`&nbsp;time  | 0.257 sec                 | 10.303 sec (40x&nbsp;slower) | 17.871 sec (69x&nbsp;slower)  | ❌ Error              | 9.448 sec (36x&nbsp;slower) | 0.535 sec (2x&nbsp;slower)    | 2.403 sec (9x&nbsp;slower)  |
| accurate digits | 26.0                      | 25.9                     | 25.9                      | ❌ Error              | 27.6                    | 26.1                      | 28.1                    |
| 400×250 `pinv`&nbsp;time  | 1.709 sec                 | 96.655 sec (56x&nbsp;slower) | 133.085 sec (77x&nbsp;slower) | ❌ Error              | ❌ Error                 | 3.668 sec (2x&nbsp;slower)    | 15.576 sec (9x&nbsp;slower) |
| accurate digits | 25.6                      | 25.8                     | 25.8                      | ❌ Error              | ❌ Error                 | 25.4                      | 27.9                    |

## Feature Comparison

|                                                        | MultiFloats | BigFloat | ArbNumerics | Decimals | DecFP | DoubleFloats | Quadmath |
|--------------------------------------------------------|-------------|----------|-------------|----------|-------|--------------|----------|
| user-selectable precision                              | ✔️          | ✔️      | ✔️          | ❌       | ❌    | ❌          | ❌       |
| avoids dynamic memory allocation                       | ✔️          | ❌      | ❌          | ❌       | ✔️    | ⚠️          | ✔️       |
| basic arithmetic `+`, `-`, `*`, `/`, `sqrt`            | ✔️          | ✔️      | ✔️          | ❌       | ✔️    | ✔️          | ✔️       |
| transcendental functions `sin`, `cos`, `exp`, `log`    | ❌ (WIP)    | ✔️      | ✔️          | ❌       | ✔️    | ✔️          | ✔️       |
| compatible with **[GenericSVD.jl](https://github.com/JuliaLinearAlgebra/GenericSVD.jl)**                         | ✔️          | ✔️      | ✔️          | ❌       | ❌    | ✔️          | ✔️       |
| floating-point introspection `minfloat`, `eps`         | ✔️          | ✔️      | ✔️          | ❌       | ✔️    | ✔️          | ✔️       |

## Complexity of operations

The number of flops per operation scales cubically for `*` and `/` and quadratically for `+` and `-` as a function of `N`. Counting explicit fma's as a single operation, the number of flops for `N` in the range `2:8` is:

| Operation | Number of flops     |
|-----------|---------------------|
| `+`       | 3N² + 4N - 9        |
| `-`       | 3N² + 4N - 9        |
| `*`       | 2N³ - 4N² + 9N - 9  |
| `/`       | 6N³ - 2N² - 11N + 5 |
