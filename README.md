# MultiFloats.jl

#### Copyright © 2019-2024 by David K. Zhang. Released under the [MIT License](https://github.com/dzhang314/MultiFloats.jl/blob/master/LICENSE).

**MultiFloats.jl** is a Julia package for extended-precision arithmetic using 100–400 bits (≈30–120 decimal digits). In this range, it is the fastest extended-precision library that I am aware of. At 100-bit precision, **MultiFloats.jl** is roughly **15x faster than [`BigFloat`](https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers/#Arbitrary-Precision-Arithmetic)** and **1.5x faster than [DoubleFloats.jl](https://github.com/JuliaMath/DoubleFloats.jl)**.

**MultiFloats.jl** is fast because it uses static data structures that do not dynamically allocate memory. In contrast, [`BigFloat`](https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers/#Arbitrary-Precision-Arithmetic) allocates memory for every individual arithmetic operation, requiring frequent pauses for garbage collection. In addition, **MultiFloats.jl** uses branch-free algorithms that can be vectorized for even faster execution on [SIMD](https://github.com/eschnett/SIMD.jl) processors.

**MultiFloats.jl** provides fast, pure-Julia implementations of the basic arithmetic operations (`+`, `-`, `*`, `/`, `sqrt`), comparison operators (`==`, `!=`, `<`, `>`, `<=`, `>=`), `exp`, `log`, and floating-point introspection methods (`isfinite`, `eps`, `minfloat`, etc.). Transcendental functions (`exp`, `log`, `sin`, `cos`, etc.) are supported through [MPFR](https://www.mpfr.org/).

**MultiFloats.jl** stores extended-precision numbers in a **multi-limb representation** that generalizes the idea of [double-double arithmetic](https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format#Double-double_arithmetic) to an arbitrary number of components. This idea takes inspiration from Jonathan Shewchuk's work on [adaptive-precision floating-point arithmetic](http://dx.doi.org/10.1007/pl00009321) and Yozo Hida, Xiaoye Li, and David Bailey's [algorithms for quad-double arithmetic](https://doi.org/10.1109/ARITH.2001.930115), combined in a novel fashion with Julia's unique JIT architecture and metaprogramming capabilities.

## New Features in v2.0

**MultiFloats.jl v2.0** now supports explicit SIMD vector programming using [SIMD.jl](https://github.com/eschnett/SIMD.jl). In addition to the basic scalar types `Float64x2`, `Float64x3`, ..., `Float64x8`, **MultiFloats.jl v2.0** also provides the vector types `v2Float64x2`, `v4Float64x2`, `v8Float64x2`, ..., `v2Float64x8`, `v4Float64x8`, `v8Float64x8`, allowing users to operate on two, four, or eight extended-precision values at a time. These are all instances of the generic type `MultiFloatVec{M,T,N}`, which represents a vector of `M` values, each represented by `N` limbs of type `T`.

**MultiFloats.jl v2.0** also provides the function `mfvgather(array, indices)` to simultaneously load multiple values from a dense array of type `Array{MultiFloat{T,N},D}`.

**MultiFloats.jl v2.0** uses Julia's `@generated function` mechanism to automatically generate code on-demand for arithmetic and comparison operations on `MultiFloat` and `MultiFloatVec` values. Thus, it is no longer necessary to call `MultiFloats.use_standard_multifloat_arithmetic(9)` in order to compute with `Float64x{9}`. Thanks to the magic of metaprogramming, it will just work!

## Breaking Changes from v1.0

**MultiFloats.jl v2.0** no longer exports the `renormalize` function by default. Internal renormalization is now performed more frequently, which should make it unnecessary for users to explicitly call `renormalize` in the vast majority of cases. It is still available for internal use as `MultiFloats.renormalize`.

**MultiFloats.jl v2.0** no longer provides the user-selectable precision modes that were available in v1.0. Consequently, the following functions no longer exist:
```
MultiFloats.use_clean_multifloat_arithmetic()
MultiFloats.use_standard_multifloat_arithmetic()
MultiFloats.use_sloppy_multifloat_arithmetic()
```
My experience has shown that `sloppy` mode causes serious problems in every nontrivial , while `clean` mode has poor performance tradeoffs. Instead of using, say, `Float64x3` in `clean` mode, it almost always makes more sense to use `Float64x4` in `standard` mode. Therefore, I made the decision to streamline development by focusing only on `standard` mode. If you have an application where `sloppy` or `clean` mode is demonstrably useful, please open an issue for discussion!

## Installation

**MultiFloats.jl** is a registered Julia package, so all you need to do is run the following line in your Julia REPL:

```
]add MultiFloats
```

## Usage

**MultiFloats.jl** provides the types `Float64x2`, `Float64x3`, ..., `Float64x8`, which represent extended-precision numbers with 2x, 3x, ..., 8x the precision of `Float64`. These are all subtypes of the parametric type `MultiFloat{T,N}`, where `T = Float64` and <code>N&nbsp;=&nbsp;2,&nbsp;3,&nbsp;...,&nbsp;8</code>.

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

## Features and Benchmarks

We use [two linear algebra tasks](https://github.com/dzhang314/MultiFloats.jl/blob/master/scripts/MultiFloatsBenchmark.jl) to compare the performance of extended-precision floating-point libraries:

* QR factorization of a random 400×400 matrix
* Pseudoinverse of a random 400×250 matrix using [GenericLinearAlgebra.jl][1]

The timings reported below are averages of 10 single-threaded runs performed on an Intel Core i9-11900KF processor.

|                          | MultiFloats `Float64x2`   | Julia Base `BigFloat`        | ArbNumerics `ArbFloat`        | DecFP `Dec128`              | DoubleFloats `Double64`    | Quadmath `Float128`        |
|--------------------------|---------------------------|------------------------------|-------------------------------|-----------------------------|----------------------------|----------------------------|
| 400×400 `qr`&nbsp;time   | 0.257 sec                 | 10.303 sec (40x&nbsp;slower) | 17.871 sec (69x&nbsp;slower)  | 9.448 sec (36x&nbsp;slower) | 0.535 sec (2x&nbsp;slower) | 2.403 sec (9x&nbsp;slower) |
| accurate digits          | 26.2                      | 25.9                         | 25.9                          | 27.7                        | 26.1                       | 27.9                       |
| 400×250 `pinv`&nbsp;time | 1.709 sec                 | 96.655 sec (56x&nbsp;slower) | ❌ Error                      | ❌ Error                   | 3.668 sec (2x&nbsp;slower) | 15.576 sec (9x&nbsp;slower) |
| accurate digits          | 26.0                      | 25.8                         | ❌ Error                      | ❌ Error                   | 25.4                       | 27.9                        |
| user-selectable precision                              | ✔️ | ✔️ | ✔️ | ❌ | ❌ | ❌ |
| avoids dynamic allocation                              | ✔️ | ❌ | ❌ | ✔️ | ✔️ | ✔️ |
| basic arithmetic<br>`+`, `-`, `*`, `/`, `sqrt`         | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ |
| transcendental functions<br>`sin`, `cos`, `exp`, `log` | ⚠️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ |
| compatible with<br>[GenericLinearAlgebra.jl][1]        | ✔️ | ✔️ | ✔️ | ❌ | ✔️ | ✔️ |
| floating-point introspection<br>`minfloat`, `eps`      | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ | ✔️ |

[1]: https://github.com/JuliaLinearAlgebra/GenericLinearAlgebra.jl

## Precision and Performance

The following tables compare the precision (in bits) and performance (in FLOPs) of the arithmetic algorithms provided by **MultiFloats.jl**.

<table>
  <thead>
    <tr>
      <th><b>Number of Accurate Bits</b></th>
      <th><b><code>+</code></b></th>
      <th><b><code>-</code></b></th>
      <th><b><code>*</code></b></th>
      <th><b><code>/</code></b></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b><code>Float64x2</code></b></td>
      <td>107</td>
      <td>107</td>
      <td>103</td>
      <td>103</td>
    </tr>
    <tr>
      <td><b><code>Float64x3</code></b></td>
      <td>161</td>
      <td>161</td>
      <td>156</td>
      <td>155</td>
    </tr>
    <tr>
      <td><b><code>Float64x4</code></b></td>
      <td>215</td>
      <td>215</td>
      <td>209</td>
      <td>207</td>
    </tr>
    <tr>
      <td><b><code>Float64x5</code></b></td>
      <td>269</td>
      <td>269</td>
      <td>262</td>
      <td>259</td>
    </tr>
    <tr>
      <td><b><code>Float64x6</code></b></td>
      <td>323</td>
      <td>323</td>
      <td>314</td>
      <td>311</td>
    </tr>
    <tr>
      <td><b><code>Float64x7</code></b></td>
      <td>377</td>
      <td>377</td>
      <td>367</td>
      <td>364</td>
    </tr>
    <tr>
      <td><b><code>Float64x8</code></b></td>
      <td>431</td>
      <td>431</td>
      <td>420</td>
      <td>416</td>
    </tr>
  </tbody>
</table>

<sup>Worst-case precision observed in ten million random trials using random numbers with uniformly distributed exponents in the range `1.0e-100` to `1.0e+100`. Here, **`+`** refers to addition of numbers with the same sign, while **`-`** refers to addition of numbers with opposite signs. The number of accurate bits was computed by comparison to exact rational arithmetic.</sup>


| Operation | FLOP Count          |
|-----------|---------------------|
| **`+`**   | 3N² + 10N - 6       |
| **`-`**   | 3N² + 10N - 6       |
| **`*`**   | 2N³ - 4N² + 9N - 9  |
| **`/`**   | 6N³ + 4N² - 14N + 2 |

## Caveats

**MultiFloats.jl** requires an underlying implementation of `Float64` with IEEE round-to-nearest semantics. It works out-of-the-box on x86 and ARM but may fail on more exotic architectures.

**MultiFloats.jl** does not attempt to propagate IEEE `Inf` and `NaN` values through arithmetic operations, as this [could cause significant performance losses](https://github.com/dzhang314/MultiFloats.jl/issues/12#issuecomment-751151737). You can pass these values through the `Float64x{N}` container types, and introspection functions (`isinf`, `isnan`, etc.) will work, but arithmetic operations will typically produce `NaN` on all non-finite inputs.
