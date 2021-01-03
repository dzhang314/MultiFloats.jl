# MultiFloats.jl

#### Copyright © 2019-2021 by David K. Zhang. Released under the [MIT License](https://github.com/dzhang314/MultiFloats.jl/blob/master/LICENSE).

**MultiFloats.jl** is a Julia package for extended-precision arithmetic using 100 – 400 bits (≈ 30 – 120 digits). In this range, it is the fastest extended-precision library that I am aware of. At 100-bit precision, **MultiFloats.jl** is roughly **40x faster than [`BigFloat`](https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers/#Arbitrary-Precision-Arithmetic)** and **2x faster than [DoubleFloats.jl](https://github.com/JuliaMath/DoubleFloats.jl)**.

**MultiFloats.jl** achieves speed by using native, vectorizable `Float64` operations and immutable data structures that do not dynamically allocate memory. In many cases, `MultiFloat` arithmetic can be performed entirely in CPU registers, eliminating memory access altogether. In contrast, [`BigFloat`](https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers/#Arbitrary-Precision-Arithmetic) allocates memory with every single arithmetic operation, requiring frequent pauses for garbage collection.

**MultiFloats.jl** currently provides basic arithmetic operations (`+`, `-`, `*`, `/`, `sqrt`), comparison operators (`==`, `!=`, `<`, `>`, `<=`, `>=`), and floating-point introspection methods (`isfinite`, `eps`, `minfloat`, etc.). Work on trigonometric functions, exponentials, and logarithms is currently in progress.

**MultiFloats.jl** stores extended-precision numbers in a generalized form of [double-double representation](https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format#Double-double_arithmetic) which supports arbitrary number of components. This idea takes inspiration from Jonathan Shewchuk's work on [adaptive-precision floating-point arithmetic](http://dx.doi.org/10.1007/pl00009321) and Yozo Hida, Xiaoye Li, and David Bailey's [algorithms for quad-double arithmetic](https://doi.org/10.1109/ARITH.2001.930115), combined in a novel fashion with Julia's unique JIT architecture and metaprogramming capabilities.

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

## Precision Modes

**MultiFloats.jl** provides three user-selectable levels of precision. The default mode is **standard mode**, which aims for a sweet-spot between performance and precision. **Clean mode** does a bunch of extra work to get the last few bits of the answer right, while **sloppy mode** throws some safety guarantees out the window in pursuit of reckless speed.

**When in doubt, stick to standard mode.** If you come across a numerical bug, then switch to clean mode. The use of sloppy mode should be limited to highly specialized cases with strong assumptions about the input data. Sloppy mode can exhibit bizarre failure modes related to (the lack of) renormalization that are difficult to reproduce.

To switch between precision modes, call any of the following three functions:

```
MultiFloats.use_clean_multifloat_arithmetic()
MultiFloats.use_standard_multifloat_arithmetic() [default]
MultiFloats.use_sloppy_multifloat_arithmetic()
```

Note that switching between precision modes is a very expensive operation that is not thread-safe. Calling any of these functions triggers recompilation of all `MultiFloat` arithmetic operations, so this should never be performed in the middle of a calculation.

Each of these functions takes an optional `Int` parameter that dictates the `MultiFloat` sizes to generate code for. For example, if you want to use `Float64x9` (which is not provided by default), you can call `MultiFloats.use_standard_multifloat_arithmetic(9)` to define the necessary arithmetic operators. Note that this will not define the _name_ `Float64x9`; you will have to refer to this type as `MultiFloat{Float64,9}` or `Float64x{9}`.

The following two tables compare the precision (in bits) and performance (in FLOPs) of the three modes provided by **MultiFloats.jl**.

<table>
  <thead>
    <tr>
      <th rowspan=2><b>Number of<br>Accurate Bits</b></th>
      <th colspan=4>Clean</th>
      <th colspan=4>Standard</th>
      <th colspan=4>Sloppy</th>
    </tr>
    <tr>
      <th><b><code>+</code></b></th>
      <th><b><code>-</code></b></th>
      <th><b><code>*</code></b></th>
      <th><b><code>/</code></b></th>
      <th><b><code>+</code></b></th>
      <th><b><code>-</code></b></th>
      <th><b><code>*</code></b></th>
      <th><b><code>/</code></b></th>
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
      <td>107</td>
      <td>106</td>
      <td>107</td>
      <td>107</td>
      <td>103</td>
      <td>103</td>
      <td>104</td>
      <td>≈50</td>
      <td>103</td>
      <td>103</td>
    </tr>
    <tr>
      <td><b><code>Float64x3</code></b></td>
      <td>161</td>
      <td>161</td>
      <td>161</td>
      <td>160</td>
      <td>161</td>
      <td>161</td>
      <td>156</td>
      <td>155</td>
      <td>157</td>
      <td>≈50</td>
      <td>156</td>
      <td>155</td>
    </tr>
    <tr>
      <td><b><code>Float64x4</code></b></td>
      <td>215</td>
      <td>215</td>
      <td>215</td>
      <td>214</td>
      <td>215</td>
      <td>215</td>
      <td>209</td>
      <td>207</td>
      <td>211</td>
      <td>≈50</td>
      <td>209</td>
      <td>207</td>
    </tr>
    <tr>
      <td><b><code>Float64x5</code></b></td>
      <td>269</td>
      <td>269</td>
      <td>269</td>
      <td>268</td>
      <td>269</td>
      <td>269</td>
      <td>262</td>
      <td>259</td>
      <td>264</td>
      <td>≈50</td>
      <td>262</td>
      <td>259</td>
    </tr>
    <tr>
      <td><b><code>Float64x6</code></b></td>
      <td>323</td>
      <td>323</td>
      <td>323</td>
      <td>322</td>
      <td>323</td>
      <td>323</td>
      <td>314</td>
      <td>311</td>
      <td>317</td>
      <td>≈50</td>
      <td>314</td>
      <td>311</td>
    </tr>
    <tr>
      <td><b><code>Float64x7</code></b></td>
      <td>377</td>
      <td>377</td>
      <td>377</td>
      <td>376</td>
      <td>377</td>
      <td>377</td>
      <td>367</td>
      <td>364</td>
      <td>371</td>
      <td>≈50</td>
      <td>367</td>
      <td>364</td>
    </tr>
    <tr>
      <td><b><code>Float64x8</code></b></td>
      <td>431</td>
      <td>431</td>
      <td>431</td>
      <td>430</td>
      <td>431</td>
      <td>431</td>
      <td>420</td>
      <td>416</td>
      <td>425</td>
      <td>≈50</td>
      <td>420</td>
      <td>416</td>
    </tr>
  </tbody>
</table>

In this table, **`+`** refers to addition of numbers with the same sign, while **`-`** refers to addition of numbers with opposite signs. Destructive cancellation in sloppy mode can cause only the leading component of a difference to be meaningful. However, this only occurs when subtracting two numbers that are _very_ close to each other (i.e., relative differences on the order of `1.0e-16`).


| FLOP Count | Clean               | Standard            | Sloppy             |
|------------|---------------------|---------------------|--------------------|
| **`+`**    | 3N² + 10N - 6       | 3N² + 10N - 6       | 3N² + N - 3        |
| **`-`**    | 3N² + 10N - 6       | 3N² + 10N - 6       | 3N² + N - 3        |
| **`*`**    | 2N³ + 2N² + 7N - 8  | 2N³ - 4N² + 9N - 9  | 2N³ - 4N² + 6N - 3 |
| **`/`**    | 6N³ + 16N² - 5N - 4 | 6N³ + 4N² - 14N + 2 | 6N³ - 8N² + 4N - 1 |

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
| compatible with **[GenericSVD.jl](https://github.com/JuliaLinearAlgebra/GenericSVD.jl)** | ✔️          | ✔️      | ✔️          | ❌       | ❌    | ✔️          | ✔️       |
| floating-point introspection `minfloat`, `eps`         | ✔️          | ✔️      | ✔️          | ❌       | ✔️    | ✔️          | ✔️       |
