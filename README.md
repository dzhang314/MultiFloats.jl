# MultiFloats.jl

_Copyright © 2019-2026 by David K. Zhang. Released under the [MIT License](https://github.com/dzhang314/MultiFloats.jl/blob/master/LICENSE)._

**MultiFloats.jl** is the world's fastest library for extended-precision floating-point arithmetic with 128–256 bits (30–60 decimal digits). At 30-digit precision, **MultiFloats.jl** is **30× faster than [`BigFloat`](https://docs.julialang.org/en/v1/manual/integers-and-floating-point-numbers/#Arbitrary-Precision-Arithmetic)**, **6× faster than [Quadmath.jl](https://github.com/JuliaMath/Quadmath.jl)**, and **2× faster than [DoubleFloats.jl](https://github.com/JuliaMath/DoubleFloats.jl)**.

**MultiFloats.jl** is the product of [significant original research](https://purl.stanford.edu/gt930wy1453) on [high-precision computer arithmetic](https://theory.stanford.edu/~aiken/publications/papers/cav25.pdf), culminating in the discovery of [a novel class of fast branch-free algorithms](https://theory.stanford.edu/~aiken/publications/papers/sc25.pdf) for floating-point arithmetic beyond machine precision. These state-of-the-art algorithms are both faster and more accurate than all previously known techniques.

**MultiFloats.jl** provides pure-Julia implementations of arithmetic (`+`, `-`, `*`, `/`, `sqrt`), comparison (`==`, `!=`, `<`, `>`, `<=`, `>=`), floating-point introspection (`isfinite`, `eps`, `nextfloat`, etc.), and round-trip-safe float-to-string conversion. Transcendental functions (`exp`, `log`, `sin`, `cos`, etc.) are supported through [MPFR](https://www.mpfr.org/).

Like all technical innovations, **MultiFloats.jl** proudly stands on the shoulders of giants. It extends the idea of [double-double arithmetic](https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format#Double-double_arithmetic), taking inspiration from Jonathan Shewchuk's [adaptive-precision methods](http://dx.doi.org/10.1007/pl00009321); Yozo Hida, Xiaoye Li, and David Bailey's [quad-double algorithms](https://doi.org/10.1109/ARITH.2001.930115); and Mioara Joldes, Jean-Michel Muller, Valentina Popescu, and Warwick Tucker's [CAMPARY library](https://doi.org/10.1007/978-3-319-42432-3_29).



## New Features in v3.0

**MultiFloats.jl v3.0** introduces new arithmetic algorithms that are **simultaneously faster and more accurate** than **MultiFloats.jl v2.0** (and, to my knowledge, all other multiprecision libraries). These new algorithms always return results in normalized form, solving the [denormalization issues](https://github.com/dzhang314/MultiFloats.jl/issues/42) that caused [spurious accuracy losses](https://github.com/dzhang314/MultiFloats.jl/issues/45) in **MultiFloats.jl v2.0**.

**MultiFloats.jl v3.0** also introduces a round-trip-safe float-to-string conversion algorithm, guaranteeing that conversion of a `MultiFloat` to string and back always yields a numerically identical result.

**MultiFloats.jl v3.0** exports the types `Float32x2`, `Float32x3`, and `Float32x4`, which are intended for use on processors lacking `Float64` hardware support (e.g., GPUs and NPUs). Note that `Float64x2` is faster and slightly more accurate than `Float32x4` and should be preferred on hardware with native `Float64` support.

Other notable changes:

- The new multiplication algorithm strictly obeys the commutative law. Previously, it was possible for `x * y` and `y * x` to return [slightly different results](https://github.com/dzhang314/MultiFloats.jl/issues/50) due to internal accumulation of floating-point rounding errors. (Addition has always strictly obeyed the commutative law.)
- Comparison operators (`==`, `!=`, `<`, `>`, `<=`, `>=`) are now significantly faster because it is no longer necessary to run a renormalization loop on their inputs.
- The formula for `precision(MultiFloat{T,N})` has been slightly changed to better reflect the known error bounds for the new arithmetic algorithms.
- `rand(Float32xN)` has been implemented.
- `Base.decompose` has been implemented, which enables [hashing](https://github.com/dzhang314/MultiFloats.jl/pull/55) and [comparison to rational numbers](https://github.com/dzhang314/MultiFloats.jl/issues/54).
- Significant correctness issues have been fixed in `prevfloat` and `nextfloat`. Previously, it was possible for these functions to skip over intermediate values because they did not properly consider interactions between multiple limbs.
- A new function, `MultiFloats.canonize`, has been added to reduce all numerically equivalent `MultiFloat`s to a single canonical form. Previously, `MultiFloats.renormalize` was thought to be sufficient, but in extremely rare (roughly 1 in 2^53) cases, it is possible for a single number to have multiple distinct normalized representations.
- Two new functions, `MultiFloats.isnormalized` and `MultiFloats.iscanonical`, have been added to determine whether a `MultiFloat` is in normalized or canonical form, respectively. These are almost always synonymous, but in extremely rare (roughly 1 in 2^53) cases, it is possible for a `MultiFloat` to be normalized but not canonical.



## Breaking Changes from v2.0

The types `Float64x5`, `Float64x6`, `Float64x7`, and `Float64x8` have been removed from the initial release of **MultiFloats.jl v3.0**. The arithmetic algorithms for these types were [fundamentally broken in **MultiFloats.jl v2.0**](https://github.com/dzhang314/MultiFloats.jl/issues/42), and new algorithms that are provably correct for all possible inputs have not yet been found. Computer searches for correct algorithms are ongoing, and these types will be reinstated once such algorithms are found.

The SIMD vector types have been renamed from `v8Float64x2` to `Vec8Float64x2` to follow Julia's convention of capitalizing type names.



## Installation

**MultiFloats.jl** is a [registered Julia package](https://juliahub.com/ui/Packages/General/MultiFloats), so it can be installed by typing

```
]add MultiFloats
```

into the Julia REPL.



## Usage

**MultiFloats.jl** provides the types `Float64x2`, `Float64x3`, and, `Float64x4`, which represent extended-precision numbers with 2×, 3×, or, 4× the precision of `Float64`. These are all instances of the parametric type `MultiFloat{T,N}` where `T = Float64` and <code>N&nbsp;=&nbsp;2,&nbsp;3,&nbsp;4</code>.

`MultiFloat`s are convertible to and from `Float64` and `BigFloat`, as shown in the following example.

```julia
julia> using MultiFloats

julia> x = Float64x4(2.0)

julia> y = sqrt(x)
1.41421356237309504880168872420969807856967187537694807317667973771

julia> y * y - x
-9.115745035929407e-64
```

<sup>Note: **MultiFloats.jl** also provides a `Float64x1` type that has the same precision as `Float64`, but behaves like `Float64x2`–`Float64x4` in terms of supported operations. This is occasionally useful for testing, since any code that works for `Float64x1` should also work for `Float64x2`–`Float64x4` and vice versa.</sup>



## Caveats

**MultiFloats.jl** requires an IEEE 754 compliant processor running in round-to-nearest mode. It works out-of-the-box on x86 and ARM, but additional setup may be necessary on more obscure architectures.

Most arithmetic algorithms in **MultiFloats.jl** treat `±Inf` inputs as if they were `NaN`. For example, `Float64x4(Inf) + Float64x4(1.0)` returns `NaN`. This is an intrinsic feature of `MultiFloat` representation, and changing this behavior would incur [a significant performance penalty](https://github.com/dzhang314/MultiFloats.jl/issues/12#issuecomment-751151737).