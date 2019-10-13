# MultiFloats.jl

**MultiFloats.jl** is a Julia package for extended-precision floating-point arithmetic that is several times faster than `BigFloat` for intermediate precision levels (100-400 bits). It achieves this speed using **native `Float64` operations** with **no dynamic memory allocation** by storing extended-precision numbers in [double-double representation](https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format#Double-double_arithmetic), generalized to an arbitrary number of `Float64` components.

**MultiFloats.jl** currently provides basic arithmetic operations (`+`, `-`, `*`, `/`), comparison operators (`==`, `!=`, `<`, `>`, `<=`, `>=`), and floating-point inspection methods (`isfinite`, `isnan`, etc.). Work on transcendental operations is in progress.

|                                                        | MultiFloats | BigFloat | ArbNumerics | Decimals | DecFP | DoubleFloats | QuadMath |
|--------------------------------------------------------|-------------|----------|-------------|----------|-------|--------------|----------|
| user-selectable precision                              | ✔️          | ✔️      | ✔️          | ❌       | ❌    | ❌          | ❌       |
| avoids dynamic memory allocation                       | ✔️          | ❌      | ❌          | ❌       | ✔️    | ⚠️          | ✔️       |
| basic arithmetic (`+`, `-`, `*`, `/`, `sqrt`)          | ✔️          | ✔️      | ✔️          | ❌       | ✔️    | ✔️          | ✔️       |
| transcendental functions (`sin`, `cos`, `exp`, `log`)  | ❌ (WIP)    | ✔️      | ✔️          | ❌       | ✔️    | ✔️          | ✔️       |
| generic linear algebra (e.g. **GenericSVD**)           | ✔️          | ✔️      | ✔️          | ❌       | ❌    | ✔️          | ✔️       |
| floating-point introspection (e.g. `minfloat`, `eps`)  | ✔️          | ✔️      | ✔️          | ❌       | ✔️    | ✔️          | ✔️       |
