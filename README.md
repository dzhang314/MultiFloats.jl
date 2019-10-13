# MultiFloats.jl

**MultiFloats.jl** is a Julia package for extended-precision floating-point arithmetic that is several times faster than `BigFloat` for intermediate precision levels (100-400 bits). It achieves this speed using **native `Float64` operations** with **no dynamic memory allocation** by storing extended-precision numbers in [double-double representation](https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format#Double-double_arithmetic), generalized to an arbitrary number of `Float64` components.

**MultiFloats.jl** currently provides basic arithmetic operations (`+`, `-`, `*`, `/`), comparison operators (`==`, `!=`, `<`, `>`, `<=`, `>=`), and floating-point inspection methods (`isfinite`, `isnan`, etc.). Work on transcendental operations is in progress.


|                 | MultiFloats `Float64x2` | Base `BigFloat`        | ArbNumerics `ArbFloat`  | Decimals `Decimal` | DecFP `Dec128`        | DoubleFloats `Double64` | Quadmath `Float128`   |
|-----------------|---------------------------|--------------------------|---------------------------|----------------------|-------------------------|---------------------------|-------------------------|
| 400×400 `qr`&nbsp;time  | **0.257 sec**                 | 10.303 sec **(40x&nbsp;slower)** | 17.871 sec **(69x&nbsp;slower)**  | ❌ Error              | 9.448 sec **(36x&nbsp;slower)** | 0.535 sec **(2x&nbsp;slower)**    | 2.403 sec **(9x&nbsp;slower)**  |
| accurate digits | 26.0                      | 25.9                     | 25.9                      | ❌ Error              | 27.6                    | 26.1                      | 28.1                    |
| 400×250 `pinv`&nbsp;time  | **1.709 sec**                 | 96.655 sec **(56x&nbsp;slower)** | 133.085 sec **(77x&nbsp;slower)** | ❌ Error              | ❌ Error                 | 3.668 sec **(2x&nbsp;slower)**    | 15.576 sec **(9x&nbsp;slower)** |
| accurate digits | 25.6                      | 25.8                     | 25.8                      | ❌ Error              | ❌ Error                 | 25.4                      | 27.9                    |


|                                                        | MultiFloats | BigFloat | ArbNumerics | Decimals | DecFP | DoubleFloats | Quadmath |
|--------------------------------------------------------|-------------|----------|-------------|----------|-------|--------------|----------|
| user-selectable precision                              | ✔️          | ✔️      | ✔️          | ❌       | ❌    | ❌          | ❌       |
| avoids dynamic memory allocation                       | ✔️          | ❌      | ❌          | ❌       | ✔️    | ⚠️          | ✔️       |
| basic arithmetic `+`, `-`, `*`, `/`, `sqrt`            | ✔️          | ✔️      | ✔️          | ❌       | ✔️    | ✔️          | ✔️       |
| transcendental functions `sin`, `cos`, `exp`, `log`    | ❌ (WIP)    | ✔️      | ✔️          | ❌       | ✔️    | ✔️          | ✔️       |
| compatible with **GenericSVD**                         | ✔️          | ✔️      | ✔️          | ❌       | ❌    | ✔️          | ✔️       |
| floating-point introspection `minfloat`, `eps`         | ✔️          | ✔️      | ✔️          | ❌       | ✔️    | ✔️          | ✔️       |

