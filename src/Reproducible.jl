module Reproducible

using Base.MPFR: MPFRRoundingMode, MPFRRoundNearest
using MPFR_jll: libmpfr

# Floating-point division and square root are implemented inconsistently across
# CPU and GPU microarchitectures. MultiFloats.jl provides reproducible variants
# of these operations that produce bit-for-bit identical results across all
# platforms. These functions have been extensively tested on current-generation
# Intel, AMD, NVIDIA, and Apple CPUs and GPUs.

export muladd_r, inv_r, div_r, sqrt_r, rsqrt_r

# MultiFloats.muladd_r is a qualified public function.
# Users are expected to call it as MultiFloats.muladd_r(x, y, z)
# or by using MultiFloats.Reproducible.

@inline muladd_r(x::Any, y::Any, z::Any) = fma(x, y, z)

# MultiFloats.inv_r is a qualified public function.
# Users are expected to call it as MultiFloats.inv_r(x)
# or by using MultiFloats.Reproducible.

@inline inv_r(x::Any) = inv(x)

# MultiFloats.div_r is a qualified public function.
# Users are expected to call it as MultiFloats.div_r(x, y)
# or by using MultiFloats.Reproducible.

@inline div_r(x::Any, y::Any) = x / y

# MultiFloats.sqrt_r is a qualified public function.
# Users are expected to call it as MultiFloats.sqrt_r(x)
# or by using MultiFloats.Reproducible.

@inline sqrt_r(x::Any) = sqrt(x)

@inline sqrt_r(x::Union{Float16,Float32,Float64}) = Base.sqrt_llvm(x)

function sqrt_r(x::BigFloat)
    result = BigFloat()
    ccall((:mpfr_sqrt, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        result, x, MPFRRoundNearest)
    return result
end

# MultiFloats.rsqrt_r is a qualified public function.
# Users are expected to call it as MultiFloats.rsqrt_r(x)
# or by using MultiFloats.Reproducible.

@inline rsqrt_r(x::Any) = inv_r(sqrt_r(x))

function rsqrt_r(x::BigFloat)
    result = BigFloat()
    ccall((:mpfr_rec_sqrt, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        result, x, MPFRRoundNearest)
    return result
end

end # module Reproducible
