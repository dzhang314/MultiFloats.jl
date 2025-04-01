module MultiFloatsV3

using Base.MPFR: libmpfr, CdoubleMax,
    MPFRRoundingMode, MPFRRoundNearest, MPFRRoundFaithful
using SIMD: Vec

################################################################################


export MultiFloat, MultiFloatVec


struct MultiFloat{T,N} <: AbstractFloat
    _limbs::NTuple{N,T}
end


struct MultiFloatVec{M,T,N}
    _limbs::NTuple{N,Vec{M,T}}
end


################################################################################


export Float16x, Float32x, Float64x,
    Float64x1, Float64x2, Float64x3, Float64x4,
    Vec1Float64x1, Vec1Float64x2, Vec1Float64x3, Vec1Float64x4,
    Vec2Float64x1, Vec2Float64x2, Vec2Float64x3, Vec2Float64x4,
    Vec4Float64x1, Vec4Float64x2, Vec4Float64x3, Vec4Float64x4,
    Vec8Float64x1, Vec8Float64x2, Vec8Float64x3, Vec8Float64x4,
    Vec16Float64x1, Vec16Float64x2, Vec16Float64x3, Vec16Float64x4


const Float16x{N} = MultiFloat{Float16,N}
const Float32x{N} = MultiFloat{Float32,N}
const Float64x{N} = MultiFloat{Float64,N}


const Float64x1 = MultiFloat{Float64,1}
const Float64x2 = MultiFloat{Float64,2}
const Float64x3 = MultiFloat{Float64,3}
const Float64x4 = MultiFloat{Float64,4}


const Vec1Float64x1 = MultiFloatVec{1,Float64,1}
const Vec1Float64x2 = MultiFloatVec{1,Float64,2}
const Vec1Float64x3 = MultiFloatVec{1,Float64,3}
const Vec1Float64x4 = MultiFloatVec{1,Float64,4}


const Vec2Float64x1 = MultiFloatVec{2,Float64,1}
const Vec2Float64x2 = MultiFloatVec{2,Float64,2}
const Vec2Float64x3 = MultiFloatVec{2,Float64,3}
const Vec2Float64x4 = MultiFloatVec{2,Float64,4}


const Vec4Float64x1 = MultiFloatVec{4,Float64,1}
const Vec4Float64x2 = MultiFloatVec{4,Float64,2}
const Vec4Float64x3 = MultiFloatVec{4,Float64,3}
const Vec4Float64x4 = MultiFloatVec{4,Float64,4}


const Vec8Float64x1 = MultiFloatVec{8,Float64,1}
const Vec8Float64x2 = MultiFloatVec{8,Float64,2}
const Vec8Float64x3 = MultiFloatVec{8,Float64,3}
const Vec8Float64x4 = MultiFloatVec{8,Float64,4}


const Vec16Float64x1 = MultiFloatVec{16,Float64,1}
const Vec16Float64x2 = MultiFloatVec{16,Float64,2}
const Vec16Float64x3 = MultiFloatVec{16,Float64,3}
const Vec16Float64x4 = MultiFloatVec{16,Float64,4}



################################################################################


@inline Base.zero(::Type{MultiFloat{T,N}}) where {T,N} =
    MultiFloat{T,N}(ntuple(_ -> zero(T), Val{N}()))

@inline Base.zero(::MultiFloat{T,N}) where {T,N} = zero(MultiFloat{T,N})

@inline Base.zero(::Type{MultiFloatVec{M,T,N}}) where {M,T,N} =
    MultiFloatVec{M,T,N}(ntuple(_ -> zero(Vec{M,T}), Val{N}()))

@inline Base.zero(::MultiFloatVec{M,T,N}) where {M,T,N} =
    zero(MultiFloatVec{M,T,N})


@inline Base.one(::Type{MultiFloat{T,N}}) where {T,N} =
    MultiFloat{T,N}(ntuple(i -> (i == 1) ? one(T) : zero(T), Val{N}()))

@inline Base.one(::MultiFloat{T,N}) where {T,N} = one(MultiFloat{T,N})

@inline Base.one(::Type{MultiFloatVec{M,T,N}}) where {M,T,N} =
    MultiFloatVec{M,T,N}(ntuple(
        i -> (i == 1) ? one(Vec{M,T}) : zero(Vec{M,T}), Val{N}()))

@inline Base.one(::MultiFloatVec{M,T,N}) where {M,T,N} =
    one(MultiFloatVec{M,T,N})


@inline Base.:-(x::MultiFloat{T,N}) where {T,N} =
    MultiFloat{T,N}(ntuple(i -> -x._limbs[i], Val{N}()))

@inline Base.:-(x::MultiFloatVec{M,T,N}) where {M,T,N} =
    MultiFloatVec{M,T,N}(ntuple(i -> -x._limbs[i], Val{N}()))


################################################################################


function mpfr_zero!(x::BigFloat)
    ccall((:mpfr_set_zero, libmpfr), Cvoid,
        (Ref{BigFloat}, Cint),
        x, 0)
    return x
end


function mpfr_add!(x::BigFloat, y::CdoubleMax)
    ccall((:mpfr_add_d, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, Cdouble, MPFRRoundingMode),
        x, x, y, MPFRRoundNearest)
    return x
end


function mpfr_sub!(x::BigFloat, y::CdoubleMax)
    ccall((:mpfr_sub_d, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, Cdouble, MPFRRoundingMode),
        x, x, y, MPFRRoundNearest)
    return x
end


################################################################################


export split!


function Base.BigFloat(x::MultiFloat{T,N}) where {T,N}
    p = exponent(floatmax(T)) - exponent(floatmin(T)) + precision(T)
    result = BigFloat(precision=p)
    mpfr_zero!(result)
    for i = 1:N
        mpfr_add!(result, x._limbs[i])
    end
    return result
end


function split!(x::BigFloat, ::Type{T}, ::Val{N}) where {T,N}
    result = ntuple(_ -> zero(T), Val{N}())
    for i = 1:N
        term = T(x)
        result = Base.setindex(result, term, i)
        mpfr_sub!(x, term)
    end
    return result
end


################################################################################


export mfadd_exact, mfmul_exact,
    mfinv_exact, mfdiv_exact, mfrsqrt_exact, mfsqrt_exact


function mfadd_exact(
    ::Val{Z}, x::MultiFloat{T,X}, y::MultiFloat{T,Y},
) where {Z,T,X,Y}
    p = exponent(floatmax(T)) - exponent(floatmin(T)) + precision(T)
    q = 2 * p + 1
    z = BigFloat(precision=q)
    ccall((:mpfr_add, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        z, BigFloat(x), BigFloat(y), MPFRRoundNearest)
    return MultiFloat{T,Z}(split!(z, T, Val{Z}()))
end


function mfmul_exact(
    ::Val{Z}, x::MultiFloat{T,X}, y::MultiFloat{T,Y},
) where {Z,T,X,Y}
    p = exponent(floatmax(T)) - exponent(floatmin(T)) + precision(T)
    q = 2 * p
    z = BigFloat(precision=q)
    ccall((:mpfr_mul, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        z, BigFloat(x), BigFloat(y), MPFRRoundNearest)
    return MultiFloat{T,Z}(split!(z, T, Val{Z}()))
end


function mfinv_exact(::Val{Z}, x::MultiFloat{T,X}) where {Z,T,X}
    p = exponent(floatmax(T)) - exponent(floatmin(T)) + precision(T)
    q = 2 * p
    z = BigFloat(precision=q)
    ccall((:mpfr_ui_div, libmpfr), Cint,
        (Ref{BigFloat}, Culong, Ref{BigFloat}, MPFRRoundingMode),
        z, 1, BigFloat(x), MPFRRoundNearest)
    return MultiFloat{T,Z}(split!(z, T, Val{Z}()))
end


function mfdiv_exact(
    ::Val{Z}, x::MultiFloat{T,X}, y::MultiFloat{T,Y},
) where {Z,T,X,Y}
    p = exponent(floatmax(T)) - exponent(floatmin(T)) + precision(T)
    q = 2 * p
    z = BigFloat(precision=q)
    ccall((:mpfr_div, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        z, BigFloat(x), BigFloat(y), MPFRRoundNearest)
    return MultiFloat{T,Z}(split!(z, T, Val{Z}()))
end


function mfrsqrt_exact(::Val{Z}, x::MultiFloat{T,X}) where {Z,T,X}
    p = exponent(floatmax(T)) - exponent(floatmin(T)) + precision(T)
    # To my knowledge, innocuous double rounding bounds for inverse square
    # roots have never been studied in the floating-point literature. This
    # is a rough estimate that combines the division and square root bounds.
    q = 4 * p + 4
    z = BigFloat(precision=q)
    ccall((:mpfr_rec_sqrt, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        z, BigFloat(x), MPFRRoundNearest)
    return MultiFloat{T,Z}(split!(z, T, Val{Z}()))
end


function mfsqrt_exact(::Val{Z}, x::MultiFloat{T,X}) where {Z,T,X}
    p = exponent(floatmax(T)) - exponent(floatmin(T)) + precision(T)
    q = 2 * p + 2
    z = BigFloat(precision=q)
    ccall((:mpfr_sqrt, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        z, BigFloat(x), MPFRRoundNearest)
    return MultiFloat{T,Z}(split!(z, T, Val{Z}()))
end


################################################################################


function mfadd_rounded(
    ::Val{Z}, x::MultiFloat{T,X}, y::MultiFloat{T,Y},
) where {Z,T,X,Y}
    q = Z * (precision(T) - 1)
    z = BigFloat(precision=q)
    ccall((:mpfr_add, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        z, BigFloat(x), BigFloat(y), MPFRRoundFaithful)
    return MultiFloat{T,Z}(split!(z, T, Val{Z}()))
end


function mfmul_rounded(
    ::Val{Z}, x::MultiFloat{T,X}, y::MultiFloat{T,Y},
) where {Z,T,X,Y}
    q = Z * (precision(T) - 1)
    z = BigFloat(precision=q)
    ccall((:mpfr_mul, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        z, BigFloat(x), BigFloat(y), MPFRRoundFaithful)
    return MultiFloat{T,Z}(split!(z, T, Val{Z}()))
end


################################################################################


export mfadd, mfmul, mfinv, mfdiv, mfrsqrt, mfsqrt


@generated function mfadd(
    ::Val{Z}, x::MultiFloat{T,X}, y::MultiFloat{T,Y},
) where {Z,T,X,Y}
    Core.println(
        Core.stderr,
        """
        WARNING: A fast algorithm for the following MultiFloat operation:
            mfadd(::Val{$Z}, x::MultiFloat{$T, $X}, y::MultiFloat{$T, $Y})
        has not yet been developed. A slow fallback algorithm using MPFR
        (BigFloat) operations will be used instead.
        """
    )
    return :(mfadd_rounded(Val{Z}(), x, y))
end


@inline Base.:+(x::MultiFloat{T,X}, y::MultiFloat{T,Y}) where {T,X,Y} =
    mfadd(Val{max(X, Y)}(), x, y)


@inline Base.:-(x::MultiFloat{T,X}, y::MultiFloat{T,Y}) where {T,X,Y} =
    mfadd(Val{max(X, Y)}(), x, -y)


@generated function mfmul(
    ::Val{Z}, x::MultiFloat{T,X}, y::MultiFloat{T,Y},
) where {Z,T,X,Y}
    Core.println(
        Core.stderr,
        """
        WARNING: A fast algorithm for the following MultiFloat operation:
            mfmul(::Val{$Z}, x::MultiFloat{$T, $X}, y::MultiFloat{$T, $Y})
        has not yet been developed. A slow fallback algorithm using MPFR
        (BigFloat) operations will be used instead.
        """
    )
    return :(mfmul_rounded(Val{Z}(), x, y))
end


@inline Base.:*(x::MultiFloat{T,X}, y::MultiFloat{T,Y}) where {T,X,Y} =
    mfmul(Val{max(X, Y)}(), x, y)


@inline function mfinv_impl(
    ::Val{Z}, x::MultiFloat{T,X}, estimate::MultiFloat{T,E},
) where {Z,T,X,E}
    _one = MultiFloat{T,1}((one(T),))
    residual = mfadd(Val{E}(), _one, -mfmul(Val{E + 1}(), estimate, x))
    correction = mfmul(Val{E}(), estimate, residual)
    if E + E >= Z
        return mfadd(Val{Z}(), estimate, correction)
    else
        next_estimate = mfadd(Val{E + E}(), estimate, correction)
        return mfinv_impl(Val{Z}(), x, next_estimate)
    end
end


@inline function mfinv(::Val{Z}, x::MultiFloat{T,X}) where {Z,T,X}
    estimate = MultiFloat{T,1}((inv(x._limbs[1]),))
    if Z == 1
        return estimate
    else
        return mfinv_impl(Val{Z}(), x, estimate)
    end
end


@inline Base.inv(x::MultiFloat{T,X}) where {T,X} =
    mfinv(Val{X}(), x)


@generated function mfdiv(
    ::Val{Z}, x::MultiFloat{T,X}, y::MultiFloat{T,Y},
) where {Z,T,X,Y}
    Core.println(
        Core.stderr,
        """
        WARNING: A fast algorithm for the following MultiFloat operation:
            mfdiv(::Val{$Z}, x::MultiFloat{$T, $X}, y::MultiFloat{$T, $Y})
        has not yet been developed. A slow fallback algorithm using MPFR
        (BigFloat) operations will be used instead.
        """
    )
    return :(mfdiv_exact(Val{Z}(), x, y))
end


@inline Base.:/(x::MultiFloat{T,X}, y::MultiFloat{T,Y}) where {T,X,Y} =
    mfdiv(Val{max(X, Y)}(), x, y)


@generated function mfrsqrt(::Val{Z}, x::MultiFloat{T,X}) where {Z,T,X}
    Core.println(
        Core.stderr,
        """
        WARNING: A fast algorithm for the following MultiFloat operation:
            mfrsqrt(::Val{$Z}, x::MultiFloat{$T, $X})
        has not yet been developed. A slow fallback algorithm using MPFR
        (BigFloat) operations will be used instead.
        """
    )
    return :(mfrsqrt_exact(Val{Z}(), x))
end


@inline rsqrt(x::MultiFloat{T,X}) where {T,X} = mfrsqrt(Val{X}(), x)


@generated function mfsqrt(::Val{Z}, x::MultiFloat{T,X}) where {Z,T,X}
    Core.println(
        Core.stderr,
        """
        WARNING: A fast algorithm for the following MultiFloat operation:
            mfsqrt(::Val{$Z}, x::MultiFloat{$T, $X})
        has not yet been developed. A slow fallback algorithm using MPFR
        (BigFloat) operations will be used instead.
        """
    )
    return :(mfsqrt_exact(Val{Z}(), x))
end


@inline Base.sqrt(x::MultiFloat{T,X}) where {T,X} = mfsqrt(Val{X}(), x)


################################################################################


@inline function two_sum(x::T, y::T) where {T}
    s = x + y
    x_prime = s - y
    y_prime = s - x_prime
    x_err = x - x_prime
    y_err = y - y_prime
    e = x_err + y_err
    return (s, e)
end


@inline function two_prod(x::T, y::T) where {T}
    p = x * y
    e = fma(x, y, -p)
    return (p, e)
end


@inline mfadd(::Val{1}, x::MultiFloat{T,1}, y::MultiFloat{T,1}) where {T} =
    MultiFloat{T,1}((x._limbs[1] + y._limbs[1],))


@inline mfmul(::Val{1}, x::MultiFloat{T,1}, y::MultiFloat{T,1}) where {T} =
    MultiFloat{T,1}((x._limbs[1] * y._limbs[1],))


@inline mfadd(::Val{2}, x::MultiFloat{T,1}, y::MultiFloat{T,1}) where {T} =
    MultiFloat{T,2}(two_sum(x._limbs[1], y._limbs[1]))


@inline mfmul(::Val{2}, x::MultiFloat{T,1}, y::MultiFloat{T,1}) where {T} =
    MultiFloat{T,2}(two_prod(x._limbs[1], y._limbs[1]))


################################################################################

end # module MultiFloatsV3
