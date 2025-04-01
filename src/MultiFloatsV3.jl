module MultiFloatsV3

using Base.MPFR: libmpfr, MPFRRoundingMode, MPFRRoundNearest, CdoubleMax
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
        z, x, y, MPFRRoundNearest)
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
        z, x, y, MPFRRoundNearest)
    return MultiFloat{T,Z}(split!(z, T, Val{Z}()))
end


function mfinv_exact(::Val{Z}, x::MultiFloat{T,X}) where {Z,T,X}
    p = exponent(floatmax(T)) - exponent(floatmin(T)) + precision(T)
    q = 2 * p
    z = BigFloat(precision=q)
    ccall((:mpfr_ui_div, libmpfr), Cint,
        (Ref{BigFloat}, Culong, Ref{BigFloat}, MPFRRoundingMode),
        z, 1, x, MPFRRoundNearest)
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
        z, x, y, MPFRRoundNearest)
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
        z, x, MPFRRoundNearest)
    return MultiFloat{T,Z}(split!(z, T, Val{Z}()))
end


function mfsqrt_exact(::Val{Z}, x::MultiFloat{T,X}) where {Z,T,X}
    p = exponent(floatmax(T)) - exponent(floatmin(T)) + precision(T)
    q = 2 * p + 2
    z = BigFloat(precision=q)
    ccall((:mpfr_sqrt, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        z, x, MPFRRoundNearest)
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
    return :(mfadd_exact(Val{Z}(), x, y))
end


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
    return :(mfmul_exact(Val{Z}(), x, y))
end


@generated function mfinv(::Val{Z}, x::MultiFloat{T,X}) where {Z,T,X}
    Core.println(
        Core.stderr,
        """
        WARNING: A fast algorithm for the following MultiFloat operation:
            mfinv(::Val{$Z}, x::MultiFloat{$T, $X})
        has not yet been developed. A slow fallback algorithm using MPFR
        (BigFloat) operations will be used instead.
        """
    )
    return :(mfinv_exact(Val{Z}(), x))
end


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


end # module MultiFloatsV3
