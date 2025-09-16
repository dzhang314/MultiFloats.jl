module MultiFloatsV3

using Base.MPFR: libmpfr, CdoubleMax,
    MPFRRoundingMode, MPFRRoundNearest, MPFRRoundFaithful
using SIMD: Vec

const ENABLE_RUNTIME_ASSERTIONS = true


################################################################################


export mfadd_exact, mfmul_exact, mfadd_rounded, mfmul_rounded,
    mfinv_exact, mfdiv_exact, mfrsqrt_exact, mfsqrt_exact


function mfadd_exact(
    ::Val{Z},
    x::Union{MultiFloat{T,X},PreciseMultiFloat{T,X}},
    y::Union{MultiFloat{T,Y},PreciseMultiFloat{T,Y}},
) where {Z,T,X,Y}
    p = exponent(floatmax(T)) - exponent(floatmin(T)) + precision(T)
    q = 2 * p + 1
    z = BigFloat(precision=q)
    ccall((:mpfr_add, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        z, BigFloat(x), BigFloat(y), MPFRRoundNearest)
    result = split!(z, T, Val{Z}())
    if (x isa MultiFloat) & (y isa MultiFloat)
        return MultiFloat{T,Z}(result)
    elseif (x isa PreciseMultiFloat) & (y isa PreciseMultiFloat)
        return PreciseMultiFloat{T,Z}(result)
    else
        @assert false
    end
end


function mfmul_exact(
    ::Val{Z},
    x::Union{MultiFloat{T,X},PreciseMultiFloat{T,X}},
    y::Union{MultiFloat{T,Y},PreciseMultiFloat{T,Y}},
) where {Z,T,X,Y}
    p = exponent(floatmax(T)) - exponent(floatmin(T)) + precision(T)
    q = 2 * p
    z = BigFloat(precision=q)
    ccall((:mpfr_mul, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        z, BigFloat(x), BigFloat(y), MPFRRoundNearest)
    result = split!(z, T, Val{Z}())
    if (x isa MultiFloat) & (y isa MultiFloat)
        return MultiFloat{T,Z}(result)
    elseif (x isa PreciseMultiFloat) & (y isa PreciseMultiFloat)
        return PreciseMultiFloat{T,Z}(result)
    else
        @assert false
    end
end


function mfadd_rounded(
    ::Val{Z},
    x::Union{MultiFloat{T,X},PreciseMultiFloat{T,X}},
    y::Union{MultiFloat{T,Y},PreciseMultiFloat{T,Y}},
) where {Z,T,X,Y}
    q = Z * (precision(T) - 1)
    z = BigFloat(precision=q)
    ccall((:mpfr_add, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        z, BigFloat(x), BigFloat(y), MPFRRoundFaithful)
    result = split!(z, T, Val{Z}())
    if (x isa MultiFloat) & (y isa MultiFloat)
        return MultiFloat{T,Z}(result)
    elseif (x isa PreciseMultiFloat) & (y isa PreciseMultiFloat)
        return PreciseMultiFloat{T,Z}(result)
    else
        @assert false
    end
end


function mfmul_rounded(
    ::Val{Z},
    x::Union{MultiFloat{T,X},PreciseMultiFloat{T,X}},
    y::Union{MultiFloat{T,Y},PreciseMultiFloat{T,Y}},
) where {Z,T,X,Y}
    q = Z * (precision(T) - 1)
    z = BigFloat(precision=q)
    ccall((:mpfr_mul, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        z, BigFloat(x), BigFloat(y), MPFRRoundFaithful)
    result = split!(z, T, Val{Z}())
    if (x isa MultiFloat) & (y isa MultiFloat)
        return MultiFloat{T,Z}(result)
    elseif (x isa PreciseMultiFloat) & (y isa PreciseMultiFloat)
        return PreciseMultiFloat{T,Z}(result)
    else
        @assert false
    end
end


function mfinv_exact(
    ::Val{Z},
    x::Union{MultiFloat{T,X},PreciseMultiFloat{T,X}},
) where {Z,T,X}
    p = exponent(floatmax(T)) - exponent(floatmin(T)) + precision(T)
    q = 2 * p
    z = BigFloat(precision=q)
    ccall((:mpfr_ui_div, libmpfr), Cint,
        (Ref{BigFloat}, Culong, Ref{BigFloat}, MPFRRoundingMode),
        z, 1, BigFloat(x), MPFRRoundNearest)
    result = split!(z, T, Val{Z}())
    if x isa MultiFloat
        return MultiFloat{T,Z}(result)
    elseif x isa PreciseMultiFloat
        return PreciseMultiFloat{T,Z}(result)
    else
        @assert false
    end
end


function mfdiv_exact(
    ::Val{Z},
    x::Union{MultiFloat{T,X},PreciseMultiFloat{T,X}},
    y::Union{MultiFloat{T,Y},PreciseMultiFloat{T,Y}},
) where {Z,T,X,Y}
    p = exponent(floatmax(T)) - exponent(floatmin(T)) + precision(T)
    q = 2 * p
    z = BigFloat(precision=q)
    ccall((:mpfr_div, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        z, BigFloat(x), BigFloat(y), MPFRRoundNearest)
    result = split!(z, T, Val{Z}())
    if (x isa MultiFloat) & (y isa MultiFloat)
        return MultiFloat{T,Z}(result)
    elseif (x isa PreciseMultiFloat) & (y isa PreciseMultiFloat)
        return PreciseMultiFloat{T,Z}(result)
    else
        @assert false
    end
end


function mfrsqrt_exact(
    ::Val{Z},
    x::Union{MultiFloat{T,X},PreciseMultiFloat{T,X}},
) where {Z,T,X}
    p = exponent(floatmax(T)) - exponent(floatmin(T)) + precision(T)
    # To my knowledge, innocuous double rounding bounds for inverse square
    # roots have never been studied in the floating-point literature. This
    # is a rough estimate that combines the division and square root bounds.
    q = 4 * p + 4
    z = BigFloat(precision=q)
    ccall((:mpfr_rec_sqrt, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        z, BigFloat(x), MPFRRoundNearest)
    result = split!(z, T, Val{Z}())
    if x isa MultiFloat
        return MultiFloat{T,Z}(result)
    elseif x isa PreciseMultiFloat
        return PreciseMultiFloat{T,Z}(result)
    else
        @assert false
    end
end


function mfsqrt_exact(
    ::Val{Z},
    x::Union{MultiFloat{T,X},PreciseMultiFloat{T,X}},
) where {Z,T,X}
    p = exponent(floatmax(T)) - exponent(floatmin(T)) + precision(T)
    q = 2 * p + 2
    z = BigFloat(precision=q)
    ccall((:mpfr_sqrt, libmpfr), Cint,
        (Ref{BigFloat}, Ref{BigFloat}, MPFRRoundingMode),
        z, BigFloat(x), MPFRRoundNearest)
    result = split!(z, T, Val{Z}())
    if x isa MultiFloat
        return MultiFloat{T,Z}(result)
    elseif x isa PreciseMultiFloat
        return PreciseMultiFloat{T,Z}(result)
    else
        @assert false
    end
end


################################################################################


function Base.:+(
    x::PreciseMultiFloat{T,N}, y::PreciseMultiFloat{T,N},
) where {T,N}
    return mfadd_exact(Val{N}(), x, y)
end


function Base.:-(
    x::PreciseMultiFloat{T,N}, y::PreciseMultiFloat{T,N},
) where {T,N}
    return mfadd_exact(Val{N}(), x, -y)
end


function Base.:*(
    x::PreciseMultiFloat{T,N}, y::PreciseMultiFloat{T,N},
) where {T,N}
    return mfmul_exact(Val{N}(), x, y)
end


function Base.:/(
    x::PreciseMultiFloat{T,N}, y::PreciseMultiFloat{T,N},
) where {T,N}
    return mfdiv_exact(Val{N}(), x, y)
end


function Base.inv(x::PreciseMultiFloat{T,N}) where {T,N}
    return mfinv_exact(Val{N}(), x)
end


function Base.sqrt(x::PreciseMultiFloat{T,N}) where {T,N}
    return mfsqrt_exact(Val{N}(), x)
end


################################################################################


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


################################################################################


@inline function is_normalized(x::NTuple{N,T}) where {N,T}
    result = true
    @inbounds for i = 1:N-1
        a, b = x[i], x[i+1]
        s, e = two_sum(a, b)
        result &= ((s == a) & (e == b)) | (!isfinite(s)) | (!isfinite(e))
    end
    return result
end


@generated function fast_combine(x::Vararg{T,N}) where {T,N}
    xs = [Symbol('x', i) for i in Base.OneTo(N)]
    body = Expr[]
    push!(body, Expr(:meta, :inline))
    push!(body, Expr(:(=), Expr(:tuple, xs...), :x))
    for _ = 1:N-1
        for i = N-1:-1:1
            push!(body, Expr(:(=), Expr(:tuple, xs[i], xs[i+1]),
                Expr(:call, :unsafe_fast_two_sum, xs[i], xs[i+1])))
        end
    end
    push!(body, Expr(:return, Expr(:tuple, xs...)))
    return Expr(:block, body...)
end


################################################################################


export mfinv, mfdiv, mfrsqrt, mfsqrt


@inline function mfinv_impl(
    ::Val{Z}, x::MultiFloat{T,X}, estimate::MultiFloat{T,E},
) where {Z,T,X,E}
    @assert Z > E > 0
    neg_one = MultiFloat{T,1}((-one(T),))
    if E + E >= Z
        residual = mfadd(Val{E}(), neg_one, mfmul(Val{E + E}(), estimate, x))
        correction = mfmul(Val{Z - E}(), estimate, residual)
        result = MultiFloat{T,Z}(fast_combine(
            estimate._limbs..., (-correction)._limbs...))
        @static if ENABLE_RUNTIME_ASSERTIONS
            @assert is_normalized(result._limbs)
        end
        return result
    else
        residual = mfadd(Val{E}(), neg_one, mfmul(Val{E + E}(), estimate, x))
        correction = mfmul(Val{E}(), estimate, residual)
        next_estimate = MultiFloat{T,E + E}(fast_combine(
            estimate._limbs..., (-correction)._limbs...))
        @static if ENABLE_RUNTIME_ASSERTIONS
            @assert is_normalized(next_estimate._limbs)
        end
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


@inline function mfdiv_impl(
    ::Val{Z}, x::MultiFloat{T,X}, y::MultiFloat{T,Y}, estimate::MultiFloat{T,E},
) where {Z,T,X,Y,E}
    @assert Z > E > 0
    neg_one = MultiFloat{T,1}((-one(T),))
    if E + E >= Z
        quotient = mfmul(Val{E}(), x, estimate)
        residual = mfadd(Val{E}(), -x, mfmul(Val{E + E}(), y, quotient))
        correction = mfmul(Val{Z - E}(), estimate, residual)
        result = MultiFloat{T,Z}(fast_combine(
            quotient._limbs..., (-correction)._limbs...))
        @static if ENABLE_RUNTIME_ASSERTIONS
            @assert is_normalized(result._limbs)
        end
        return result
    else
        residual = mfadd(Val{E}(), neg_one, mfmul(Val{E + E}(), y, estimate))
        correction = mfmul(Val{E}(), estimate, residual)
        next_estimate = MultiFloat{T,E + E}(fast_combine(
            estimate._limbs..., (-correction)._limbs...))
        @static if ENABLE_RUNTIME_ASSERTIONS
            @assert is_normalized(next_estimate._limbs)
        end
        return mfdiv_impl(Val{Z}(), x, y, next_estimate)
    end
end


@inline function mfdiv(
    ::Val{Z}, x::MultiFloat{T,X}, y::MultiFloat{T,Y},
) where {Z,T,X,Y}
    if Z == 1
        return MultiFloat{T,1}((x._limbs[1] / y._limbs[1],))
    else
        estimate = MultiFloat{T,1}((inv(y._limbs[1]),))
        return mfdiv_impl(Val{Z}(), x, y, estimate)
    end
end


@inline Base.:/(x::MultiFloat{T,X}, y::MultiFloat{T,Y}) where {T,X,Y} =
    mfdiv(Val{max(X, Y)}(), x, y)


@inline function mfrsqrt_impl(
    ::Val{Z}, x::MultiFloat{T,X}, estimate::MultiFloat{T,E},
) where {Z,T,X,E}
    @assert Z > E > 0
    neg_one = MultiFloat{T,1}((-one(T),))
    if E + E >= Z
        square = mfmul(Val{E + E}(), estimate, estimate)
        residual = mfadd(Val{E}(), neg_one, mfmul(Val{E + E}(), x, square))
        correction = mfmul(Val{Z - E}(), halve(estimate), residual)
        result = MultiFloat{T,Z}(fast_combine(
            estimate._limbs..., (-correction)._limbs...))
        @static if ENABLE_RUNTIME_ASSERTIONS
            @assert is_normalized(result._limbs)
        end
        return result
    else
        square = mfmul(Val{E + E}(), estimate, estimate)
        residual = mfadd(Val{E}(), neg_one, mfmul(Val{E + E}(), x, square))
        correction = mfmul(Val{E}(), halve(estimate), residual)
        next_estimate = MultiFloat{T,E + E}(fast_combine(
            estimate._limbs..., (-correction)._limbs...))
        @static if ENABLE_RUNTIME_ASSERTIONS
            @assert is_normalized(next_estimate._limbs)
        end
        return mfrsqrt_impl(Val{Z}(), x, next_estimate)
    end
end


@inline function mfrsqrt(::Val{Z}, x::MultiFloat{T,X}) where {Z,T,X}
    estimate = MultiFloat{T,1}((inv(sqrt(x._limbs[1])),))
    if Z == 1
        return estimate
    else
        return mfrsqrt_impl(Val{Z}(), x, estimate)
    end
end


@inline rsqrt(x::MultiFloat{T,X}) where {T,X} = mfrsqrt(Val{X}(), x)


@inline function mfsqrt_impl(
    ::Val{Z}, x::MultiFloat{T,X}, estimate::MultiFloat{T,E},
) where {Z,T,X,E}
    @assert Z > E > 0
    neg_one = MultiFloat{T,1}((-one(T),))
    if E + E >= Z
        root = mfmul(Val{E}(), x, estimate)
        square = mfmul(Val{E + E}(), root, root)
        residual = mfadd(Val{E}(), -x, square)
        correction = mfmul(Val{Z - E}(), halve(estimate), residual)
        result = MultiFloat{T,Z}(fast_combine(
            root._limbs..., (-correction)._limbs...))
        @static if ENABLE_RUNTIME_ASSERTIONS
            @assert is_normalized(result._limbs)
        end
        return result
    else
        square = mfmul(Val{E + E}(), estimate, estimate)
        residual = mfadd(Val{E}(), neg_one, mfmul(Val{E + E}(), x, square))
        correction = mfmul(Val{E}(), halve(estimate), residual)
        next_estimate = MultiFloat{T,E + E}(fast_combine(
            estimate._limbs..., (-correction)._limbs...))
        @static if ENABLE_RUNTIME_ASSERTIONS
            @assert is_normalized(next_estimate._limbs)
        end
        return mfsqrt_impl(Val{Z}(), x, next_estimate)
    end
end


@inline function mfsqrt(::Val{Z}, x::MultiFloat{T,X}) where {Z,T,X}
    if Z == 1
        return MultiFloat{T,1}((sqrt(x._limbs[1]),))
    else
        estimate = MultiFloat{T,1}((inv(sqrt(x._limbs[1])),))
        return mfsqrt_impl(Val{Z}(), x, estimate)
    end
end


@inline Base.sqrt(x::MultiFloat{T,X}) where {T,X} = mfsqrt(Val{X}(), x)


end # module MultiFloatsV3
