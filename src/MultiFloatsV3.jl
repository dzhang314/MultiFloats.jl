module MultiFloatsV3


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


end # module MultiFloatsV3
