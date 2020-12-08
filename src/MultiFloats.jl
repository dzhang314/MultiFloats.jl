module MultiFloats

export MultiFloat, renormalize,
       Float64x1, Float64x2, Float64x3, Float64x4,
       Float64x5, Float64x6, Float64x7, Float64x8,
       use_clean_multifloat_arithmetic,
       use_sloppy_multifloat_arithmetic,
       use_very_sloppy_multifloat_arithmetic

include("./MultiFloatsCodeGen.jl")
using .MultiFloatsCodeGen

####################################################### DEFINITION OF MULTIFLOAT

struct MultiFloat{T<:AbstractFloat,N} <: AbstractFloat
    _limbs::NTuple{N,T}
end

const Float16x{N} = MultiFloat{Float16,N}
const Float32x{N} = MultiFloat{Float32,N}
const Float64x{N} = MultiFloat{Float64,N}

const AF = Core.AbstractFloat
const MF = MultiFloat

const Float64x1 = Float64x{1}
const Float64x2 = Float64x{2}
const Float64x3 = Float64x{3}
const Float64x4 = Float64x{4}
const Float64x5 = Float64x{5}
const Float64x6 = Float64x{6}
const Float64x7 = Float64x{7}
const Float64x8 = Float64x{8}

################################################ CONVERSION FROM PRIMITIVE TYPES

@inline MultiFloat{T,N}(x::MultiFloat{T,N}) where {T<:AF,N} = x

@inline MultiFloat{T,N}(x::T) where {T<:AF,N} =
    MultiFloat{T,N}((x, ntuple(_ -> zero(T), N - 1)...))

@inline MultiFloat{T,N}(x::MultiFloat{T,M}) where {T<:AbstractFloat,M,N} =
    MultiFloat{T,N}((
        ntuple(i -> x._limbs[i], min(M, N))...,
        ntuple(_ -> zero(T), max(N - M, 0))...))

# Values of the types Bool, Int8, UInt8, Int16, UInt16, Float16, Int32, UInt32,
# and Float32 can be converted losslessly to a single Float64, which has 53
# bits of integer precision.

@inline Float64x{N}(x::Bool   ) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::Int8   ) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::UInt8  ) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::Int16  ) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::UInt16 ) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::Float16) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::Int32  ) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::UInt32 ) where {N} = Float64x{N}(Float64(x))
@inline Float64x{N}(x::Float32) where {N} = Float64x{N}(Float64(x))

# Values of the types Int64, UInt64, Int128, and UInt128 cannot be converted
# losslessly to a single Float64 and must be split into multiple components.

@inline Float64x1(x::Int64  ) = Float64x1(Float64(x))
@inline Float64x1(x::UInt64 ) = Float64x1(Float64(x))
@inline Float64x1(x::Int128 ) = Float64x1(Float64(x))
@inline Float64x1(x::UInt128) = Float64x1(Float64(x))

@inline function Float64x2(x::Int128)
    x0 = Float64(x)
    x1 = Float64(x - Int128(x0))
    Float64x2((x0, x1))
end

@inline function Float64x2(x::UInt128)
    x0 = Float64(x)
    x1 = Float64(reinterpret(Int128, x - UInt128(x0)))
    Float64x2((x0, x1))
end

@inline function Float64x{N}(x::Int64) where {N}
    x0 = Float64(x)
    x1 = Float64(x - Int64(x0))
    Float64x{N}((x0, x1, ntuple(_ -> 0.0, N - 2)...))
end

@inline function Float64x{N}(x::UInt64) where {N}
    x0 = Float64(x)
    x1 = Float64(reinterpret(Int64, x - UInt64(x0)))
    Float64x{N}((x0, x1, ntuple(_ -> 0.0, N - 2)...))
end

@inline function Float64x{N}(x::Int128) where {N}
    x0 = Float64(x)
    r1 = x - Int128(x0)
    x1 = Float64(r1)
    x2 = Float64(r1 - Int128(x1))
    Float64x{N}((x0, x1, x2, ntuple(_ -> 0.0, N - 3)...))
end

@inline function Float64x{N}(x::UInt128) where {N}
    x0 = Float64(x)
    r1 = reinterpret(Int128, x - UInt128(x0))
    x1 = Float64(r1)
    x2 = Float64(r1 - Int128(x1))
    Float64x{N}((x0, x1, x2, ntuple(_ -> 0.0, N - 3)...))
end

################################################## CONVERSION TO PRIMITIVE TYPES

@inline Base.Float16(x::Float64x{N}) where {N} = Float16(x._limbs[1])
@inline Base.Float32(x::Float64x{N}) where {N} = Float32(x._limbs[1])

@inline Base.Float16(x::Float16x{N}) where {N} = x._limbs[1]
@inline Base.Float32(x::Float32x{N}) where {N} = x._limbs[1]
@inline Base.Float64(x::Float64x{N}) where {N} = x._limbs[1]

###################################################### CONVERSION FROM BIG TYPES

function MultiFloat{T,N}(x::BigFloat) where {T<:AF,N}
    setprecision(Int(precision(x))) do
        r = Vector{BigFloat}(undef, N)
        y = Vector{T}(undef, N)
        r[1] = x
        y[1] = T(r[1])
        for i = 2 : N
            r[i] = r[i-1] - y[i-1]
            y[i] = T(r[i])
        end
        MultiFloat{T,N}((y...,))
    end
end

function MultiFloat{T,N}(x::BigInt) where {T<:AF,N}
    y = Vector{T}(undef, N)
    for i = 1 : N
        y[i] = T(x)
        x -= BigInt(y[i])
    end
    MultiFloat{T,N}((y...,))
end

MultiFloat{T,N}(x::Rational{U}) where {T<:AF,N,U} =
    MF{T,N}(numerator(x)) / MF{T,N}(denominator(x))

######################################################## CONVERSION TO BIG TYPES

Base.BigFloat(x::MultiFloat{T,N}) where {T<:AF,N} =
    +(ntuple(i -> BigFloat(x._limbs[N-i+1]), N)...)

################################################################ PROMOTION RULES

Base.promote_rule(::Type{MF{T,N}}, ::Type{T      }) where {T<:AF,N} = MF{T,N}
Base.promote_rule(::Type{MF{T,N}}, ::Type{Int8   }) where {T<:AF,N} = MF{T,N}
Base.promote_rule(::Type{MF{T,N}}, ::Type{Int16  }) where {T<:AF,N} = MF{T,N}
Base.promote_rule(::Type{MF{T,N}}, ::Type{Int32  }) where {T<:AF,N} = MF{T,N}
Base.promote_rule(::Type{MF{T,N}}, ::Type{Int64  }) where {T<:AF,N} = MF{T,N}
Base.promote_rule(::Type{MF{T,N}}, ::Type{Int128 }) where {T<:AF,N} = MF{T,N}
Base.promote_rule(::Type{MF{T,N}}, ::Type{Bool   }) where {T<:AF,N} = MF{T,N}
Base.promote_rule(::Type{MF{T,N}}, ::Type{UInt8  }) where {T<:AF,N} = MF{T,N}
Base.promote_rule(::Type{MF{T,N}}, ::Type{UInt16 }) where {T<:AF,N} = MF{T,N}
Base.promote_rule(::Type{MF{T,N}}, ::Type{UInt32 }) where {T<:AF,N} = MF{T,N}
Base.promote_rule(::Type{MF{T,N}}, ::Type{UInt64 }) where {T<:AF,N} = MF{T,N}
Base.promote_rule(::Type{MF{T,N}}, ::Type{UInt128}) where {T<:AF,N} = MF{T,N}

Base.promote_rule(::Type{MF{T,N}}, ::Type{BigFloat}) where {T<:AF,N} = BigFloat

Base.promote_rule(::Type{Float32x{N}}, ::Type{Float16}) where {N} = Float32x{N}
Base.promote_rule(::Type{Float64x{N}}, ::Type{Float16}) where {N} = Float64x{N}
Base.promote_rule(::Type{Float64x{N}}, ::Type{Float32}) where {N} = Float64x{N}

####################################################################### PRINTING

@inline function renormalize(x::MF{T,N}) where {T<:AF,N}
    total = +(x._limbs...)
    if isfinite(total)
        while true
            x0::MF{T,N} = x + zero(T)
            if !(x0._limbs != x._limbs); break; end
            x = x0
        end
        x
    else
        MultiFloat{T,N}(ntuple(_ -> total, N))
    end
end

@inline renormalize(x::T) where {T<:Number} = x

function call_normalized(callback, x::MultiFloat{T,N}) where {T<:AF,N}
    x = renormalize(x)
    if !isfinite(x._limbs[1])
        callback(x._limbs[1])
    else
        i = N
        while (i > 0) && iszero(x._limbs[i])
            i -= 1
        end
        if iszero(i)
            callback(zero(T))
        else
            setprecision(() -> callback(BigFloat(x)),
                precision(T) + exponent(x._limbs[1]) - exponent(x._limbs[i]))
        end
    end
end

function Base.show(io::IO, x::MultiFloat{T,N}) where {T<:AF,N}
    call_normalized(y -> show(io, y), x)
end

################################################################################

# Thanks to Greg Plowman (https://github.com/GregPlowman) for suggesting
# implementations of Printf.fix_dec and Printf.ini_dec for @printf support.

import Printf: fix_dec, ini_dec

if VERSION < v"1.1"

    fix_dec(out, x::MultiFloat{T,N}, flags::String, width::Int, precision::Int, c::Char) where {T<:AF,N} =
        call_normalized(d -> fix_dec(out, BigFloat(d), flags, width, precision, c), x)

    ini_dec(out, x::MultiFloat{T,N}, ndigits::Int, flags::String, width::Int, precision::Int, c::Char) where {T<:AF,N} =
        call_normalized(d -> ini_dec(out, BigFloat(d), ndigits, flags, width, precision, c), x)

else

    fix_dec(out, x::MultiFloat{T,N}, flags::String, width::Int, precision::Int, c::Char, digits) where {T<:AF,N} =
        call_normalized(d -> fix_dec(out, BigFloat(d), flags, width, precision, c, digits), x)

    ini_dec(out, x::MultiFloat{T,N}, ndigits::Int, flags::String, width::Int, precision::Int, c::Char, digits) where {T<:AF,N} =
        call_normalized(d -> ini_dec(out, BigFloat(d), ndigits, flags, width, precision, c, digits), x)

end

################################################################################

@inline Base.:(==)(x::MF{T,N}, y::MF{T,N}) where {T<:AF,N} = _eq(renormalize(x), renormalize(y))
@inline Base.:(!=)(x::MF{T,N}, y::MF{T,N}) where {T<:AF,N} = _ne(renormalize(x), renormalize(y))
@inline Base.:(< )(x::MF{T,N}, y::MF{T,N}) where {T<:AF,N} = _lt(renormalize(x), renormalize(y))
@inline Base.:(> )(x::MF{T,N}, y::MF{T,N}) where {T<:AF,N} = _gt(renormalize(x), renormalize(y))
@inline Base.:(<=)(x::MF{T,N}, y::MF{T,N}) where {T<:AF,N} = _le(renormalize(x), renormalize(y))
@inline Base.:(>=)(x::MF{T,N}, y::MF{T,N}) where {T<:AF,N} = _ge(renormalize(x), renormalize(y))

@inline scale(a::T, x::MultiFloat{T,N}) where {T<:AF,N} =
    MultiFloat{T,N}(ntuple(i -> a * x._limbs[i], N))

@inline function Base.ldexp(x::MF{T,N}, n::U) where {T<:AF,N,U<:Integer}
    x = renormalize(x)
    MultiFloat{T,N}(ntuple(i -> ldexp(x._limbs[i], n), N))
end

################################################################################

@inline Base.zero(::Type{MF{T,N}}) where {T<:AF,N} = MF{T,N}(zero(T)  )
@inline Base.one( ::Type{MF{T,N}}) where {T<:AF,N} = MF{T,N}(one( T)  )
@inline Base.eps( ::Type{MF{T,N}}) where {T<:AF,N} = MF{T,N}(eps( T)^N)

@inline _iszero(x::MF{T,N}) where {T<:AF,N} =
    (&)(ntuple(i -> iszero(x._limbs[i]), N)...)
@inline _isone( x::MF{T,N}) where {T<:AF,N} =
    isone(x._limbs[1]) & (&)(ntuple(i -> iszero(x._limbs[i + 1]), N - 1)...)

@inline Base.iszero(x::MF{T,1}) where {T<:AF  } =  iszero(x._limbs[1])
@inline Base.isone( x::MF{T,1}) where {T<:AF  } =  isone( x._limbs[1])
@inline Base.iszero(x::MF{T,N}) where {T<:AF,N} = _iszero(renormalize(x))
@inline Base.isone( x::MF{T,N}) where {T<:AF,N} = _isone( renormalize(x))

################################################################################

@inline Base.precision(::Type{MF{T,N}}) where {T<:AF,N} = N * precision(T)

@inline Base.floatmin(::Type{MF{T,N}}) where {T<:AF,N} = MF{T,N}(floatmin(T))
@inline Base.floatmax(::Type{MF{T,N}}) where {T<:AF,N} = MF{T,N}(floatmax(T))

@inline Base.typemin(::Type{MF{T,N}}) where {T<:AF,N} = MF{T,N}(ntuple(_ -> typemin(T), N))
@inline Base.typemax(::Type{MF{T,N}}) where {T<:AF,N} = MF{T,N}(ntuple(_ -> typemax(T), N))

@inline Base.exponent(   x::MF{T,N}) where {T<:AF,N} = exponent(   renormalize(x)._limbs[1])
@inline Base.signbit(    x::MF{T,N}) where {T<:AF,N} = signbit(    renormalize(x)._limbs[1])
@inline Base.issubnormal(x::MF{T,N}) where {T<:AF,N} = issubnormal(renormalize(x)._limbs[1])
@inline Base.isfinite(   x::MF{T,N}) where {T<:AF,N} = isfinite(   renormalize(x)._limbs[1])
@inline Base.isinf(      x::MF{T,N}) where {T<:AF,N} = isinf(      renormalize(x)._limbs[1])
@inline Base.isnan(      x::MF{T,N}) where {T<:AF,N} = isnan(      renormalize(x)._limbs[1])

@inline function Base.nextfloat(x::MF{T,N}) where {T<:AF,N}
    y = renormalize(x)
    MF{T,N}((ntuple(i -> y._limbs[i], N - 1)..., nextfloat(y._limbs[N])))
end

@inline function Base.prevfloat(x::MF{T,N}) where {T<:AF,N}
    y = renormalize(x)
    MF{T,N}((ntuple(i -> y._limbs[i], N - 1)..., prevfloat(y._limbs[N])))
end

import LinearAlgebra: floatmin2
@inline floatmin2(::Type{MF{T,N}}) where {T<:AF,N} =
    MF{T,N}(ldexp(one(T), div(exponent(floatmin(T)) - N * exponent(eps(T)), 2)))

################################################################################

@inline Base.inv(x::MF{T,N}) where {T<:AF,N} = one(MF{T,N}) / x

@inline function Base.abs(x::MF{T,N}) where {T<:AF,N}
    x = renormalize(x)
    ifelse(signbit(x._limbs[1]), -x, x)
end

@inline function Base.abs2(x::MF{T,N}) where {T<:AF,N}
    x = renormalize(x)
    renormalize(x * x)
end

################################################################################

# To-do list of transcendental math functions to be implemented

Base.exp(  x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.exp(Base.BigFloat(x)))
Base.expm1(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.expm1(Base.BigFloat(x)))
Base.log(  x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.log(Base.BigFloat(x)))
Base.log2( x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.log2(Base.BigFloat(x)))
Base.log10(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.log10(Base.BigFloat(x)))
Base.log1p(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.log1p(Base.BigFloat(x)))

Base.sin(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.sin(Base.BigFloat(x)))
Base.cos(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.cos(Base.BigFloat(x)))
Base.tan(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.tan(Base.BigFloat(x)))
Base.sec(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.sec(Base.BigFloat(x)))
Base.csc(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.csc(Base.BigFloat(x)))
Base.cot(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.cot(Base.BigFloat(x)))

Base.sinh(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.sinh(Base.BigFloat(x)))
Base.cosh(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.cosh(Base.BigFloat(x)))
Base.tanh(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.tanh(Base.BigFloat(x)))
Base.sech(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.sech(Base.BigFloat(x)))
Base.csch(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.csch(Base.BigFloat(x)))
Base.coth(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.coth(Base.BigFloat(x)))

Base.sind(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.sind(Base.BigFloat(x)))
Base.cosd(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.cosd(Base.BigFloat(x)))
Base.tand(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.tand(Base.BigFloat(x)))
Base.secd(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.secd(Base.BigFloat(x)))
Base.cscd(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.cscd(Base.BigFloat(x)))
Base.cotd(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.cotd(Base.BigFloat(x)))

Base.asin(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.asin(Base.BigFloat(x)))
Base.acos(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.acos(Base.BigFloat(x)))
Base.atan(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.atan(Base.BigFloat(x)))
Base.asec(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.asec(Base.BigFloat(x)))
Base.acsc(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.acsc(Base.BigFloat(x)))
Base.acot(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.acot(Base.BigFloat(x)))

Base.asinh(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.asinh(Base.BigFloat(x)))
Base.acosh(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.acosh(Base.BigFloat(x)))
Base.atanh(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.atanh(Base.BigFloat(x)))
Base.asech(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.asech(Base.BigFloat(x)))
Base.acsch(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.acsch(Base.BigFloat(x)))
Base.acoth(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.acoth(Base.BigFloat(x)))

Base.asind(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.asind(Base.BigFloat(x)))
Base.acosd(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.acosd(Base.BigFloat(x)))
Base.atand(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.atand(Base.BigFloat(x)))
Base.asecd(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.asecd(Base.BigFloat(x)))
Base.acscd(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.acscd(Base.BigFloat(x)))
Base.acotd(x::MF{T,N}) where {T<:AF,N} = MultiFloat{T,N}(Base.acotd(Base.BigFloat(x)))

################################################################################

@inline function two_sum(a::T, b::T) where {T<:AF}
    s = a + b
    v = s - a
    s, (a - (s - v)) + (b - v)
end

@inline function quick_two_sum(a::T, b::T) where {T<:AF}
    s = a + b
    s, b - (s - a)
end

@inline function two_prod(a::T, b::T) where {T<:AF}
    p = a * b
    p, fma(a, b, -p)
end

################################################################################

@inline unsafe_sqrt(x::Float32) = Base.sqrt_llvm(x)
@inline unsafe_sqrt(x::Float64) = Base.sqrt_llvm(x)
@inline unsafe_sqrt(x::T) where {T <: Real} = sqrt(x)

@inline Base.:(==)(x::MF{T,1}, y::MF{T,1}) where {T<:AF} = (x._limbs[1] == y._limbs[1])
@inline Base.:(!=)(x::MF{T,1}, y::MF{T,1}) where {T<:AF} = (x._limbs[1] != y._limbs[1])
@inline Base.:(< )(x::MF{T,1}, y::MF{T,1}) where {T<:AF} = (x._limbs[1] <  y._limbs[1])
@inline Base.:(> )(x::MF{T,1}, y::MF{T,1}) where {T<:AF} = (x._limbs[1] >  y._limbs[1])
@inline Base.:(<=)(x::MF{T,1}, y::MF{T,1}) where {T<:AF} = (x._limbs[1] <= y._limbs[1])
@inline Base.:(>=)(x::MF{T,1}, y::MF{T,1}) where {T<:AF} = (x._limbs[1] >= y._limbs[1])

@inline Base.:+(a::MF{T,1}, b::MF{T,1}) where {T<:AF} = MF{T,1}(a._limbs[1] + b._limbs[1])
@inline Base.:+(a::MF{T,1}, b::T      ) where {T<:AF} = MF{T,1}(a._limbs[1] + b          )
@inline Base.:*(a::MF{T,1}, b::MF{T,1}) where {T<:AF} = MF{T,1}(a._limbs[1] * b._limbs[1])
@inline Base.:*(a::MF{T,1}, b::T      ) where {T<:AF} = MF{T,1}(a._limbs[1] * b          )
@inline Base.:/(a::MF{T,1}, b::MF{T,1}) where {T<:AF} = MF{T,1}(a._limbs[1] / b._limbs[1])
@inline _sqrt(  x::MF{T,1}            ) where {T<:AF} = MF{T,1}(unsafe_sqrt(x._limbs[1]) )

@inline function Base.sqrt(x::MF{T,N}) where {T<:AF,N}
    x = renormalize(x)
    if iszero(x)
        return x
    else
        return _sqrt(x)
    end
end

function use_clean_multifloat_arithmetic(n::Integer=8)
    for i = 2 : n
        eval(multifloat_eq_func(       i              ))
        eval(multifloat_ne_func(       i              ))
        eval(multifloat_lt_func(       i              ))
        eval(multifloat_gt_func(       i              ))
        eval(multifloat_le_func(       i              ))
        eval(multifloat_ge_func(       i              ))
        eval(two_pass_renorm_func(     i, sloppy=false))
        eval(multifloat_add_func(      i, sloppy=false))
        eval(multifloat_float_add_func(i, sloppy=false))
        eval(multifloat_mul_func(      i, sloppy=false))
        eval(multifloat_float_mul_func(i, sloppy=false))
        eval(multifloat_div_func(      i, sloppy=false))
        eval(multifloat_sqrt_func(     i, sloppy=false))
    end
    for (_, v) in MultiFloatsCodeGen.MPADD_CACHE
        eval(v)
    end
end

function use_sloppy_multifloat_arithmetic(n::Integer=8)
    for i = 2 : n
        eval(multifloat_eq_func(       i             ))
        eval(multifloat_ne_func(       i             ))
        eval(multifloat_lt_func(       i             ))
        eval(multifloat_gt_func(       i             ))
        eval(multifloat_le_func(       i             ))
        eval(multifloat_ge_func(       i             ))
        eval(two_pass_renorm_func(     i, sloppy=true))
        eval(multifloat_add_func(      i, sloppy=true))
        eval(multifloat_float_add_func(i, sloppy=true))
        eval(multifloat_mul_func(      i, sloppy=true))
        eval(multifloat_float_mul_func(i, sloppy=true))
        eval(multifloat_div_func(      i, sloppy=true))
        eval(multifloat_sqrt_func(     i, sloppy=true))
    end
    for (_, v) in MultiFloatsCodeGen.MPADD_CACHE
        eval(v)
    end
end

function use_very_sloppy_multifloat_arithmetic(n::Integer=8)
    for i = 2 : n
        eval(multifloat_eq_func(       i             ))
        eval(multifloat_ne_func(       i             ))
        eval(multifloat_lt_func(       i             ))
        eval(multifloat_gt_func(       i             ))
        eval(multifloat_le_func(       i             ))
        eval(multifloat_ge_func(       i             ))
        eval(one_pass_renorm_func(     i, sloppy=true))
        eval(multifloat_add_func(      i, sloppy=true))
        eval(multifloat_float_add_func(i, sloppy=true))
        eval(multifloat_mul_func(      i, sloppy=true))
        eval(multifloat_float_mul_func(i, sloppy=true))
        eval(multifloat_div_func(      i, sloppy=true))
        eval(multifloat_sqrt_func(     i, sloppy=true))
    end
    for (_, v) in MultiFloatsCodeGen.MPADD_CACHE
        eval(v)
    end
end

################################################################################

@inline Base.:+(x::T, y::MF{T,N}) where {T<:AF,N} = y + x
@inline Base.:-(x::T, y::MF{T,N}) where {T<:AF,N} = -(y + (-x))
@inline Base.:*(x::T, y::MF{T,N}) where {T<:AF,N} = y * x

@inline Base.:-(x::MF{T,N}) where {T<:AF,N} = MF{T,N}(ntuple(i -> -x._limbs[i], N))
@inline Base.:-(x::MF{T,N}, y::MF{T,N}) where {T<:AF,N} = x + (-y)
@inline Base.:-(x::MF{T,N}, y::T      ) where {T<:AF,N} = x + (-y)

@inline Base.hypot(x::MF{T,N}, y::MF{T,N}) where {T<:AF,N} = sqrt(x*x + y*y)

################################################################################

use_sloppy_multifloat_arithmetic()

end # module MultiFloats
