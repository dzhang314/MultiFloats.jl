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
    x::NTuple{N,T}
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
        ntuple(i -> x.x[i], min(M, N))...,
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

@inline Base.Float16(x::Float64x{N}) where {N} = Float16(x.x[1])
@inline Base.Float32(x::Float64x{N}) where {N} = Float32(x.x[1])

@inline Base.Float16(x::Float16x{N}) where {N} = x.x[1]
@inline Base.Float32(x::Float32x{N}) where {N} = x.x[1]
@inline Base.Float64(x::Float64x{N}) where {N} = x.x[1]

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
    +(ntuple(i -> BigFloat(x.x[N-i+1]), N)...)

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
    total = +(x.x...)
    if isfinite(total)
        while true
            x0::MF{T,N} = x + zero(T)
            if !(x0.x != x.x); break; end
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
    if !isfinite(x.x[1])
        callback(x.x[1])
    else
        i = N
        while (i > 0) && iszero(x.x[i])
            i -= 1
        end
        if iszero(i)
            callback(zero(T))
        else
            setprecision(() -> callback(BigFloat(x)),
                precision(T) + exponent(x.x[1]) - exponent(x.x[i]))
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
    MultiFloat{T,N}(ntuple(i -> a * x.x[i], N))

@inline function Base.ldexp(x::MF{T,N}, n::U) where {T<:AF,N,U<:Integer}
    x = renormalize(x)
    MultiFloat{T,N}(ntuple(i -> ldexp(x.x[i], n), N))
end

################################################################################

@inline Base.zero(::Type{MF{T,N}}) where {T<:AF,N} = MF{T,N}(zero(T)  )
@inline Base.one( ::Type{MF{T,N}}) where {T<:AF,N} = MF{T,N}(one( T)  )
@inline Base.eps( ::Type{MF{T,N}}) where {T<:AF,N} = MF{T,N}(eps( T)^N)

@inline _iszero(x::MF{T,N}) where {T<:AF,N} =
    (&)(ntuple(i -> iszero(x.x[i]), N)...)
@inline _isone( x::MF{T,N}) where {T<:AF,N} =
    isone(x.x[1]) & (&)(ntuple(i -> iszero(x.x[i + 1]), N - 1)...)

@inline Base.iszero(x::MF{T,1}) where {T<:AF  } =  iszero(x.x[1])
@inline Base.isone( x::MF{T,1}) where {T<:AF  } =  isone( x.x[1])
@inline Base.iszero(x::MF{T,N}) where {T<:AF,N} = _iszero(renormalize(x))
@inline Base.isone( x::MF{T,N}) where {T<:AF,N} = _isone( renormalize(x))

################################################################################

@inline Base.precision(::Type{MF{T,N}}) where {T<:AF,N} = N * precision(T)

@inline Base.floatmin(::Type{MF{T,N}}) where {T<:AF,N} = MF{T,N}(floatmin(T))
@inline Base.floatmax(::Type{MF{T,N}}) where {T<:AF,N} = MF{T,N}(floatmax(T))

@inline Base.typemin(::Type{MF{T,N}}) where {T<:AF,N} = MF{T,N}(ntuple(_ -> typemin(T), N))
@inline Base.typemax(::Type{MF{T,N}}) where {T<:AF,N} = MF{T,N}(ntuple(_ -> typemax(T), N))

@inline Base.exponent(   x::MF{T,N}) where {T<:AF,N} = exponent(   renormalize(x).x[1])
@inline Base.signbit(    x::MF{T,N}) where {T<:AF,N} = signbit(    renormalize(x).x[1])
@inline Base.issubnormal(x::MF{T,N}) where {T<:AF,N} = issubnormal(renormalize(x).x[1])
@inline Base.isfinite(   x::MF{T,N}) where {T<:AF,N} = isfinite(   renormalize(x).x[1])
@inline Base.isinf(      x::MF{T,N}) where {T<:AF,N} = isinf(      renormalize(x).x[1])
@inline Base.isnan(      x::MF{T,N}) where {T<:AF,N} = isnan(      renormalize(x).x[1])

import LinearAlgebra: floatmin2
@inline floatmin2(::Type{MF{T,N}}) where {T<:AF,N} =
    MF{T,N}(ldexp(one(T), div(exponent(floatmin(T)) - N * exponent(eps(T)), 2)))

################################################################################

@inline Base.inv(x::MF{T,N}) where {T<:AF,N} = one(MF{T,N}) / x

@inline function Base.abs(x::MF{T,N}) where {T<:AF,N}
    x = renormalize(x)
    ifelse(signbit(x.x[1]), -x, x)
end

@inline function Base.abs2(x::MF{T,N}) where {T<:AF,N}
    x = renormalize(x)
    renormalize(x * x)
end

################################################################################

# To-do list of transcendental math functions to be implemented

Base.exp(  ::MF{T,N}) where {T<:AF,N} = error(  "exp(MultiFloat) not yet implemented")
Base.expm1(::MF{T,N}) where {T<:AF,N} = error("expm1(MultiFloat) not yet implemented")
Base.log(  ::MF{T,N}) where {T<:AF,N} = error(  "log(MultiFloat) not yet implemented")
Base.log2( ::MF{T,N}) where {T<:AF,N} = error( "log2(MultiFloat) not yet implemented")
Base.log10(::MF{T,N}) where {T<:AF,N} = error("log10(MultiFloat) not yet implemented")
Base.log1p(::MF{T,N}) where {T<:AF,N} = error("log1p(MultiFloat) not yet implemented")

Base.sin(::MF{T,N}) where {T<:AF,N} = error("sin(MultiFloat) not yet implemented")
Base.cos(::MF{T,N}) where {T<:AF,N} = error("cos(MultiFloat) not yet implemented")
Base.tan(::MF{T,N}) where {T<:AF,N} = error("tan(MultiFloat) not yet implemented")
Base.sec(::MF{T,N}) where {T<:AF,N} = error("sec(MultiFloat) not yet implemented")
Base.csc(::MF{T,N}) where {T<:AF,N} = error("csc(MultiFloat) not yet implemented")
Base.cot(::MF{T,N}) where {T<:AF,N} = error("cot(MultiFloat) not yet implemented")

Base.sinh(::MF{T,N}) where {T<:AF,N} = error("sinh(MultiFloat) not yet implemented")
Base.cosh(::MF{T,N}) where {T<:AF,N} = error("cosh(MultiFloat) not yet implemented")
Base.tanh(::MF{T,N}) where {T<:AF,N} = error("tanh(MultiFloat) not yet implemented")
Base.sech(::MF{T,N}) where {T<:AF,N} = error("sech(MultiFloat) not yet implemented")
Base.csch(::MF{T,N}) where {T<:AF,N} = error("csch(MultiFloat) not yet implemented")
Base.coth(::MF{T,N}) where {T<:AF,N} = error("coth(MultiFloat) not yet implemented")

Base.sind(::MF{T,N}) where {T<:AF,N} = error("sind(MultiFloat) not yet implemented")
Base.cosd(::MF{T,N}) where {T<:AF,N} = error("cosd(MultiFloat) not yet implemented")
Base.tand(::MF{T,N}) where {T<:AF,N} = error("tand(MultiFloat) not yet implemented")
Base.secd(::MF{T,N}) where {T<:AF,N} = error("secd(MultiFloat) not yet implemented")
Base.cscd(::MF{T,N}) where {T<:AF,N} = error("cscd(MultiFloat) not yet implemented")
Base.cotd(::MF{T,N}) where {T<:AF,N} = error("cotd(MultiFloat) not yet implemented")

Base.asin(::MF{T,N}) where {T<:AF,N} = error("asin(MultiFloat) not yet implemented")
Base.acos(::MF{T,N}) where {T<:AF,N} = error("acos(MultiFloat) not yet implemented")
Base.atan(::MF{T,N}) where {T<:AF,N} = error("atan(MultiFloat) not yet implemented")
Base.asec(::MF{T,N}) where {T<:AF,N} = error("asec(MultiFloat) not yet implemented")
Base.acsc(::MF{T,N}) where {T<:AF,N} = error("acsc(MultiFloat) not yet implemented")
Base.acot(::MF{T,N}) where {T<:AF,N} = error("acot(MultiFloat) not yet implemented")

Base.asinh(::MF{T,N}) where {T<:AF,N} = error("asinh(MultiFloat) not yet implemented")
Base.acosh(::MF{T,N}) where {T<:AF,N} = error("acosh(MultiFloat) not yet implemented")
Base.atanh(::MF{T,N}) where {T<:AF,N} = error("atanh(MultiFloat) not yet implemented")
Base.asech(::MF{T,N}) where {T<:AF,N} = error("asech(MultiFloat) not yet implemented")
Base.acsch(::MF{T,N}) where {T<:AF,N} = error("acsch(MultiFloat) not yet implemented")
Base.acoth(::MF{T,N}) where {T<:AF,N} = error("acoth(MultiFloat) not yet implemented")

Base.asind(::MF{T,N}) where {T<:AF,N} = error("asind(MultiFloat) not yet implemented")
Base.acosd(::MF{T,N}) where {T<:AF,N} = error("acosd(MultiFloat) not yet implemented")
Base.atand(::MF{T,N}) where {T<:AF,N} = error("atand(MultiFloat) not yet implemented")
Base.asecd(::MF{T,N}) where {T<:AF,N} = error("asecd(MultiFloat) not yet implemented")
Base.acscd(::MF{T,N}) where {T<:AF,N} = error("acscd(MultiFloat) not yet implemented")
Base.acotd(::MF{T,N}) where {T<:AF,N} = error("acotd(MultiFloat) not yet implemented")

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

@inline Base.:(==)(x::MF{T,1}, y::MF{T,1}) where {T<:AF} = (x.x[1] == y.x[1])
@inline Base.:(!=)(x::MF{T,1}, y::MF{T,1}) where {T<:AF} = (x.x[1] != y.x[1])
@inline Base.:(< )(x::MF{T,1}, y::MF{T,1}) where {T<:AF} = (x.x[1] <  y.x[1])
@inline Base.:(> )(x::MF{T,1}, y::MF{T,1}) where {T<:AF} = (x.x[1] >  y.x[1])
@inline Base.:(<=)(x::MF{T,1}, y::MF{T,1}) where {T<:AF} = (x.x[1] <= y.x[1])
@inline Base.:(>=)(x::MF{T,1}, y::MF{T,1}) where {T<:AF} = (x.x[1] >= y.x[1])

@inline Base.:+(a::MF{T,1}, b::MF{T,1}) where {T<:AF} = MF{T,1}(a.x[1] + b.x[1])
@inline Base.:+(a::MF{T,1}, b::T      ) where {T<:AF} = MF{T,1}(a.x[1] + b     )
@inline Base.:*(a::MF{T,1}, b::MF{T,1}) where {T<:AF} = MF{T,1}(a.x[1] * b.x[1])
@inline Base.:*(a::MF{T,1}, b::T      ) where {T<:AF} = MF{T,1}(a.x[1] * b     )
@inline Base.:/(a::MF{T,1}, b::MF{T,1}) where {T<:AF} = MF{T,1}(a.x[1] / b.x[1])
@inline _sqrt(  x::MF{T,1}            ) where {T<:AF} = MF{T,1}(unsafe_sqrt(x.x[1]))

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

@inline Base.:-(x::MF{T,N}) where {T<:AF,N} = MF{T,N}(ntuple(i -> -x.x[i], N))
@inline Base.:-(x::MF{T,N}, y::MF{T,N}) where {T<:AF,N} = x + (-y)
@inline Base.:-(x::MF{T,N}, y::T      ) where {T<:AF,N} = x + (-y)

################################################################################

use_sloppy_multifloat_arithmetic()

end # module MultiFloats
