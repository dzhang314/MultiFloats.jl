module MultiFloats

export MultiFloat, Float16x, Float32x, Float64x,
                  Float64x2, Float64x3, Float64x4,
       Float64x5, Float64x6, Float64x7, Float64x8

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

@inline Float64x{1}(x::Int64  ) = Float64x{1}(Float64(x))
@inline Float64x{1}(x::UInt64 ) = Float64x{1}(Float64(x))
@inline Float64x{1}(x::Int128 ) = Float64x{1}(Float64(x))
@inline Float64x{1}(x::UInt128) = Float64x{1}(Float64(x))

@inline function Float64x{2}(x::Int128)
    x0 = Float64(x)
    x1 = Float64(x - Int128(x0))
    Float64x{2}((x0, x1))
end

@inline function Float64x{2}(x::UInt128)
    x0 = Float64(x)
    x1 = Float64(reinterpret(Int128, x - UInt128(x0)))
    Float64x{2}((x0, x1))
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

Base.promote_rule(::Type{Float64x{N}}, ::Type{Float16}) where {N} = Float64x{N}
Base.promote_rule(::Type{Float64x{N}}, ::Type{Float32}) where {N} = Float64x{N}

####################################################################### PRINTING

function Base.show(io::IO, x::MultiFloat{T,N}) where {T<:AF,N}
    x_renorm = x + zero(T)
    while x_renorm.x != x.x
        x = x_renorm
        x_renorm += zero(T)
    end
    i = N
    while (i > 0) && iszero(x.x[i])
        i -= 1
    end
    if iszero(i)
        show(io, zero(T))
    else
        show(io, setprecision(() -> BigFloat(x),
             precision(T) + exponent(x.x[1]) - exponent(x.x[i])))
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

# TODO: Special-case these operations for a single operand.
@inline Base.inv(x::Float64x{N}) where {N} = one(Float64x{N}) / x
@inline Base.abs2(x::Float64x{N}) where {N} = x * x

# TODO: Add accurate comparison operators. Sloppy stop-gap operators for now.
import Base: ==, !=, <, >, <=, >=
@inline ==(x::Float64x{N}, y::Float64x{N}) where {N} =
    (x.x[1] == y.x[1]) & (x.x[2] == y.x[2])
@inline !=(x::Float64x{N}, y::Float64x{N}) where {N} =
    (x.x[1] != y.x[1]) | (x.x[2] != y.x[2])
@inline <(x::Float64x{N}, y::Float64x{N}) where {N} =
    (x.x[1] < y.x[1]) | ((x.x[1] == y.x[1]) & (x.x[2] < y.x[2]))
@inline >(x::Float64x{N}, y::Float64x{N}) where {N} =
    (x.x[1] > y.x[1]) | ((x.x[1] == y.x[1]) & (x.x[2] > y.x[2]))
@inline <=(x::Float64x{N}, y::Float64x{N}) where {N} =
    (x.x[1] < y.x[1]) | ((x.x[1] == y.x[1]) & (x.x[2] <= y.x[2]))
@inline >=(x::Float64x{N}, y::Float64x{N}) where {N} =
    (x.x[1] > y.x[1]) | ((x.x[1] == y.x[1]) & (x.x[2] >= y.x[2]))

# import DZMisc: scale
@inline scale(a::T, x::MultiFloat{T,N}) where {T<:AF,N} =
    MultiFloat{T,N}(ntuple(i -> a * x.x[i], N))

# import DZMisc: dbl
@inline dbl(x::MultiFloat{T,N}) where {T<:AF,N} = scale(T(2.0), x)

################################################################################

@inline Base.zero(::Type{MF{T,N}}) where {T<:AF,N} = MF{T,N}(zero(T))
@inline Base.one( ::Type{MF{T,N}}) where {T<:AF,N} = MF{T,N}(one( T))

@inline Base.iszero(x::MultiFloat{T,N}) where {T<:AF,N} =
    (&)(ntuple(i -> iszero(x.x[i]), N)...)
@inline Base.isone( x::MultiFloat{T,N}) where {T<:AF,N} =
    isone(x.x[1]) & (&)(ntuple(i -> iszero(x.x[i + 1]), N - 1)...)

@inline Base.signbit(    x::MF{T,N}) where {T<:AF,N} = signbit(    x.x[1])
@inline Base.issubnormal(x::MF{T,N}) where {T<:AF,N} = issubnormal(x.x[1])
@inline Base.isfinite(   x::MF{T,N}) where {T<:AF,N} = isfinite(   x.x[1])
@inline Base.isinf(      x::MF{T,N}) where {T<:AF,N} = isinf(      x.x[1])
@inline Base.isnan(      x::MF{T,N}) where {T<:AF,N} = isnan(      x.x[1])

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

for i = 2 : 8
    eval(one_pass_renorm_func(     i, sloppy=true))
    eval(multifloat_add_func(      i, sloppy=true))
    eval(multifloat_float_add_func(i, sloppy=true))
    eval(multifloat_mul_func(      i, sloppy=true))
    eval(multifloat_float_mul_func(i, sloppy=true))
    eval(multifloat_div_func(      i, sloppy=true))
    eval(multifloat_sqrt_func(     i, sloppy=true))
end

for (_, v) in MultiFloatsCodeGen.MPADD_CACHE; eval(v); end

end # module MultiFloats
