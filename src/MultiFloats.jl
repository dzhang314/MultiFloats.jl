module MultiFloats

export MultiFloat, renormalize,
       Float16x, Float32x, Float64x,
       Float64x1, Float64x2, Float64x3, Float64x4,
       Float64x5, Float64x6, Float64x7, Float64x8,
       use_clean_multifloat_arithmetic,
       use_standard_multifloat_arithmetic,
       use_sloppy_multifloat_arithmetic

include("./MultiFloatsCodeGen.jl")
using .MultiFloatsCodeGen

####################################################### DEFINITION OF MULTIFLOAT

struct MultiFloat{T,N} <: AbstractFloat
    _limbs::NTuple{N,T}
end

const Float16x{N} = MultiFloat{Float16,N}
const Float32x{N} = MultiFloat{Float32,N}
const Float64x{N} = MultiFloat{Float64,N}

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

@inline MultiFloat{T,N}(x::MultiFloat{T,N}) where {T,N} = x

@inline MultiFloat{T,N}(x::T) where {T,N} =
    MultiFloat{T,N}((x, ntuple(_ -> zero(T), N - 1)...))

@inline MultiFloat{T,N}(x::MultiFloat{T,M}) where {T,M,N} =
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

function MultiFloat{T,N}(x::BigFloat) where {T,N}
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

function MultiFloat{T,N}(x::BigInt) where {T,N}
    y = Vector{T}(undef, N)
    for i = 1 : N
        y[i] = T(x)
        x -= BigInt(y[i])
    end
    MultiFloat{T,N}((y...,))
end

MultiFloat{T,N}(x::Rational{U}) where {T,N,U} =
    MF{T,N}(numerator(x)) / MF{T,N}(denominator(x))

######################################################## CONVERSION TO BIG TYPES

Base.BigFloat(x::MultiFloat{T,N}) where {T,N} =
    +(ntuple(i -> BigFloat(x._limbs[N-i+1]), N)...)

Base.Rational{BigInt}(x::MultiFloat{T,N}) where {T,N} =
    sum(Rational{BigInt}.(x._limbs))

################################################################ PROMOTION RULES

Base.promote_rule(::Type{MF{T,N}}, ::Type{T      }) where {T,N} = MF{T,N}
Base.promote_rule(::Type{MF{T,N}}, ::Type{Int8   }) where {T,N} = MF{T,N}
Base.promote_rule(::Type{MF{T,N}}, ::Type{Int16  }) where {T,N} = MF{T,N}
Base.promote_rule(::Type{MF{T,N}}, ::Type{Int32  }) where {T,N} = MF{T,N}
Base.promote_rule(::Type{MF{T,N}}, ::Type{Int64  }) where {T,N} = MF{T,N}
Base.promote_rule(::Type{MF{T,N}}, ::Type{Int128 }) where {T,N} = MF{T,N}
Base.promote_rule(::Type{MF{T,N}}, ::Type{Bool   }) where {T,N} = MF{T,N}
Base.promote_rule(::Type{MF{T,N}}, ::Type{UInt8  }) where {T,N} = MF{T,N}
Base.promote_rule(::Type{MF{T,N}}, ::Type{UInt16 }) where {T,N} = MF{T,N}
Base.promote_rule(::Type{MF{T,N}}, ::Type{UInt32 }) where {T,N} = MF{T,N}
Base.promote_rule(::Type{MF{T,N}}, ::Type{UInt64 }) where {T,N} = MF{T,N}
Base.promote_rule(::Type{MF{T,N}}, ::Type{UInt128}) where {T,N} = MF{T,N}

Base.promote_rule(::Type{MF{T,N}}, ::Type{BigFloat}) where {T,N} = BigFloat

Base.promote_rule(::Type{Float32x{N}}, ::Type{Float16}) where {N} = Float32x{N}
Base.promote_rule(::Type{Float64x{N}}, ::Type{Float16}) where {N} = Float64x{N}
Base.promote_rule(::Type{Float64x{N}}, ::Type{Float32}) where {N} = Float64x{N}

####################################################################### PRINTING

@inline function renormalize(x::MF{T,N}) where {T,N}
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

function call_normalized(callback, x::MultiFloat{T,N}) where {T,N}
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

function Base.show(io::IO, x::MultiFloat{T,N}) where {T,N}
    call_normalized(y -> show(io, y), x)
end

################################################################# PRINTF SUPPORT

# Thanks to Greg Plowman (https://github.com/GregPlowman) for suggesting
# implementations of Printf.fix_dec and Printf.ini_dec for @printf support.

import Printf: fix_dec, ini_dec

if VERSION < v"1.1"

    fix_dec(out, x::MultiFloat{T,N}, flags::String, width::Int,
            precision::Int, c::Char) where {T,N} =
        call_normalized(d -> fix_dec(out, BigFloat(d), flags,
                                     width, precision, c), x)

    ini_dec(out, x::MultiFloat{T,N}, ndigits::Int, flags::String,
            width::Int, precision::Int, c::Char) where {T,N} =
        call_normalized(d -> ini_dec(out, BigFloat(d), ndigits,
                                     flags, width, precision, c), x)

else

    fix_dec(out, x::MultiFloat{T,N}, flags::String, width::Int,
            precision::Int, c::Char, digits) where {T,N} =
        call_normalized(d -> fix_dec(out, BigFloat(d), flags,
                                     width, precision, c, digits), x)

    ini_dec(out, x::MultiFloat{T,N}, ndigits::Int, flags::String,
            width::Int, precision::Int, c::Char, digits) where {T,N} =
        call_normalized(d -> ini_dec(out, BigFloat(d), ndigits,
                                     flags, width, precision, c, digits), x)

end

######################################################## EXPONENTIATION (BASE 2)

@inline scale(a::T, x::MultiFloat{T,N}) where {T,N} =
    MultiFloat{T,N}(ntuple(i -> a * x._limbs[i], N))

@inline function Base.ldexp(x::MF{T,N}, n::U) where {T,N,U<:Integer}
    x = renormalize(x)
    MultiFloat{T,N}(ntuple(i -> ldexp(x._limbs[i], n), N))
end

######################################################## EXPONENTIATION (BASE E)

const INVERSE_FACTORIALS_F64 = setprecision(() ->
    [MultiFloat{Float64,20}(inv(BigFloat(factorial(BigInt(i)))))
        for i = 1 : 170],
    BigFloat, 1200)

const LOG2_F64 = setprecision(() ->
    MultiFloat{Float64,20}(log(BigFloat(2))),
    BigFloat, 1200)

log2_f64_literal(n::Int) = :(
    MultiFloat{Float64,$n}($(LOG2_F64._limbs[1:n])))

inverse_factorial_f64_literal(n::Int, i::Int) = :(
    MultiFloat{Float64,$n}($(INVERSE_FACTORIALS_F64[i]._limbs[1:n])))

meta_y(n::Int) = Symbol("y_", n)

meta_y_definition(n::Int) = Expr(:(=), meta_y(n),
    Expr(:call, :*, meta_y(div(n, 2)), meta_y(div(n+1, 2))))

meta_exp_term(n::Int, i::Int) = :(
    $(Symbol("y_", i)) * $(inverse_factorial_f64_literal(n, i)))

function multifloat_exp_func(n::Int, num_terms::Int,
                             reduction_power::Int; sloppy::Bool=false)
    return Expr(:function,
        :(multifloat_exp(x::MultiFloat{Float64,$n})),
        Expr(:block,
            :(exponent_f = Base.rint_llvm(
                x._limbs[1] / $(LOG2_F64._limbs[1]))),
            :(exponent_i = Base.fptosi(Int, exponent_f)),
            :(y_1 = scale($(ldexp(1.0, -reduction_power)),
                MultiFloat{Float64,$n}(MultiFloat{Float64,$(n + !sloppy)}(x) -
                    exponent_f * $(log2_f64_literal(n + !sloppy))))),
            [meta_y_definition(i) for i = 2 : num_terms]...,
            :(exp_y = $(meta_exp_term(n, num_terms))),
            [:(exp_y += $(meta_exp_term(n, i)))
                for i = num_terms-1 : -1 : 3]...,
            :(exp_y += scale(0.5, y_2)),
            :(exp_y += y_1),
            :(exp_y += 1.0),
            [:(exp_y *= exp_y) for _ = 1 : reduction_power]...,
            :(return scale(reinterpret(Float64,
                UInt64(1023 + exponent_i) << 52), exp_y))))
end

################################################################################

@inline Base.zero(::Type{MF{T,N}}) where {T,N} = MF{T,N}(zero(T)  )
@inline Base.one( ::Type{MF{T,N}}) where {T,N} = MF{T,N}(one( T)  )
@inline Base.eps( ::Type{MF{T,N}}) where {T,N} = MF{T,N}(eps( T)^N)

################################################### FLOATING-POINT INTROSPECTION

@inline Base.precision(::Type{MF{T,N}}) where {T,N} = N * precision(T) + (N - 1)

@inline _iszero(x::MF{T,N}) where {T,N} = (&)(ntuple(i -> iszero(x._limbs[i]), N)...)
@inline _isone( x::MF{T,N}) where {T,N} = isone(x._limbs[1]) & (&)(ntuple(i -> iszero(x._limbs[i + 1]), N - 1)...)

@inline Base.iszero(x::MF{T,1}) where {T  } =  iszero(x._limbs[1])
@inline Base.isone( x::MF{T,1}) where {T  } =  isone( x._limbs[1])
@inline Base.iszero(x::MF{T,N}) where {T,N} = _iszero(renormalize(x))
@inline Base.isone( x::MF{T,N}) where {T,N} = _isone( renormalize(x))

# TODO: This is technically not the maximum/minimum representable MultiFloat.
@inline Base.floatmin(::Type{MF{T,N}}) where {T,N} = MF{T,N}(floatmin(T))
@inline Base.floatmax(::Type{MF{T,N}}) where {T,N} = MF{T,N}(floatmax(T))

@inline Base.typemin(::Type{MF{T,N}}) where {T,N} = MF{T,N}(ntuple(_ -> typemin(T), N))
@inline Base.typemax(::Type{MF{T,N}}) where {T,N} = MF{T,N}(ntuple(_ -> typemax(T), N))

@inline Base.exponent(   x::MF{T,N}) where {T,N} = exponent(   renormalize(x)._limbs[1])
@inline Base.signbit(    x::MF{T,N}) where {T,N} = signbit(    renormalize(x)._limbs[1])
@inline Base.issubnormal(x::MF{T,N}) where {T,N} = issubnormal(renormalize(x)._limbs[1])
@inline Base.isfinite(   x::MF{T,N}) where {T,N} = isfinite(   renormalize(x)._limbs[1])
@inline Base.isinf(      x::MF{T,N}) where {T,N} = isinf(      renormalize(x)._limbs[1])
@inline Base.isnan(      x::MF{T,N}) where {T,N} = isnan(      renormalize(x)._limbs[1])
@inline Base.isinteger(  x::MF{T,N}) where {T,N} = all(isinteger.(renormalize(x)._limbs))

@inline function Base.nextfloat(x::MF{T,N}) where {T,N}
    y = renormalize(x)
    MF{T,N}((ntuple(i -> y._limbs[i], N - 1)..., nextfloat(y._limbs[N])))
end

@inline function Base.prevfloat(x::MF{T,N}) where {T,N}
    y = renormalize(x)
    MF{T,N}((ntuple(i -> y._limbs[i], N - 1)..., prevfloat(y._limbs[N])))
end

import LinearAlgebra: floatmin2
@inline floatmin2(::Type{MF{T,N}}) where {T,N} =
    MF{T,N}(ldexp(one(T), div(exponent(floatmin(T)) - N * exponent(eps(T)), 2)))

################################################################################

@inline Base.inv(x::MF{T,N}) where {T,N} = one(MF{T,N}) / x

@inline function Base.abs(x::MF{T,N}) where {T,N}
    x = renormalize(x)
    ifelse(signbit(x._limbs[1]), -x, x)
end

@inline function Base.abs2(x::MF{T,N}) where {T,N}
    x = renormalize(x)
    renormalize(x * x)
end

################################################################################

BASE_TRANSCENDENTAL_FUNCTIONS = [
    :exp2, :exp10, :expm1, :log, :log2, :log10, :log1p,
    :sin, :cos, :tan, :sec, :csc, :cot,
    :sinh, :cosh, :tanh, :sech, :csch, :coth,
    :sind, :cosd, :tand, :secd, :cscd, :cotd,
    :asin, :acos, :atan, :asec, :acsc, :acot,
    :asinh, :acosh, :atanh, :asech, :acsch, :acoth,
    :asind, :acosd, :atand, :asecd, :acscd, :acotd
]

for name in BASE_TRANSCENDENTAL_FUNCTIONS
    eval(:(Base.$name(x::MultiFloat{T,N}) where {T,N} = error($(
        "$name(MultiFloat) is not yet implemented. For a temporary workaround,\n" *
        "call MultiFloats.use_bigfloat_transcendentals() immediately after\n" *
        "importing MultiFloats. This will use the BigFloat implementation of\n" *
        "$name, which will not be as fast as a pure-MultiFloat implementation.\n"
    ))))
end

eval_bigfloat(f::Function, x::MultiFloat{T,N}, k::Int) where {T,N} =
    setprecision(() -> MultiFloat{T,N}(f(BigFloat(x))),
                 BigFloat, precision(MultiFloat{T,N}) + k)

function use_bigfloat_transcendentals(k::Int=20)
    for name in BASE_TRANSCENDENTAL_FUNCTIONS
        eval(:(Base.$name(x::MultiFloat{T,N}) where {T,N} =
                eval_bigfloat($name, x, $k)))
    end
end

################################################################################

@inline function two_sum(a::T, b::T) where {T}
    s = a + b
    v = s - a
    s, (a - (s - v)) + (b - v)
end

@inline function quick_two_sum(a::T, b::T) where {T}
    s = a + b
    s, b - (s - a)
end

@inline function two_prod(a::T, b::T) where {T}
    p = a * b
    p, fma(a, b, -p)
end

################################################################################

@inline unsafe_sqrt(x::Float32) = Base.sqrt_llvm(x)
@inline unsafe_sqrt(x::Float64) = Base.sqrt_llvm(x)
@inline unsafe_sqrt(x::T) where {T <: Real} = sqrt(x)

################################################################################

import Random
using Random: AbstractRNG, SamplerTrivial, CloseOpen01

@inline function _rand_f64(rng::AbstractRNG, k::Int)
    expnt = reinterpret(UInt64,
        exponent(floatmax(Float64)) + k) << (precision(Float64) - 1)
    mntsa = rand(rng, Random.UInt52())
    return reinterpret(Float64, expnt | mntsa)
end

@inline function _rand_sf64(rng::AbstractRNG, k::Int)
    expnt = reinterpret(UInt64,
        exponent(floatmax(Float64)) + k) << (precision(Float64) - 1)
    mntsa = rand(rng, UInt64) & 0x800FFFFFFFFFFFFF
    return reinterpret(Float64, expnt | mntsa)
end

@inline function _rand_mf64(rng::AbstractRNG, offset::Int,
                            padding::NTuple{N,Int}) where {N}
    exponents = (cumsum(padding) .+
        (precision(Float64) + 1) .* ntuple(identity, N))
    return Float64x{N+1}((_rand_f64(rng, offset),
                          _rand_sf64.(rng, offset .- exponents)...))
end

function multifloat_rand_func(n::Int)
    :(
        Random.rand(rng::AbstractRNG,
                    ::SamplerTrivial{CloseOpen01{Float64x{$n}}}) =
        _rand_mf64(
            rng,
            -leading_zeros(rand(rng, UInt64)) - 1,
            $(Expr(:tuple,
                   ntuple(_ -> :(leading_zeros(rand(rng, UInt64))), n - 1)...
            ))
        )
    )
end

function Base.exp(x::MultiFloat{T,N}) where {T,N}
    x = renormalize(x)
    if x._limbs[1] >= log(floatmax(Float64))
        return typemax(MultiFloat{T,N})
    elseif x._limbs[1] <= log(floatmin(Float64))
        return zero(MultiFloat{T,N})
    else
        return multifloat_exp(x)
    end
end

################################################################################

@inline multifloat_add(a::MF{T,1}, b::MF{T,1}) where {T} = MF{T,1}(a._limbs[1] + b._limbs[1])
@inline multifloat_mul(a::MF{T,1}, b::MF{T,1}) where {T} = MF{T,1}(a._limbs[1] * b._limbs[1])
@inline multifloat_div(a::MF{T,1}, b::MF{T,1}) where {T} = MF{T,1}(a._limbs[1] / b._limbs[1])
@inline multifloat_float_add(a::MF{T,1}, b::T) where {T} = MF{T,1}(a._limbs[1] + b)
@inline multifloat_float_mul(a::MF{T,1}, b::T) where {T} = MF{T,1}(a._limbs[1] * b)
@inline multifloat_sqrt(x::MF{T,1}) where {T} = MF{T,1}(unsafe_sqrt(x._limbs[1]))
@inline multifloat_exp(x::MF{T,1}) where {T} = MF{T,1}(exp(x._limbs[1]))

function use_clean_multifloat_arithmetic(n::Integer=8)
    for i = 1 : n
        eval(multifloat_eq_func(i))
        eval(multifloat_ne_func(i))
        eval(multifloat_lt_func(i))
        eval(multifloat_gt_func(i))
        eval(multifloat_le_func(i))
        eval(multifloat_ge_func(i))
        eval(multifloat_rand_func(i))
    end
    for i = 2 : n+1
        eval(two_pass_renorm_func(     i, sloppy=false))
        eval(multifloat_add_func(      i, sloppy=false))
        eval(multifloat_mul_func(      i, sloppy=false))
        eval(multifloat_div_func(      i, sloppy=false))
        eval(multifloat_float_add_func(i, sloppy=false))
        eval(multifloat_float_mul_func(i, sloppy=false))
        eval(multifloat_sqrt_func(     i, sloppy=false))
    end
    eval(MultiFloats.multifloat_exp_func(2, 20, 1, sloppy=false))
    eval(MultiFloats.multifloat_exp_func(3, 28, 1, sloppy=false))
    eval(MultiFloats.multifloat_exp_func(4, 35, 1, sloppy=false))
    eval(MultiFloats.multifloat_exp_func(5, 42, 1, sloppy=false))
    eval(MultiFloats.multifloat_exp_func(6, 49, 1, sloppy=false))
    eval(MultiFloats.multifloat_exp_func(7, 56, 1, sloppy=false))
    eval(MultiFloats.multifloat_exp_func(8, 63, 1, sloppy=false))
    for (_, v) in MultiFloatsCodeGen.MPADD_CACHE
        eval(v)
    end
end

function use_standard_multifloat_arithmetic(n::Integer=8)
    for i = 1 : n
        eval(multifloat_eq_func(i))
        eval(multifloat_ne_func(i))
        eval(multifloat_lt_func(i))
        eval(multifloat_gt_func(i))
        eval(multifloat_le_func(i))
        eval(multifloat_ge_func(i))
        eval(multifloat_rand_func(i))
    end
    for i = 2 : n
        eval(two_pass_renorm_func(     i, sloppy=true ))
        eval(two_pass_renorm_func(     i, sloppy=false))
        eval(multifloat_add_func(      i, sloppy=false))
        eval(multifloat_mul_func(      i, sloppy=true ))
        eval(multifloat_div_func(      i, sloppy=true ))
        eval(multifloat_float_add_func(i, sloppy=false))
        eval(multifloat_float_mul_func(i, sloppy=true ))
        eval(multifloat_sqrt_func(     i, sloppy=true ))
    end
    eval(MultiFloats.multifloat_exp_func(2, 17, 2, sloppy=false))
    eval(MultiFloats.multifloat_exp_func(3, 19, 4, sloppy=true))
    eval(MultiFloats.multifloat_exp_func(4, 20, 6, sloppy=true))
    eval(MultiFloats.multifloat_exp_func(5, 23, 7, sloppy=true))
    eval(MultiFloats.multifloat_exp_func(6, 23, 9, sloppy=true))
    eval(MultiFloats.multifloat_exp_func(7, 22, 12, sloppy=true))
    eval(MultiFloats.multifloat_exp_func(8, 24, 13, sloppy=true))
    for (_, v) in MultiFloatsCodeGen.MPADD_CACHE
        eval(v)
    end
end

function use_sloppy_multifloat_arithmetic(n::Integer=8)
    for i = 1 : n
        eval(multifloat_eq_func(i))
        eval(multifloat_ne_func(i))
        eval(multifloat_lt_func(i))
        eval(multifloat_gt_func(i))
        eval(multifloat_le_func(i))
        eval(multifloat_ge_func(i))
        eval(multifloat_rand_func(i))
    end
    for i = 2 : n
        eval(one_pass_renorm_func(     i, sloppy=true))
        eval(multifloat_add_func(      i, sloppy=true))
        eval(multifloat_mul_func(      i, sloppy=true))
        eval(multifloat_div_func(      i, sloppy=true))
        eval(multifloat_float_add_func(i, sloppy=true))
        eval(multifloat_float_mul_func(i, sloppy=true))
        eval(multifloat_sqrt_func(     i, sloppy=true))
    end
    eval(MultiFloats.multifloat_exp_func(2, 17, 2, sloppy=false))
    eval(MultiFloats.multifloat_exp_func(3, 19, 4, sloppy=true))
    eval(MultiFloats.multifloat_exp_func(4, 20, 6, sloppy=true))
    eval(MultiFloats.multifloat_exp_func(5, 23, 7, sloppy=true))
    eval(MultiFloats.multifloat_exp_func(6, 23, 9, sloppy=true))
    eval(MultiFloats.multifloat_exp_func(7, 22, 12, sloppy=true))
    eval(MultiFloats.multifloat_exp_func(8, 24, 13, sloppy=true))
    for (_, v) in MultiFloatsCodeGen.MPADD_CACHE
        eval(v)
    end
end

################################################################################

@inline Base.:(==)(x::MF{T,N}, y::MF{T,N}) where {T,N} = multifloat_eq(renormalize(x), renormalize(y))
@inline Base.:(!=)(x::MF{T,N}, y::MF{T,N}) where {T,N} = multifloat_ne(renormalize(x), renormalize(y))
@inline Base.:(< )(x::MF{T,N}, y::MF{T,N}) where {T,N} = multifloat_lt(renormalize(x), renormalize(y))
@inline Base.:(> )(x::MF{T,N}, y::MF{T,N}) where {T,N} = multifloat_gt(renormalize(x), renormalize(y))
@inline Base.:(<=)(x::MF{T,N}, y::MF{T,N}) where {T,N} = multifloat_le(renormalize(x), renormalize(y))
@inline Base.:(>=)(x::MF{T,N}, y::MF{T,N}) where {T,N} = multifloat_ge(renormalize(x), renormalize(y))

@inline Base.:+(x::MF{T,N}, y::MF{T,N}) where {T,N} = multifloat_add(x, y)
@inline Base.:*(x::MF{T,N}, y::MF{T,N}) where {T,N} = multifloat_mul(x, y)
@inline Base.:/(x::MF{T,N}, y::MF{T,N}) where {T,N} = multifloat_div(x, y)
@inline Base.:+(x::MF{T,N}, y::T) where {T        ,N} = multifloat_float_add(x, y)
@inline Base.:+(x::MF{T,N}, y::T) where {T<:Number,N} = multifloat_float_add(x, y)
@inline Base.:*(x::MF{T,N}, y::T) where {T        ,N} = multifloat_float_mul(x, y)
@inline Base.:*(x::MF{T,N}, y::T) where {T<:Number,N} = multifloat_float_mul(x, y)

@inline function Base.sqrt(x::MF{T,N}) where {T,N}
    x = renormalize(x)
    if iszero(x)
        return x
    else
        return multifloat_sqrt(x)
    end
end

@inline Base.:+(x::T, y::MF{T,N}) where {T        ,N} = y + x
@inline Base.:+(x::T, y::MF{T,N}) where {T<:Number,N} = y + x
@inline Base.:-(x::T, y::MF{T,N}) where {T        ,N} = -(y + (-x))
@inline Base.:-(x::T, y::MF{T,N}) where {T<:Number,N} = -(y + (-x))
@inline Base.:*(x::T, y::MF{T,N}) where {T        ,N} = y * x
@inline Base.:*(x::T, y::MF{T,N}) where {T<:Number,N} = y * x

@inline Base.:-(x::MF{T,N}) where {T,N} = MF{T,N}(ntuple(i -> -x._limbs[i], N))
@inline Base.:-(x::MF{T,N}, y::MF{T,N}) where {T        ,N} = x + (-y)
@inline Base.:-(x::MF{T,N}, y::T      ) where {T        ,N} = x + (-y)
@inline Base.:-(x::MF{T,N}, y::T      ) where {T<:Number,N} = x + (-y)

@inline Base.hypot(x::MF{T,N}, y::MF{T,N}) where {T,N} = sqrt(x*x + y*y)

################################################################################

use_standard_multifloat_arithmetic()

end # module MultiFloats
